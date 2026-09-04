"""Offline Go2 leg kinematics derived from the pinned Unitree URDF.

This module deliberately stops at an offline prior.  It does not import the
Unitree SDK, publish ``LowCmd``, or claim that the URDF frame, joint zero, and
joint directions have been verified against a physical Go2.  Every geometric
translation, rotation, axis, and limit used by forward/inverse kinematics is
read from the same hash-checked URDF used to build
:class:`Go2UrdfMassProperties`.

中文说明：四条腿的 FK/Jacobian/IK 均从同一哈希锁定 URDF 构建，避免手写连杆长度
与质量模型版本不一致。IK 使用多初值阻尼迭代并限制在 URDF 机械范围；这些范围不是
已验证的 LowCmd 安全范围。B 到 URDF root、SDK 零位和正方向完成实机核对前，本
模块只允许离线使用。
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerogo2.landing.impact_aware.go2_urdf import (
    MAX_URDF_BYTES,
    Go2UrdfError,
    Go2UrdfJointLimit,
    Go2UrdfMassProperties,
)
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FootLeverArmsFromComBody,
    FootPositionsFromBodyOriginB,
    foot_positions_from_body_origin_B_to_com_lever_arms,
    validate_four_foot_leg_order,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]

# Compatibility export.  The tuple itself is defined once in ``types`` so
# kinematics, force adapters, and dynamics assembly can share one identity.
SDK_LEG_ORDER: Tuple[str, str, str, str] = GO2_SDK_LEG_ORDER
JOINT_ROLE_ORDER: Tuple[str, ...] = ("hip", "thigh", "calf")
OFFLINE_PRIOR_ONLY = True
HARDWARE_VALIDATED = False

_FOOT_ROLE = "foot"
_IK_POSITION_TOLERANCE_M = 1.0e-7
_IK_MAX_ITERATIONS = 100
_IK_MAX_STEP_RAD = 0.35
_IK_INITIAL_DAMPING = 1.0e-5
_IK_SOLUTION_EQUIVALENCE_TOLERANCE_RAD = 1.0e-6


class Go2KinematicsError(ValueError):
    """Raised when a URDF leg chain or a requested kinematic state is invalid."""


def _readonly(value: object, shape: Tuple[int, ...], name: str) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise Go2KinematicsError(f"{name} must be a real numeric array with shape {shape}") from exc
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise Go2KinematicsError(f"{name} must be a real numeric array with shape {shape}")
    result = cast(FloatArray, np.array(raw, dtype=float, copy=True))
    if not np.all(np.isfinite(result)):
        raise Go2KinematicsError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_copy(value: FloatArray) -> FloatArray:
    result = cast(FloatArray, np.array(value, dtype=float, copy=True))
    result.setflags(write=False)
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise Go2KinematicsError(f"{name} must be finite")
    if isinstance(value, Real):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError as exc:
            raise Go2KinematicsError(f"{name} must be finite") from exc
    else:
        raise Go2KinematicsError(f"{name} must be finite")
    if not math.isfinite(result):
        raise Go2KinematicsError(f"{name} must be finite")
    return result


def _vector3(text: Optional[str], name: str, *, default: Sequence[float]) -> FloatArray:
    if text is None:
        return _readonly(default, (3,), name)
    fields = text.split()
    if len(fields) != 3:
        raise Go2KinematicsError(f"{name} must contain exactly three values")
    return _readonly([_finite(field, name) for field in fields], (3,), name)


def _rotation_from_rpy(rpy: FloatArray) -> FloatArray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cast(
        FloatArray,
        np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        ),
    )


def _axis_angle(axis: FloatArray, angle_rad: float) -> FloatArray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise Go2KinematicsError("movable joint axis must be nonzero")
    x, y, z = axis / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    d = 1.0 - c
    return cast(
        FloatArray,
        np.array(
            [
                [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
                [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
                [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
            ],
            dtype=float,
        ),
    )


def _normalized_hash(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized.startswith("sha256:"):
        normalized = "sha256:" + normalized
    digest = normalized[len("sha256:") :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Go2KinematicsError("mass-property source_sha256 is malformed")
    return normalized


@dataclass(frozen=True)
class Go2UrdfKinematicJoint:
    """One joint in a leg chain, with geometry copied from the pinned URDF."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    translation_parent_m: FloatArray
    rotation_joint_to_parent: FloatArray
    axis_joint: FloatArray
    lower_rad: Optional[float]
    upper_rad: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_parent_m",
            _readonly(self.translation_parent_m, (3,), f"{self.name}.translation"),
        )
        object.__setattr__(
            self,
            "rotation_joint_to_parent",
            _readonly(self.rotation_joint_to_parent, (3, 3), f"{self.name}.rotation"),
        )
        object.__setattr__(
            self,
            "axis_joint",
            _readonly(self.axis_joint, (3,), f"{self.name}.axis"),
        )


@dataclass(frozen=True)
class _ParsedJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    translation_parent_m: FloatArray
    rotation_joint_to_parent: FloatArray
    axis_joint: FloatArray
    lower_rad: Optional[float]
    upper_rad: Optional[float]


def _optional_limit(element: Optional[ET.Element], key: str, name: str) -> Optional[float]:
    if element is None or key not in element.attrib:
        return None
    return _finite(element.attrib[key], name)


def _parse_joint(element: ET.Element) -> _ParsedJoint:
    name = element.attrib.get("name", "").strip()
    joint_type = element.attrib.get("type", "").strip()
    parent_element = element.find("parent")
    child_element = element.find("child")
    if not name or joint_type not in {"fixed", "revolute"}:
        raise Go2KinematicsError(f"unsupported leg joint {name!r}/{joint_type!r}")
    if parent_element is None or child_element is None:
        raise Go2KinematicsError(f"joint {name} has incomplete parent/child data")
    parent = parent_element.attrib.get("link", "").strip()
    child = child_element.attrib.get("link", "").strip()
    if not parent or not child or parent == child:
        raise Go2KinematicsError(f"joint {name} has invalid parent/child data")
    origin = element.find("origin")
    translation = _vector3(
        None if origin is None else origin.attrib.get("xyz"),
        f"{name}.origin.xyz",
        default=(0.0, 0.0, 0.0),
    )
    rpy = _vector3(
        None if origin is None else origin.attrib.get("rpy"),
        f"{name}.origin.rpy",
        default=(0.0, 0.0, 0.0),
    )
    axis_element = element.find("axis")
    axis = _vector3(
        None if axis_element is None else axis_element.attrib.get("xyz"),
        f"{name}.axis",
        default=(1.0, 0.0, 0.0),
    )
    limit = element.find("limit")
    lower = _optional_limit(limit, "lower", f"{name}.limit.lower")
    upper = _optional_limit(limit, "upper", f"{name}.limit.upper")
    if joint_type == "revolute":
        if lower is None or upper is None or lower >= upper:
            raise Go2KinematicsError(f"joint {name} has invalid position limits")
        if float(np.linalg.norm(axis)) <= 0.0:
            raise Go2KinematicsError(f"joint {name} has a zero axis")
    return _ParsedJoint(
        name=name,
        joint_type=joint_type,
        parent_link=parent,
        child_link=child,
        translation_parent_m=translation,
        rotation_joint_to_parent=_rotation_from_rpy(rpy),
        axis_joint=axis,
        lower_rad=lower,
        upper_rad=upper,
    )


def _matches_limit(parsed: _ParsedJoint, verified: Go2UrdfJointLimit) -> bool:
    return (
        parsed.joint_type == verified.joint_type
        and np.allclose(parsed.axis_joint, verified.axis_joint, rtol=0.0, atol=1.0e-12)
        and parsed.lower_rad == verified.lower_rad
        and parsed.upper_rad == verified.upper_rad
    )


class Go2LegKinematics:
    """Three-DOF FK/IK for one leg, expressed in the user-defined B axes.

    ``inverse`` accepts the current/previous joint command as
    ``preferred_q_rad``.  All converged branches are checked and the branch
    nearest that seed is returned, preventing discontinuous branch switches
    along a Cartesian trajectory.  Omitting the seed preserves the historical
    behaviour by using the pinned reference pose as the preference.  Every
    result is rejected unless forward substitution reaches the target within a
    strict metric tolerance.
    """

    offline_prior_only = OFFLINE_PRIOR_ONLY
    hardware_validated = HARDWARE_VALIDATED

    def __init__(
        self,
        *,
        leg_name: str,
        movable_joints: Tuple[Go2UrdfKinematicJoint, ...],
        foot_joint: Go2UrdfKinematicJoint,
        home_q_rad: ArrayLike,
        urdf_root_from_body_origin_B_m: ArrayLike,
    ) -> None:
        if leg_name not in SDK_LEG_ORDER:
            raise Go2KinematicsError(f"unknown Go2 leg {leg_name!r}")
        if len(movable_joints) != 3 or any(
            joint.joint_type != "revolute" for joint in movable_joints
        ):
            raise Go2KinematicsError(f"{leg_name} must contain three revolute joints")
        if foot_joint.joint_type != "fixed":
            raise Go2KinematicsError(f"{leg_name} foot joint must be fixed")
        self._leg_name = leg_name
        self._movable_joints = movable_joints
        self._foot_joint = foot_joint
        self._root_from_B = _readonly(
            urdf_root_from_body_origin_B_m,
            (3,),
            "urdf_root_from_body_origin_B_m",
        )
        lower = [joint.lower_rad for joint in movable_joints]
        upper = [joint.upper_rad for joint in movable_joints]
        if any(value is None for value in lower + upper):
            raise Go2KinematicsError(f"{leg_name} movable joints require finite limits")
        self._lower = _readonly(cast(Sequence[float], lower), (3,), f"{leg_name}.lower")
        self._upper = _readonly(cast(Sequence[float], upper), (3,), f"{leg_name}.upper")
        self._home = self._checked_q(home_q_rad, "home_q_rad")

    @property
    def leg_name(self) -> str:
        return self._leg_name

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple(joint.name for joint in self._movable_joints)

    @property
    def joints(self) -> Tuple[Go2UrdfKinematicJoint, ...]:
        return self._movable_joints + (self._foot_joint,)

    @property
    def lower_rad(self) -> FloatArray:
        return _readonly_copy(self._lower)

    @property
    def upper_rad(self) -> FloatArray:
        return _readonly_copy(self._upper)

    @property
    def home_q_rad(self) -> FloatArray:
        return _readonly_copy(self._home)

    @property
    def urdf_root_from_body_origin_B_m(self) -> FloatArray:
        return _readonly_copy(self._root_from_B)

    def _checked_q(self, value: ArrayLike, name: str) -> FloatArray:
        q = _readonly(value, (3,), name)
        if np.any(q < self._lower - 1.0e-12) or np.any(q > self._upper + 1.0e-12):
            raise Go2KinematicsError(f"{self._leg_name} {name} violates URDF joint limits")
        return q

    def _forward_and_jacobian(self, q: FloatArray) -> Tuple[FloatArray, FloatArray]:
        rotation = cast(FloatArray, np.eye(3, dtype=float))
        position = cast(FloatArray, np.array(self._root_from_B, dtype=float, copy=True))
        joint_origins = []
        joint_axes = []
        for joint, angle in zip(self._movable_joints, q):
            position = position + rotation @ joint.translation_parent_m
            rotation_at_zero = rotation @ joint.rotation_joint_to_parent
            axis_B = rotation_at_zero @ joint.axis_joint
            axis_B = axis_B / float(np.linalg.norm(axis_B))
            joint_origins.append(position.copy())
            joint_axes.append(axis_B)
            rotation = rotation_at_zero @ _axis_angle(joint.axis_joint, float(angle))
        position = position + rotation @ self._foot_joint.translation_parent_m
        rotation = rotation @ self._foot_joint.rotation_joint_to_parent
        del rotation
        jacobian = np.column_stack(
            [np.cross(axis, position - origin) for origin, axis in zip(joint_origins, joint_axes)]
        )
        return (
            _readonly(position, (3,), "foot_position_body"),
            _readonly(jacobian, (3, 3), "foot_position_jacobian"),
        )

    def forward(self, q_rad: ArrayLike) -> FloatArray:
        """Return this leg's URDF foot origin in B coordinates."""

        q = self._checked_q(q_rad, "q_rad")
        foot, _ = self._forward_and_jacobian(q)
        return foot

    def jacobian(self, q_rad: ArrayLike) -> FloatArray:
        """Return the translational foot Jacobian in B coordinates."""

        q = self._checked_q(q_rad, "q_rad")
        _, jacobian = self._forward_and_jacobian(q)
        return jacobian

    def _candidate_seeds(self, preferred_q_rad: FloatArray) -> Tuple[FloatArray, ...]:
        span = self._upper - self._lower
        fractions = (0.12, 0.5, 0.88)
        seeds = [preferred_q_rad, self._home, 0.5 * (self._lower + self._upper)]
        for first in fractions:
            for second in fractions:
                for third in fractions:
                    seeds.append(self._lower + span * np.array([first, second, third]))
        unique: List[FloatArray] = []
        for seed in seeds:
            checked = _readonly(seed, (3,), "IK seed")
            if not any(
                np.allclose(
                    checked,
                    existing,
                    rtol=0.0,
                    atol=_IK_SOLUTION_EQUIVALENCE_TOLERANCE_RAD,
                )
                for existing in unique
            ):
                unique.append(checked)
        return tuple(unique)

    def _refine(self, target: FloatArray, seed: FloatArray) -> Tuple[FloatArray, float]:
        q = np.clip(np.asarray(seed, dtype=float), self._lower, self._upper)
        damping = _IK_INITIAL_DAMPING
        best_q = q.copy()
        best_error = math.inf
        for _ in range(_IK_MAX_ITERATIONS):
            current, jacobian = self._forward_and_jacobian(q)
            residual = target - current
            error = float(np.linalg.norm(residual))
            if error < best_error:
                best_error = error
                best_q = q.copy()
            if error <= _IK_POSITION_TOLERANCE_M:
                break
            normal = jacobian.T @ jacobian + damping * np.eye(3)
            gradient = jacobian.T @ residual
            try:
                step = cast(FloatArray, np.linalg.solve(normal, gradient))
            except np.linalg.LinAlgError:
                damping = min(1.0e6, damping * 10.0)
                continue
            step_norm = float(np.linalg.norm(step))
            if step_norm > _IK_MAX_STEP_RAD:
                step = step * (_IK_MAX_STEP_RAD / step_norm)
            accepted = False
            scale = 1.0
            for _ in range(10):
                candidate = np.clip(q + scale * step, self._lower, self._upper)
                candidate_foot, _ = self._forward_and_jacobian(candidate)
                candidate_error = float(np.linalg.norm(target - candidate_foot))
                if candidate_error < error - 1.0e-14:
                    q = candidate
                    damping = max(1.0e-10, damping * 0.35)
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                damping = min(1.0e6, damping * 10.0)
                if damping >= 1.0e6:
                    break
        return _readonly(best_q, (3,), "IK result"), best_error

    def inverse(
        self,
        foot_position_body: ArrayLike,
        *,
        preferred_q_rad: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Solve bounded IK and select the branch nearest ``preferred_q_rad``.

        ``preferred_q_rad`` should normally be the measured joint position or
        the last applied command.  It is a selection seed only: it must satisfy
        the URDF limits, and every returned candidate still has to satisfy the
        same bounds and strict FK substitution check.  If omitted, the pinned
        home pose is used for backward-compatible deterministic selection.
        """

        target = _readonly(foot_position_body, (3,), "foot_position_body")
        preferred = (
            self._home
            if preferred_q_rad is None
            else self._checked_q(preferred_q_rad, "preferred_q_rad")
        )
        feasible: List[Tuple[FloatArray, float]] = []
        best_error = math.inf
        for seed in self._candidate_seeds(preferred):
            candidate, error = self._refine(target, seed)
            best_error = min(best_error, error)
            if error > _IK_POSITION_TOLERANCE_M:
                continue
            verified = self.forward(candidate)
            verified_error = float(np.linalg.norm(verified - target))
            best_error = min(best_error, verified_error)
            if verified_error > _IK_POSITION_TOLERANCE_M:
                continue
            if any(
                np.allclose(
                    candidate,
                    existing,
                    rtol=0.0,
                    atol=_IK_SOLUTION_EQUIVALENCE_TOLERANCE_RAD,
                )
                for existing, _ in feasible
            ):
                continue
            feasible.append((candidate, verified_error))
        if not feasible:
            raise Go2KinematicsError(
                f"{self._leg_name} foot target is unreachable within URDF limits; "
                f"best residual={best_error:.6g} m"
            )

        best_q, _ = min(
            feasible,
            key=lambda item: (
                float(np.linalg.norm(item[0] - preferred)),
                item[1],
            ),
        )
        verified = self.forward(best_q)
        verified_error = float(np.linalg.norm(verified - target))
        if verified_error > _IK_POSITION_TOLERANCE_M:
            raise Go2KinematicsError(
                f"{self._leg_name} IK failed forward-substitution verification"
            )
        return _readonly_copy(best_q)


class Go2UrdfKinematics:
    """Four-leg SDK-order view of a hash-pinned Go2 URDF model."""

    offline_prior_only = OFFLINE_PRIOR_ONLY
    hardware_validated = HARDWARE_VALIDATED
    leg_order = SDK_LEG_ORDER

    def __init__(
        self,
        mass_properties: Go2UrdfMassProperties,
        *,
        urdf_root_from_body_origin_B_m: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        if not isinstance(mass_properties, Go2UrdfMassProperties):
            raise TypeError("mass_properties must be Go2UrdfMassProperties")
        root_from_B = _readonly(
            urdf_root_from_body_origin_B_m,
            (3,),
            "urdf_root_from_body_origin_B_m",
        )
        parsed = _read_verified_leg_joints(mass_properties)
        positions = dict(mass_properties.joint_positions_rad)
        legs: Dict[str, Go2LegKinematics] = {}
        for leg_name in SDK_LEG_ORDER:
            movable_names = tuple(f"{leg_name}_{role}_joint" for role in JOINT_ROLE_ORDER)
            foot_name = f"{leg_name}_{_FOOT_ROLE}_joint"
            chain = tuple(parsed[name] for name in movable_names + (foot_name,))
            _validate_chain(leg_name, mass_properties.root_link, chain)
            public_chain = tuple(_public_joint(joint) for joint in chain)
            home = [positions[name] for name in movable_names]
            legs[leg_name] = Go2LegKinematics(
                leg_name=leg_name,
                movable_joints=public_chain[:3],
                foot_joint=public_chain[3],
                home_q_rad=home,
                urdf_root_from_body_origin_B_m=root_from_B,
            )
        self._mass_properties = mass_properties
        self._root_from_B = root_from_B
        self._legs = MappingProxyType(legs)

    @property
    def source_path(self) -> Path:
        return self._mass_properties.source_path

    @property
    def source_sha256(self) -> str:
        return self._mass_properties.source_sha256

    @property
    def sdk_joint_order(self) -> Tuple[str, ...]:
        return tuple(
            f"{leg_name}_{role}_joint" for leg_name in SDK_LEG_ORDER for role in JOINT_ROLE_ORDER
        )

    @property
    def urdf_root_from_body_origin_B_m(self) -> FloatArray:
        return _readonly_copy(self._root_from_B)

    @property
    def legs(self) -> Mapping[str, Go2LegKinematics]:
        return self._legs

    def leg(self, leg_name: str) -> Go2LegKinematics:
        try:
            return self._legs[leg_name]
        except KeyError as exc:
            raise Go2KinematicsError(f"unknown Go2 leg {leg_name!r}") from exc

    def forward(self, leg_name: str, q_rad: ArrayLike) -> FloatArray:
        return self.leg(leg_name).forward(q_rad)

    def inverse(
        self,
        leg_name: str,
        foot_position_body: ArrayLike,
        *,
        preferred_q_rad: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Solve one leg while preserving the branch nearest a joint seed."""

        return self.leg(leg_name).inverse(
            foot_position_body,
            preferred_q_rad=preferred_q_rad,
        )

    def forward_all(
        self,
        joint_positions_by_leg: Mapping[str, ArrayLike],
        *,
        output_leg_order: Sequence[str] = SDK_LEG_ORDER,
    ) -> FootPositionsFromBodyOriginB:
        """Return labeled ``B``-referenced FK results for all four feet.

        Joint positions are accepted as a name-keyed mapping instead of an
        unlabeled ``(4, 3)`` matrix.  ``output_leg_order`` must be a permutation
        of the canonical Go2 SDK order, making every non-identity mapping
        explicit at the call site.
        """

        if not isinstance(joint_positions_by_leg, Mapping):
            raise TypeError("joint_positions_by_leg must be a leg-name mapping")
        supplied = set(joint_positions_by_leg)
        expected = set(SDK_LEG_ORDER)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise Go2KinematicsError(
                "joint_positions_by_leg must contain exactly the canonical Go2 legs; "
                f"missing={missing}, extra={extra}"
            )
        try:
            order = validate_four_foot_leg_order(
                output_leg_order,
                name="output_leg_order",
            )
        except (TypeError, ValueError) as exc:
            raise Go2KinematicsError(str(exc)) from exc
        if set(order) != expected:
            raise Go2KinematicsError(
                "output_leg_order must be a permutation of the canonical Go2 SDK leg order"
            )
        values = np.vstack(
            [self.forward(leg_name, joint_positions_by_leg[leg_name]) for leg_name in order]
        )
        return FootPositionsFromBodyOriginB(values_m=values, leg_order=order)

    def foot_lever_arms_from_com(
        self,
        joint_positions_by_leg: Mapping[str, ArrayLike],
        total_com_C_from_go2_body_origin_B_body_m: ArrayLike,
        *,
        output_leg_order: Sequence[str] = SDK_LEG_ORDER,
    ) -> FootLeverArmsFromComBody:
        """Return labeled ``{}^B r_CF`` values using ``r_BF - r_BC``."""

        feet_from_B = self.forward_all(
            joint_positions_by_leg,
            output_leg_order=output_leg_order,
        )
        return foot_positions_from_body_origin_B_to_com_lever_arms(
            feet_from_B,
            total_com_C_from_go2_body_origin_B_body_m,
        )


def _read_verified_leg_joints(
    mass_properties: Go2UrdfMassProperties,
) -> Mapping[str, _ParsedJoint]:
    try:
        path = mass_properties.source_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Go2KinematicsError("verified URDF source no longer exists") from exc
    if path != mass_properties.source_path:
        raise Go2KinematicsError("verified URDF source path identity changed")
    payload = path.read_bytes()
    if len(payload) <= 0 or len(payload) > MAX_URDF_BYTES:
        raise Go2KinematicsError("URDF size is outside the accepted range")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise Go2KinematicsError("URDF DTD/entity declarations are prohibited")
    actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual_hash != _normalized_hash(mass_properties.source_sha256):
        raise Go2KinematicsError("verified URDF changed after mass-property evaluation")
    try:
        robot = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise Go2KinematicsError(f"invalid URDF XML: {exc}") from exc
    if robot.tag != "robot" or robot.attrib.get("name") != mass_properties.robot_name:
        raise Go2KinematicsError("URDF robot identity changed after mass-property evaluation")

    required_names = {
        f"{leg_name}_{role}_joint"
        for leg_name in SDK_LEG_ORDER
        for role in JOINT_ROLE_ORDER + (_FOOT_ROLE,)
    }
    matching_elements: Dict[str, ET.Element] = {}
    for element in robot.findall("joint"):
        name = element.attrib.get("name", "").strip()
        if name in required_names:
            if name in matching_elements:
                raise Go2KinematicsError(f"URDF contains duplicate leg joint {name}")
            matching_elements[name] = element
    if set(matching_elements) != required_names:
        missing = sorted(required_names - set(matching_elements))
        raise Go2KinematicsError(f"URDF is missing required leg joints: {missing}")

    verified_limits = {limit.name: limit for limit in mass_properties.movable_joint_limits}
    parsed: Dict[str, _ParsedJoint] = {}
    for name, element in matching_elements.items():
        joint = _parse_joint(element)
        if joint.joint_type == "revolute":
            verified = verified_limits.get(name)
            if verified is None or not _matches_limit(joint, verified):
                raise Go2KinematicsError(f"joint {name} differs from verified mass-property data")
        parsed[name] = joint
    return MappingProxyType(parsed)


def _validate_chain(
    leg_name: str,
    root_link: str,
    chain: Tuple[_ParsedJoint, ...],
) -> None:
    hip, thigh, calf, foot = chain
    if (
        hip.parent_link != root_link
        or thigh.parent_link != hip.child_link
        or calf.parent_link != thigh.child_link
        or foot.parent_link != calf.child_link
    ):
        raise Go2KinematicsError(f"{leg_name} URDF leg chain is disconnected or reordered")
    if tuple(joint.joint_type for joint in chain) != (
        "revolute",
        "revolute",
        "revolute",
        "fixed",
    ):
        raise Go2KinematicsError(f"{leg_name} URDF leg chain has unexpected joint types")


def _public_joint(joint: _ParsedJoint) -> Go2UrdfKinematicJoint:
    return Go2UrdfKinematicJoint(
        name=joint.name,
        joint_type=joint.joint_type,
        parent_link=joint.parent_link,
        child_link=joint.child_link,
        translation_parent_m=joint.translation_parent_m,
        rotation_joint_to_parent=joint.rotation_joint_to_parent,
        axis_joint=joint.axis_joint,
        lower_rad=joint.lower_rad,
        upper_rad=joint.upper_rad,
    )


def load_go2_urdf_kinematics(
    mass_properties: Go2UrdfMassProperties,
    *,
    urdf_root_from_body_origin_B_m: ArrayLike = (0.0, 0.0, 0.0),
) -> Go2UrdfKinematics:
    """Build an offline-only four-leg model from verified mass properties."""

    try:
        return Go2UrdfKinematics(
            mass_properties,
            urdf_root_from_body_origin_B_m=urdf_root_from_body_origin_B_m,
        )
    except Go2UrdfError as exc:
        raise Go2KinematicsError(str(exc)) from exc


__all__ = [
    "HARDWARE_VALIDATED",
    "JOINT_ROLE_ORDER",
    "OFFLINE_PRIOR_ONLY",
    "SDK_LEG_ORDER",
    "Go2KinematicsError",
    "Go2LegKinematics",
    "Go2UrdfKinematicJoint",
    "Go2UrdfKinematics",
    "load_go2_urdf_kinematics",
]
