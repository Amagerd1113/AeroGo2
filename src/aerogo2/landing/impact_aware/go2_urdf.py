"""Strict, dependency-light URDF mass-property reader for the Go2 prior.

The reader intentionally supports only the URDF features used by the pinned
Unitree Go2 model.  It never resolves meshes, network resources, packages, or
XML entities.  All inertia is assembled from each link's ``<inertial>`` data
at an explicitly supplied joint pose.

中文说明：读取前验证文件大小与 SHA-256，只解析固定版本官方 Go2 URDF 的惯性和
关节树，不加载 mesh、package URI 或外部实体。各连杆惯量经旋转与平行轴定理合成
到整机参考系；结果是裸机 URDF 先验，不包含上装，也不代表实机 CAD/称重辨识值。
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]

MAX_URDF_BYTES = 512 * 1024
_SUPPORTED_JOINT_TYPES = frozenset({"fixed", "revolute", "continuous"})


class Go2UrdfError(ValueError):
    """Raised when the pinned URDF or requested pose is incomplete or unsafe."""


def _readonly(value: object, shape: Tuple[int, ...], name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise Go2UrdfError(f"{name} must contain real numbers")
    result = cast(FloatArray, np.array(raw, dtype=float, copy=True))
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise Go2UrdfError(f"{name} must be a finite array with shape {shape}")
    result.setflags(write=False)
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise Go2UrdfError(f"{name} must be finite")
    if isinstance(value, Real):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError as exc:
            raise Go2UrdfError(f"{name} must be finite") from exc
    else:
        raise Go2UrdfError(f"{name} must be finite")
    if not math.isfinite(result):
        raise Go2UrdfError(f"{name} must be finite")
    return result


def _normalized_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise Go2UrdfError("expected_sha256 must be a hexadecimal SHA-256 string")
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:") :]
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Go2UrdfError("expected_sha256 must be a hexadecimal SHA-256 string")
    return "sha256:" + normalized


def _vector3(text: Optional[str], name: str, *, default: Sequence[float]) -> FloatArray:
    if text is None:
        return _readonly(default, (3,), name)
    fields = text.split()
    if len(fields) != 3:
        raise Go2UrdfError(f"{name} must contain exactly three numbers")
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
        raise Go2UrdfError("movable joint axis must be nonzero")
    x, y, z = axis / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return cast(
        FloatArray,
        np.array(
            [
                [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
                [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
                [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
            ],
            dtype=float,
        ),
    )


def _parallel_axis(mass_kg: float, displacement_m: FloatArray) -> FloatArray:
    return cast(
        FloatArray,
        mass_kg
        * (
            float(displacement_m @ displacement_m) * np.eye(3)
            - np.outer(displacement_m, displacement_m)
        ),
    )


@dataclass(frozen=True)
class Go2UrdfJointLimit:
    """One movable joint definition read directly from the pinned URDF."""

    name: str
    joint_type: str
    axis_joint: FloatArray
    lower_rad: Optional[float]
    upper_rad: Optional[float]
    effort_nm: Optional[float]
    velocity_rad_s: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_joint", _readonly(self.axis_joint, (3,), "axis_joint"))


@dataclass(frozen=True)
class Go2UrdfMassProperties:
    """Composite Go2 properties at one explicit pose, expressed in root axes."""

    source_path: Path
    source_sha256: str
    robot_name: str
    root_link: str
    total_link_count: int
    inertial_link_count: int
    positive_mass_link_count: int
    mass_kg: float
    com_from_root_m: FloatArray
    inertia_about_com_root_axes_kg_m2: FloatArray
    inertia_about_root_origin_root_axes_kg_m2: FloatArray
    movable_joint_limits: Tuple[Go2UrdfJointLimit, ...]
    joint_positions_rad: Tuple[Tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "com_from_root_m", _readonly(self.com_from_root_m, (3,), "com"))
        object.__setattr__(
            self,
            "inertia_about_com_root_axes_kg_m2",
            _readonly(self.inertia_about_com_root_axes_kg_m2, (3, 3), "inertia_about_com"),
        )
        object.__setattr__(
            self,
            "inertia_about_root_origin_root_axes_kg_m2",
            _readonly(
                self.inertia_about_root_origin_root_axes_kg_m2,
                (3, 3),
                "inertia_about_root",
            ),
        )


@dataclass(frozen=True)
class PointMassCompositeEstimate:
    """A base rigid body plus one zero-intrinsic-inertia point mass."""

    mass_kg: float
    com_from_root_m: FloatArray
    inertia_about_com_root_axes_kg_m2: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "com_from_root_m", _readonly(self.com_from_root_m, (3,), "com"))
        object.__setattr__(
            self,
            "inertia_about_com_root_axes_kg_m2",
            _readonly(self.inertia_about_com_root_axes_kg_m2, (3, 3), "inertia"),
        )


@dataclass(frozen=True)
class _LinkInertial:
    mass_kg: float
    com_link_m: FloatArray
    rotation_inertial_to_link: FloatArray
    inertia_inertial_kg_m2: FloatArray


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    translation_parent_m: FloatArray
    rotation_joint_to_parent: FloatArray
    axis_joint: FloatArray
    lower_rad: Optional[float]
    upper_rad: Optional[float]
    effort_nm: Optional[float]
    velocity_rad_s: Optional[float]


def _optional_attribute(element: Optional[ET.Element], key: str, name: str) -> Optional[float]:
    if element is None or key not in element.attrib:
        return None
    return _finite(element.attrib[key], name)


def _parse_inertial(link: ET.Element, link_name: str) -> Optional[_LinkInertial]:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None or "value" not in mass_element.attrib:
        raise Go2UrdfError(f"link {link_name} has incomplete inertial data")
    mass = _finite(mass_element.attrib["value"], f"{link_name}.mass")
    if mass < 0.0:
        raise Go2UrdfError(f"link {link_name} mass cannot be negative")
    origin = inertial.find("origin")
    xyz = _vector3(
        None if origin is None else origin.attrib.get("xyz"),
        f"{link_name}.inertial.xyz",
        default=(0.0, 0.0, 0.0),
    )
    rpy = _vector3(
        None if origin is None else origin.attrib.get("rpy"),
        f"{link_name}.inertial.rpy",
        default=(0.0, 0.0, 0.0),
    )
    required = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    if any(key not in inertia_element.attrib for key in required):
        raise Go2UrdfError(f"link {link_name} has incomplete inertia tensor")
    ixx, ixy, ixz, iyy, iyz, izz = (
        _finite(inertia_element.attrib[key], f"{link_name}.inertia.{key}") for key in required
    )
    tensor = _readonly(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        (3, 3),
        f"{link_name}.inertia",
    )
    eigenvalues = np.linalg.eigvalsh(tensor)
    if float(np.min(eigenvalues)) < -1.0e-12:
        raise Go2UrdfError(f"link {link_name} inertia must be positive semidefinite")
    principal = np.maximum(eigenvalues, 0.0)
    triangle_tolerance = max(1.0e-12, float(np.max(principal)) * 1.0e-10)
    if float(principal[2]) > float(principal[0] + principal[1]) + triangle_tolerance:
        raise Go2UrdfError(f"link {link_name} inertia violates the triangle inequality")
    if mass == 0.0 and not np.allclose(tensor, 0.0, rtol=0.0, atol=1.0e-15):
        raise Go2UrdfError(f"massless link {link_name} cannot have nonzero inertia")
    return _LinkInertial(
        mass_kg=mass,
        com_link_m=xyz,
        rotation_inertial_to_link=_rotation_from_rpy(rpy),
        inertia_inertial_kg_m2=tensor,
    )


def _parse_joint(element: ET.Element) -> _Joint:
    name = element.attrib.get("name", "").strip()
    joint_type = element.attrib.get("type", "").strip()
    if not name or joint_type not in _SUPPORTED_JOINT_TYPES:
        raise Go2UrdfError(f"unsupported or unnamed URDF joint: {name!r}/{joint_type!r}")
    parent_element = element.find("parent")
    child_element = element.find("child")
    if parent_element is None or child_element is None:
        raise Go2UrdfError(f"joint {name} must name parent and child links")
    parent = parent_element.attrib.get("link", "").strip()
    child = child_element.attrib.get("link", "").strip()
    if not parent or not child or parent == child:
        raise Go2UrdfError(f"joint {name} has invalid parent/child links")
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
    lower = _optional_attribute(limit, "lower", f"{name}.limit.lower")
    upper = _optional_attribute(limit, "upper", f"{name}.limit.upper")
    effort = _optional_attribute(limit, "effort", f"{name}.limit.effort")
    velocity = _optional_attribute(limit, "velocity", f"{name}.limit.velocity")
    if joint_type == "revolute" and (lower is None or upper is None or lower >= upper):
        raise Go2UrdfError(f"revolute joint {name} requires valid lower/upper limits")
    if joint_type != "fixed" and float(np.linalg.norm(axis)) <= 0.0:
        raise Go2UrdfError(f"movable joint {name} requires a nonzero axis")
    if effort is not None and effort <= 0.0:
        raise Go2UrdfError(f"joint {name} effort limit must be positive")
    if velocity is not None and velocity <= 0.0:
        raise Go2UrdfError(f"joint {name} velocity limit must be positive")
    return _Joint(
        name=name,
        joint_type=joint_type,
        parent=parent,
        child=child,
        translation_parent_m=translation,
        rotation_joint_to_parent=_rotation_from_rpy(rpy),
        axis_joint=axis,
        lower_rad=lower,
        upper_rad=upper,
        effort_nm=effort,
        velocity_rad_s=velocity,
    )


def load_go2_urdf_mass_properties(
    path: Path,
    *,
    expected_sha256: str,
    expected_robot_name: str,
    root_link: str,
    joint_positions_rad: Mapping[str, float],
) -> Go2UrdfMassProperties:
    """Read and assemble the pinned Go2 URDF at ``joint_positions_rad``."""

    try:
        source_path = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Go2UrdfError(f"cannot resolve URDF path: {path}") from exc
    if not source_path.is_file():
        raise Go2UrdfError(f"URDF path is not a file: {source_path}")
    size = source_path.stat().st_size
    if size <= 0 or size > MAX_URDF_BYTES:
        raise Go2UrdfError("URDF size is outside the accepted 1..512 KiB range")
    payload = source_path.read_bytes()
    if len(payload) <= 0 or len(payload) > MAX_URDF_BYTES:
        raise Go2UrdfError("URDF size changed while reading or is outside the accepted range")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise Go2UrdfError("URDF DTD/entity declarations are prohibited")
    actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    normalized_expected_hash = _normalized_sha256(expected_sha256)
    if actual_hash != normalized_expected_hash:
        raise Go2UrdfError(
            f"URDF SHA-256 mismatch: expected {normalized_expected_hash}, got {actual_hash}"
        )
    try:
        robot = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise Go2UrdfError(f"invalid URDF XML: {exc}") from exc
    if robot.tag != "robot" or robot.attrib.get("name") != expected_robot_name:
        raise Go2UrdfError("URDF robot name does not match the pinned configuration")

    links: Dict[str, Optional[_LinkInertial]] = {}
    for link in robot.findall("link"):
        name = link.attrib.get("name", "").strip()
        if not name or name in links:
            raise Go2UrdfError(f"URDF contains a duplicate or empty link name: {name!r}")
        links[name] = _parse_inertial(link, name)
    if root_link not in links:
        raise Go2UrdfError(f"URDF root link {root_link!r} does not exist")

    joints = tuple(_parse_joint(element) for element in robot.findall("joint"))
    joint_names = [joint.name for joint in joints]
    if len(set(joint_names)) != len(joint_names):
        raise Go2UrdfError("URDF joint names must be unique")
    movable = tuple(joint for joint in joints if joint.joint_type != "fixed")
    expected_position_names = {joint.name for joint in movable}
    supplied_position_names = set(joint_positions_rad)
    if supplied_position_names != expected_position_names:
        missing = sorted(expected_position_names - supplied_position_names)
        extra = sorted(supplied_position_names - expected_position_names)
        raise Go2UrdfError(f"joint pose does not match URDF; missing={missing}, extra={extra}")
    positions: Dict[str, float] = {}
    for joint in movable:
        position = _finite(joint_positions_rad[joint.name], f"pose.{joint.name}")
        if joint.lower_rad is not None and position < joint.lower_rad - 1.0e-12:
            raise Go2UrdfError(f"pose for {joint.name} is below its URDF limit")
        if joint.upper_rad is not None and position > joint.upper_rad + 1.0e-12:
            raise Go2UrdfError(f"pose for {joint.name} is above its URDF limit")
        positions[joint.name] = position

    children: Dict[str, list[_Joint]] = {name: [] for name in links}
    child_links = set()
    for joint in joints:
        if joint.parent not in links or joint.child not in links:
            raise Go2UrdfError(f"joint {joint.name} references an unknown link")
        if joint.child in child_links:
            raise Go2UrdfError(f"link {joint.child} has more than one parent")
        child_links.add(joint.child)
        children[joint.parent].append(joint)
    if root_link in child_links:
        raise Go2UrdfError("configured root_link cannot have a parent")

    transforms: Dict[str, Tuple[FloatArray, FloatArray]] = {root_link: (np.eye(3), np.zeros(3))}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        rotation_parent, translation_parent = transforms[parent]
        for joint in children[parent]:
            if joint.child in transforms:
                raise Go2UrdfError("URDF kinematic graph contains a cycle")
            motion = (
                np.eye(3)
                if joint.joint_type == "fixed"
                else _axis_angle(joint.axis_joint, positions[joint.name])
            )
            rotation_child = rotation_parent @ joint.rotation_joint_to_parent @ motion
            translation_child = translation_parent + rotation_parent @ joint.translation_parent_m
            transforms[joint.child] = (rotation_child, translation_child)
            queue.append(joint.child)
    if set(transforms) != set(links):
        missing_links = sorted(set(links) - set(transforms))
        raise Go2UrdfError(f"URDF contains links unreachable from {root_link}: {missing_links}")

    bodies: list[Tuple[float, FloatArray, FloatArray]] = []
    for link_name, inertial in links.items():
        if inertial is None or inertial.mass_kg == 0.0:
            continue
        rotation_link, translation_link = transforms[link_name]
        com_root = translation_link + rotation_link @ inertial.com_link_m
        rotation_inertial_to_root = rotation_link @ inertial.rotation_inertial_to_link
        inertia_com_root = (
            rotation_inertial_to_root
            @ inertial.inertia_inertial_kg_m2
            @ rotation_inertial_to_root.T
        )
        bodies.append((inertial.mass_kg, com_root, inertia_com_root))
    if not bodies:
        raise Go2UrdfError("URDF contains no positive-mass inertial links")
    total_mass = sum(mass for mass, _, _ in bodies)
    total_com = sum((mass * com for mass, com, _ in bodies), np.zeros(3)) / total_mass
    inertia_about_com = sum(
        (
            inertia_at_link_com + _parallel_axis(mass, com - total_com)
            for mass, com, inertia_at_link_com in bodies
        ),
        np.zeros((3, 3)),
    )
    inertia_about_root = inertia_about_com + _parallel_axis(total_mass, total_com)
    if not np.allclose(inertia_about_com, inertia_about_com.T, rtol=0.0, atol=1.0e-10):
        raise Go2UrdfError("assembled inertia is not symmetric")
    if float(np.min(np.linalg.eigvalsh(inertia_about_com))) <= 0.0:
        raise Go2UrdfError("assembled inertia is not positive definite")

    limits = tuple(
        Go2UrdfJointLimit(
            name=joint.name,
            joint_type=joint.joint_type,
            axis_joint=joint.axis_joint,
            lower_rad=joint.lower_rad,
            upper_rad=joint.upper_rad,
            effort_nm=joint.effort_nm,
            velocity_rad_s=joint.velocity_rad_s,
        )
        for joint in movable
    )
    ordered_positions = tuple((joint.name, positions[joint.name]) for joint in movable)
    return Go2UrdfMassProperties(
        source_path=source_path,
        source_sha256=actual_hash,
        robot_name=expected_robot_name,
        root_link=root_link,
        total_link_count=len(links),
        inertial_link_count=sum(inertial is not None for inertial in links.values()),
        positive_mass_link_count=len(bodies),
        mass_kg=total_mass,
        com_from_root_m=total_com,
        inertia_about_com_root_axes_kg_m2=inertia_about_com,
        inertia_about_root_origin_root_axes_kg_m2=inertia_about_root,
        movable_joint_limits=limits,
        joint_positions_rad=ordered_positions,
    )


def combine_with_point_mass(
    base: Go2UrdfMassProperties,
    *,
    point_mass_kg: float,
    point_position_from_root_m: object,
) -> PointMassCompositeEstimate:
    """Combine URDF properties with a zero-intrinsic-inertia point mass."""

    added_mass = _finite(point_mass_kg, "point_mass_kg")
    if added_mass <= 0.0:
        raise Go2UrdfError("point_mass_kg must be positive")
    point = _readonly(point_position_from_root_m, (3,), "point_position_from_root_m")
    total_mass = base.mass_kg + added_mass
    total_com = (base.mass_kg * base.com_from_root_m + added_mass * point) / total_mass
    inertia = (
        base.inertia_about_com_root_axes_kg_m2
        + _parallel_axis(base.mass_kg, base.com_from_root_m - total_com)
        + _parallel_axis(added_mass, point - total_com)
    )
    return PointMassCompositeEstimate(
        mass_kg=total_mass,
        com_from_root_m=total_com,
        inertia_about_com_root_axes_kg_m2=inertia,
    )


__all__ = [
    "Go2UrdfError",
    "Go2UrdfJointLimit",
    "Go2UrdfMassProperties",
    "PointMassCompositeEstimate",
    "combine_with_point_mass",
    "load_go2_urdf_mass_properties",
]
