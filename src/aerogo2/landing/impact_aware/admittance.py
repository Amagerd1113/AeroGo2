"""Paper-faithful leg-side execution for the impact-aware landing controller.

This module implements Eqs. (51)--(56) of ``LeggedRobotWithWings``.  It is a
pure numerical boundary: it neither imports the Unitree SDK nor writes to
hardware.  The caller remains responsible for contact filtering, state
estimation, MPC execution, and sending the returned joint command.

中文说明：每条腿按论文式 (55) 实现笛卡尔空间二阶导纳
``M_a Δp¨ + D_a Δp˙ + (1-η)K_r Δp = δ_m(f_est-η f_des)``，并在触地后按时间平滑切换
参数。这里的正负号与“地面对机器人”的接触力定义及 ``p_cmd=p_nom+Δp`` 配套，
不能单独改成 ``f_des-f_est``。导纳位移经工作空间限幅、逆运动学和关节每周期变化率限制后才形成候选关节
位置。控制器采用 preview/commit 事务语义：四条腿全部校验成功后才能一起提交内部
状态，任何一腿失败都可回滚，防止部分腿状态提前推进。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Real
from threading import Lock
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
    MatrixSpec: TypeAlias = Union[float, ArrayLike]
else:
    # Keep Python 3.8 runtime reflection free of an unresolved ``TypeAlias``
    # annotation while exposing the same subscriptable NumPy aliases.
    FloatArray = NDArray[np.float64]
    MatrixSpec = Union[float, ArrayLike]

_CARTESIAN_DIMENSION = 3
_JOINTS_PER_LEG = 3
_ROTATION_TOLERANCE = 1e-6
_SYMMETRY_TOLERANCE = 1e-10


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _positive_scalar(name: str, value: object) -> float:
    converted = _finite_scalar(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return converted


def _numeric_array(name: str, value: ArrayLike, shape: Tuple[int, ...]) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array with shape {shape}") from exc
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    converted = np.asarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _vector3(name: str, value: ArrayLike) -> FloatArray:
    return _numeric_array(name, value, (_CARTESIAN_DIMENSION,))


def _joint_vector(name: str, value: ArrayLike) -> FloatArray:
    return _numeric_array(name, value, (_JOINTS_PER_LEG,))


def _nonnegative_vector3(name: str, value: ArrayLike) -> FloatArray:
    result = _vector3(name, value)
    if np.any(result < 0.0):
        raise ValueError(f"{name} entries must be non-negative")
    return result


def _positive_vector3(name: str, value: ArrayLike) -> FloatArray:
    result = _vector3(name, value)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} entries must be greater than zero")
    return result


def _componentwise_deadband(value: FloatArray, deadband: FloatArray) -> FloatArray:
    """Remove a symmetric per-axis deadband without a boundary discontinuity."""

    return np.sign(value) * np.maximum(np.abs(value) - deadband, 0.0)


def _rotation_body_to_world(value: ArrayLike) -> FloatArray:
    rotation = _numeric_array(
        "rotation_body_to_world",
        value,
        (_CARTESIAN_DIMENSION, _CARTESIAN_DIMENSION),
    )
    gram = rotation.T @ rotation
    if not np.allclose(
        gram,
        np.eye(_CARTESIAN_DIMENSION),
        rtol=0.0,
        atol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("rotation_body_to_world must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not math.isclose(
        determinant,
        1.0,
        rel_tol=0.0,
        abs_tol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("rotation_body_to_world must be a proper rotation with determinant +1")
    return rotation


def _spd_matrix(name: str, value: MatrixSpec) -> FloatArray:
    if isinstance(value, Real) and not isinstance(value, bool):
        scalar = _positive_scalar(name, value)
        matrix = np.eye(_CARTESIAN_DIMENSION, dtype=np.float64) * scalar
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a scalar, diagonal, or 3x3 SPD matrix") from exc
        if raw.ndim == 1:
            diagonal = _vector3(name, value)
            if np.any(diagonal <= 0.0):
                raise ValueError(f"{name} diagonal entries must be greater than zero")
            matrix = np.diag(diagonal)
        else:
            matrix = _numeric_array(
                name,
                value,
                (_CARTESIAN_DIMENSION, _CARTESIAN_DIMENSION),
            )

    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_TOLERANCE):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return matrix


def _readonly_copy(value: FloatArray) -> FloatArray:
    copied = value.copy()
    copied.setflags(write=False)
    return copied


def world_to_body_force(
    force_world: ArrayLike,
    rotation_body_to_world: ArrayLike,
) -> FloatArray:
    """Transform a world-frame force to the body frame as in Eq. (51).

    ``rotation_body_to_world`` maps body-frame coordinates into world-frame
    coordinates, so its transpose performs the inverse transformation.
    """

    force = _vector3("force_world", force_world)
    rotation = _rotation_body_to_world(rotation_body_to_world)
    return rotation.T @ force


def nominal_foot_position_body(
    foot_position_world: ArrayLike,
    body_position_world: ArrayLike,
    rotation_body_to_world: ArrayLike,
) -> FloatArray:
    """Return the nominal body-frame foot reference from Eq. (52)."""

    foot = _vector3("foot_position_world", foot_position_world)
    body = _vector3("body_position_world", body_position_world)
    rotation = _rotation_body_to_world(rotation_body_to_world)
    return rotation.T @ (foot - body)


@dataclass(frozen=True)
class TouchdownBlend:
    """Normalized touchdown time and cubic blending factor from Eq. (53)."""

    xi: float
    eta: float


def touchdown_blend(
    current_time_s: float,
    touchdown_time_s: Optional[float],
    transition_duration_s: float,
) -> TouchdownBlend:
    """Evaluate the clamped linear phase and smoothstep touchdown blend."""

    current = _finite_scalar("current_time_s", current_time_s)
    duration = _positive_scalar("transition_duration_s", transition_duration_s)
    if touchdown_time_s is None:
        return TouchdownBlend(xi=0.0, eta=0.0)
    touchdown = _finite_scalar("touchdown_time_s", touchdown_time_s)
    xi = min(1.0, max(0.0, (current - touchdown) / duration))
    eta = xi * xi * (3.0 - 2.0 * xi)
    return TouchdownBlend(xi=xi, eta=eta)


def scheduled_spd_matrix(
    touchdown_value: MatrixSpec,
    stance_value: MatrixSpec,
    eta: float,
) -> FloatArray:
    """Linearly schedule SPD touchdown/stance matrices as in Eq. (54)."""

    blend = _finite_scalar("eta", eta)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("eta must be in the closed interval [0, 1]")
    touchdown = _spd_matrix("touchdown_value", touchdown_value)
    stance = _spd_matrix("stance_value", stance_value)
    return (1.0 - blend) * touchdown + blend * stance


class InverseKinematics(Protocol):
    """Callback protocol for one leg's body-frame inverse kinematics."""

    def __call__(self, foot_position_body: FloatArray) -> ArrayLike:
        """Map a three-axis body-frame foot position to three joint angles."""


class ForwardKinematics(Protocol):
    """Optional callback used to close joint limiting back to Cartesian state."""

    def __call__(self, joint_position: FloatArray) -> ArrayLike:
        """Map three joint angles to a body-frame foot position."""


class WorkspaceLimiter(Protocol):
    """Abstraction for a leg-specific admissible workspace projection."""

    def limit(self, foot_position_body: FloatArray) -> ArrayLike:
        """Return an admissible body-frame foot position."""


class AxisAlignedWorkspace:
    """Componentwise body-frame foot workspace bounds."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        lower_array = _vector3("workspace lower", lower)
        upper_array = _vector3("workspace upper", upper)
        if np.any(lower_array >= upper_array):
            raise ValueError("workspace lower bounds must be strictly below upper bounds")
        self._lower = lower_array
        self._upper = upper_array

    @property
    def lower(self) -> FloatArray:
        return _readonly_copy(self._lower)

    @property
    def upper(self) -> FloatArray:
        return _readonly_copy(self._upper)

    def limit(self, foot_position_body: FloatArray) -> FloatArray:
        position = _vector3("foot_position_body", foot_position_body)
        return np.clip(position, self._lower, self._upper)


@dataclass(frozen=True)
class AdmittanceConfig:
    """Validated Eq. (53)--(56) parameters for one three-DOF Go2 leg.

    Matrix parameters accept an isotropic positive scalar, three positive
    diagonal entries, or a symmetric positive-definite 3x3 matrix.
    ``anti_windup_enabled`` is an optional safety extension and is disabled by
    default so that the internal state follows Eq. (55) without projection.

    中文：M/D/K 可写成标量、三个对角元素或 3×3 对称正定矩阵。flight 与
    touchdown 参数经 ``transition_duration_s`` 平滑过渡。这里的关节限值是算法
    边界，不能替代 LowCmd owner 最后一层的位置、速度、力矩和温度保护。
    """

    transition_duration_s: float
    touchdown_inertia: MatrixSpec
    stance_inertia: MatrixSpec
    touchdown_damping: MatrixSpec
    stance_damping: MatrixSpec
    restoring_stiffness: MatrixSpec
    joint_lower: ArrayLike
    joint_upper: ArrayLike
    joint_rate_limit: ArrayLike
    anti_windup_enabled: bool = False
    stance_stiffness: Optional[MatrixSpec] = None
    force_error_deadband_n: ArrayLike = (0.0, 0.0, 0.0)
    correction_position_limit_m: ArrayLike = (0.25, 0.25, 0.25)
    correction_velocity_limit_m_per_s: ArrayLike = (1.0, 1.0, 1.0)
    contact_release_policy: str = "reset"

    def __post_init__(self) -> None:
        if type(self.anti_windup_enabled) is not bool:
            raise ValueError("anti_windup_enabled must be a bool")

        duration = _positive_scalar("transition_duration_s", self.transition_duration_s)
        touchdown_inertia = _readonly_copy(_spd_matrix("touchdown_inertia", self.touchdown_inertia))
        stance_inertia = _readonly_copy(_spd_matrix("stance_inertia", self.stance_inertia))
        touchdown_damping = _readonly_copy(_spd_matrix("touchdown_damping", self.touchdown_damping))
        stance_damping = _readonly_copy(_spd_matrix("stance_damping", self.stance_damping))
        restoring_stiffness = _readonly_copy(
            _spd_matrix("restoring_stiffness", self.restoring_stiffness)
        )
        stance_stiffness = _readonly_copy(
            restoring_stiffness
            if self.stance_stiffness is None
            else _spd_matrix("stance_stiffness", self.stance_stiffness)
        )
        force_error_deadband = _readonly_copy(
            _nonnegative_vector3("force_error_deadband_n", self.force_error_deadband_n)
        )
        correction_position_limit = _readonly_copy(
            _positive_vector3(
                "correction_position_limit_m",
                self.correction_position_limit_m,
            )
        )
        correction_velocity_limit = _readonly_copy(
            _positive_vector3(
                "correction_velocity_limit_m_per_s",
                self.correction_velocity_limit_m_per_s,
            )
        )
        if not isinstance(self.contact_release_policy, str):
            raise ValueError("contact_release_policy must be 'reset' or 'freeze'")
        contact_release_policy = self.contact_release_policy.strip().lower()
        if contact_release_policy not in {"reset", "freeze"}:
            raise ValueError("contact_release_policy must be 'reset' or 'freeze'")
        joint_lower = _readonly_copy(_joint_vector("joint_lower", self.joint_lower))
        joint_upper = _readonly_copy(_joint_vector("joint_upper", self.joint_upper))
        joint_rate_limit = _readonly_copy(_joint_vector("joint_rate_limit", self.joint_rate_limit))
        if np.any(joint_lower >= joint_upper):
            raise ValueError("joint_lower must be strictly below joint_upper componentwise")
        if np.any(joint_rate_limit <= 0.0):
            raise ValueError("joint_rate_limit entries must be greater than zero")

        object.__setattr__(self, "transition_duration_s", duration)
        object.__setattr__(self, "touchdown_inertia", touchdown_inertia)
        object.__setattr__(self, "stance_inertia", stance_inertia)
        object.__setattr__(self, "touchdown_damping", touchdown_damping)
        object.__setattr__(self, "stance_damping", stance_damping)
        object.__setattr__(self, "restoring_stiffness", restoring_stiffness)
        object.__setattr__(self, "stance_stiffness", stance_stiffness)
        object.__setattr__(self, "force_error_deadband_n", force_error_deadband)
        object.__setattr__(self, "correction_position_limit_m", correction_position_limit)
        object.__setattr__(
            self,
            "correction_velocity_limit_m_per_s",
            correction_velocity_limit,
        )
        object.__setattr__(self, "contact_release_policy", contact_release_policy)
        object.__setattr__(self, "joint_lower", joint_lower)
        object.__setattr__(self, "joint_upper", joint_upper)
        object.__setattr__(self, "joint_rate_limit", joint_rate_limit)


@dataclass(frozen=True)
class AdmittanceState:
    """Body-frame compliant displacement and velocity for one foot."""

    correction_position_body: FloatArray
    correction_velocity_body: FloatArray
    contact_seen: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correction_position_body",
            _readonly_copy(
                _vector3("correction_position_body", self.correction_position_body)
            ),
        )
        object.__setattr__(
            self,
            "correction_velocity_body",
            _readonly_copy(
                _vector3("correction_velocity_body", self.correction_velocity_body)
            ),
        )
        if type(self.contact_seen) is not bool:
            raise TypeError("contact_seen must be a bool")


@dataclass(frozen=True)
class LegAdmittanceOutput:
    """Numerical leg command plus Eq. (51)--(56) observability signals."""

    blend: TouchdownBlend
    desired_force_body: FloatArray
    estimated_force_body: FloatArray
    raw_force_error_body: FloatArray
    admittance_force_body: FloatArray
    virtual_inertia: FloatArray
    virtual_damping: FloatArray
    effective_stiffness: FloatArray
    state: AdmittanceState
    nominal_foot_position_body: FloatArray
    unlimited_foot_position_body: FloatArray
    foot_position_command_body: FloatArray
    raw_joint_position: FloatArray
    bounded_joint_position: FloatArray
    joint_position_command: FloatArray
    workspace_limited: bool
    joint_position_limited: bool
    joint_rate_limited: bool
    correction_position_limited: bool
    correction_velocity_limited: bool
    contact_release_state_handled: bool
    joint_chain_state_projected: bool
    downstream_feedback_anti_windup_applied: bool
    anti_windup_applied: bool


@dataclass(frozen=True)
class AdmittanceTransition:
    """Opaque, immutable result of a non-mutating admittance preview.

    A transition is bound to both the controller instance that created it and
    that controller's current generation.  Callers may inspect ``output`` and
    submit its joint command, but only the originating controller can commit
    the tentative state.  A successful commit or reset advances the
    generation, invalidating every other outstanding transition.
    """

    output: LegAdmittanceOutput
    generation: int
    _owner_identity: object = field(repr=False, compare=False)
    _next_position: FloatArray = field(repr=False, compare=False)
    _next_velocity: FloatArray = field(repr=False, compare=False)
    _next_joint_command: FloatArray = field(repr=False, compare=False)
    _previous_position: FloatArray = field(repr=False, compare=False)
    _nominal_foot_position: FloatArray = field(repr=False, compare=False)
    _next_contact_seen: bool = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.output, LegAdmittanceOutput):
            raise TypeError("output must be a LegAdmittanceOutput")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an int")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        object.__setattr__(
            self,
            "_next_position",
            _readonly_copy(_vector3("next correction position", self._next_position)),
        )
        object.__setattr__(
            self,
            "_next_velocity",
            _readonly_copy(_vector3("next correction velocity", self._next_velocity)),
        )
        object.__setattr__(
            self,
            "_next_joint_command",
            _readonly_copy(_joint_vector("next joint command", self._next_joint_command)),
        )
        object.__setattr__(
            self,
            "_previous_position",
            _readonly_copy(_vector3("previous correction position", self._previous_position)),
        )
        object.__setattr__(
            self,
            "_nominal_foot_position",
            _readonly_copy(_vector3("nominal foot position", self._nominal_foot_position)),
        )
        if type(self._next_contact_seen) is not bool:
            raise TypeError("next_contact_seen must be a bool")


def _integrate_backward_euler(
    position: FloatArray,
    velocity: FloatArray,
    force: FloatArray,
    inertia: FloatArray,
    damping: FloatArray,
    stiffness: FloatArray,
    dt_s: float,
) -> Tuple[FloatArray, FloatArray]:
    """Implicitly integrate Eq. (55), preserving its continuous dynamics."""

    # Backward Euler with p[n+1] = p[n] + dt*v[n+1] gives this SPD system.
    # It is dissipative for the passive M/D/K model and avoids explicit-step
    # instability when landing stiffness or damping is large.
    try:
        with np.errstate(over="raise", invalid="raise"):
            system = inertia + dt_s * damping + dt_s * dt_s * stiffness
            right_hand_side = inertia @ velocity + dt_s * (force - stiffness @ position)
            velocity_next = np.asarray(
                np.linalg.solve(system, right_hand_side),
                dtype=np.float64,
            )
            position_next = position + dt_s * velocity_next
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("admittance integration failed numerically") from exc
    if not np.all(np.isfinite(position_next)) or not np.all(np.isfinite(velocity_next)):
        raise ValueError("admittance integration produced non-finite state")
    return position_next, velocity_next


class LegAdmittanceController:
    """Stateful low-level execution layer for one leg.

    The controller returns a bounded joint-position command only.  It has no
    transport or hardware side effects.  One instance should be used per leg.

    中文：``preview`` 只计算候选状态，不改变积分器；``commit`` 仅接受同一控制器、
    同一 generation 的 transition。上层可先计算四腿，在 IK/限幅全部成功后共同
    提交；``step`` 只是单腿便捷接口，不提供四腿原子性。
    """

    def __init__(
        self,
        config: AdmittanceConfig,
        inverse_kinematics: InverseKinematics,
        workspace_limiter: WorkspaceLimiter,
        initial_joint_command: ArrayLike,
        *,
        forward_kinematics: Optional[ForwardKinematics] = None,
    ) -> None:
        if not isinstance(config, AdmittanceConfig):
            raise TypeError("config must be an AdmittanceConfig")
        if not callable(inverse_kinematics):
            raise TypeError("inverse_kinematics must be callable")
        if not callable(getattr(workspace_limiter, "limit", None)):
            raise TypeError("workspace_limiter must provide a callable limit method")
        if forward_kinematics is not None and not callable(forward_kinematics):
            raise TypeError("forward_kinematics must be callable when provided")

        self._config = config
        self._inverse_kinematics = inverse_kinematics
        self._workspace_limiter = workspace_limiter
        self._touchdown_inertia = _spd_matrix(
            "touchdown_inertia",
            config.touchdown_inertia,
        )
        self._stance_inertia = _spd_matrix("stance_inertia", config.stance_inertia)
        self._touchdown_damping = _spd_matrix(
            "touchdown_damping",
            config.touchdown_damping,
        )
        self._stance_damping = _spd_matrix("stance_damping", config.stance_damping)
        self._restoring_stiffness = _spd_matrix(
            "restoring_stiffness",
            config.restoring_stiffness,
        )
        self._stance_stiffness = _spd_matrix(
            "stance_stiffness",
            config.stance_stiffness,
        )
        self._force_error_deadband = _nonnegative_vector3(
            "force_error_deadband_n",
            config.force_error_deadband_n,
        )
        self._correction_position_limit = _positive_vector3(
            "correction_position_limit_m",
            config.correction_position_limit_m,
        )
        self._correction_velocity_limit = _positive_vector3(
            "correction_velocity_limit_m_per_s",
            config.correction_velocity_limit_m_per_s,
        )
        self._joint_lower = _joint_vector("joint_lower", config.joint_lower)
        self._joint_upper = _joint_vector("joint_upper", config.joint_upper)
        self._joint_rate_limit = _joint_vector("joint_rate_limit", config.joint_rate_limit)
        self._forward_kinematics = forward_kinematics
        initial = _joint_vector("initial_joint_command", initial_joint_command)
        if np.any(initial < self._joint_lower) or np.any(initial > self._joint_upper):
            raise ValueError("initial_joint_command must lie within the joint bounds")

        self._position: FloatArray = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
        self._velocity: FloatArray = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
        self._previous_joint_command: FloatArray = initial
        self._contact_seen = False
        self._generation = 0
        self._owner_identity = object()
        self._state_lock = Lock()

    @property
    def state(self) -> AdmittanceState:
        with self._state_lock:
            return AdmittanceState(
                correction_position_body=_readonly_copy(self._position),
                correction_velocity_body=_readonly_copy(self._velocity),
                contact_seen=self._contact_seen,
            )

    @property
    def previous_joint_command(self) -> FloatArray:
        with self._state_lock:
            return _readonly_copy(self._previous_joint_command)

    @property
    def generation(self) -> int:
        """Return the generation used to invalidate speculative transitions."""

        with self._state_lock:
            return self._generation

    def reset(
        self,
        previous_joint_command: ArrayLike,
        *,
        correction_position_body: Optional[ArrayLike] = None,
        correction_velocity_body: Optional[ArrayLike] = None,
        contact_seen: bool = False,
    ) -> None:
        """Atomically reset Eq. (55) state and Eq. (56) command history."""

        previous = _joint_vector("previous_joint_command", previous_joint_command)
        if np.any(previous < self._joint_lower) or np.any(previous > self._joint_upper):
            raise ValueError("previous_joint_command must lie within the joint bounds")
        position: FloatArray
        if correction_position_body is None:
            position = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
        else:
            position = _vector3("correction_position_body", correction_position_body)
        velocity: FloatArray
        if correction_velocity_body is None:
            velocity = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
        else:
            velocity = _vector3("correction_velocity_body", correction_velocity_body)
        if np.any(np.abs(position) > self._correction_position_limit):
            raise ValueError("correction_position_body exceeds correction_position_limit_m")
        if np.any(np.abs(velocity) > self._correction_velocity_limit):
            raise ValueError(
                "correction_velocity_body exceeds correction_velocity_limit_m_per_s"
            )
        if type(contact_seen) is not bool:
            raise ValueError("contact_seen must be a bool")

        with self._state_lock:
            self._previous_joint_command = previous.copy()
            self._position = position.copy()
            self._velocity = velocity.copy()
            self._contact_seen = contact_seen
            self._generation += 1

    def preview(
        self,
        *,
        current_time_s: float,
        dt_s: float,
        measured_contact: bool,
        touchdown_time_s: Optional[float],
        rotation_body_to_world: ArrayLike,
        body_position_world: ArrayLike,
        nominal_foot_position_world: ArrayLike,
        desired_force_world: ArrayLike,
        estimated_force_world: ArrayLike,
    ) -> AdmittanceTransition:
        """Evaluate Eqs. (51)--(56) without changing controller state."""

        if type(measured_contact) is not bool:
            raise ValueError("measured_contact must be a bool")
        current = _finite_scalar("current_time_s", current_time_s)
        dt = _positive_scalar("dt_s", dt_s)
        if measured_contact and touchdown_time_s is None:
            raise ValueError("touchdown_time_s is required when measured_contact is true")
        if touchdown_time_s is not None:
            touchdown = _finite_scalar("touchdown_time_s", touchdown_time_s)
            if measured_contact and touchdown > current:
                raise ValueError("touchdown_time_s cannot be in the future for measured contact")

        rotation = _rotation_body_to_world(rotation_body_to_world)
        desired_world = _vector3("desired_force_world", desired_force_world)
        estimated_world = _vector3("estimated_force_world", estimated_force_world)
        body_world = _vector3("body_position_world", body_position_world)
        nominal_world = _vector3(
            "nominal_foot_position_world",
            nominal_foot_position_world,
        )

        # Capture a consistent starting point, then release the lock before
        # invoking numerical routines and caller-provided callbacks.  A reset
        # or competing commit during the preview makes this token stale rather
        # than allowing the old result to overwrite newer state.
        with self._state_lock:
            generation = self._generation
            position = self._position.copy()
            velocity = self._velocity.copy()
            previous_joint_command = self._previous_joint_command.copy()
            contact_seen = self._contact_seen

        blend = touchdown_blend(current, touchdown_time_s, self._config.transition_duration_s)
        desired_body = rotation.T @ desired_world
        estimated_body = rotation.T @ estimated_world
        nominal_body = rotation.T @ (nominal_world - body_world)

        inertia = (1.0 - blend.eta) * self._touchdown_inertia + blend.eta * self._stance_inertia
        damping = (1.0 - blend.eta) * self._touchdown_damping + blend.eta * self._stance_damping
        stiffness = (
            (1.0 - blend.eta) * self._restoring_stiffness
            + blend.eta * self._stance_stiffness
        )
        contact_multiplier = 1.0 if measured_contact else 0.0
        raw_force_error = contact_multiplier * (estimated_body - blend.eta * desired_body)
        admittance_force = _componentwise_deadband(
            raw_force_error,
            self._force_error_deadband,
        )

        contact_release_state_handled = contact_seen and not measured_contact
        correction_position_limited = False
        correction_velocity_limited = False
        if contact_release_state_handled:
            velocity_next = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
            if self._config.contact_release_policy == "reset":
                position_next = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
            else:
                position_next = position.copy()
        else:
            integrated_position, integrated_velocity = _integrate_backward_euler(
                position,
                velocity,
                admittance_force,
                inertia,
                damping,
                stiffness,
                dt,
            )
            velocity_next = np.clip(
                integrated_velocity,
                -self._correction_velocity_limit,
                self._correction_velocity_limit,
            )
            correction_velocity_limited = not np.allclose(
                integrated_velocity,
                velocity_next,
                rtol=0.0,
                atol=1.0e-12,
            )
            # Preserve p[n+1] = p[n] + dt*v[n+1] after velocity saturation.
            integrated_position = position + dt * velocity_next
            position_next = np.clip(
                integrated_position,
                -self._correction_position_limit,
                self._correction_position_limit,
            )
            correction_position_limited = not np.allclose(
                integrated_position,
                position_next,
                rtol=0.0,
                atol=1.0e-12,
            )
            if correction_position_limited:
                projection_error = integrated_position - position_next
                pushing_further_out = projection_error * velocity_next > 0.0
                velocity_next[pushing_further_out] = 0.0

        unlimited_foot = nominal_body + position_next
        limited_value = self._workspace_limiter.limit(unlimited_foot.copy())
        limited_foot = _vector3("workspace_limiter result", limited_value)
        workspace_limited = not np.allclose(
            unlimited_foot,
            limited_foot,
            rtol=0.0,
            atol=1e-12,
        )

        anti_windup_applied = False
        committed_position = position_next
        committed_velocity = velocity_next
        if self._config.anti_windup_enabled and workspace_limited:
            anti_windup_applied = True
            workspace_position = limited_foot - nominal_body
            if np.any(np.abs(workspace_position) > self._correction_position_limit + 1.0e-12):
                raise ValueError(
                    "workspace and correction_position_limit_m have no admissible "
                    "intersection for the nominal foot target"
                )
            committed_position = np.clip(
                workspace_position,
                -self._correction_position_limit,
                self._correction_position_limit,
            )
            committed_velocity = velocity_next.copy()
            projection_error = unlimited_foot - limited_foot
            pushing_further_out = projection_error * committed_velocity > 0.0
            committed_velocity[pushing_further_out] = 0.0

        raw_joint_value = self._inverse_kinematics(limited_foot.copy())
        raw_joint = _joint_vector("inverse_kinematics result", raw_joint_value)
        bounded_joint = np.clip(raw_joint, self._joint_lower, self._joint_upper)
        maximum_change = self._joint_rate_limit * dt
        rate_lower = previous_joint_command - maximum_change
        rate_upper = previous_joint_command + maximum_change
        joint_command = np.clip(bounded_joint, rate_lower, rate_upper)
        position_limited = not np.allclose(raw_joint, bounded_joint, rtol=0.0, atol=1e-12)
        rate_limited = not np.allclose(bounded_joint, joint_command, rtol=0.0, atol=1e-12)

        joint_chain_state_projected = False
        mapped_joint_foot: Optional[FloatArray] = None
        if self._forward_kinematics is not None:
            mapped_joint_foot = _vector3(
                "forward_kinematics result",
                self._forward_kinematics(joint_command.copy()),
            )
        ik_or_joint_mismatch = (
            position_limited
            or rate_limited
            or (
                mapped_joint_foot is not None
                and not np.allclose(
                    mapped_joint_foot,
                    limited_foot,
                    rtol=0.0,
                    atol=1.0e-7,
                )
            )
        )
        if (
            self._config.anti_windup_enabled
            and ik_or_joint_mismatch
            and not contact_release_state_handled
        ):
            anti_windup_applied = True
            joint_chain_state_projected = True
            committed_velocity = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
            if mapped_joint_foot is None:
                # Without FK there is no defensible Cartesian inverse for a
                # clipped q.  Preserve the last valid bounded state instead of
                # integrating farther into an unknown/unreachable target.
                committed_position = position.copy()
            else:
                mapped_position = mapped_joint_foot - nominal_body
                bounded_mapped_position = np.clip(
                    mapped_position,
                    -self._correction_position_limit,
                    self._correction_position_limit,
                )
                correction_position_limited = correction_position_limited or not np.allclose(
                    mapped_position,
                    bounded_mapped_position,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                committed_position = bounded_mapped_position

        output_state = AdmittanceState(
            correction_position_body=_readonly_copy(committed_position),
            correction_velocity_body=_readonly_copy(committed_velocity),
            contact_seen=(contact_seen or measured_contact),
        )
        output = LegAdmittanceOutput(
            blend=blend,
            desired_force_body=_readonly_copy(desired_body),
            estimated_force_body=_readonly_copy(estimated_body),
            raw_force_error_body=_readonly_copy(raw_force_error),
            admittance_force_body=_readonly_copy(admittance_force),
            virtual_inertia=_readonly_copy(inertia),
            virtual_damping=_readonly_copy(damping),
            effective_stiffness=_readonly_copy(stiffness),
            state=output_state,
            nominal_foot_position_body=_readonly_copy(nominal_body),
            unlimited_foot_position_body=_readonly_copy(unlimited_foot),
            foot_position_command_body=_readonly_copy(limited_foot),
            raw_joint_position=_readonly_copy(raw_joint),
            bounded_joint_position=_readonly_copy(bounded_joint),
            joint_position_command=_readonly_copy(joint_command),
            workspace_limited=workspace_limited,
            joint_position_limited=position_limited,
            joint_rate_limited=rate_limited,
            correction_position_limited=correction_position_limited,
            correction_velocity_limited=correction_velocity_limited,
            contact_release_state_handled=contact_release_state_handled,
            joint_chain_state_projected=joint_chain_state_projected,
            downstream_feedback_anti_windup_applied=False,
            anti_windup_applied=anti_windup_applied,
        )
        return AdmittanceTransition(
            output=output,
            generation=generation,
            _owner_identity=self._owner_identity,
            _next_position=committed_position,
            _next_velocity=committed_velocity,
            _next_joint_command=joint_command,
            _previous_position=position,
            _nominal_foot_position=nominal_body,
            _next_contact_seen=(contact_seen or measured_contact),
        )

    def commit(
        self,
        transition: AdmittanceTransition,
        applied_joint_position: Optional[ArrayLike] = None,
    ) -> LegAdmittanceOutput:
        """Commit one fresh preview, optionally using downstream applied-q feedback.

        ``applied_joint_position`` is the position actually accepted by the
        downstream command boundary after any additional safety clamping.  It
        becomes the next rate-limit reference; the previewed command remains
        available in the returned output for observability.
        """

        return self.commit_many(
            (self,),
            (transition,),
            (applied_joint_position,),
        )[0]

    @staticmethod
    def commit_many(
        controllers: Sequence[LegAdmittanceController],
        transitions: Sequence[AdmittanceTransition],
        applied_joint_positions: Optional[Sequence[Optional[ArrayLike]]] = None,
    ) -> Tuple[LegAdmittanceOutput, ...]:
        """Commit a group of leg previews atomically under a fixed lock order.

        Every token and downstream applied position is validated first.  All
        controller locks are then acquired in object-id order, every generation
        is revalidated, and only then are any states changed.  A concurrent
        reset therefore rejects the whole group instead of advancing a prefix.
        """

        group = tuple(controllers)
        tokens = tuple(transitions)
        if not group or len(group) != len(tokens):
            raise ValueError("controllers and transitions must have the same nonzero length")
        if not all(isinstance(controller, LegAdmittanceController) for controller in group):
            raise TypeError("controllers must contain only LegAdmittanceController values")
        if len({id(controller) for controller in group}) != len(group):
            raise ValueError("controllers cannot contain duplicate instances")
        if applied_joint_positions is None:
            applied_values: Tuple[Optional[ArrayLike], ...] = (None,) * len(group)
        else:
            applied_values = tuple(applied_joint_positions)
            if len(applied_values) != len(group):
                raise ValueError("applied_joint_positions must match the controller count")

        prepared = []
        for controller, transition, applied_value in zip(group, tokens, applied_values):
            controller._validate_transition_identity(transition)
            applied = controller._validated_applied_joint_position(
                transition,
                applied_value,
            )
            next_position, next_velocity, committed_output = (
                controller._state_for_applied_joint_position(
                    transition,
                    applied,
                )
            )
            prepared.append(
                (
                    controller,
                    transition,
                    applied,
                    next_position,
                    next_velocity,
                    committed_output,
                )
            )

        lock_order = sorted(group, key=id)
        for controller in lock_order:
            controller._state_lock.acquire()
        try:
            if any(
                transition.generation != controller._generation
                for controller, transition, *_ in prepared
            ):
                raise ValueError("one or more transitions are stale or already committed")
            for (
                controller,
                transition,
                applied,
                next_position,
                next_velocity,
                _,
            ) in prepared:
                controller._position = next_position.copy()
                controller._velocity = next_velocity.copy()
                controller._previous_joint_command = applied.copy()
                controller._contact_seen = transition._next_contact_seen
                controller._generation += 1
        finally:
            for controller in reversed(lock_order):
                controller._state_lock.release()
        return tuple(item[-1] for item in prepared)

    def validate_transition(
        self,
        transition: AdmittanceTransition,
        applied_joint_position: Optional[ArrayLike] = None,
    ) -> None:
        """Validate a tentative commit without changing controller state.

        This supports an all-legs preflight check: in a single-threaded caller,
        four transitions that all pass validation can then be committed without
        a partial state advance caused by a stale, replayed, or foreign token.
        If downstream applied-q feedback is available, pass it here as well so
        its shape, finiteness, and joint bounds are checked before any commit.
        """

        self._validate_transition_identity(transition)
        applied = self._validated_applied_joint_position(
            transition,
            applied_joint_position,
        )
        self._state_for_applied_joint_position(transition, applied)
        with self._state_lock:
            if transition.generation != self._generation:
                raise ValueError("transition generation is stale or already committed")

    def _validate_transition_identity(self, transition: AdmittanceTransition) -> None:
        if not isinstance(transition, AdmittanceTransition):
            raise TypeError("transition must be an AdmittanceTransition")
        if transition._owner_identity is not self._owner_identity:
            raise ValueError("transition belongs to a different controller")

    def _validated_applied_joint_position(
        self,
        transition: AdmittanceTransition,
        applied_joint_position: Optional[ArrayLike],
    ) -> FloatArray:
        if applied_joint_position is None:
            return transition._next_joint_command.copy()
        applied = _joint_vector("applied_joint_position", applied_joint_position)
        if np.any(applied < self._joint_lower) or np.any(applied > self._joint_upper):
            raise ValueError("applied_joint_position must lie within the joint bounds")
        return applied

    def _state_for_applied_joint_position(
        self,
        transition: AdmittanceTransition,
        applied_joint_position: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, LegAdmittanceOutput]:
        """Close a downstream q clamp back into the Cartesian integrator.

        This method is pure with respect to controller state and is therefore
        safe to execute during the all-leg preflight phase.  With an FK
        callback the state is projected to the actually staged joint target;
        without FK it is conservatively frozen at the previous valid state.
        """

        if (
            not self._config.anti_windup_enabled
            or transition.output.contact_release_state_handled
            or np.allclose(
                applied_joint_position,
                transition._next_joint_command,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            return (
                transition._next_position.copy(),
                transition._next_velocity.copy(),
                transition.output,
            )

        if self._forward_kinematics is None:
            next_position = transition._previous_position.copy()
            feedback_position_limited = False
        else:
            applied_foot = _vector3(
                "forward_kinematics result",
                self._forward_kinematics(applied_joint_position.copy()),
            )
            mapped_position = applied_foot - transition._nominal_foot_position
            next_position = np.clip(
                mapped_position,
                -self._correction_position_limit,
                self._correction_position_limit,
            )
            feedback_position_limited = not np.allclose(
                mapped_position,
                next_position,
                rtol=0.0,
                atol=1.0e-12,
            )
        next_velocity = np.zeros(_CARTESIAN_DIMENSION, dtype=np.float64)
        output = replace(
            transition.output,
            state=AdmittanceState(
                correction_position_body=_readonly_copy(next_position),
                correction_velocity_body=_readonly_copy(next_velocity),
                contact_seen=transition._next_contact_seen,
            ),
            correction_position_limited=(
                transition.output.correction_position_limited
                or feedback_position_limited
            ),
            joint_chain_state_projected=True,
            downstream_feedback_anti_windup_applied=True,
            anti_windup_applied=True,
        )
        return next_position, next_velocity, output

    def step(
        self,
        *,
        current_time_s: float,
        dt_s: float,
        measured_contact: bool,
        touchdown_time_s: Optional[float],
        rotation_body_to_world: ArrayLike,
        body_position_world: ArrayLike,
        nominal_foot_position_world: ArrayLike,
        desired_force_world: ArrayLike,
        estimated_force_world: ArrayLike,
    ) -> LegAdmittanceOutput:
        """Evaluate and atomically commit one compatible controller step."""

        transition = self.preview(
            current_time_s=current_time_s,
            dt_s=dt_s,
            measured_contact=measured_contact,
            touchdown_time_s=touchdown_time_s,
            rotation_body_to_world=rotation_body_to_world,
            body_position_world=body_position_world,
            nominal_foot_position_world=nominal_foot_position_world,
            desired_force_world=desired_force_world,
            estimated_force_world=estimated_force_world,
        )
        return self.commit(transition)


__all__ = [
    "AdmittanceConfig",
    "AdmittanceState",
    "AdmittanceTransition",
    "AxisAlignedWorkspace",
    "FloatArray",
    "ForwardKinematics",
    "InverseKinematics",
    "LegAdmittanceController",
    "LegAdmittanceOutput",
    "TouchdownBlend",
    "WorkspaceLimiter",
    "nominal_foot_position_body",
    "scheduled_spd_matrix",
    "touchdown_blend",
    "world_to_body_force",
]
