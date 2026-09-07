"""Strict one-dimensional normal-force observation and admittance model.

This module is deliberately isolated from every hardware writer.  It closes an
offline, auditable chain from one explicitly typed force observation to a
Cartesian correction that is *only* parallel to the supplied ground normal.
It must not be treated as permission to command a Go2 robot.

中文说明：本文件只实现“一维法向观测 -> 一维导纳位移”的离线数值边界。
Go2 的原始 ``foot_force`` 计数只能用于接触事件；只有完成标定后得到的牛顿值，
或独立三维力传感器给出的世界系力，才能进入导纳方程。所有输出位移和速度都严格
沿单位地面法向，不生成切向命令；本模块也永远不授予任何实机输出权限。
控制器在第一次有效接触候选被 ``commit`` 时绑定地面法向；同一会话内不得更换，
只有显式调用 ``reset`` 清空积分状态和方向身份后，才能绑定新的地面法向。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from threading import Lock
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    # Python 3.8 在运行时不解析 ``TypeAlias``，但 NumPy 别名仍保持可下标。
    FloatArray = NDArray[np.float64]

_VECTOR_DIMENSION = 3
_UNIT_VECTOR_TOLERANCE = 1e-8
_NEGATIVE_NORMAL_FORCE_TOLERANCE_N = 1e-9

# 本模块是离线参考实现；常量不可由配置或调用方改写。
HARDWARE_OUTPUT_PERMITTED = False


class NormalAdmittanceError(ValueError):
    """Raised when observation semantics or a transition are unsafe/ambiguous."""


class ForceObservationMode(str, Enum):
    """Mutually exclusive semantics of the input force observation."""

    CONTACT_EVENT_ONLY_COUNTS = "CONTACT_EVENT_ONLY_COUNTS"
    CALIBRATED_NORMAL_ONLY_N = "CALIBRATED_NORMAL_ONLY_N"
    INDEPENDENT_3D_WORLD_N = "INDEPENDENT_3D_WORLD_N"


class ContactLossPolicy(str, Enum):
    """State action when the contact detector reports contact loss."""

    RESET = "RESET"
    FREEZE = "FREEZE"


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_scalar(name: str, value: object) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _nonnegative_scalar(name: str, value: object) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _readonly_vector3(name: str, value: ArrayLike) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real vector with shape (3,)") from exc
    if raw.shape != (_VECTOR_DIMENSION,) or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a finite real vector with shape (3,)")
    result: FloatArray = np.asarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector with shape (3,)")
    result.setflags(write=False)
    return result


def _unit_ground_normal(value: ArrayLike) -> FloatArray:
    normal = _readonly_vector3("ground_normal_world", value)
    norm = float(np.linalg.norm(normal))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_VECTOR_TOLERANCE):
        raise ValueError("ground_normal_world must be a unit vector")
    return normal


@dataclass(frozen=True)
class ForceObservation:
    """One force observation with exactly one declared physical meaning.

    Construct the object with only the field corresponding to ``mode``.  This
    prevents a caller from supplying a scalar calibrated normal force and a
    conflicting three-dimensional force in the same sample.
    """

    mode: ForceObservationMode
    contact_detected: bool
    contact_count: Optional[int] = None
    calibrated_normal_force_n: Optional[float] = None
    independent_force_world_n: Optional[ArrayLike] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ForceObservationMode):
            raise TypeError("mode must be a ForceObservationMode")
        if type(self.contact_detected) is not bool:
            raise TypeError("contact_detected must be a bool")

        if self.mode is ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS:
            if (
                isinstance(self.contact_count, bool)
                or not isinstance(self.contact_count, Integral)
            ):
                raise TypeError("contact_count must be an integer SDK count")
            count = int(self.contact_count)
            if count < -32768 or count > 32767:
                raise ValueError("contact_count must fit the Go2 signed int16 field")
            if self.calibrated_normal_force_n is not None:
                raise NormalAdmittanceError(
                    "contact-only counts cannot also contain calibrated newtons"
                )
            if self.independent_force_world_n is not None:
                raise NormalAdmittanceError(
                    "contact-only counts cannot also contain a three-dimensional force"
                )
            object.__setattr__(self, "contact_count", count)
            return

        if self.mode is ForceObservationMode.CALIBRATED_NORMAL_ONLY_N:
            if self.contact_count is not None or self.independent_force_world_n is not None:
                raise NormalAdmittanceError(
                    "calibrated normal-only mode accepts only one scalar normal force"
                )
            normal_force = _nonnegative_scalar(
                "calibrated_normal_force_n",
                self.calibrated_normal_force_n,
            )
            object.__setattr__(self, "calibrated_normal_force_n", normal_force)
            return

        if self.contact_count is not None or self.calibrated_normal_force_n is not None:
            raise NormalAdmittanceError(
                "independent 3D mode accepts only one world-frame force vector"
            )
        if self.independent_force_world_n is None:
            raise TypeError("independent_force_world_n is required in independent 3D mode")
        object.__setattr__(
            self,
            "independent_force_world_n",
            _readonly_vector3(
                "independent_force_world_n",
                self.independent_force_world_n,
            ),
        )


@dataclass(frozen=True)
class ResolvedNormalForce:
    """A scalar normal force and its uniquely derived normal-only world vector."""

    normal_force_n: float
    force_world_n: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_force_n",
            _nonnegative_scalar("normal_force_n", self.normal_force_n),
        )
        object.__setattr__(
            self,
            "force_world_n",
            _readonly_vector3("force_world_n", self.force_world_n),
        )


def resolve_normal_force(
    observation: ForceObservation,
    ground_normal_world: ArrayLike,
) -> ResolvedNormalForce:
    """Resolve an eligible observation into the one-dimensional normal channel.

    Raw SDK counts are intentionally rejected: without robot-specific force
    calibration they do not have newton units and cannot appear in the
    admittance equation.
    """

    if not isinstance(observation, ForceObservation):
        raise TypeError("observation must be a ForceObservation")
    normal = _unit_ground_normal(ground_normal_world)

    if observation.mode is ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS:
        raise NormalAdmittanceError(
            "CONTACT_EVENT_ONLY_COUNTS has no newton semantics and cannot drive admittance"
        )
    if observation.mode is ForceObservationMode.CALIBRATED_NORMAL_ONLY_N:
        assert observation.calibrated_normal_force_n is not None
        scalar = observation.calibrated_normal_force_n
    else:
        assert observation.independent_force_world_n is not None
        scalar = float(np.dot(observation.independent_force_world_n, normal))
        if scalar < -_NEGATIVE_NORMAL_FORCE_TOLERANCE_N:
            raise NormalAdmittanceError(
                "independent world force has a negative projection on the ground normal"
            )
        # 仅吸收浮点舍入产生的极小负值，不掩盖符号或坐标系错误。
        scalar = max(0.0, scalar)

    return ResolvedNormalForce(
        normal_force_n=scalar,
        force_world_n=scalar * normal,
    )


@dataclass(frozen=True)
class NormalAdmittanceConfig:
    """Positive scalar parameters and hard bounds for one normal axis."""

    virtual_mass_kg: float
    damping_n_s_per_m: float
    stance_stiffness_n_per_m: float
    force_error_deadband_n: float
    correction_position_limit_m: float
    correction_velocity_limit_m_per_s: float
    contact_loss_policy: ContactLossPolicy = ContactLossPolicy.RESET

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "virtual_mass_kg",
            _positive_scalar("virtual_mass_kg", self.virtual_mass_kg),
        )
        object.__setattr__(
            self,
            "damping_n_s_per_m",
            _positive_scalar("damping_n_s_per_m", self.damping_n_s_per_m),
        )
        # 着陆后仍保留严格正刚度，避免测力偏差造成无界速度/位移漂移。
        object.__setattr__(
            self,
            "stance_stiffness_n_per_m",
            _positive_scalar(
                "stance_stiffness_n_per_m",
                self.stance_stiffness_n_per_m,
            ),
        )
        object.__setattr__(
            self,
            "force_error_deadband_n",
            _nonnegative_scalar("force_error_deadband_n", self.force_error_deadband_n),
        )
        object.__setattr__(
            self,
            "correction_position_limit_m",
            _positive_scalar(
                "correction_position_limit_m",
                self.correction_position_limit_m,
            ),
        )
        object.__setattr__(
            self,
            "correction_velocity_limit_m_per_s",
            _positive_scalar(
                "correction_velocity_limit_m_per_s",
                self.correction_velocity_limit_m_per_s,
            ),
        )
        if not isinstance(self.contact_loss_policy, ContactLossPolicy):
            raise TypeError("contact_loss_policy must be a ContactLossPolicy")


@dataclass(frozen=True)
class NormalAdmittanceState:
    """Committed scalar displacement/velocity along the current ground normal."""

    correction_position_m: float = 0.0
    correction_velocity_m_per_s: float = 0.0
    contact_seen: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correction_position_m",
            _finite_scalar("correction_position_m", self.correction_position_m),
        )
        object.__setattr__(
            self,
            "correction_velocity_m_per_s",
            _finite_scalar(
                "correction_velocity_m_per_s",
                self.correction_velocity_m_per_s,
            ),
        )
        if type(self.contact_seen) is not bool:
            raise TypeError("contact_seen must be a bool")


@dataclass(frozen=True)
class NormalAdmittanceOutput:
    """Observable result of one preview; vectors have no tangential component."""

    observation_mode: ForceObservationMode
    contact_detected: bool
    ground_normal_world: FloatArray
    estimated_normal_force_n: Optional[float]
    desired_normal_force_n: float
    raw_force_error_n: Optional[float]
    admittance_force_error_n: Optional[float]
    state: NormalAdmittanceState
    correction_position_world_m: FloatArray
    correction_velocity_world_m_per_s: FloatArray
    correction_position_limited: bool
    correction_velocity_limited: bool
    contact_loss_state_handled: bool
    hardware_output_permitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_mode, ForceObservationMode):
            raise TypeError("observation_mode must be a ForceObservationMode")
        if type(self.contact_detected) is not bool:
            raise TypeError("contact_detected must be a bool")
        object.__setattr__(
            self,
            "ground_normal_world",
            _unit_ground_normal(self.ground_normal_world),
        )
        if self.estimated_normal_force_n is not None:
            object.__setattr__(
                self,
                "estimated_normal_force_n",
                _nonnegative_scalar(
                    "estimated_normal_force_n",
                    self.estimated_normal_force_n,
                ),
            )
        object.__setattr__(
            self,
            "desired_normal_force_n",
            _nonnegative_scalar("desired_normal_force_n", self.desired_normal_force_n),
        )
        for name in ("raw_force_error_n", "admittance_force_error_n"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_scalar(name, value))
        if not isinstance(self.state, NormalAdmittanceState):
            raise TypeError("state must be a NormalAdmittanceState")
        object.__setattr__(
            self,
            "correction_position_world_m",
            _readonly_vector3(
                "correction_position_world_m",
                self.correction_position_world_m,
            ),
        )
        object.__setattr__(
            self,
            "correction_velocity_world_m_per_s",
            _readonly_vector3(
                "correction_velocity_world_m_per_s",
                self.correction_velocity_world_m_per_s,
            ),
        )
        for name in (
            "correction_position_limited",
            "correction_velocity_limited",
            "contact_loss_state_handled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.hardware_output_permitted is not HARDWARE_OUTPUT_PERMITTED:
            raise NormalAdmittanceError("normal admittance never permits hardware output")


@dataclass(frozen=True)
class NormalAdmittanceTransition:
    """Opaque preview token that can be committed once or explicitly aborted."""

    output: NormalAdmittanceOutput
    generation: int
    nonce: int
    _owner_identity: object = field(repr=False, compare=False)
    _next_state: NormalAdmittanceState = field(repr=False, compare=False)
    _ground_normal_to_bind: Optional[FloatArray] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.output, NormalAdmittanceOutput):
            raise TypeError("output must be a NormalAdmittanceOutput")
        for name in ("generation", "nonce"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self._next_state, NormalAdmittanceState):
            raise TypeError("next_state must be a NormalAdmittanceState")
        if self._ground_normal_to_bind is not None:
            object.__setattr__(
                self,
                "_ground_normal_to_bind",
                _unit_ground_normal(self._ground_normal_to_bind),
            )


def _deadband(value: float, half_width: float) -> float:
    return math.copysign(max(abs(value) - half_width, 0.0), value)


class NormalAdmittanceController:
    """Thread-safe scalar normal admittance with preview/commit/abort semantics."""

    def __init__(
        self,
        config: NormalAdmittanceConfig,
        initial_state: Optional[NormalAdmittanceState] = None,
        initial_ground_normal_world: Optional[ArrayLike] = None,
    ) -> None:
        if not isinstance(config, NormalAdmittanceConfig):
            raise TypeError("config must be a NormalAdmittanceConfig")
        if initial_state is not None and not isinstance(initial_state, NormalAdmittanceState):
            raise TypeError("initial_state must be a NormalAdmittanceState or None")
        state = initial_state or NormalAdmittanceState()
        if abs(state.correction_position_m) > config.correction_position_limit_m:
            raise ValueError("initial correction position exceeds its hard limit")
        if abs(state.correction_velocity_m_per_s) > config.correction_velocity_limit_m_per_s:
            raise ValueError("initial correction velocity exceeds its hard limit")
        bound_normal = (
            None
            if initial_ground_normal_world is None
            else _unit_ground_normal(initial_ground_normal_world)
        )
        # 非零标量状态若没有方向身份就无法安全解释；禁止在第一次 preview 时
        # 把历史位移/速度任意投影到调用方临时给出的新方向。
        state_has_directional_history = (
            state.correction_position_m != 0.0
            or state.correction_velocity_m_per_s != 0.0
            or state.contact_seen
        )
        if state_has_directional_history and bound_normal is None:
            raise NormalAdmittanceError(
                "a nonzero/contact-seen initial state requires initial_ground_normal_world"
            )
        self._config = config
        self._state = state
        self._bound_ground_normal = bound_normal
        self._generation = 0
        self._next_nonce = 0
        self._aborted_nonces: set[int] = set()
        self._owner_identity = object()
        self._lock = Lock()

    @property
    def hardware_output_permitted(self) -> bool:
        """Always false: this controller is not an actuator authorization boundary."""

        return HARDWARE_OUTPUT_PERMITTED

    def state(self) -> NormalAdmittanceState:
        """Return the immutable committed state."""

        with self._lock:
            return self._state

    def bound_ground_normal_world(self) -> Optional[FloatArray]:
        """Return the session's bound normal, or ``None`` before first commit.

        The first committed contact-bearing transition binds the direction.
        Preview and abort remain non-mutating; only an explicit ``reset``
        clears the binding and permits a different ground normal.
        """

        with self._lock:
            if self._bound_ground_normal is None:
                return None
            return _readonly_vector3(
                "bound_ground_normal_world",
                self._bound_ground_normal,
            )

    def reset(self) -> NormalAdmittanceState:
        """Start a new session: reset states, normal identity, and preview tokens."""

        with self._lock:
            self._state = NormalAdmittanceState()
            self._bound_ground_normal = None
            self._generation += 1
            self._aborted_nonces.clear()
            return self._state

    def preview(
        self,
        *,
        observation: ForceObservation,
        desired_normal_force_n: float,
        ground_normal_world: ArrayLike,
        dt_s: float,
    ) -> NormalAdmittanceTransition:
        """Compute a bounded candidate without advancing the committed integrator."""

        if not isinstance(observation, ForceObservation):
            raise TypeError("observation must be a ForceObservation")
        desired = _nonnegative_scalar("desired_normal_force_n", desired_normal_force_n)
        dt = _positive_scalar("dt_s", dt_s)
        normal = _unit_ground_normal(ground_normal_world)

        with self._lock:
            if self._bound_ground_normal is not None and not np.allclose(
                normal,
                self._bound_ground_normal,
                rtol=0.0,
                atol=_UNIT_VECTOR_TOLERANCE,
            ):
                raise NormalAdmittanceError(
                    "ground normal differs from the committed session; call reset first"
                )
            previous = self._state
            generation = self._generation
            nonce = self._next_nonce
            self._next_nonce += 1

            estimated: Optional[float]
            raw_error: Optional[float]
            applied_error: Optional[float]
            position_limited = False
            velocity_limited = False
            loss_handled = not observation.contact_detected

            if not observation.contact_detected:
                # 接触计数在此只决定“已失去接触”，绝不换算为牛顿或进入积分器。
                estimated = None
                raw_error = None
                applied_error = None
                if self._config.contact_loss_policy is ContactLossPolicy.RESET:
                    next_state = NormalAdmittanceState()
                else:
                    next_state = NormalAdmittanceState(
                        correction_position_m=previous.correction_position_m,
                        correction_velocity_m_per_s=0.0,
                        contact_seen=False,
                    )
            else:
                resolved = resolve_normal_force(observation, normal)
                estimated = resolved.normal_force_n
                raw_error = estimated - desired
                applied_error = _deadband(
                    raw_error,
                    self._config.force_error_deadband_n,
                )

                # 后向欧拉离散化：
                # M(v1-v0)/dt + D*v1 + K*x1 = e, x1=x0+dt*v1。
                denominator = (
                    self._config.virtual_mass_kg
                    + dt * self._config.damping_n_s_per_m
                    + dt * dt * self._config.stance_stiffness_n_per_m
                )
                velocity_unlimited = (
                    self._config.virtual_mass_kg * previous.correction_velocity_m_per_s
                    + dt
                    * (
                        applied_error
                        - self._config.stance_stiffness_n_per_m
                        * previous.correction_position_m
                    )
                ) / denominator
                velocity = min(
                    self._config.correction_velocity_limit_m_per_s,
                    max(
                        -self._config.correction_velocity_limit_m_per_s,
                        velocity_unlimited,
                    ),
                )
                velocity_limited = not math.isclose(
                    velocity,
                    velocity_unlimited,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                position_unlimited = previous.correction_position_m + dt * velocity
                position = min(
                    self._config.correction_position_limit_m,
                    max(-self._config.correction_position_limit_m, position_unlimited),
                )
                position_limited = not math.isclose(
                    position,
                    position_unlimited,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                # 位置饱和后清除继续向外的速度，形成简单的积分 anti-windup。
                if position_limited and position_unlimited * velocity > 0.0:
                    velocity = 0.0
                next_state = NormalAdmittanceState(
                    correction_position_m=position,
                    correction_velocity_m_per_s=velocity,
                    contact_seen=True,
                )

            output = NormalAdmittanceOutput(
                observation_mode=observation.mode,
                contact_detected=observation.contact_detected,
                ground_normal_world=normal,
                estimated_normal_force_n=estimated,
                desired_normal_force_n=desired,
                raw_force_error_n=raw_error,
                admittance_force_error_n=applied_error,
                state=next_state,
                correction_position_world_m=next_state.correction_position_m * normal,
                correction_velocity_world_m_per_s=(
                    next_state.correction_velocity_m_per_s * normal
                ),
                correction_position_limited=position_limited,
                correction_velocity_limited=velocity_limited,
                contact_loss_state_handled=loss_handled,
            )
            return NormalAdmittanceTransition(
                output=output,
                generation=generation,
                nonce=nonce,
                _owner_identity=self._owner_identity,
                _next_state=next_state,
                # A no-contact preview cannot establish a physical correction
                # direction.  It therefore leaves an unbound session unbound.
                _ground_normal_to_bind=(normal if observation.contact_detected else None),
            )

    def commit(self, transition: NormalAdmittanceTransition) -> NormalAdmittanceOutput:
        """Commit one fresh candidate and invalidate all same-generation previews."""

        if not isinstance(transition, NormalAdmittanceTransition):
            raise TypeError("transition must be a NormalAdmittanceTransition")
        with self._lock:
            self._validate_transition_locked(transition)
            self._state = transition._next_state
            if self._bound_ground_normal is None:
                self._bound_ground_normal = transition._ground_normal_to_bind
            self._generation += 1
            self._aborted_nonces.clear()
            return transition.output

    def abort(self, transition: NormalAdmittanceTransition) -> None:
        """Invalidate one preview while leaving the committed state unchanged."""

        if not isinstance(transition, NormalAdmittanceTransition):
            raise TypeError("transition must be a NormalAdmittanceTransition")
        with self._lock:
            self._validate_transition_locked(transition)
            self._aborted_nonces.add(transition.nonce)

    def _validate_transition_locked(self, transition: NormalAdmittanceTransition) -> None:
        if transition._owner_identity is not self._owner_identity:
            raise NormalAdmittanceError("transition belongs to a different controller")
        if transition.generation != self._generation:
            raise NormalAdmittanceError("transition is stale or already committed")
        if transition.nonce in self._aborted_nonces:
            raise NormalAdmittanceError("transition was aborted")


__all__ = [
    "HARDWARE_OUTPUT_PERMITTED",
    "ContactLossPolicy",
    "ForceObservation",
    "ForceObservationMode",
    "NormalAdmittanceConfig",
    "NormalAdmittanceController",
    "NormalAdmittanceError",
    "NormalAdmittanceOutput",
    "NormalAdmittanceState",
    "NormalAdmittanceTransition",
    "ResolvedNormalForce",
    "resolve_normal_force",
]
