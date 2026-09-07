"""Finite-horizon impact-aware nonlinear MPC from paper Eqs. (45)-(50).

This module is a pure numerical reference implementation.  It has no bridge,
SDK, flight-controller, or hardware imports and cannot transmit commands.

The nonlinear program uses direct multiple shooting: every post-step state,
control, selected state-bound slack, and touchdown impulse is an explicit
decision variable as in Eq. (45).  The continuous step followed by the
optional momentum reset is imposed as an equality (Eq. 46), rather than being
hidden in a single-shooting rollout.  Attitude decision variables use a
three-dimensional rotation-vector chart; public trajectories remain
``ReducedState`` objects with proper rotation matrices.

All physical values, references, bounds, weights, contact schedules, foot
geometry/velocity data, and solver limits are caller supplied.  There are no
paper-specific numerical defaults.  SciPy SLSQP is imported lazily and is
intended as a deterministic dry-run/reference backend, not a hard real-time
solver.

中文说明：该 NLP 是论文公式的可审计数值基准，不是已经满足实时性的上机求解器。
状态、控制量、触地冲量和松弛变量均作为直接多重射击决策变量；动力学、冲击重置、
接触力和旋翼执行约束分别形成等式/不等式。只有 ``success``、残差和变量边界同时
通过时，首个控制量才可交给下游。原生 SLSQP 运行在可终止的独立 ``spawn`` 进程中；
即便如此，它仍只是离线/影子模式参考求解器，必须在目标 aarch64 机载计算机完成
WCET、抖动、资源隔离和故障注入验证后，才能讨论实时部署。
"""

from __future__ import annotations

import math
import multiprocessing
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from aerogo2.landing.impact_aware.dynamics import reduced_discrete_step
from aerogo2.landing.impact_aware.impact import (
    foot_post_impact_velocity,
    momentum_reset,
)
from aerogo2.landing.impact_aware.math_utils import (
    require_rotation_matrix,
    skew,
    so3_exp,
)
from aerogo2.landing.impact_aware.rotor import (
    evaluate_rotor_constraints,
    first_order_thrust_rate,
)
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    ROTOR_COUNT,
    FloatArray,
    FootLeverArmsFromComBodyHorizon,
    ImpactLimits,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
    validate_four_foot_leg_order,
)

STATE_DIM = 16
CONTROL_DIM = 16
TRACKING_DIM = 12

_POSITION = slice(0, 3)
_LINEAR_VELOCITY = slice(3, 6)
_ROTATION_VECTOR = slice(6, 9)
_ANGULAR_VELOCITY = slice(9, 12)
_ROTOR_THRUST = slice(12, 16)
_CONTACT_FORCE = slice(0, 12)
_ROTOR_COMMAND = slice(12, 16)

# Inverting the transport blend below this gain magnifies an applied-command
# round-off error into an unbounded raw correction.  Gains below this explicit
# numerical floor are therefore treated as disabled at the API boundary.
MINIMUM_RECONSTRUCTABLE_CORRECTION_GAIN = 1.0e-6


def _readonly(values: object, *, dtype: DTypeLike = float) -> NDArray[Any]:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return cast(FloatArray, result)


def _finite_array(name: str, values: object, shape: Tuple[int, ...]) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, result)


def _strict_finite_array(name: str, values: object, shape: Tuple[int, ...]) -> FloatArray:
    """Validate numeric arrays without silently coercing bool or text entries."""

    try:
        object_values = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    for item in object_values.flat:
        if isinstance(item, (bool, np.bool_, str, np.str_, bytes, np.bytes_)):
            raise TypeError(f"{name} must contain real numeric values, not bool or text")
    return _finite_array(name, values, shape)


def _bounded_array(name: str, values: object, shape: Tuple[int, ...]) -> FloatArray:
    """Validate an array that may use infinities to denote absent bounds."""

    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} cannot contain NaN")
    return cast(FloatArray, result)


def _finite_scalar(
    name: str,
    value: object,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _binary_schedule(name: str, values: object, shape: Tuple[int, ...]) -> NDArray[np.int8]:
    numeric = _finite_array(name, values, shape)
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only 0 or 1")
    return numeric.astype(np.int8)


def _boolean_mask(name: str, values: object, shape: Tuple[int, ...]) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind != "b":
        raise TypeError(f"{name} must be a boolean array")
    return np.array(raw, dtype=bool, copy=True)


def _weight_matrix(name: str, values: object, dimension: int) -> FloatArray:
    matrix = _finite_array(name, values, (dimension, dimension))
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _quadratic(vector: FloatArray, weight: FloatArray) -> float:
    return float(vector @ weight @ vector)


def _rotation_matrix_to_vector(rotation: object) -> FloatArray:
    """Return the principal rotation vector for a validated SO(3) matrix."""

    matrix = require_rotation_matrix(rotation, name="rotation_body_to_world", atol=1e-7)
    trace = float(np.trace(matrix))
    quaternion = np.empty(4, dtype=float)  # scalar first
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion[0] = 0.25 * scale
        quaternion[1] = (matrix[2, 1] - matrix[1, 2]) / scale
        quaternion[2] = (matrix[0, 2] - matrix[2, 0]) / scale
        quaternion[3] = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion[0] = (matrix[2, 1] - matrix[1, 2]) / scale
            quaternion[1] = 0.25 * scale
            quaternion[2] = (matrix[0, 1] + matrix[1, 0]) / scale
            quaternion[3] = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion[0] = (matrix[0, 2] - matrix[2, 0]) / scale
            quaternion[1] = (matrix[0, 1] + matrix[1, 0]) / scale
            quaternion[2] = 0.25 * scale
            quaternion[3] = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion[0] = (matrix[1, 0] - matrix[0, 1]) / scale
            quaternion[1] = (matrix[0, 2] + matrix[2, 0]) / scale
            quaternion[2] = (matrix[1, 2] + matrix[2, 1]) / scale
            quaternion[3] = 0.25 * scale

    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("rotation could not be converted to a finite quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-12:
        return cast(FloatArray, np.zeros(3, dtype=float))
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    return cast(FloatArray, quaternion[1:] * (angle / vector_norm))


def _state_to_vector(state: ReducedState) -> FloatArray:
    if not isinstance(state, ReducedState):
        raise TypeError("state must be a ReducedState")
    return cast(
        FloatArray,
        np.concatenate(
            (
                state.position_world_m,
                state.linear_velocity_world_m_per_s,
                _rotation_matrix_to_vector(state.rotation_body_to_world),
                state.angular_velocity_body_rad_per_s,
                state.rotor_thrusts_n,
            ),
        ),
    )


def _vector_to_state(vector: object) -> ReducedState:
    values = _finite_array("state_vector", vector, (STATE_DIM,))
    return ReducedState(
        position_world_m=values[_POSITION],
        linear_velocity_world_m_per_s=values[_LINEAR_VELOCITY],
        rotation_body_to_world=so3_exp(values[_ROTATION_VECTOR]),
        angular_velocity_body_rad_per_s=values[_ANGULAR_VELOCITY],
        rotor_thrusts_n=values[_ROTOR_THRUST],
    )


def _input_to_vector(control: ReducedInput) -> FloatArray:
    if not isinstance(control, ReducedInput):
        raise TypeError("control must be a ReducedInput")
    return cast(
        FloatArray,
        np.concatenate(
            (control.contact_forces_world_n.reshape(12), control.rotor_thrust_commands_n)
        ),
    )


def _vector_to_input(vector: object) -> ReducedInput:
    values = _finite_array("control_vector", vector, (CONTROL_DIM,))
    return ReducedInput(
        contact_forces_world_n=values[_CONTACT_FORCE].reshape(FOOT_COUNT, 3),
        rotor_thrust_commands_n=values[_ROTOR_COMMAND],
    )


def _state_transition_residual(actual: ReducedState, expected: ReducedState) -> FloatArray:
    attitude_error = _rotation_matrix_to_vector(
        expected.rotation_body_to_world.T @ actual.rotation_body_to_world
    )
    return cast(
        FloatArray,
        np.concatenate(
            (
                actual.position_world_m - expected.position_world_m,
                actual.linear_velocity_world_m_per_s - expected.linear_velocity_world_m_per_s,
                attitude_error,
                actual.angular_velocity_body_rad_per_s - expected.angular_velocity_body_rad_per_s,
                actual.rotor_thrusts_n - expected.rotor_thrusts_n,
            ),
        ),
    )


def _tracking_error(state: ReducedState, references: MPCReferences, step: int) -> FloatArray:
    reference_rotation = references.rotation_body_to_world[step]
    # The skew/vee attitude error vanishes at an exact pi rotation.  The
    # principal SO(3) logarithm remains a geodesic error with norm pi there;
    # _rotation_matrix_to_vector uses a quaternion branch that is well-defined
    # at the cut locus (the sign of the axis is immaterial to a quadratic cost).
    attitude_error = _rotation_matrix_to_vector(
        reference_rotation.T @ state.rotation_body_to_world
    )
    return cast(
        FloatArray,
        np.concatenate(
            (
                state.position_world_m - references.position_world_m[step],
                state.linear_velocity_world_m_per_s
                - references.linear_velocity_world_m_per_s[step],
                attitude_error,
                state.angular_velocity_body_rad_per_s
                - references.angular_velocity_body_rad_per_s[step],
            ),
        ),
    )


@dataclass(frozen=True)
class MPCReferences:
    """Caller-supplied phase references for Eqs. (47)-(49).

    State reference arrays have ``N + 1`` samples.  Input reference arrays
    have ``N`` samples.  ``rotor_thrust_commands_n`` always references the
    actual applied total command after any safety-gain blend.  No interpolation
    or phase values are invented here.
    """

    position_world_m: FloatArray
    linear_velocity_world_m_per_s: FloatArray
    rotation_body_to_world: FloatArray
    angular_velocity_body_rad_per_s: FloatArray
    contact_forces_world_n: FloatArray
    rotor_thrust_commands_n: FloatArray

    def __post_init__(self) -> None:
        position_raw = np.asarray(self.position_world_m)
        if position_raw.ndim != 2 or position_raw.shape[1:] != (3,):
            raise ValueError("position_world_m must have shape (N + 1, 3)")
        horizon = position_raw.shape[0] - 1
        if horizon < 1:
            raise ValueError("references must contain at least one prediction interval")
        state_count = horizon + 1
        position = _finite_array("position_world_m", position_raw, (state_count, 3))
        linear_velocity = _finite_array(
            "linear_velocity_world_m_per_s",
            self.linear_velocity_world_m_per_s,
            (state_count, 3),
        )
        rotations = _finite_array(
            "rotation_body_to_world",
            self.rotation_body_to_world,
            (state_count, 3, 3),
        )
        for step in range(state_count):
            rotations[step] = require_rotation_matrix(
                rotations[step],
                name=f"rotation_body_to_world[{step}]",
            )
        angular_velocity = _finite_array(
            "angular_velocity_body_rad_per_s",
            self.angular_velocity_body_rad_per_s,
            (state_count, 3),
        )
        contact_forces = _finite_array(
            "contact_forces_world_n",
            self.contact_forces_world_n,
            (horizon, FOOT_COUNT, 3),
        )
        rotor_commands = _finite_array(
            "rotor_thrust_commands_n",
            self.rotor_thrust_commands_n,
            (horizon, ROTOR_COUNT),
        )
        for name, value in (
            ("position_world_m", position),
            ("linear_velocity_world_m_per_s", linear_velocity),
            ("rotation_body_to_world", rotations),
            ("angular_velocity_body_rad_per_s", angular_velocity),
            ("contact_forces_world_n", contact_forces),
            ("rotor_thrust_commands_n", rotor_commands),
        ):
            object.__setattr__(self, name, _readonly(value))

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.position_world_m).shape[0] - 1)


@dataclass(frozen=True)
class MPCWeights:
    """Caller-supplied positive-semidefinite weights from Eqs. (48)-(49)."""

    tracking: FloatArray
    input: FloatArray
    input_rate: FloatArray
    slack: FloatArray
    terminal_tracking: FloatArray
    impulse: FloatArray
    touchdown_velocity: FloatArray

    def __post_init__(self) -> None:
        specifications = (
            ("tracking", TRACKING_DIM),
            ("input", CONTROL_DIM),
            ("input_rate", CONTROL_DIM),
            ("slack", STATE_DIM),
            ("terminal_tracking", TRACKING_DIM),
            ("impulse", 3),
            ("touchdown_velocity", 3),
        )
        for name, dimension in specifications:
            object.__setattr__(
                self,
                name,
                _readonly(_weight_matrix(name, getattr(self, name), dimension)),
            )


@dataclass(frozen=True)
class StateBounds:
    """Phase state bounds ``X_k(s_k)`` and hard terminal bounds ``X_N``.

    The 16 columns are ``[p_G, v_G, rotation_vector, omega_B, T]``.  Infinite
    endpoints explicitly mean that the caller does not impose that bound.
    ``soft_mask`` has one row per nonterminal stage; terminal bounds are always
    hard, matching Eq. (50).  One nonnegative slack per state coordinate
    symmetrically relaxes selected lower/upper bounds.
    """

    lower: FloatArray
    upper: FloatArray
    soft_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        lower_raw = np.asarray(self.lower)
        if lower_raw.ndim != 2 or lower_raw.shape[1:] != (STATE_DIM,):
            raise ValueError(f"lower must have shape (N + 1, {STATE_DIM})")
        state_count = lower_raw.shape[0]
        if state_count < 2:
            raise ValueError("state bounds must cover at least one interval")
        shape = (state_count, STATE_DIM)
        lower = _bounded_array("lower", lower_raw, shape)
        upper = _bounded_array("upper", self.upper, shape)
        if np.any(np.isposinf(lower)):
            raise ValueError("lower bounds cannot contain +inf")
        if np.any(np.isneginf(upper)):
            raise ValueError("upper bounds cannot contain -inf")
        if np.any(lower > upper):
            raise ValueError("every lower state bound must not exceed its upper bound")
        soft = _boolean_mask("soft_mask", self.soft_mask, (state_count - 1, STATE_DIM))
        object.__setattr__(self, "lower", _readonly(lower))
        object.__setattr__(self, "upper", _readonly(upper))
        object.__setattr__(self, "soft_mask", _readonly(soft, dtype=bool))

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.lower).shape[0] - 1)


@dataclass(frozen=True)
class ContactForceLimits:
    """Caller-supplied unilateral/friction limits for continuous contacts."""

    friction_coefficients: FloatArray
    maximum_normal_force_n: FloatArray

    def __post_init__(self) -> None:
        friction = _finite_array("friction_coefficients", self.friction_coefficients, (FOOT_COUNT,))
        maximum = _finite_array(
            "maximum_normal_force_n", self.maximum_normal_force_n, (FOOT_COUNT,)
        )
        if np.any(friction < 0.0):
            raise ValueError("friction_coefficients cannot be negative")
        if np.any(maximum < 0.0):
            raise ValueError("maximum_normal_force_n cannot be negative")
        object.__setattr__(self, "friction_coefficients", _readonly(friction))
        object.__setattr__(self, "maximum_normal_force_n", _readonly(maximum))


@dataclass(frozen=True)
class LandingContactGeometry:
    """Explicit ground plane and touchdown guard parameters.

    The plane is ``ground_normal_world @ point_world = ground_plane_offset_m``;
    signed distance is positive on the free-space side.  The normal must be a
    unit vector so that the distance and velocity parameters retain SI units.
    At touchdown, a foot must be no farther than
    ``touchdown_position_tolerance_m`` from the plane and its pre-impact
    velocity must have at least ``minimum_downward_speed_m_per_s`` toward it.
    The body +Z axis must also remain inside the hard
    ``maximum_tilt_from_ground_normal_rad`` cone, which is required to be
    strictly smaller than 90 degrees.  Every landing MPC problem must provide
    this object explicitly; the solver never invents a default ground plane.
    """

    ground_normal_world: FloatArray
    ground_plane_offset_m: float
    touchdown_position_tolerance_m: float
    minimum_downward_speed_m_per_s: float
    maximum_tilt_from_ground_normal_rad: float

    def __post_init__(self) -> None:
        normal = _strict_finite_array(
            "ground_normal_world",
            self.ground_normal_world,
            (3,),
        )
        if not math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("ground_normal_world must be a unit vector")
        offset = _finite_scalar("ground_plane_offset_m", self.ground_plane_offset_m)
        position_tolerance = _finite_scalar(
            "touchdown_position_tolerance_m",
            self.touchdown_position_tolerance_m,
            minimum=0.0,
        )
        minimum_speed = _finite_scalar(
            "minimum_downward_speed_m_per_s",
            self.minimum_downward_speed_m_per_s,
            minimum=0.0,
        )
        maximum_tilt = _finite_scalar(
            "maximum_tilt_from_ground_normal_rad",
            self.maximum_tilt_from_ground_normal_rad,
            strictly_positive=True,
        )
        if maximum_tilt >= 0.5 * math.pi:
            raise ValueError(
                "maximum_tilt_from_ground_normal_rad must be strictly below pi/2"
            )
        object.__setattr__(self, "ground_normal_world", _readonly(normal))
        object.__setattr__(self, "ground_plane_offset_m", offset)
        object.__setattr__(self, "touchdown_position_tolerance_m", position_tolerance)
        object.__setattr__(self, "minimum_downward_speed_m_per_s", minimum_speed)
        object.__setattr__(
            self,
            "maximum_tilt_from_ground_normal_rad",
            maximum_tilt,
        )


@dataclass(frozen=True)
class RotorExecutionPlan:
    """Per-stage safe-gain envelope for the *applied total* rotor command.

    ``baseline_thrusts_n`` is the flight-controller baseline, and
    ``correction_gains`` is the scheduled safety gain κ in
    ``[0, 1]``.  The NLP control remains the actual total command applied after
    blending.  Thus its stage-wise envelope is

    ``baseline +/- correction_gain * maximum_raw_correction_n``.

    No division by ``correction_gains`` is used inside the optimization.
    """

    baseline_thrusts_n: FloatArray
    correction_gains: FloatArray
    maximum_raw_correction_n: FloatArray

    def __post_init__(self) -> None:
        try:
            baseline_raw = np.asarray(self.baseline_thrusts_n)
        except (TypeError, ValueError) as exc:
            raise TypeError("baseline_thrusts_n must be a numeric array") from exc
        if baseline_raw.ndim != 2 or baseline_raw.shape[1:] != (ROTOR_COUNT,):
            raise ValueError("baseline_thrusts_n must have shape (N, 4)")
        horizon = int(baseline_raw.shape[0])
        if horizon < 1:
            raise ValueError("baseline_thrusts_n must contain at least one stage")
        baseline = _strict_finite_array(
            "baseline_thrusts_n",
            self.baseline_thrusts_n,
            (horizon, ROTOR_COUNT),
        )
        gains = _strict_finite_array(
            "correction_gains",
            self.correction_gains,
            (horizon,),
        )
        maximum_raw = _strict_finite_array(
            "maximum_raw_correction_n",
            self.maximum_raw_correction_n,
            (ROTOR_COUNT,),
        )
        if np.any((gains < 0.0) | (gains > 1.0)):
            raise ValueError("correction_gains must lie in [0, 1]")
        if np.any(
            (gains > 0.0)
            & (gains < MINIMUM_RECONSTRUCTABLE_CORRECTION_GAIN)
        ):
            raise ValueError(
                "each correction gain must be zero or at least "
                f"{MINIMUM_RECONSTRUCTABLE_CORRECTION_GAIN:g}"
            )
        if np.any(maximum_raw <= 0.0):
            raise ValueError("maximum_raw_correction_n must be strictly positive")
        object.__setattr__(self, "baseline_thrusts_n", _readonly(baseline))
        object.__setattr__(self, "correction_gains", _readonly(gains))
        object.__setattr__(self, "maximum_raw_correction_n", _readonly(maximum_raw))

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.baseline_thrusts_n).shape[0])


@dataclass(frozen=True)
class RotorTransportTarget:
    """Transport-layer target reconstructed from an optimized applied command."""

    target_thrusts_n: FloatArray
    is_gain_limited_reconstruction: bool

    def __post_init__(self) -> None:
        target = _strict_finite_array(
            "target_thrusts_n",
            self.target_thrusts_n,
            (ROTOR_COUNT,),
        )
        if not isinstance(self.is_gain_limited_reconstruction, (bool, np.bool_)):
            raise TypeError("is_gain_limited_reconstruction must be a boolean")
        object.__setattr__(self, "target_thrusts_n", _readonly(target))
        object.__setattr__(
            self,
            "is_gain_limited_reconstruction",
            bool(self.is_gain_limited_reconstruction),
        )


def reconstruct_transport_target(
    execution_plan: RotorExecutionPlan,
    step: int,
    applied_total_thrusts_n: object,
) -> Optional[RotorTransportTarget]:
    """Invert one safe-gain blend for transport after the NLP has solved.

    The stage index is zero based.  At κ=0, the optimizer command
    is fixed to the baseline and there is no uniquely recoverable target, so
    this function returns ``None``.  For ``0 < kappa < 1``, the returned value
    is explicitly marked as a gain-limited transport reconstruction; it is not
    the solution of a separate ``kappa=1`` optimization.
    """

    if not isinstance(execution_plan, RotorExecutionPlan):
        raise TypeError("execution_plan must be a RotorExecutionPlan")
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
        raise TypeError("step must be an integer")
    index = int(step)
    if index < 0 or index >= execution_plan.horizon:
        raise ValueError("step is outside the execution-plan horizon")
    applied = _strict_finite_array(
        "applied_total_thrusts_n",
        applied_total_thrusts_n,
        (ROTOR_COUNT,),
    )
    baseline = execution_plan.baseline_thrusts_n[index]
    gain = float(execution_plan.correction_gains[index])
    delta = applied - baseline
    if gain == 0.0:
        rounding_tolerance = 16.0 * np.finfo(float).eps * np.maximum(
            1.0,
            np.maximum(np.abs(baseline), np.abs(applied)),
        )
        if np.any(np.abs(delta) > rounding_tolerance):
            raise ValueError("applied_total_thrusts_n violates the zero-gain baseline")
        return None
    raw_delta = delta / gain
    maximum_raw = execution_plan.maximum_raw_correction_n
    # Audit after inversion in the raw-command domain.  The conditioning term
    # accounts only for floating-point subtraction at the applied baseline; it
    # cannot become unbounded because RotorExecutionPlan rejects tiny gains.
    raw_tolerance = 1.0e-10 * np.maximum(1.0, maximum_raw) + (
        16.0
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(baseline), np.abs(applied)))
        / gain
    )
    if np.any(np.abs(raw_delta) > maximum_raw + raw_tolerance):
        raise ValueError(
            "applied_total_thrusts_n violates the execution-plan envelope "
            "in the raw-correction domain"
        )
    raw_delta = np.clip(raw_delta, -maximum_raw, maximum_raw)
    target = baseline + raw_delta
    return RotorTransportTarget(
        target_thrusts_n=target,
        is_gain_limited_reconstruction=gain < 1.0,
    )


@dataclass(frozen=True)
class ImpactEvent:
    """Known touchdown/reset metadata for one prediction step.

    ``step`` is in ``1..N``.  ``touchdown`` is ``delta_td`` from Eq. (39),
    while ``participation`` is ``delta_imp``.  The problem validates both
    against the supplied contact schedule rather than silently changing them.
    """

    step: int
    touchdown: NDArray[np.int8]
    participation: NDArray[np.int8]
    post_impact_joint_velocities_rad_per_s: FloatArray
    impulse_limits: ImpactLimits

    def __post_init__(self) -> None:
        if isinstance(self.step, (bool, np.bool_)) or not isinstance(self.step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(self.step)
        if step < 1:
            raise ValueError("step must be at least 1")
        touchdown = _binary_schedule("touchdown", self.touchdown, (FOOT_COUNT,))
        participation = _binary_schedule("participation", self.participation, (FOOT_COUNT,))
        if not bool(np.any(touchdown)):
            raise ValueError("an impact event must contain at least one touchdown foot")
        if np.any(touchdown > participation):
            raise ValueError("every touchdown foot must participate in the impact reset")
        if not isinstance(self.impulse_limits, ImpactLimits):
            raise TypeError("impulse_limits must be an ImpactLimits")
        post_impact_joint_velocities = _strict_finite_array(
            "post_impact_joint_velocities_rad_per_s",
            self.post_impact_joint_velocities_rad_per_s,
            (FOOT_COUNT, 3),
        )
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "touchdown", _readonly(touchdown, dtype=np.int8))
        object.__setattr__(self, "participation", _readonly(participation, dtype=np.int8))
        object.__setattr__(
            self,
            "post_impact_joint_velocities_rad_per_s",
            _readonly(post_impact_joint_velocities),
        )


@dataclass(frozen=True)
class ImpactAwareMPCProblem:
    """Fully specified finite-horizon problem data for Eqs. (45)-(50).

    Every ``ReducedInput.rotor_thrust_commands_n`` value, including
    ``previous_input``, is an actual applied total thrust command.  An optional
    execution plan restricts those variables to commands realizable by the
    caller's baseline and scheduled safety gain.

    ``foot_lever_arms_from_com_body_m`` is explicitly referenced to the
    reduced-state CoM C.  ``foot_leg_order`` binds every four-row contact,
    kinematic, force, and impulse array in this problem to one validated order.
    Landing schedules are monotonic: once a foot contacts, release within this
    horizon is rejected.  Every problem, including a pre-touchdown horizon
    whose schedule is still all zero, requires explicit
    ``landing_contact_geometry``; the model never invents or omits a ground
    plane merely because touchdown lies beyond the current horizon.

    中文：``contact_schedule[k]`` 表示第 k 个结点已经建立的接触；从 0 到 1 的
    跳变必须在同一步提供唯一 ``ImpactEvent``。构造阶段还检查触地后指定关节速度
    是否与刚体粘着约束相容，从源头拒绝无解问题。旋翼决策变量始终表示最终总推力，
    不是 residual；有执行计划时再限制为飞控基线和 κ/headroom 实际可实现的范围。
    """

    initial_state: ReducedState
    previous_input: ReducedInput
    dt_s: float
    contact_schedule: NDArray[np.int8]
    foot_leg_order: Tuple[str, str, str, str]
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBodyHorizon
    leg_jacobians_body: FloatArray
    joint_velocities_rad_per_s: FloatArray
    references: MPCReferences
    state_bounds: StateBounds
    contact_limits: ContactForceLimits
    impact_events: Sequence[ImpactEvent]
    dynamics_config: ReducedDynamicsConfig
    rotor_actuator_config: RotorActuatorConfig
    weights: MPCWeights
    landing_contact_geometry: LandingContactGeometry
    rotor_execution_plan: Optional[RotorExecutionPlan] = None

    def __post_init__(self) -> None:
        if not isinstance(self.initial_state, ReducedState):
            raise TypeError("initial_state must be a ReducedState")
        if not isinstance(self.previous_input, ReducedInput):
            raise TypeError("previous_input must be a ReducedInput")
        if not isinstance(self.references, MPCReferences):
            raise TypeError("references must be MPCReferences")
        if not isinstance(self.state_bounds, StateBounds):
            raise TypeError("state_bounds must be StateBounds")
        if not isinstance(self.contact_limits, ContactForceLimits):
            raise TypeError("contact_limits must be ContactForceLimits")
        if not isinstance(self.dynamics_config, ReducedDynamicsConfig):
            raise TypeError("dynamics_config must be ReducedDynamicsConfig")
        if not isinstance(self.rotor_actuator_config, RotorActuatorConfig):
            raise TypeError("rotor_actuator_config must be RotorActuatorConfig")
        if not isinstance(self.weights, MPCWeights):
            raise TypeError("weights must be MPCWeights")
        dt = _finite_scalar("dt_s", self.dt_s, strictly_positive=True)
        horizon = self.references.horizon
        execution_plan = self.rotor_execution_plan
        if execution_plan is not None:
            if not isinstance(execution_plan, RotorExecutionPlan):
                raise TypeError("rotor_execution_plan must be a RotorExecutionPlan or None")
            if execution_plan.horizon != horizon:
                raise ValueError("rotor_execution_plan and references must use the same horizon")
            actuator = self.rotor_actuator_config
            baseline = execution_plan.baseline_thrusts_n
            if np.any(baseline < actuator.thrust_min_n) or np.any(baseline > actuator.thrust_max_n):
                raise ValueError("rotor execution baselines must lie within actuator bounds")
        if self.state_bounds.horizon != horizon:
            raise ValueError("state_bounds and references must use the same horizon")
        schedule = _binary_schedule(
            "contact_schedule", self.contact_schedule, (horizon + 1, FOOT_COUNT)
        )
        foot_leg_order = validate_four_foot_leg_order(
            self.foot_leg_order,
            name="foot_leg_order",
        )
        foot_lever_arms = self.foot_lever_arms_from_com_body_m
        if not isinstance(foot_lever_arms, FootLeverArmsFromComBodyHorizon):
            raise TypeError(
                "foot_lever_arms_from_com_body_m must be "
                "FootLeverArmsFromComBodyHorizon; unlabeled arrays and "
                "B-referenced positions are forbidden"
            )
        if foot_lever_arms.node_count != horizon + 1:
            raise ValueError(
                "foot lever-arm horizon and references must use the same horizon"
            )
        if foot_lever_arms.leg_order != foot_leg_order:
            raise ValueError(
                "foot lever-arm leg_order must exactly match foot_leg_order"
            )
        jacobians = _finite_array(
            "leg_jacobians_body",
            self.leg_jacobians_body,
            (horizon + 1, FOOT_COUNT, 3, 3),
        )
        joint_velocities = _finite_array(
            "joint_velocities_rad_per_s",
            self.joint_velocities_rad_per_s,
            (horizon + 1, FOOT_COUNT, 3),
        )

        events: Dict[int, ImpactEvent] = {}
        for event in self.impact_events:
            if not isinstance(event, ImpactEvent):
                raise TypeError("impact_events must contain only ImpactEvent values")
            if event.step > horizon:
                raise ValueError(f"impact event step {event.step} exceeds horizon {horizon}")
            if event.step in events:
                raise ValueError(f"duplicate impact event at step {event.step}")
            events[event.step] = event

        detected_steps = []
        for step in range(1, horizon + 1):
            detected = (1 - schedule[step - 1]) * schedule[step]
            if bool(np.any(detected)):
                detected_steps.append(step)
                scheduled_event = events.get(step)
                if scheduled_event is None:
                    raise ValueError(
                        f"contact_schedule predicts touchdown at step {step} without an ImpactEvent"
                    )
                if not np.array_equal(scheduled_event.touchdown, detected):
                    raise ValueError(
                        f"ImpactEvent.touchdown at step {step} disagrees with contact_schedule"
                    )
                if not np.array_equal(scheduled_event.participation, schedule[step]):
                    raise ValueError(
                        f"ImpactEvent.participation at step {step} must equal post-impact contacts"
                    )
        unexpected = sorted(set(events) - set(detected_steps))
        if unexpected:
            raise ValueError(
                f"impact events {unexpected} do not correspond to contact_schedule touchdowns"
            )

        released = (schedule[:-1] == 1) & (schedule[1:] == 0)
        if bool(np.any(released)):
            release_step, release_foot = np.argwhere(released)[0]
            raise ValueError(
                "landing contact_schedule must be monotonic; "
                f"foot {int(release_foot)} releases at step {int(release_step) + 1}"
            )

        geometry = self.landing_contact_geometry
        if geometry is not None and not isinstance(geometry, LandingContactGeometry):
            raise TypeError(
                "landing_contact_geometry must be a LandingContactGeometry or None"
            )
        if geometry is None:
            raise ValueError(
                "landing_contact_geometry is required for every impact-aware MPC problem"
            )
        initial_foot_positions_world = (
            self.initial_state.position_world_m[np.newaxis, :]
            + foot_lever_arms.values_m[0] @ self.initial_state.rotation_body_to_world.T
        )
        initial_signed_distances = (
            initial_foot_positions_world @ geometry.ground_normal_world
            - geometry.ground_plane_offset_m
        )
        if np.any(initial_signed_distances < -1.0e-9):
            raise ValueError("initial foot geometry penetrates the ground plane")
        initial_contacts = schedule[0].astype(bool)
        if np.any(
            initial_signed_distances[initial_contacts]
            > geometry.touchdown_position_tolerance_m + 1.0e-9
        ):
            raise ValueError(
                "initial scheduled contact must lie inside the ground contact band"
            )

        for event in events.values():
            participating = np.flatnonzero(event.participation)
            rigid_velocity_map = np.vstack(
                tuple(
                    np.hstack(
                        (
                            np.eye(3),
                            -skew(foot_lever_arms.values_m[event.step, int(foot)]),
                        )
                    )
                    for foot in participating
                )
            )
            prescribed_relative_velocity = np.concatenate(
                tuple(
                    jacobians[event.step, int(foot)]
                    @ event.post_impact_joint_velocities_rad_per_s[int(foot)]
                    for foot in participating
                )
            )
            left_vectors, singular_values, _ = np.linalg.svd(
                rigid_velocity_map,
                full_matrices=False,
            )
            scale = max(1.0, float(singular_values[0]))
            rank_tolerance = max(rigid_velocity_map.shape) * np.finfo(float).eps * scale * 100.0
            rank = int(np.count_nonzero(singular_values > rank_tolerance))
            attainable_basis = left_vectors[:, :rank]
            orthogonal_residual = prescribed_relative_velocity - attainable_basis @ (
                attainable_basis.T @ prescribed_relative_velocity
            )
            compatibility_tolerance = 1e-9 * max(
                1.0,
                float(np.linalg.norm(prescribed_relative_velocity)),
            )
            if float(np.linalg.norm(orthogonal_residual)) > compatibility_tolerance:
                raise ValueError(
                    "prescribed impact joint velocities are incompatible with rigid-body "
                    f"sticking at step {event.step}"
                )

        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "contact_schedule", _readonly(schedule, dtype=np.int8))
        object.__setattr__(self, "foot_leg_order", foot_leg_order)
        object.__setattr__(self, "foot_lever_arms_from_com_body_m", foot_lever_arms)
        object.__setattr__(self, "leg_jacobians_body", _readonly(jacobians))
        object.__setattr__(self, "joint_velocities_rad_per_s", _readonly(joint_velocities))
        object.__setattr__(
            self,
            "impact_events",
            tuple(events[step] for step in sorted(events)),
        )

    @property
    def horizon(self) -> int:
        return self.references.horizon


@dataclass(frozen=True)
class SLSQPSettings:
    """Explicit SciPy reference-solver settings; no controller defaults."""

    max_iterations: int
    ftol: float
    constraint_tolerance: float
    timeout_s: float
    display: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, (bool, np.bool_)) or not isinstance(
            self.max_iterations, (int, np.integer)
        ):
            raise TypeError("max_iterations must be an integer")
        iterations = int(self.max_iterations)
        if iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        ftol = _finite_scalar("ftol", self.ftol, strictly_positive=True)
        tolerance = _finite_scalar("constraint_tolerance", self.constraint_tolerance, minimum=0.0)
        timeout = _finite_scalar("timeout_s", self.timeout_s, strictly_positive=True)
        if not isinstance(self.display, (bool, np.bool_)):
            raise TypeError("display must be a boolean")
        object.__setattr__(self, "max_iterations", iterations)
        object.__setattr__(self, "ftol", ftol)
        object.__setattr__(self, "constraint_tolerance", tolerance)
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "display", bool(self.display))


@dataclass(frozen=True)
class MPCWarmStart:
    """Explicit decision trajectories usable as an SLSQP initial point."""

    states: Sequence[ReducedState]
    controls: Sequence[ReducedInput]
    slacks: FloatArray
    impulses_by_step: Mapping[int, FloatArray]

    def __post_init__(self) -> None:
        states = tuple(self.states)
        controls = tuple(self.controls)
        if not all(isinstance(state, ReducedState) for state in states):
            raise TypeError("states must contain only ReducedState values")
        if not all(isinstance(control, ReducedInput) for control in controls):
            raise TypeError("controls must contain only ReducedInput values")
        slack_raw = np.asarray(self.slacks)
        if slack_raw.ndim != 2 or slack_raw.shape[1:] != (STATE_DIM,):
            raise ValueError(f"slacks must have shape (N, {STATE_DIM})")
        slacks = _finite_array("slacks", slack_raw, slack_raw.shape)
        if np.any(slacks < 0.0):
            raise ValueError("slacks must be nonnegative")
        impulses: Dict[int, FloatArray] = {}
        for raw_step, values in self.impulses_by_step.items():
            if isinstance(raw_step, (bool, np.bool_)) or not isinstance(
                raw_step, (int, np.integer)
            ):
                raise TypeError("impulse step keys must be integers")
            step = int(raw_step)
            impulses[step] = _readonly(
                _finite_array(f"impulses_by_step[{step}]", values, (FOOT_COUNT, 3))
            )
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "slacks", _readonly(slacks))
        object.__setattr__(self, "impulses_by_step", MappingProxyType(impulses))

    @classmethod
    def from_result(cls, result: MPCSolveResult) -> MPCWarmStart:
        """Reuse a complete prior solution without modifying it."""

        if not isinstance(result, MPCSolveResult):
            raise TypeError("result must be an MPCSolveResult")
        return cls(
            states=result.states,
            controls=result.controls,
            slacks=result.slacks,
            impulses_by_step=result.impulses_by_step,
        )


@dataclass(frozen=True)
class MPCSolveResult:
    """Solver output plus feasibility diagnostics; never a hardware command."""

    success: bool
    status: str
    message: str
    objective: float
    solve_time_s: float
    iterations: int
    max_equality_violation: float
    min_inequality_residual: float
    states: Tuple[ReducedState, ...]
    pre_impact_states: Mapping[int, ReducedState]
    controls: Tuple[ReducedInput, ...]
    slacks: FloatArray
    impulses_by_step: Mapping[int, FloatArray]
    raw_solver_status: Optional[int]
    min_variable_bound_residual: float = math.inf

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "slacks", _readonly(self.slacks))
        object.__setattr__(
            self, "pre_impact_states", MappingProxyType(dict(self.pre_impact_states))
        )
        impulses = {int(step): _readonly(values) for step, values in self.impulses_by_step.items()}
        object.__setattr__(self, "impulses_by_step", MappingProxyType(impulses))

    @property
    def first_input(self) -> Optional[ReducedInput]:
        """Return the first executable MPC input only for a successful solve.

        Failed solves retain candidate trajectories for diagnostics, but this
        property deliberately withholds them from command-oriented callers.
        """

        return self.controls[0] if self.success and self.controls else None

    @property
    def diagnostic_first_candidate(self) -> Optional[ReducedInput]:
        """Return the first stored candidate regardless of solver success."""

        return self.controls[0] if self.controls else None


@dataclass(frozen=True)
class _DecisionLayout:
    state_slice: slice
    control_slice: slice
    slack_slice: slice
    impulse_slices: Mapping[Tuple[int, int], slice]
    size: int


class _SolveTimedOut(RuntimeError):
    pass


@dataclass(frozen=True)
class _SLSQPOutcome:
    kind: str
    candidate: FloatArray
    solver_success: bool
    message: str
    iterations: int
    raw_status: Optional[int]


def _slsqp_process_entry(
    problem: ImpactAwareMPCProblem,
    settings: SLSQPSettings,
    candidate: FloatArray,
    sender: Any,
) -> None:
    """Run native SLSQP behind a process boundary that the parent can kill."""

    try:
        nlp = ImpactAwareNLP(problem)
        local_result = nlp._solve_in_current_process(
            settings,
            warm_start=nlp.unpack(candidate),
        )
        completed_statuses = {"success", "constraint_violation", "solver_failure"}
        outcome = _SLSQPOutcome(
            kind=(
                "completed"
                if local_result.status in completed_statuses
                else local_result.status
            ),
            candidate=nlp.pack(MPCWarmStart.from_result(local_result)),
            solver_success=(local_result.raw_solver_status == 0),
            message=local_result.message,
            iterations=local_result.iterations,
            raw_status=local_result.raw_solver_status,
        )
    except Exception as exc:  # pragma: no cover - last-resort child containment
        outcome = _SLSQPOutcome(
            kind="solver_error",
            candidate=np.array(candidate, copy=True),
            solver_success=False,
            message=f"SLSQP worker raised {type(exc).__name__}: {exc}",
            iterations=0,
            raw_status=None,
        )
    try:
        sender.send(outcome)
    finally:
        sender.close()


WarmStartLike = Union[MPCWarmStart, MPCSolveResult]


class ImpactAwareNLP:
    """Validated multiple-shooting transcription of paper Eq. (50).

    中文：决策向量按全部状态、全部控制、软约束松弛量和各触地脚三维冲量排列。
    对外 ``equality_residual`` 保留完整物理粘着方程；给 SLSQP 的版本用 SVD 去除
    线性相关行，避免冗余等式导致数值秩亏。求解后仍用完整残差独立复核。
    """

    def __init__(self, problem: ImpactAwareMPCProblem) -> None:
        if not isinstance(problem, ImpactAwareMPCProblem):
            raise TypeError("problem must be an ImpactAwareMPCProblem")
        self.problem = problem
        self._events = {event.step: event for event in problem.impact_events}
        self._layout = self._build_layout()
        self._solver_sticking_bases = self._build_solver_sticking_bases()

    @property
    def decision_size(self) -> int:
        return self._layout.size

    def _rotor_command_bounds(self) -> Tuple[FloatArray, FloatArray]:
        """Return per-stage bounds for actual applied total rotor commands."""

        problem = self.problem
        horizon = problem.horizon
        actuator = problem.rotor_actuator_config
        lower = np.repeat(actuator.thrust_min_n[None, :], horizon, axis=0)
        upper = np.repeat(actuator.thrust_max_n[None, :], horizon, axis=0)
        plan = problem.rotor_execution_plan
        if plan is not None:
            scaled_raw_limit = (
                plan.correction_gains[:, None] * plan.maximum_raw_correction_n[None, :]
            )
            lower = np.maximum(lower, plan.baseline_thrusts_n - scaled_raw_limit)
            upper = np.minimum(upper, plan.baseline_thrusts_n + scaled_raw_limit)
        return cast(FloatArray, lower), cast(FloatArray, upper)

    def _build_layout(self) -> _DecisionLayout:
        horizon = self.problem.horizon
        offset = 0
        state_slice = slice(offset, offset + (horizon + 1) * STATE_DIM)
        offset = state_slice.stop
        control_slice = slice(offset, offset + horizon * CONTROL_DIM)
        offset = control_slice.stop
        slack_slice = slice(offset, offset + horizon * STATE_DIM)
        offset = slack_slice.stop
        impulse_slices: Dict[Tuple[int, int], slice] = {}
        for event in self.problem.impact_events:
            for foot in np.flatnonzero(event.participation):
                impulse_slices[(event.step, int(foot))] = slice(offset, offset + 3)
                offset += 3
        return _DecisionLayout(
            state_slice=state_slice,
            control_slice=control_slice,
            slack_slice=slack_slice,
            impulse_slices=MappingProxyType(impulse_slices),
            size=offset,
        )

    def _build_solver_sticking_bases(self) -> Mapping[int, FloatArray]:
        """Build body-frame column-space bases for SLSQP sticking equations.

        For participating feet, body-frame foot velocity has the affine form
        ``A [v_B, omega_B] + b`` with blocks ``A_i = [I, -S(p_i)]``.  The full
        Eq. (43) residual can contain more rows than the rank of this rigid-body
        map.  Projecting it through ``U_r.T`` from the SVD of ``A`` preserves
        every attainable sticking constraint without orientation-dependent row
        selection.  Full world-frame residuals are still checked after solve.
        """

        bases: Dict[int, FloatArray] = {}
        for event in self.problem.impact_events:
            participating = np.flatnonzero(event.participation)
            blocks = [
                np.hstack(
                    (
                        np.eye(3),
                        -skew(
                            self.problem.foot_lever_arms_from_com_body_m.values_m[
                                event.step,
                                int(foot),
                            ]
                        ),
                    )
                )
                for foot in participating
            ]
            matrix = np.vstack(blocks)
            left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
            scale = max(1.0, float(singular_values[0]))
            tolerance = max(matrix.shape) * np.finfo(float).eps * scale * 100.0
            rank = int(np.count_nonzero(singular_values > tolerance))
            bases[event.step] = _readonly(left_vectors[:, :rank])
        return MappingProxyType(bases)

    def _impact_initial_impulses(
        self,
        pre_impact_state: ReducedState,
        event: ImpactEvent,
    ) -> FloatArray:
        """Return a minimum-norm, Eq. (44)-projected impact warm start.

        The impulse-to-post-impact-foot-velocity map is linear for the frozen
        impact configuration.  Building it with unit impulses avoids a second
        handwritten frame derivation and naturally includes nonzero moment
        arms, inertia, leg Jacobians, and prescribed joint rates.
        """

        problem = self.problem
        participating = np.flatnonzero(event.participation)
        impulse_count = len(participating)
        impulses = np.zeros((FOOT_COUNT, 3), dtype=float)
        if impulse_count == 0:
            return cast(FloatArray, impulses)

        foot_lever_arms = problem.foot_lever_arms_from_com_body_m.at_step(event.step)
        jacobians = problem.leg_jacobians_body[event.step]
        joint_velocities = event.post_impact_joint_velocities_rad_per_s
        base_state = momentum_reset(
            pre_impact_state,
            impulses,
            event.participation,
            foot_lever_arms,
            problem.dynamics_config,
            impulse_leg_order=problem.foot_leg_order,
        )
        base_velocity = foot_post_impact_velocity(
            base_state,
            foot_lever_arms,
            jacobians,
            joint_velocities,
            leg_order=problem.foot_leg_order,
        )[participating].reshape(-1)
        response = np.empty((3 * impulse_count, 3 * impulse_count), dtype=float)
        for local_foot, foot in enumerate(participating):
            for axis in range(3):
                unit_impulses = np.zeros((FOOT_COUNT, 3), dtype=float)
                unit_impulses[int(foot), axis] = 1.0
                unit_state = momentum_reset(
                    pre_impact_state,
                    unit_impulses,
                    event.participation,
                    foot_lever_arms,
                    problem.dynamics_config,
                    impulse_leg_order=problem.foot_leg_order,
                )
                unit_velocity = foot_post_impact_velocity(
                    unit_state,
                    foot_lever_arms,
                    jacobians,
                    joint_velocities,
                    leg_order=problem.foot_leg_order,
                )[participating].reshape(-1)
                response[:, 3 * local_foot + axis] = unit_velocity - base_velocity

        solution = np.linalg.lstsq(response, -base_velocity, rcond=None)[0]
        if not np.all(np.isfinite(solution)):
            return cast(FloatArray, impulses)
        for local_foot, foot in enumerate(participating):
            index = int(foot)
            candidate = np.array(solution[3 * local_foot : 3 * local_foot + 3], copy=True)
            limits = event.impulse_limits
            geometry = problem.landing_contact_geometry
            if geometry is None:  # guarded by ImpactAwareMPCProblem validation
                raise RuntimeError("impact event has no landing contact geometry")
            ground_normal = geometry.ground_normal_world
            maximum_normal = min(
                limits.maximum_normal_impulse_ns,
                limits.maximum_average_normal_force_n * limits.impact_duration_s,
            )
            normal_impulse = min(
                max(float(candidate @ ground_normal), 0.0),
                maximum_normal,
            )
            tangential = candidate - normal_impulse * ground_normal
            tangential_norm = float(np.linalg.norm(tangential))
            tangential_limit = limits.friction_coefficients[index] * normal_impulse
            if tangential_norm > tangential_limit and tangential_norm > 0.0:
                tangential *= tangential_limit / tangential_norm
            candidate = normal_impulse * ground_normal + tangential
            candidate[np.abs(candidate) < 1e-12] = 0.0
            impulses[index] = candidate
        return cast(FloatArray, impulses)

    def pack(self, trajectories: MPCWarmStart) -> FloatArray:
        """Pack validated public trajectories into the solver decision vector."""

        if not isinstance(trajectories, MPCWarmStart):
            raise TypeError("trajectories must be an MPCWarmStart")
        horizon = self.problem.horizon
        if len(trajectories.states) != horizon + 1:
            raise ValueError(f"warm-start states must contain {horizon + 1} values")
        if len(trajectories.controls) != horizon:
            raise ValueError(f"warm-start controls must contain {horizon} values")
        if np.asarray(trajectories.slacks).shape != (horizon, STATE_DIM):
            raise ValueError(f"warm-start slacks must have shape {(horizon, STATE_DIM)}")
        expected_steps = set(self._events)
        if set(trajectories.impulses_by_step) != expected_steps:
            raise ValueError(
                "warm-start impulse keys must exactly match the problem impact-event steps"
            )

        decision = np.zeros(self.decision_size, dtype=float)
        decision[self._layout.state_slice] = np.concatenate(
            tuple(_state_to_vector(state) for state in trajectories.states)
        )
        decision[self._layout.control_slice] = np.concatenate(
            tuple(_input_to_vector(control) for control in trajectories.controls)
        )
        decision[self._layout.slack_slice] = np.asarray(trajectories.slacks).reshape(-1)
        for key, target_slice in self._layout.impulse_slices.items():
            step, foot = key
            decision[target_slice] = trajectories.impulses_by_step[step][foot]
        if not np.all(np.isfinite(decision)):
            raise ValueError("packed decision vector must be finite")
        return cast(FloatArray, decision)

    def unpack(self, decision: object) -> MPCWarmStart:
        """Convert a solver vector to immutable state/control trajectories."""

        values = _finite_array("decision", decision, (self.decision_size,))
        horizon = self.problem.horizon
        state_matrix = values[self._layout.state_slice].reshape(horizon + 1, STATE_DIM)
        control_matrix = values[self._layout.control_slice].reshape(horizon, CONTROL_DIM)
        slacks = values[self._layout.slack_slice].reshape(horizon, STATE_DIM)
        states = tuple(_vector_to_state(row) for row in state_matrix)
        controls = tuple(_vector_to_input(row) for row in control_matrix)
        impulses: Dict[int, FloatArray] = {
            event.step: np.zeros((FOOT_COUNT, 3), dtype=float)
            for event in self.problem.impact_events
        }
        for (step, foot), source_slice in self._layout.impulse_slices.items():
            impulses[step][foot] = values[source_slice]
        return MPCWarmStart(
            states=states,
            controls=controls,
            slacks=slacks,
            impulses_by_step=impulses,
        )

    def initial_guess(self, warm_start: Optional[WarmStartLike] = None) -> FloatArray:
        """Build a dynamics rollout or pack an explicit prior trajectory."""

        if warm_start is not None:
            if isinstance(warm_start, MPCSolveResult):
                warm_start = MPCWarmStart.from_result(warm_start)
            if not isinstance(warm_start, MPCWarmStart):
                raise TypeError("warm_start must be MPCWarmStart, MPCSolveResult, or None")
            return self.pack(warm_start)

        problem = self.problem
        states = [problem.initial_state]
        controls = []
        contact_normal = problem.landing_contact_geometry.ground_normal_world
        rotor_command_lower, rotor_command_upper = self._rotor_command_bounds()
        impulses: Dict[int, FloatArray] = {
            event.step: np.zeros((FOOT_COUNT, 3), dtype=float) for event in problem.impact_events
        }
        for step in range(problem.horizon):
            forces = np.array(problem.references.contact_forces_world_n[step], copy=True)
            for foot in range(FOOT_COUNT):
                if problem.contact_schedule[step, foot] == 0:
                    forces[foot] = 0.0
                    continue
                normal_force = min(
                    max(float(forces[foot] @ contact_normal), 0.0),
                    float(problem.contact_limits.maximum_normal_force_n[foot]),
                )
                tangential = forces[foot] - normal_force * contact_normal
                tangential_norm = float(np.linalg.norm(tangential))
                tangential_limit = (
                    float(problem.contact_limits.friction_coefficients[foot]) * normal_force
                )
                if tangential_norm > tangential_limit and tangential_norm > 0.0:
                    tangential = tangential * (tangential_limit / tangential_norm)
                forces[foot] = normal_force * contact_normal + tangential
            commands = np.clip(
                problem.references.rotor_thrust_commands_n[step],
                rotor_command_lower[step],
                rotor_command_upper[step],
            )
            control = ReducedInput(
                contact_forces_world_n=forces,
                rotor_thrust_commands_n=commands,
            )
            controls.append(control)
            predicted = reduced_discrete_step(
                states[-1],
                control,
                problem.contact_schedule[step],
                problem.foot_lever_arms_from_com_body_m.at_step(step),
                problem.dynamics_config,
                problem.rotor_actuator_config,
                problem.dt_s,
                contact_force_leg_order=problem.foot_leg_order,
            )
            event = self._events.get(step + 1)
            if event is not None:
                impulses[step + 1] = self._impact_initial_impulses(predicted, event)
                predicted = momentum_reset(
                    predicted,
                    impulses[step + 1],
                    event.participation,
                    problem.foot_lever_arms_from_com_body_m.at_step(step + 1),
                    problem.dynamics_config,
                    impulse_leg_order=problem.foot_leg_order,
                )
            states.append(predicted)

        slacks = np.zeros((problem.horizon, STATE_DIM), dtype=float)
        for step in range(problem.horizon):
            state_vector = _state_to_vector(states[step])
            lower_violation = np.maximum(problem.state_bounds.lower[step] - state_vector, 0.0)
            upper_violation = np.maximum(state_vector - problem.state_bounds.upper[step], 0.0)
            required = np.maximum(lower_violation, upper_violation)
            slacks[step, problem.state_bounds.soft_mask[step]] = required[
                problem.state_bounds.soft_mask[step]
            ]
        return self.pack(
            MPCWarmStart(
                states=states,
                controls=controls,
                slacks=slacks,
                impulses_by_step=impulses,
            )
        )

    def _pre_impact_states(self, trajectories: MPCWarmStart) -> Dict[int, ReducedState]:
        problem = self.problem
        result: Dict[int, ReducedState] = {}
        for event in problem.impact_events:
            interval = event.step - 1
            result[event.step] = reduced_discrete_step(
                trajectories.states[interval],
                trajectories.controls[interval],
                problem.contact_schedule[interval],
                problem.foot_lever_arms_from_com_body_m.at_step(interval),
                problem.dynamics_config,
                problem.rotor_actuator_config,
                problem.dt_s,
                contact_force_leg_order=problem.foot_leg_order,
            )
        return result

    def objective(self, decision: object) -> float:
        """Evaluate tracking/input/delta/slack/terminal/impact cost.

        中文：阶段代价依次惩罚状态跟踪、控制偏差、控制变化率和软约束；终端状态
        单独加权。触地步同时惩罚冲量与触地前足端速度，但这些是优化偏好，不能替代
        后面的硬约束与求解后可行性检查。
        """

        trajectories = self.unpack(decision)
        problem = self.problem
        references = problem.references
        weights = problem.weights
        total = 0.0
        previous = _input_to_vector(problem.previous_input)
        for step in range(problem.horizon):
            state_error = _tracking_error(trajectories.states[step], references, step)
            control_vector = _input_to_vector(trajectories.controls[step])
            reference_control = np.concatenate(
                (
                    references.contact_forces_world_n[step].reshape(12),
                    references.rotor_thrust_commands_n[step],
                )
            )
            input_error = control_vector - reference_control
            input_delta = control_vector - previous
            slack = np.asarray(trajectories.slacks[step])
            total += _quadratic(state_error, weights.tracking)
            total += _quadratic(input_error, weights.input)
            total += _quadratic(input_delta, weights.input_rate)
            total += _quadratic(slack, weights.slack)
            previous = control_vector

        terminal_error = _tracking_error(
            trajectories.states[problem.horizon], references, problem.horizon
        )
        total += _quadratic(terminal_error, weights.terminal_tracking)

        pre_impact_states = self._pre_impact_states(trajectories)
        for event in problem.impact_events:
            impulses = trajectories.impulses_by_step[event.step]
            for foot in np.flatnonzero(event.participation):
                total += _quadratic(impulses[int(foot)], weights.impulse)
            pre_impact_foot_velocity = foot_post_impact_velocity(
                pre_impact_states[event.step],
                problem.foot_lever_arms_from_com_body_m.at_step(event.step),
                problem.leg_jacobians_body[event.step],
                problem.joint_velocities_rad_per_s[event.step],
                leg_order=problem.foot_leg_order,
            )
            for foot in np.flatnonzero(event.touchdown):
                total += _quadratic(pre_impact_foot_velocity[int(foot)], weights.touchdown_velocity)
        if not math.isfinite(total):
            raise ValueError("objective became nonfinite")
        return total

    def _equality_parts(self, decision: object) -> Tuple[FloatArray, FloatArray]:
        trajectories = self.unpack(decision)
        problem = self.problem
        base_residuals = [
            _state_to_vector(trajectories.states[0]) - _state_to_vector(problem.initial_state)
        ]
        for step in range(problem.horizon):
            predicted = reduced_discrete_step(
                trajectories.states[step],
                trajectories.controls[step],
                problem.contact_schedule[step],
                problem.foot_lever_arms_from_com_body_m.at_step(step),
                problem.dynamics_config,
                problem.rotor_actuator_config,
                problem.dt_s,
                contact_force_leg_order=problem.foot_leg_order,
            )
            event = self._events.get(step + 1)
            if event is not None:
                predicted = momentum_reset(
                    predicted,
                    trajectories.impulses_by_step[step + 1],
                    event.participation,
                    problem.foot_lever_arms_from_com_body_m.at_step(step + 1),
                    problem.dynamics_config,
                    impulse_leg_order=problem.foot_leg_order,
                )
            base_residuals.append(
                _state_transition_residual(trajectories.states[step + 1], predicted)
            )
            inactive = np.flatnonzero(1 - problem.contact_schedule[step])
            for foot in inactive:
                base_residuals.append(
                    np.asarray(trajectories.controls[step].contact_forces_world_n[int(foot)])
                )

        sticking_residuals = []
        for event in problem.impact_events:
            post_velocity = foot_post_impact_velocity(
                trajectories.states[event.step],
                problem.foot_lever_arms_from_com_body_m.at_step(event.step),
                problem.leg_jacobians_body[event.step],
                event.post_impact_joint_velocities_rad_per_s,
                leg_order=problem.foot_leg_order,
            )
            for foot in np.flatnonzero(event.participation):
                sticking_residuals.append(post_velocity[int(foot)])
        base = np.concatenate(base_residuals)
        sticking = (
            np.concatenate(sticking_residuals) if sticking_residuals else np.empty(0, dtype=float)
        )
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(sticking)):
            raise ValueError("equality residual became nonfinite")
        return cast(FloatArray, base), cast(FloatArray, sticking)

    def equality_residual(self, decision: object) -> FloatArray:
        """Return every paper equality, including all sticking equations."""

        base, sticking = self._equality_parts(decision)
        return cast(FloatArray, np.concatenate((base, sticking)))

    def _solver_equality_residual(self, decision: object) -> FloatArray:
        """Return SVD-projected, full-rank equalities for SciPy SLSQP."""

        base, _ = self._equality_parts(decision)
        trajectories = self.unpack(decision)
        projected = []
        for event in self.problem.impact_events:
            state = trajectories.states[event.step]
            participating = np.flatnonzero(event.participation)
            velocity_world = foot_post_impact_velocity(
                state,
                self.problem.foot_lever_arms_from_com_body_m.at_step(event.step),
                self.problem.leg_jacobians_body[event.step],
                event.post_impact_joint_velocities_rad_per_s,
                leg_order=self.problem.foot_leg_order,
            )[participating]
            velocity_body = velocity_world @ state.rotation_body_to_world
            basis = self._solver_sticking_bases[event.step]
            projected.append(basis.T @ velocity_body.reshape(-1))
        sticking = np.concatenate(projected) if projected else np.empty(0, dtype=float)
        result = np.concatenate((base, sticking))
        if not np.all(np.isfinite(result)):
            raise ValueError("solver equality residual became nonfinite")
        return cast(FloatArray, result)

    def _physical_inequality_residual(self, decision: object) -> FloatArray:
        """Return model constraints not already represented by execution bounds.

        中文：所有返回值采用“非负即满足”的统一约定，包括状态上下界、旋翼推力/
        变化率、接触法向力、摩擦锥以及冲量/平均冲击力限制。未激活足的接触力不在
        此处处理，而是在等式约束中强制为零。
        """

        trajectories = self.unpack(decision)
        problem = self.problem
        residuals = []
        geometry = problem.landing_contact_geometry
        if geometry is None:  # only reachable through an invalid runtime cast/replace
            raise RuntimeError("impact-aware MPC has no landing contact geometry")
        contact_normal = geometry.ground_normal_world

        for step, state in enumerate(trajectories.states):
            state_vector = _state_to_vector(state)
            slack = (
                np.asarray(trajectories.slacks[step])
                if step < problem.horizon
                else np.zeros(STATE_DIM, dtype=float)
            )
            soften = (
                problem.state_bounds.soft_mask[step]
                if step < problem.horizon
                else np.zeros(STATE_DIM, dtype=bool)
            )
            lower = problem.state_bounds.lower[step]
            upper = problem.state_bounds.upper[step]
            for index in range(STATE_DIM):
                relaxation = float(slack[index]) if soften[index] else 0.0
                if math.isfinite(float(lower[index])):
                    residuals.append(state_vector[index] - lower[index] + relaxation)
                if math.isfinite(float(upper[index])):
                    residuals.append(upper[index] - state_vector[index] + relaxation)

            foot_positions_world = (
                state.position_world_m[np.newaxis, :]
                + problem.foot_lever_arms_from_com_body_m.values_m[step]
                @ state.rotation_body_to_world.T
            )
            signed_distances = (
                foot_positions_world @ geometry.ground_normal_world
                - geometry.ground_plane_offset_m
            )
            # No predicted foot may cross to the solid side of the plane,
            # independent of the caller-supplied contact schedule.
            residuals.extend(signed_distances)
            # A scheduled contact may not exert force while geometrically
            # separated from the plane.  Together with nonpenetration,
            # this keeps every active stance foot inside the configured
            # contact band, not only at the first 0->1 transition.
            for foot in np.flatnonzero(problem.contact_schedule[step]):
                residuals.append(
                    geometry.touchdown_position_tolerance_m
                    - signed_distances[int(foot)]
                )
            # Hard landing-attitude cone: the body +Z axis must remain
            # inside the caller-selected cone about the ground normal.
            body_up_world = state.rotation_body_to_world[:, 2]
            residuals.append(
                float(body_up_world @ geometry.ground_normal_world)
                - math.cos(geometry.maximum_tilt_from_ground_normal_rad)
            )

        actuator = problem.rotor_actuator_config
        for step, control in enumerate(trajectories.controls):
            state = trajectories.states[step]
            rates = first_order_thrust_rate(
                state.rotor_thrusts_n,
                control.rotor_thrust_commands_n,
                actuator,
            )
            rotor = evaluate_rotor_constraints(
                state.rotor_thrusts_n,
                rates,
                control.rotor_thrust_commands_n,
                actuator,
            )
            residuals.extend(rotor.thrust_lower_margin_n)
            residuals.extend(rotor.thrust_upper_margin_n)
            residuals.extend(rotor.thrust_rate_lower_margin_n_per_s)
            residuals.extend(rotor.thrust_rate_upper_margin_n_per_s)
            residuals.extend(rotor.command_lower_margin_n)
            residuals.extend(rotor.command_upper_margin_n)

            for foot in np.flatnonzero(problem.contact_schedule[step]):
                index = int(foot)
                force = control.contact_forces_world_n[index]
                normal_force = float(force @ contact_normal)
                tangential_force = force - normal_force * contact_normal
                residuals.append(normal_force)
                residuals.append(
                    problem.contact_limits.maximum_normal_force_n[index] - normal_force
                )
                residuals.append(
                    problem.contact_limits.friction_coefficients[index] * normal_force
                    - float(np.linalg.norm(tangential_force))
                )

        terminal_thrust = trajectories.states[-1].rotor_thrusts_n
        residuals.extend(terminal_thrust - actuator.thrust_min_n)
        residuals.extend(actuator.thrust_max_n - terminal_thrust)

        pre_impact_states = self._pre_impact_states(trajectories)
        for event in problem.impact_events:
            event_state = trajectories.states[event.step]
            foot_positions_world = (
                event_state.position_world_m[np.newaxis, :]
                + problem.foot_lever_arms_from_com_body_m.values_m[event.step]
                @ event_state.rotation_body_to_world.T
            )
            signed_distances = (
                foot_positions_world @ geometry.ground_normal_world
                - geometry.ground_plane_offset_m
            )
            pre_impact_foot_velocities = foot_post_impact_velocity(
                pre_impact_states[event.step],
                problem.foot_lever_arms_from_com_body_m.at_step(event.step),
                problem.leg_jacobians_body[event.step],
                problem.joint_velocities_rad_per_s[event.step],
                leg_order=problem.foot_leg_order,
            )
            impulses = trajectories.impulses_by_step[event.step]
            for foot in np.flatnonzero(event.participation):
                index = int(foot)
                impulse = impulses[index]
                normal_impulse = float(impulse @ geometry.ground_normal_world)
                tangential_impulse = impulse - normal_impulse * geometry.ground_normal_world
                residuals.append(normal_impulse)
                residuals.append(
                    event.impulse_limits.maximum_normal_impulse_ns - normal_impulse
                )
                residuals.append(
                    event.impulse_limits.friction_coefficients[index] * normal_impulse
                    - float(np.linalg.norm(tangential_impulse))
                )
                residuals.append(
                    event.impulse_limits.maximum_average_normal_force_n
                    - normal_impulse / event.impulse_limits.impact_duration_s
                )
            for foot in np.flatnonzero(event.touchdown):
                index = int(foot)
                # Nonpenetration is included above; this upper margin places
                # the touchdown foot inside the caller-selected guard band.
                residuals.append(
                    geometry.touchdown_position_tolerance_m - signed_distances[index]
                )
                normal_velocity = float(
                    pre_impact_foot_velocities[index] @ geometry.ground_normal_world
                )
                residuals.append(
                    -geometry.minimum_downward_speed_m_per_s - normal_velocity
                )

        result = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("inequality residual became nonfinite")
        return cast(FloatArray, result)

    def rotor_execution_residual(self, decision: object) -> FloatArray:
        """Audit applied commands against the scheduled baseline/gain envelope.

        Rows are returned in nonnegative form, with four lower margins followed
        by four upper margins for each stage.  They remain part of the public
        feasibility report but are omitted from SLSQP's inequality callback
        because the identical restrictions are already variable bounds.
        """

        plan = self.problem.rotor_execution_plan
        if plan is None:
            return cast(FloatArray, np.empty(0, dtype=float))
        trajectories = self.unpack(decision)
        residuals = []
        for step, control in enumerate(trajectories.controls):
            applied = control.rotor_thrust_commands_n
            baseline = plan.baseline_thrusts_n[step]
            scaled_limit = float(plan.correction_gains[step]) * plan.maximum_raw_correction_n
            delta = applied - baseline
            residuals.extend(scaled_limit + delta)
            residuals.extend(scaled_limit - delta)
        result = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("rotor execution residual became nonfinite")
        return cast(FloatArray, result)

    def inequality_residual(self, decision: object) -> FloatArray:
        """Return every public inequality for full post-solve auditing."""

        physical = self._physical_inequality_residual(decision)
        execution = self.rotor_execution_residual(decision)
        return cast(FloatArray, np.concatenate((physical, execution)))

    def _variable_bounds(self) -> Tuple[FloatArray, FloatArray]:
        problem = self.problem
        horizon = problem.horizon
        lower = np.full(self.decision_size, -np.inf, dtype=float)
        upper = np.full(self.decision_size, np.inf, dtype=float)

        state_bounds = lower[self._layout.state_slice].reshape(horizon + 1, STATE_DIM)
        state_upper = upper[self._layout.state_slice].reshape(horizon + 1, STATE_DIM)
        state_bounds[:, _ROTOR_THRUST] = problem.rotor_actuator_config.thrust_min_n
        state_upper[:, _ROTOR_THRUST] = problem.rotor_actuator_config.thrust_max_n

        controls_lower = lower[self._layout.control_slice].reshape(horizon, CONTROL_DIM)
        controls_upper = upper[self._layout.control_slice].reshape(horizon, CONTROL_DIM)
        rotor_lower, rotor_upper = self._rotor_command_bounds()
        controls_lower[:, _ROTOR_COMMAND] = rotor_lower
        controls_upper[:, _ROTOR_COMMAND] = rotor_upper

        slack_lower = lower[self._layout.slack_slice].reshape(horizon, STATE_DIM)
        slack_upper = upper[self._layout.slack_slice].reshape(horizon, STATE_DIM)
        slack_lower[:] = 0.0
        slack_upper[:] = np.where(problem.state_bounds.soft_mask, np.inf, 0.0)

        for event in problem.impact_events:
            maximum_normal = min(
                event.impulse_limits.maximum_normal_impulse_ns,
                event.impulse_limits.maximum_average_normal_force_n
                * event.impulse_limits.impact_duration_s,
            )
            for foot in np.flatnonzero(event.participation):
                target = self._layout.impulse_slices[(event.step, int(foot))]
                friction = event.impulse_limits.friction_coefficients[int(foot)]
                component_limit = maximum_normal * math.sqrt(1.0 + friction * friction)
                # A general plane's unilateral direction is not axis-aligned;
                # use a safe Cartesian box and enforce the exact normal/cone
                # restrictions in _physical_inequality_residual.
                lower[target] = -component_limit
                upper[target] = component_limit
        return lower, upper

    def variable_bound_residual(self, decision: object) -> FloatArray:
        """Return finite lower/upper decision-bound margins for independent audits."""

        values = _finite_array("decision", decision, (self.decision_size,))
        lower, upper = self._variable_bounds()
        margins = np.concatenate(
            (
                (values - lower)[np.isfinite(lower)],
                (upper - values)[np.isfinite(upper)],
            )
        )
        return _readonly(margins)

    def _solve_in_current_process(
        self,
        settings: SLSQPSettings,
        warm_start: Optional[WarmStartLike] = None,
    ) -> MPCSolveResult:
        """Solve with optional SciPy SLSQP and return fail-closed diagnostics.

        中文：优化器报告 success 只是必要条件；本函数还重新计算完整等式、不等式
        和变量边界残差，只有全部在 ``constraint_tolerance`` 内才允许结果成功。
        超时回调只能阻止下一次 Python 回调，无法杀死正在执行的原生求解例程。
        """

        if not isinstance(settings, SLSQPSettings):
            raise TypeError("settings must be SLSQPSettings")
        started = time.perf_counter()
        candidate = self.initial_guess(warm_start)
        best_candidate = np.array(candidate, copy=True)
        best_objective = self.objective(candidate)
        raw_status: Optional[int] = None
        iterations = 0

        try:
            from scipy.optimize import Bounds, minimize  # type: ignore[import-untyped]
        except ImportError as exc:
            return self._make_result(
                candidate,
                success=False,
                status="backend_unavailable",
                message=f"SciPy SLSQP is unavailable: {exc}",
                started=started,
                iterations=0,
                raw_status=None,
            )

        deadline = started + settings.timeout_s

        def check_timeout() -> None:
            if time.perf_counter() >= deadline:
                raise _SolveTimedOut("SLSQP exceeded the caller-supplied timeout")

        def objective_with_timeout(values: FloatArray) -> float:
            nonlocal best_candidate, best_objective
            check_timeout()
            value = self.objective(values)
            if value < best_objective:
                best_candidate = np.array(values, copy=True)
                best_objective = value
            return value

        def equality_with_timeout(values: FloatArray) -> FloatArray:
            check_timeout()
            return self._solver_equality_residual(values)

        def inequality_with_timeout(values: FloatArray) -> FloatArray:
            check_timeout()
            return self._physical_inequality_residual(values)

        def callback(values: FloatArray) -> None:
            nonlocal iterations, best_candidate
            iterations += 1
            check_timeout()
            best_candidate = np.array(values, copy=True)

        lower, upper = self._variable_bounds()
        try:
            result = minimize(
                objective_with_timeout,
                candidate,
                method="SLSQP",
                bounds=Bounds(lower, upper),
                constraints=(
                    {"type": "eq", "fun": equality_with_timeout},
                    {"type": "ineq", "fun": inequality_with_timeout},
                ),
                callback=callback,
                options={
                    "maxiter": settings.max_iterations,
                    "ftol": settings.ftol,
                    "disp": settings.display,
                },
            )
            raw_status = int(result.status)
            iterations = int(getattr(result, "nit", iterations))
            if np.asarray(result.x).shape == (self.decision_size,) and np.all(
                np.isfinite(result.x)
            ):
                candidate = np.array(result.x, dtype=float, copy=True)
            else:
                candidate = best_candidate
            equality = self.equality_residual(candidate)
            inequality = self.inequality_residual(candidate)
            variable_bounds = self.variable_bound_residual(candidate)
            maximum_equality = float(np.max(np.abs(equality))) if equality.size else 0.0
            minimum_inequality = float(np.min(inequality)) if inequality.size else math.inf
            minimum_variable_bound = (
                float(np.min(variable_bounds)) if variable_bounds.size else math.inf
            )
            elapsed = time.perf_counter() - started
            timed_out = elapsed >= settings.timeout_s
            feasible = (
                maximum_equality <= settings.constraint_tolerance
                and minimum_inequality >= -settings.constraint_tolerance
                and minimum_variable_bound >= -settings.constraint_tolerance
            )
            success = bool(result.success) and feasible and not timed_out
            if timed_out:
                status = "timeout"
            elif success:
                status = "success"
            elif bool(result.success):
                status = "constraint_violation"
            else:
                status = "solver_failure"
            message = (
                f"{result.message}; max|eq|={maximum_equality:.6g}, "
                f"min(ineq)={minimum_inequality:.6g}, "
                f"min(bounds)={minimum_variable_bound:.6g}, elapsed={elapsed:.6g}s"
            )
            if timed_out:
                message += f"; exceeded timeout_s={settings.timeout_s:.6g}"
            return self._make_result(
                candidate,
                success=success,
                status=status,
                message=message,
                started=started,
                iterations=iterations,
                raw_status=raw_status,
                timeout_s=settings.timeout_s,
            )
        except _SolveTimedOut as exc:
            return self._make_result(
                best_candidate,
                success=False,
                status="timeout",
                message=str(exc),
                started=started,
                iterations=iterations,
                raw_status=raw_status,
                timeout_s=settings.timeout_s,
            )
        except Exception as exc:
            return self._make_result(
                best_candidate,
                success=False,
                status="solver_error",
                message=f"SLSQP raised {type(exc).__name__}: {exc}",
                started=started,
                iterations=iterations,
                raw_status=raw_status,
                timeout_s=settings.timeout_s,
            )

    def _solve_with_hard_timeout(
        self,
        settings: SLSQPSettings,
        candidate: FloatArray,
        deadline: float,
    ) -> _SLSQPOutcome:
        """Run native SLSQP in a process that can be killed at the deadline."""

        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_slsqp_process_entry,
            args=(self.problem, settings, np.array(candidate, copy=True), sender),
            name="impact-aware-slsqp",
        )

        def stop_worker() -> bool:
            """Best-effort termination with an explicit, auditable postcondition."""

            if not process.is_alive():
                return True
            process.terminate()
            process.join(0.2)
            if process.is_alive():  # pragma: no cover - platform fallback
                process.kill()
                process.join(0.2)
            return not process.is_alive()

        try:
            process.start()
        except Exception as exc:
            receiver.close()
            sender.close()
            return _SLSQPOutcome(
                "solver_error",
                np.array(candidate, copy=True),
                False,
                f"could not start isolated SLSQP worker: {type(exc).__name__}: {exc}",
                0,
                None,
            )
        sender.close()
        try:
            # Read the pipe before joining.  Joining first can deadlock when a
            # long-horizon candidate exceeds the OS pipe buffer: the child is
            # then blocked in ``send`` while the parent waits for child exit.
            result_available = receiver.poll(max(0.0, deadline - time.perf_counter()))
            if not result_available:
                terminated = stop_worker()
                return _SLSQPOutcome(
                    "timeout" if terminated else "termination_failure",
                    np.array(candidate, copy=True),
                    False,
                    (
                        "SLSQP worker was terminated at the caller-supplied hard timeout"
                        if terminated
                        else "SLSQP worker survived terminate/kill after the hard timeout"
                    ),
                    0,
                    None,
                )
            try:
                outcome = receiver.recv()
            except EOFError:
                process.join(0.05)
                return _SLSQPOutcome(
                    "solver_error",
                    np.array(candidate, copy=True),
                    False,
                    f"isolated SLSQP worker exited with code {process.exitcode} without a result",
                    0,
                    None,
                )
            # ``send`` is the worker's last application-level action, but the
            # spawned interpreter can still need more than an arbitrary
            # 50 ms to flush/tear down SciPy and BLAS state.  Give it exactly
            # the time that remains in the caller's *same* hard deadline;
            # never introduce a second, hidden timeout budget here.
            process.join(max(0.0, deadline - time.perf_counter()))
            if process.is_alive():
                terminated = stop_worker()
                return _SLSQPOutcome(
                    "timeout" if terminated else "termination_failure",
                    np.array(candidate, copy=True),
                    False,
                    (
                        "isolated SLSQP worker returned a payload but did not exit "
                        "before the caller-supplied hard timeout"
                        if terminated
                        else "isolated SLSQP worker survived terminate/kill after returning"
                    ),
                    0,
                    None,
                )
            if process.exitcode != 0:
                return _SLSQPOutcome(
                    "solver_error",
                    np.array(candidate, copy=True),
                    False,
                    f"isolated SLSQP worker exited abnormally with code {process.exitcode}",
                    0,
                    None,
                )
            if not isinstance(outcome, _SLSQPOutcome):
                return _SLSQPOutcome(
                    "solver_error",
                    np.array(candidate, copy=True),
                    False,
                    "isolated SLSQP worker returned an invalid payload",
                    0,
                    None,
                )
            return outcome
        except (EOFError, OSError) as exc:
            terminated = stop_worker()
            return _SLSQPOutcome(
                "solver_error" if terminated else "termination_failure",
                np.array(candidate, copy=True),
                False,
                (
                    f"isolated SLSQP worker communication failed: {type(exc).__name__}: {exc}"
                    if terminated
                    else "isolated SLSQP worker communication failed and the worker "
                    "survived terminate/kill"
                ),
                0,
                None,
            )
        finally:
            receiver.close()
            # Communication failures are not permission to orphan a native
            # optimizer.  An isolated worker that is still alive here can no
            # longer return an auditable result, so terminate it just like a
            # hard-timeout worker.
            if process.is_alive():
                stop_worker()
            if not process.is_alive():
                process.close()

    def solve(
        self,
        settings: SLSQPSettings,
        warm_start: Optional[WarmStartLike] = None,
    ) -> MPCSolveResult:
        """Solve in a killable process and independently audit all constraints."""

        if not isinstance(settings, SLSQPSettings):
            raise TypeError("settings must be SLSQPSettings")
        started = time.perf_counter()
        deadline = started + settings.timeout_s
        candidate = self.initial_guess(warm_start)
        initial_candidate = np.array(candidate, copy=True)
        if time.perf_counter() >= deadline:
            return self._make_unverified_failure_result(
                candidate,
                fallback_decision=initial_candidate,
                status="timeout",
                message="initial-guess construction exceeded the caller-supplied timeout",
                started=started,
                iterations=0,
                raw_status=None,
            )

        outcome = self._solve_with_hard_timeout(settings, candidate, deadline)
        try:
            candidate = np.array(outcome.candidate, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            return self._make_unverified_failure_result(
                initial_candidate,
                fallback_decision=initial_candidate,
                status="solver_error",
                message=f"isolated SLSQP worker returned an invalid candidate: {exc}",
                started=started,
                iterations=outcome.iterations,
                raw_status=outcome.raw_status,
            )
        if candidate.shape != (self.decision_size,) or not np.all(np.isfinite(candidate)):
            return self._make_unverified_failure_result(
                candidate,
                fallback_decision=initial_candidate,
                status="solver_error",
                message="isolated SLSQP worker returned a malformed or nonfinite candidate",
                started=started,
                iterations=outcome.iterations,
                raw_status=outcome.raw_status,
            )
        if outcome.kind != "completed":
            return self._make_unverified_failure_result(
                candidate,
                fallback_decision=initial_candidate,
                status=outcome.kind,
                message=outcome.message,
                started=started,
                iterations=outcome.iterations,
                raw_status=outcome.raw_status,
            )

        equality = self.equality_residual(candidate)
        inequality = self.inequality_residual(candidate)
        variable_bounds = self.variable_bound_residual(candidate)
        maximum_equality = float(np.max(np.abs(equality))) if equality.size else 0.0
        minimum_inequality = float(np.min(inequality)) if inequality.size else math.inf
        minimum_variable_bound = (
            float(np.min(variable_bounds)) if variable_bounds.size else math.inf
        )
        feasible = (
            maximum_equality <= settings.constraint_tolerance
            and minimum_inequality >= -settings.constraint_tolerance
            and minimum_variable_bound >= -settings.constraint_tolerance
        )
        elapsed = time.perf_counter() - started
        timed_out = elapsed >= settings.timeout_s
        success = outcome.solver_success and feasible and not timed_out
        if timed_out:
            status = "timeout"
        elif success:
            status = "success"
        elif outcome.solver_success:
            status = "constraint_violation"
        else:
            status = "solver_failure"
        message = (
            f"{outcome.message}; max|eq|={maximum_equality:.6g}, "
            f"min(ineq)={minimum_inequality:.6g}, "
            f"min(bounds)={minimum_variable_bound:.6g}, elapsed={elapsed:.6g}s"
        )
        if timed_out:
            message += f"; exceeded timeout_s={settings.timeout_s:.6g}"
        return self._make_result(
            candidate,
            success=success,
            status=status,
            message=message,
            started=started,
            iterations=outcome.iterations,
            raw_status=outcome.raw_status,
            timeout_s=settings.timeout_s,
        )

    def _make_unverified_failure_result(
        self,
        decision: FloatArray,
        *,
        fallback_decision: FloatArray,
        status: str,
        message: str,
        started: float,
        iterations: int,
        raw_status: Optional[int],
    ) -> MPCSolveResult:
        """Build a non-executable failure without spending time on post-timeout audits.

        Once the isolated solver has timed out, failed communication, or shown
        an abnormal lifecycle, the parent must not run objective/constraint
        callbacks with no remaining budget.  A structurally valid trajectory
        is retained only for diagnostics; every feasibility metric is set to
        the conservative failing value and ``first_input`` remains withheld.
        """

        try:
            trajectories = self.unpack(decision)
        except (TypeError, ValueError, OverflowError):
            trajectories = self.unpack(fallback_decision)
        return MPCSolveResult(
            success=False,
            status=status,
            message=message + "; candidate was not post-audited after solver failure",
            objective=math.inf,
            solve_time_s=time.perf_counter() - started,
            iterations=iterations,
            max_equality_violation=math.inf,
            min_inequality_residual=-math.inf,
            min_variable_bound_residual=-math.inf,
            states=tuple(trajectories.states),
            pre_impact_states={},
            controls=tuple(trajectories.controls),
            slacks=np.asarray(trajectories.slacks),
            impulses_by_step=trajectories.impulses_by_step,
            raw_solver_status=raw_status,
        )

    def _make_result(
        self,
        decision: FloatArray,
        *,
        success: bool,
        status: str,
        message: str,
        started: float,
        iterations: int,
        raw_status: Optional[int],
        timeout_s: Optional[float] = None,
    ) -> MPCSolveResult:
        trajectories = self.unpack(decision)
        equality = self.equality_residual(decision)
        inequality = self.inequality_residual(decision)
        variable_bounds = self.variable_bound_residual(decision)
        maximum_equality = float(np.max(np.abs(equality))) if equality.size else 0.0
        minimum_inequality = float(np.min(inequality)) if inequality.size else math.inf
        minimum_variable_bound = (
            float(np.min(variable_bounds)) if variable_bounds.size else math.inf
        )
        objective = self.objective(decision)
        pre_impact_states = self._pre_impact_states(trajectories)
        solve_time = time.perf_counter() - started
        if timeout_s is not None and solve_time >= timeout_s:
            success = False
            status = "timeout"
            if "timeout" not in message.lower():
                message += f"; total elapsed time exceeded timeout_s={timeout_s:.6g}"
        return MPCSolveResult(
            success=success,
            status=status,
            message=message,
            objective=objective,
            solve_time_s=solve_time,
            iterations=iterations,
            max_equality_violation=maximum_equality,
            min_inequality_residual=minimum_inequality,
            min_variable_bound_residual=minimum_variable_bound,
            states=tuple(trajectories.states),
            pre_impact_states=pre_impact_states,
            controls=tuple(trajectories.controls),
            slacks=np.asarray(trajectories.slacks),
            impulses_by_step=trajectories.impulses_by_step,
            raw_solver_status=raw_status,
        )


def solve_impact_aware_mpc(
    problem: ImpactAwareMPCProblem,
    settings: SLSQPSettings,
    warm_start: Optional[WarmStartLike] = None,
) -> MPCSolveResult:
    """Convenience entry point for the pure numerical reference solver."""

    return ImpactAwareNLP(problem).solve(settings, warm_start=warm_start)


__all__ = [
    "CONTROL_DIM",
    "MINIMUM_RECONSTRUCTABLE_CORRECTION_GAIN",
    "STATE_DIM",
    "TRACKING_DIM",
    "ContactForceLimits",
    "ImpactAwareMPCProblem",
    "ImpactAwareNLP",
    "ImpactEvent",
    "LandingContactGeometry",
    "MPCReferences",
    "MPCSolveResult",
    "MPCWarmStart",
    "MPCWeights",
    "RotorExecutionPlan",
    "RotorTransportTarget",
    "SLSQPSettings",
    "StateBounds",
    "reconstruct_transport_target",
    "solve_impact_aware_mpc",
]
