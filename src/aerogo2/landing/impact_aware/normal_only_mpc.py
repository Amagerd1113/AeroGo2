"""Hardware-isolated one-dimensional landing MPC.

This module is the executable form of the preliminary normal-only model.  It
does not obtain actuator authority and it deliberately does not reinterpret
uncalibrated Unitree scalar fields as newtons.  The optimisation state is the
total-system CoM height/vertical velocity; rotor and foot inputs are positive
world-Z forces in newtons.

中文：这是近期可实施的一维法向着陆模型，而不是把三维摩擦系数简单设为零。
所有足端高度均相对整机质心 C；接触表只能从 0 变为 1；每个结点都检查地面
非穿透，触地结点还检查位置容差和触地前向下运动。该模块永久不连接硬件，输出
的足力只是导纳控制器的期望值，不能直接当作 Go2 LowCmd。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from aerogo2.landing.impact_aware.preliminary import NormalOnlyVerticalState
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    ROTOR_COUNT,
    validate_four_foot_leg_order,
)

_ROTOR_ORDER = ("RR", "LF", "LR", "RF")

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]


def _finite_scalar(
    value: object,
    name: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_array(value: object, shape: Tuple[int, ...], name: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    copied = np.array(result, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _binary_schedule(value: object, horizon: int) -> NDArray[np.int8]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("contact_schedule must be an array") from exc
    shape = (horizon + 1, FOOT_COUNT)
    if raw.shape != shape:
        raise ValueError(f"contact_schedule must have shape {shape}")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("contact_schedule must contain only zero or one")
    schedule = np.array(raw, dtype=np.int8, copy=True)
    if np.any(np.diff(schedule, axis=0) < 0):
        raise ValueError("normal-only landing contact_schedule must be monotone (no 1->0 release)")
    schedule.setflags(write=False)
    return schedule


def _fixed_allocation(
    value: object,
    horizon: int,
    name: str,
    active_mask: NDArray[np.bool_],
) -> FloatArray:
    """Validate and freeze a per-step four-channel allocation.

    ``active_mask`` defines which channels physically exist in each interval.
    A row with at least one active channel must be nonnegative, exactly zero on
    inactive channels, and sum to one. A row with no active channel must be all
    zero. The returned array is backed by immutable ``bytes`` so callers cannot
    re-enable NumPy writes and silently change the optimisation problem.

    中文：一维模型只决定“总法向量”，不能自行决定四通道之间如何分配；这里把
    分配比例作为不可变配置冻结。无效通道必须为零，有效通道比例之和必须为 1。
    """

    allocation = _finite_array(value, (horizon, FOOT_COUNT), name)
    mask = np.asarray(active_mask, dtype=np.bool_)
    if mask.shape != (horizon, FOOT_COUNT):
        raise ValueError(f"{name} active mask must have shape {(horizon, FOOT_COUNT)}")
    if np.any(allocation < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    for step in range(horizon):
        active = mask[step]
        if np.any(allocation[step, ~active] != 0.0):
            raise ValueError(f"{name}[{step}] must be zero on inactive channels")
        row_sum = float(np.sum(allocation[step, active]))
        expected = 1.0 if np.any(active) else 0.0
        if not math.isclose(row_sum, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{name}[{step}] must sum to {expected:.0f} over active channels"
            )

    # ``bytes`` has no writable buffer, unlike a merely write-disabled owning
    # ndarray whose WRITEABLE flag can later be turned back on by the caller.
    frozen = np.frombuffer(allocation.tobytes(order="C"), dtype=np.float64).reshape(
        horizon,
        FOOT_COUNT,
    )
    return cast(FloatArray, frozen)


def _allocation_equalities(values: FloatArray, allocation: FloatArray) -> Tuple[float, ...]:
    """Return three independent equations for ``values = allocation * sum(values)``.

    All four direct equations sum to zero and would be rank deficient. Dropping
    the row corresponding to the largest allocation leaves exactly three
    independent equations whenever the allocation is normalized. An all-zero
    allocation needs no equation because bounds already fix every channel to
    zero for an inactive interval.
    """

    if not np.any(allocation):
        return ()
    pivot = int(np.argmax(allocation))
    total = float(np.sum(values))
    return tuple(
        float(values[channel] - allocation[channel] * total)
        for channel in range(FOOT_COUNT)
        if channel != pivot
    )


def _allocated_reference(
    reference: FloatArray,
    allocation: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    """Project a four-channel reference onto one fixed allocation and bounds."""

    if not np.any(allocation):
        return cast(FloatArray, np.zeros(FOOT_COUNT, dtype=np.float64))
    positive = allocation > 0.0
    if np.any(lower[~positive] > 0.0) or np.any(upper[~positive] < 0.0):
        raise ValueError("zero allocation is incompatible with channel bounds")
    total_lower = float(np.max(lower[positive] / allocation[positive]))
    total_upper = float(np.min(upper[positive] / allocation[positive]))
    if total_lower > total_upper + 1.0e-12:
        raise ValueError("fixed allocation is incompatible with channel bounds")
    desired_total = max(0.0, float(np.sum(reference)))
    total = float(np.clip(desired_total, total_lower, total_upper))
    return cast(FloatArray, np.asarray(allocation * total, dtype=np.float64))


@dataclass(frozen=True)
class NormalOnlyMPCWeights:
    """Nonnegative scalar weights for the one-dimensional reference problem."""

    height: float = 40.0
    vertical_velocity: float = 20.0
    rotor_force: float = 0.002
    contact_force: float = 0.01
    rotor_rate: float = 0.001
    impulse: float = 5.0
    terminal_height: float = 80.0
    terminal_vertical_velocity: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "height",
            "vertical_velocity",
            "rotor_force",
            "contact_force",
            "rotor_rate",
            "impulse",
            "terminal_height",
            "terminal_vertical_velocity",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name, minimum=0.0),
            )


@dataclass(frozen=True)
class NormalOnlyMPCProblem:
    """Finite-horizon vertical landing problem with explicit ground geometry."""

    initial_state: NormalOnlyVerticalState
    dt_s: float
    contact_schedule: NDArray[np.int8]
    leg_order: Tuple[str, str, str, str]
    rotor_order: Tuple[str, str, str, str]
    foot_heights_from_com_m: FloatArray
    ground_height_world_m: float
    touchdown_position_tolerance_m: float
    minimum_downward_speed_m_per_s: float
    mass_kg: float
    gravity_m_per_s2: float
    previous_rotor_forces_n: FloatArray
    rotor_force_min_n: FloatArray
    rotor_force_max_n: FloatArray
    rotor_force_rate_max_n_per_s: FloatArray
    contact_force_max_n: FloatArray
    normal_impulse_max_ns: FloatArray
    impact_duration_s: float
    average_impact_force_max_n: FloatArray
    rotor_force_allocation: FloatArray
    contact_force_allocation: FloatArray
    normal_impulse_allocation: FloatArray
    reference_height_world_m: FloatArray
    reference_vertical_velocity_world_m_per_s: FloatArray
    reference_rotor_forces_n: FloatArray
    reference_contact_forces_n: FloatArray
    minimum_com_height_world_m: float
    maximum_com_height_world_m: float
    maximum_abs_vertical_velocity_m_per_s: float
    weights: NormalOnlyMPCWeights = NormalOnlyMPCWeights()

    def __post_init__(self) -> None:
        if not isinstance(self.initial_state, NormalOnlyVerticalState):
            raise TypeError("initial_state must be NormalOnlyVerticalState")
        if not isinstance(self.weights, NormalOnlyMPCWeights):
            raise TypeError("weights must be NormalOnlyMPCWeights")
        try:
            schedule_raw = np.asarray(self.contact_schedule)
        except (TypeError, ValueError) as exc:
            raise TypeError("contact_schedule must be an array") from exc
        if schedule_raw.ndim != 2 or schedule_raw.shape[1:] != (FOOT_COUNT,):
            raise ValueError("contact_schedule must have shape (N + 1, 4)")
        horizon = int(schedule_raw.shape[0] - 1)
        if horizon < 1:
            raise ValueError("normal-only MPC requires at least one interval")
        schedule = _binary_schedule(schedule_raw, horizon)
        leg_order = validate_four_foot_leg_order(self.leg_order)
        rotor_order = validate_four_foot_leg_order(self.rotor_order, name="rotor_order")
        if rotor_order != _ROTOR_ORDER:
            raise ValueError("rotor_order must be exactly [RR, LF, LR, RF]")
        dt = _finite_scalar(self.dt_s, "dt_s", strictly_positive=True)
        mass = _finite_scalar(self.mass_kg, "mass_kg", strictly_positive=True)
        gravity = _finite_scalar(
            self.gravity_m_per_s2,
            "gravity_m_per_s2",
            strictly_positive=True,
        )
        ground = _finite_scalar(self.ground_height_world_m, "ground_height_world_m")
        tolerance = _finite_scalar(
            self.touchdown_position_tolerance_m,
            "touchdown_position_tolerance_m",
            minimum=0.0,
        )
        minimum_downward_speed = _finite_scalar(
            self.minimum_downward_speed_m_per_s,
            "minimum_downward_speed_m_per_s",
            minimum=0.0,
        )
        minimum_height = _finite_scalar(
            self.minimum_com_height_world_m,
            "minimum_com_height_world_m",
        )
        maximum_height = _finite_scalar(
            self.maximum_com_height_world_m,
            "maximum_com_height_world_m",
        )
        if minimum_height >= maximum_height:
            raise ValueError("minimum COM height must be below maximum COM height")
        maximum_speed = _finite_scalar(
            self.maximum_abs_vertical_velocity_m_per_s,
            "maximum_abs_vertical_velocity_m_per_s",
            strictly_positive=True,
        )
        impact_duration = _finite_scalar(
            self.impact_duration_s,
            "impact_duration_s",
            strictly_positive=True,
        )

        foot_heights = _finite_array(
            self.foot_heights_from_com_m,
            (horizon + 1, FOOT_COUNT),
            "foot_heights_from_com_m",
        )
        previous_rotor = _finite_array(
            self.previous_rotor_forces_n,
            (ROTOR_COUNT,),
            "previous_rotor_forces_n",
        )
        rotor_minimum = _finite_array(
            self.rotor_force_min_n,
            (ROTOR_COUNT,),
            "rotor_force_min_n",
        )
        rotor_maximum = _finite_array(
            self.rotor_force_max_n,
            (ROTOR_COUNT,),
            "rotor_force_max_n",
        )
        rotor_rate = _finite_array(
            self.rotor_force_rate_max_n_per_s,
            (ROTOR_COUNT,),
            "rotor_force_rate_max_n_per_s",
        )
        contact_maximum = _finite_array(
            self.contact_force_max_n,
            (FOOT_COUNT,),
            "contact_force_max_n",
        )
        impulse_maximum = _finite_array(
            self.normal_impulse_max_ns,
            (FOOT_COUNT,),
            "normal_impulse_max_ns",
        )
        average_maximum = _finite_array(
            self.average_impact_force_max_n,
            (FOOT_COUNT,),
            "average_impact_force_max_n",
        )
        if np.any(rotor_minimum < 0.0) or np.any(rotor_maximum < rotor_minimum):
            raise ValueError("rotor force bounds are invalid")
        if np.any(previous_rotor < rotor_minimum) or np.any(previous_rotor > rotor_maximum):
            raise ValueError("previous rotor forces violate actuator bounds")
        if np.any(rotor_rate <= 0.0):
            raise ValueError("rotor force-rate limits must be strictly positive")
        if np.any(contact_maximum < 0.0):
            raise ValueError("contact force maxima cannot be negative")
        if np.any(impulse_maximum < 0.0) or np.any(average_maximum < 0.0):
            raise ValueError("impact limits cannot be negative")

        touchdown = (1 - schedule[:-1]) * schedule[1:]
        rotor_allocation = _fixed_allocation(
            self.rotor_force_allocation,
            horizon,
            "rotor_force_allocation",
            np.ones((horizon, ROTOR_COUNT), dtype=np.bool_),
        )
        contact_allocation = _fixed_allocation(
            self.contact_force_allocation,
            horizon,
            "contact_force_allocation",
            schedule[:-1].astype(np.bool_),
        )
        impulse_allocation = _fixed_allocation(
            self.normal_impulse_allocation,
            horizon,
            "normal_impulse_allocation",
            touchdown.astype(np.bool_),
        )
        # 静态边界与固定比例若本身矛盾，应在求解前立即拒绝，而不是把一个必然
        # 不可行的问题交给 SLSQP。速率约束仍由 NLP 在完整时域内审核。
        for step in range(horizon):
            _allocated_reference(
                rotor_allocation[step],
                rotor_allocation[step],
                rotor_minimum,
                rotor_maximum,
            )
            _allocated_reference(
                contact_allocation[step],
                contact_allocation[step],
                np.zeros(FOOT_COUNT, dtype=np.float64),
                schedule[step] * contact_maximum,
            )
            _allocated_reference(
                impulse_allocation[step],
                impulse_allocation[step],
                np.zeros(FOOT_COUNT, dtype=np.float64),
                touchdown[step]
                * np.minimum(impulse_maximum, average_maximum * impact_duration),
            )

        reference_height = _finite_array(
            self.reference_height_world_m,
            (horizon + 1,),
            "reference_height_world_m",
        )
        reference_velocity = _finite_array(
            self.reference_vertical_velocity_world_m_per_s,
            (horizon + 1,),
            "reference_vertical_velocity_world_m_per_s",
        )
        reference_rotor = _finite_array(
            self.reference_rotor_forces_n,
            (horizon, ROTOR_COUNT),
            "reference_rotor_forces_n",
        )
        reference_contact = _finite_array(
            self.reference_contact_forces_n,
            (horizon, FOOT_COUNT),
            "reference_contact_forces_n",
        )

        initial_height = self.initial_state.height_world_m
        initial_velocity = self.initial_state.vertical_velocity_world_m_per_s
        if not minimum_height <= initial_height <= maximum_height:
            raise ValueError("initial COM height violates configured bounds")
        if abs(initial_velocity) > maximum_speed:
            raise ValueError("initial vertical speed violates configured bound")
        initial_clearance = initial_height + foot_heights[0] - ground
        if np.any(initial_clearance < -tolerance):
            raise ValueError("initial foot geometry penetrates the ground")
        current_contacts = schedule[0].astype(bool)
        if np.any(np.abs(initial_clearance[current_contacts]) > tolerance):
            raise ValueError("initial scheduled contact is not on the ground")
        if np.any(current_contacts) and abs(initial_velocity) > 1.0e-12:
            raise ValueError("initial scheduled contact requires zero vertical COM velocity")
        # The reduced state has no independent stance-leg velocity.  Freezing
        # each contacted foot's C-relative height is therefore required for a
        # well-defined one-dimensional sticking constraint.
        stance_intervals = schedule[:-1].astype(bool)
        foot_height_change = np.diff(foot_heights, axis=0)
        if np.any(np.abs(foot_height_change[stance_intervals]) > 1.0e-12):
            raise ValueError(
                "foot_heights_from_com_m must remain fixed after each foot enters contact"
            )

        for name, value in (
            ("dt_s", dt),
            ("mass_kg", mass),
            ("gravity_m_per_s2", gravity),
            ("ground_height_world_m", ground),
            ("touchdown_position_tolerance_m", tolerance),
            ("minimum_downward_speed_m_per_s", minimum_downward_speed),
            ("impact_duration_s", impact_duration),
            ("minimum_com_height_world_m", minimum_height),
            ("maximum_com_height_world_m", maximum_height),
            ("maximum_abs_vertical_velocity_m_per_s", maximum_speed),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "leg_order", leg_order)
        object.__setattr__(self, "rotor_order", rotor_order)
        for name, value in (
            ("contact_schedule", schedule),
            ("foot_heights_from_com_m", foot_heights),
            ("previous_rotor_forces_n", previous_rotor),
            ("rotor_force_min_n", rotor_minimum),
            ("rotor_force_max_n", rotor_maximum),
            ("rotor_force_rate_max_n_per_s", rotor_rate),
            ("contact_force_max_n", contact_maximum),
            ("normal_impulse_max_ns", impulse_maximum),
            ("average_impact_force_max_n", average_maximum),
            ("rotor_force_allocation", rotor_allocation),
            ("contact_force_allocation", contact_allocation),
            ("normal_impulse_allocation", impulse_allocation),
            ("reference_height_world_m", reference_height),
            ("reference_vertical_velocity_world_m_per_s", reference_velocity),
            ("reference_rotor_forces_n", reference_rotor),
            ("reference_contact_forces_n", reference_contact),
        ):
            object.__setattr__(self, name, value)

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.contact_schedule).shape[0] - 1)


@dataclass(frozen=True)
class NormalOnlyMPCResult:
    """Audited one-dimensional candidate; commands exist only on success."""

    success: bool
    status: str
    message: str
    states: Tuple[NormalOnlyVerticalState, ...]
    rotor_forces_n: FloatArray
    desired_contact_normal_forces_n: FloatArray
    normal_impulses_ns: FloatArray
    rotor_force_allocation: FloatArray
    contact_force_allocation: FloatArray
    normal_impulse_allocation: FloatArray
    leg_order: Tuple[str, str, str, str]
    rotor_order: Tuple[str, str, str, str]
    objective: float
    solve_time_s: float
    max_equality_violation: float
    min_inequality_residual: float
    min_variable_bound_residual: float

    @property
    def first_rotor_forces_n(self) -> Optional[FloatArray]:
        if not self.success:
            return None
        value = np.array(self.rotor_forces_n[0], dtype=np.float64, copy=True)
        value.setflags(write=False)
        return cast(FloatArray, value)

    @property
    def hardware_output_permitted(self) -> bool:
        """Normal-only reference results never authorize an actuator."""

        return False

    @property
    def first_desired_contact_normal_forces_n(self) -> Optional[FloatArray]:
        if not self.success:
            return None
        value = np.array(
            self.desired_contact_normal_forces_n[0],
            dtype=np.float64,
            copy=True,
        )
        value.setflags(write=False)
        return cast(FloatArray, value)


class _Layout:
    def __init__(self, horizon: int) -> None:
        cursor = 0
        self.state = slice(cursor, cursor + 2 * (horizon + 1))
        cursor = self.state.stop
        self.rotor = slice(cursor, cursor + ROTOR_COUNT * horizon)
        cursor = self.rotor.stop
        self.contact = slice(cursor, cursor + FOOT_COUNT * horizon)
        cursor = self.contact.stop
        self.impulse = slice(cursor, cursor + FOOT_COUNT * horizon)
        cursor = self.impulse.stop
        self.size = cursor


class _NormalOnlyNLP:
    def __init__(self, problem: NormalOnlyMPCProblem) -> None:
        self.problem = problem
        self.layout = _Layout(problem.horizon)

    def unpack(self, decision: object) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        values = _finite_array(decision, (self.layout.size,), "decision")
        horizon = self.problem.horizon
        states = values[self.layout.state].reshape(horizon + 1, 2)
        rotor = values[self.layout.rotor].reshape(horizon, ROTOR_COUNT)
        contact = values[self.layout.contact].reshape(horizon, FOOT_COUNT)
        impulse = values[self.layout.impulse].reshape(horizon, FOOT_COUNT)
        return states, rotor, contact, impulse

    def bounds(self) -> Tuple[FloatArray, FloatArray]:
        problem = self.problem
        horizon = problem.horizon
        lower = np.full(self.layout.size, -np.inf, dtype=np.float64)
        upper = np.full(self.layout.size, np.inf, dtype=np.float64)
        states_lower = lower[self.layout.state].reshape(horizon + 1, 2)
        states_upper = upper[self.layout.state].reshape(horizon + 1, 2)
        states_lower[:, 0] = problem.minimum_com_height_world_m
        states_upper[:, 0] = problem.maximum_com_height_world_m
        states_lower[:, 1] = -problem.maximum_abs_vertical_velocity_m_per_s
        states_upper[:, 1] = problem.maximum_abs_vertical_velocity_m_per_s
        rotor_lower = lower[self.layout.rotor].reshape(horizon, ROTOR_COUNT)
        rotor_upper = upper[self.layout.rotor].reshape(horizon, ROTOR_COUNT)
        rotor_lower[:] = problem.rotor_force_min_n
        rotor_upper[:] = problem.rotor_force_max_n
        contact_lower = lower[self.layout.contact].reshape(horizon, FOOT_COUNT)
        contact_upper = upper[self.layout.contact].reshape(horizon, FOOT_COUNT)
        contact_lower[:] = 0.0
        contact_upper[:] = problem.contact_schedule[:-1] * problem.contact_force_max_n
        impulse_lower = lower[self.layout.impulse].reshape(horizon, FOOT_COUNT)
        impulse_upper = upper[self.layout.impulse].reshape(horizon, FOOT_COUNT)
        impulse_lower[:] = 0.0
        touchdown = (1 - problem.contact_schedule[:-1]) * problem.contact_schedule[1:]
        impulse_upper[:] = touchdown * problem.normal_impulse_max_ns
        return cast(FloatArray, lower), cast(FloatArray, upper)

    def bound_residual(self, decision: object) -> FloatArray:
        values = _finite_array(decision, (self.layout.size,), "decision")
        lower, upper = self.bounds()
        result = np.concatenate(
            ((values - lower)[np.isfinite(lower)], (upper - values)[np.isfinite(upper)])
        )
        return cast(FloatArray, result)

    def equality(self, decision: object) -> FloatArray:
        states, rotor, contact, impulse = self.unpack(decision)
        problem = self.problem
        residuals = [
            states[0, 0] - problem.initial_state.height_world_m,
            states[0, 1] - problem.initial_state.vertical_velocity_world_m_per_s,
        ]
        for step in range(problem.horizon):
            acceleration = (
                -problem.gravity_m_per_s2
                + (float(np.sum(rotor[step])) + float(np.sum(contact[step])))
                / problem.mass_kg
            )
            expected_height = (
                states[step, 0]
                + states[step, 1] * problem.dt_s
                + 0.5 * acceleration * problem.dt_s * problem.dt_s
            )
            pre_impact_velocity = states[step, 1] + acceleration * problem.dt_s
            expected_velocity = pre_impact_velocity + float(np.sum(impulse[step])) / problem.mass_kg
            residuals.extend(
                (states[step + 1, 0] - expected_height, states[step + 1, 1] - expected_velocity)
            )
            if np.any(problem.contact_schedule[step + 1]):
                # 一维冲击采用规划器给出的触地后竖直速度作为 sticking/reset
                # 目标。每个触地步只增加一条独立约束，避免四足同时触地时
                # 产生四条完全重复、秩亏的 SLSQP 等式。
                # Perfectly inelastic normal sticking.  Since stance-foot
                # height relative to C is frozen by problem validation, one
                # non-duplicated v_C^+=0 equation proves zero normal velocity
                # for every simultaneous touchdown foot.
                residuals.append(states[step + 1, 1])
            # 一维动力学只包含总竖直力/冲量，若让优化器自由分配四个通道，可能
            # 产生模型没有描述的滚转、俯仰或偏航力矩。每类输入因此必须严格服从
            # 调用者冻结的比例。每行只加 3 条独立等式，避免秩亏。
            residuals.extend(
                _allocation_equalities(rotor[step], problem.rotor_force_allocation[step])
            )
            residuals.extend(
                _allocation_equalities(contact[step], problem.contact_force_allocation[step])
            )
            residuals.extend(
                _allocation_equalities(impulse[step], problem.normal_impulse_allocation[step])
            )
        result = np.asarray(residuals, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("normal-only equality residual became nonfinite")
        return cast(FloatArray, result)

    def inequality(self, decision: object) -> FloatArray:
        states, rotor, contact, impulse = self.unpack(decision)
        problem = self.problem
        residuals = []
        for step in range(problem.horizon + 1):
            clearance = (
                states[step, 0]
                + problem.foot_heights_from_com_m[step]
                - problem.ground_height_world_m
            )
            for foot in range(FOOT_COUNT):
                residuals.append(clearance[foot])
                if problem.contact_schedule[step, foot]:
                    residuals.append(problem.touchdown_position_tolerance_m - clearance[foot])

        previous = problem.previous_rotor_forces_n
        maximum_delta = problem.rotor_force_rate_max_n_per_s * problem.dt_s
        for step in range(problem.horizon):
            delta = rotor[step] - previous
            residuals.extend(maximum_delta - delta)
            residuals.extend(maximum_delta + delta)
            previous = rotor[step]

            touchdown = (1 - problem.contact_schedule[step]) * problem.contact_schedule[step + 1]
            if np.any(touchdown):
                acceleration = (
                    -problem.gravity_m_per_s2
                    + (float(np.sum(rotor[step])) + float(np.sum(contact[step])))
                    / problem.mass_kg
                )
                pre_impact_com_velocity = states[step, 1] + acceleration * problem.dt_s
                foot_relative_velocity = (
                    problem.foot_heights_from_com_m[step + 1]
                    - problem.foot_heights_from_com_m[step]
                ) / problem.dt_s
                for foot in np.flatnonzero(touchdown):
                    index = int(foot)
                    residuals.append(
                        -(
                            pre_impact_com_velocity
                            + foot_relative_velocity[index]
                            + problem.minimum_downward_speed_m_per_s
                        )
                    )
                    residuals.append(
                        problem.average_impact_force_max_n[index]
                        - impulse[step, index] / problem.impact_duration_s
                    )
        result = np.asarray(residuals, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("normal-only inequality residual became nonfinite")
        return cast(FloatArray, result)

    def objective(self, decision: object) -> float:
        states, rotor, contact, impulse = self.unpack(decision)
        problem = self.problem
        weights = problem.weights
        height_error = states[:-1, 0] - problem.reference_height_world_m[:-1]
        velocity_error = states[:-1, 1] - problem.reference_vertical_velocity_world_m_per_s[:-1]
        total = weights.height * float(height_error @ height_error)
        total += weights.vertical_velocity * float(velocity_error @ velocity_error)
        rotor_error = rotor - problem.reference_rotor_forces_n
        contact_error = contact - problem.reference_contact_forces_n
        total += weights.rotor_force * float(np.sum(rotor_error * rotor_error))
        total += weights.contact_force * float(np.sum(contact_error * contact_error))
        previous = np.vstack((problem.previous_rotor_forces_n, rotor[:-1]))
        rotor_delta = rotor - previous
        total += weights.rotor_rate * float(np.sum(rotor_delta * rotor_delta))
        total += weights.impulse * float(np.sum(impulse * impulse))
        terminal_height_error = states[-1, 0] - problem.reference_height_world_m[-1]
        terminal_velocity_error = (
            states[-1, 1] - problem.reference_vertical_velocity_world_m_per_s[-1]
        )
        total += weights.terminal_height * terminal_height_error * terminal_height_error
        total += (
            weights.terminal_vertical_velocity
            * terminal_velocity_error
            * terminal_velocity_error
        )
        if not math.isfinite(total):
            raise ValueError("normal-only objective became nonfinite")
        return total

    def initial_guess(self) -> FloatArray:
        problem = self.problem
        horizon = problem.horizon
        lower, upper = self.bounds()
        values = np.zeros(self.layout.size, dtype=np.float64)
        rotor = values[self.layout.rotor].reshape(horizon, ROTOR_COUNT)
        contact = values[self.layout.contact].reshape(horizon, FOOT_COUNT)
        impulse = values[self.layout.impulse].reshape(horizon, FOOT_COUNT)
        previous_rotor = problem.previous_rotor_forces_n
        maximum_delta = problem.rotor_force_rate_max_n_per_s * problem.dt_s
        for step in range(horizon):
            dynamic_lower = np.maximum(problem.rotor_force_min_n, previous_rotor - maximum_delta)
            dynamic_upper = np.minimum(problem.rotor_force_max_n, previous_rotor + maximum_delta)
            try:
                rotor[step] = _allocated_reference(
                    problem.reference_rotor_forces_n[step],
                    problem.rotor_force_allocation[step],
                    dynamic_lower,
                    dynamic_upper,
                )
            except ValueError:
                # A changing allocation may make the greedy rate-feasible interval
                # empty even when the horizon problem is feasible. Preserve the
                # fixed allocation and physical bounds; the NLP then decides rate
                # feasibility and fails closed if no trajectory exists.
                rotor[step] = _allocated_reference(
                    problem.reference_rotor_forces_n[step],
                    problem.rotor_force_allocation[step],
                    problem.rotor_force_min_n,
                    problem.rotor_force_max_n,
                )
            previous_rotor = rotor[step]
            contact[step] = _allocated_reference(
                problem.reference_contact_forces_n[step],
                problem.contact_force_allocation[step],
                np.zeros(FOOT_COUNT, dtype=np.float64),
                problem.contact_schedule[step] * problem.contact_force_max_n,
            )
        states = values[self.layout.state].reshape(horizon + 1, 2)
        states[0] = (
            problem.initial_state.height_world_m,
            problem.initial_state.vertical_velocity_world_m_per_s,
        )
        for step in range(horizon):
            acceleration = (
                -problem.gravity_m_per_s2
                + (float(np.sum(rotor[step])) + float(np.sum(contact[step])))
                / problem.mass_kg
            )
            pre_velocity = states[step, 1] + acceleration * problem.dt_s
            touchdown = (1 - problem.contact_schedule[step]) * problem.contact_schedule[step + 1]
            touchdown_feet = np.flatnonzero(touchdown)
            if touchdown_feet.size and pre_velocity < 0.0:
                required = -problem.mass_kg * pre_velocity
                impulse_upper = touchdown * np.minimum(
                    problem.normal_impulse_max_ns,
                    problem.average_impact_force_max_n * problem.impact_duration_s,
                )
                impulse[step] = _allocated_reference(
                    problem.normal_impulse_allocation[step] * required,
                    problem.normal_impulse_allocation[step],
                    np.zeros(FOOT_COUNT, dtype=np.float64),
                    impulse_upper,
                )
            states[step + 1, 0] = (
                states[step, 0]
                + states[step, 1] * problem.dt_s
                + 0.5 * acceleration * problem.dt_s * problem.dt_s
            )
            states[step + 1, 1] = pre_velocity + float(np.sum(impulse[step])) / problem.mass_kg
        return cast(FloatArray, np.clip(values, lower, upper))


def solve_normal_only_mpc(
    problem: NormalOnlyMPCProblem,
    *,
    max_iterations: int = 100,
    ftol: float = 1.0e-8,
    constraint_tolerance: float = 1.0e-6,
    timeout_s: float = 5.0,
) -> NormalOnlyMPCResult:
    """Solve and independently audit one normal-only problem.

    The timeout is checked at every Python callback and again after SciPy
    returns.  It is not a preemptive native-code deadline; this function is an
    offline reference and never grants hardware authority.
    """

    if not isinstance(problem, NormalOnlyMPCProblem):
        raise TypeError("problem must be NormalOnlyMPCProblem")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    tolerance = _finite_scalar(
        constraint_tolerance,
        "constraint_tolerance",
        strictly_positive=True,
    )
    objective_tolerance = _finite_scalar(ftol, "ftol", strictly_positive=True)
    timeout = _finite_scalar(timeout_s, "timeout_s", strictly_positive=True)
    nlp = _NormalOnlyNLP(problem)
    candidate = nlp.initial_guess()
    started = time.perf_counter()
    deadline = started + timeout
    raw_success = False
    raw_status = "backend_unavailable"
    raw_message = "SciPy SLSQP is unavailable"

    class _TimedOut(RuntimeError):
        pass

    def check_deadline() -> None:
        if time.perf_counter() >= deadline:
            raise _TimedOut("normal-only SLSQP exceeded timeout_s")

    try:
        from scipy.optimize import Bounds, minimize  # type: ignore[import-untyped]

        lower, upper = nlp.bounds()

        def objective(values: FloatArray) -> float:
            check_deadline()
            return nlp.objective(values)

        def equality(values: FloatArray) -> FloatArray:
            check_deadline()
            return nlp.equality(values)

        def inequality(values: FloatArray) -> FloatArray:
            check_deadline()
            return nlp.inequality(values)

        result = minimize(
            objective,
            candidate,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=(
                {"type": "eq", "fun": equality},
                {"type": "ineq", "fun": inequality},
            ),
            callback=lambda _values: check_deadline(),
            options={"maxiter": max_iterations, "ftol": objective_tolerance, "disp": False},
        )
        elapsed_after_return = time.perf_counter() - started
        if elapsed_after_return >= timeout:
            raw_status = "timeout"
            raw_message = "normal-only SLSQP returned after timeout_s"
        else:
            values = np.asarray(result.x, dtype=np.float64)
            if values.shape == (nlp.layout.size,) and np.all(np.isfinite(values)):
                candidate = cast(FloatArray, np.array(values, copy=True))
            raw_success = bool(result.success)
            raw_status = "success" if raw_success else "solver_failure"
            raw_message = str(result.message)
    except _TimedOut as exc:
        raw_status = "timeout"
        raw_message = str(exc)
    except ImportError as exc:
        raw_message = f"SciPy SLSQP is unavailable: {exc}"
    except Exception as exc:
        raw_status = "solver_error"
        raw_message = f"SLSQP raised {type(exc).__name__}: {exc}"

    states_raw, rotor, contact, impulse = nlp.unpack(candidate)
    equality_residual = nlp.equality(candidate)
    inequality_residual = nlp.inequality(candidate)
    bound_residual = nlp.bound_residual(candidate)
    maximum_equality = (
        float(np.max(np.abs(equality_residual))) if equality_residual.size else 0.0
    )
    minimum_inequality = (
        float(np.min(inequality_residual)) if inequality_residual.size else math.inf
    )
    minimum_bound = float(np.min(bound_residual)) if bound_residual.size else math.inf
    feasible = (
        maximum_equality <= tolerance
        and minimum_inequality >= -tolerance
        and minimum_bound >= -tolerance
    )
    objective_value = nlp.objective(candidate)
    # The deadline covers the complete independent audit and result
    # reconstruction, not merely SciPy's return boundary.
    solve_time = time.perf_counter() - started
    timed_out = solve_time >= timeout
    success = raw_success and raw_status == "success" and feasible and not timed_out
    if timed_out:
        raw_status = "timeout"
        raw_message = "normal-only solve or post-solve audit exceeded timeout_s"
    elif raw_status == "success" and not feasible:
        raw_status = "constraint_violation"
    states = tuple(
        NormalOnlyVerticalState(
            height_world_m=float(row[0]),
            vertical_velocity_world_m_per_s=float(row[1]),
        )
        for row in states_raw
    )
    for value in (rotor, contact, impulse):
        value.setflags(write=False)
    return NormalOnlyMPCResult(
        success=success,
        status=raw_status,
        message=(
            f"{raw_message}; max|eq|={maximum_equality:.6g}, "
            f"min(ineq)={minimum_inequality:.6g}, min(bound)={minimum_bound:.6g}"
        ),
        states=states,
        rotor_forces_n=rotor,
        desired_contact_normal_forces_n=contact,
        normal_impulses_ns=impulse,
        rotor_force_allocation=problem.rotor_force_allocation,
        contact_force_allocation=problem.contact_force_allocation,
        normal_impulse_allocation=problem.normal_impulse_allocation,
        leg_order=problem.leg_order,
        rotor_order=problem.rotor_order,
        objective=objective_value,
        solve_time_s=solve_time,
        max_equality_violation=maximum_equality,
        min_inequality_residual=minimum_inequality,
        min_variable_bound_residual=minimum_bound,
    )


__all__ = [
    "NormalOnlyMPCProblem",
    "NormalOnlyMPCResult",
    "NormalOnlyMPCWeights",
    "solve_normal_only_mpc",
]
