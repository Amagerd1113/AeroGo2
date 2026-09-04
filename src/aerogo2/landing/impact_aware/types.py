"""Validated immutable values for the impact-aware mathematical core.

Every array is defensively copied and marked read-only.  Dataclasses validate
shape, finiteness, and the physical properties stated by the paper, but they do
not supply numerical robot parameters.  Callers must provide identified values.

中文说明：这里集中定义数学内核的数据契约。所有 NumPy 数组在构造时复制并设为
只读，避免求解过程中被外部线程原地修改；形状、有限性、旋转矩阵、正定性及上下界
关系在进入算法前统一检查。``ReducedState.position_world_m`` 表示整机总质心 C，
不是 Go2 机身原点 B；腿部运动学使用 B 时必须显式完成 B/C 坐标转换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]

from aerogo2.landing.impact_aware.math_utils import (
    _as_finite_array,
    _finite_scalar,
    _readonly,
    require_rotation_matrix,
)

ROTOR_COUNT = 4
FOOT_COUNT = 4
GO2_SDK_LEG_ORDER: Tuple[str, str, str, str] = ("FR", "FL", "RR", "RL")


def _set_array(instance: object, field_name: str, value: FloatArray) -> None:
    object.__setattr__(instance, field_name, _readonly(value))


def _validate_tolerance(value: object) -> float:
    return _finite_scalar(value, "atol", minimum=0.0)


def validate_four_foot_leg_order(
    leg_order: object,
    *,
    name: str = "leg_order",
) -> Tuple[str, str, str, str]:
    """Return one unambiguous four-foot order contract.

    Four-foot arrays cannot be safely exchanged using shape alone: a valid
    ``(4, 3)`` array in ``FR, FL, RR, RL`` order is numerically
    indistinguishable from the same rows in another order.  This validator is
    shared by the typed B/C geometry helpers and the Go2 URDF adapter.
    """

    if isinstance(leg_order, (str, bytes)):
        raise TypeError(f"{name} must be a four-item sequence of leg names")
    try:
        raw: Tuple[object, ...] = tuple(leg_order)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a four-item sequence of leg names") from exc
    if len(raw) != FOOT_COUNT:
        raise ValueError(f"{name} must contain exactly {FOOT_COUNT} leg names")
    normalized = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name}[{index}] must be a non-empty string")
        if value != value.strip():
            raise ValueError(f"{name}[{index}] must not contain surrounding whitespace")
        normalized.append(value)
    result = tuple(normalized)
    if len(set(result)) != FOOT_COUNT:
        raise ValueError(f"{name} must contain four unique leg names")
    return result  # type: ignore[return-value]


def four_foot_reorder_indices(
    source_leg_order: object,
    target_leg_order: object,
) -> Tuple[int, int, int, int]:
    """Return source-row indices that produce ``target_leg_order``.

    The two orders must name exactly the same four legs.  Consequently a
    caller either proves an identity mapping or supplies an explicit,
    inspectable permutation; silently relabelling four rows is impossible.
    """

    source = validate_four_foot_leg_order(source_leg_order, name="source_leg_order")
    target = validate_four_foot_leg_order(target_leg_order, name="target_leg_order")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(
            "source_leg_order and target_leg_order must name the same legs; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(source.index(name) for name in target)  # type: ignore[return-value]


@dataclass(frozen=True)
class FootPositionsFromBodyOriginB:
    """Four foot positions ``{}^B r_BF`` with an explicit row-order identity."""

    values_m: FloatArray
    leg_order: Tuple[str, str, str, str]

    def __post_init__(self) -> None:
        values = _as_finite_array(
            self.values_m,
            (FOOT_COUNT, 3),
            "foot_positions_from_body_origin_B_body_m",
        )
        order = validate_four_foot_leg_order(self.leg_order)
        _set_array(self, "values_m", values)
        object.__setattr__(self, "leg_order", order)


@dataclass(frozen=True)
class FootLeverArmsFromComBody:
    """Four CoM-referenced foot lever arms ``{}^B r_CF`` and row order."""

    values_m: FloatArray
    leg_order: Tuple[str, str, str, str]

    def __post_init__(self) -> None:
        values = _as_finite_array(
            self.values_m,
            (FOOT_COUNT, 3),
            "foot_lever_arms_from_com_body_m",
        )
        order = validate_four_foot_leg_order(self.leg_order)
        _set_array(self, "values_m", values)
        object.__setattr__(self, "leg_order", order)


@dataclass(frozen=True)
class FootLeverArmsFromComBodyHorizon:
    """Horizon of four CoM-referenced foot lever arms with one row order.

    The leading dimension contains MPC nodes.  Keeping the reference point and
    row identity attached to the complete trajectory prevents a numerically
    valid ``(N + 1, 4, 3)`` array from silently carrying body-origin ``B``
    positions or a different leg permutation into the CoM dynamics.
    """

    values_m: FloatArray
    leg_order: Tuple[str, str, str, str]

    def __post_init__(self) -> None:
        raw = np.asarray(self.values_m)
        if raw.dtype.kind not in "fiu":
            raise TypeError(
                "foot_lever_arms_from_com_body_horizon_m must contain real "
                "numeric values"
            )
        values = np.array(raw, dtype=float, copy=True)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (FOOT_COUNT, 3):
            raise ValueError(
                "foot_lever_arms_from_com_body_horizon_m must have shape "
                "(node_count, 4, 3) with node_count >= 1"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "foot_lever_arms_from_com_body_horizon_m must contain only finite values"
            )
        order = validate_four_foot_leg_order(self.leg_order)
        _set_array(self, "values_m", values)
        object.__setattr__(self, "leg_order", order)

    @property
    def node_count(self) -> int:
        return int(self.values_m.shape[0])

    def at_step(self, step: int) -> FootLeverArmsFromComBody:
        """Return one typed node without discarding the leg-order contract."""

        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        index = int(step)
        if index < 0 or index >= self.node_count:
            raise IndexError(f"step {index} is outside [0, {self.node_count})")
        return FootLeverArmsFromComBody(
            values_m=self.values_m[index],
            leg_order=self.leg_order,
        )


def require_com_foot_lever_arms(
    value: object,
    *,
    data_leg_order: object,
) -> FootLeverArmsFromComBody:
    """Validate a CoM lever-arm value against the paired four-foot data order."""

    if isinstance(value, FootPositionsFromBodyOriginB):
        raise TypeError(
            "B-referenced foot positions must be converted to CoM lever arms first"
        )
    if not isinstance(value, FootLeverArmsFromComBody):
        raise TypeError(
            "foot_lever_arms_from_com_body_m must be FootLeverArmsFromComBody; "
            "unlabeled arrays are forbidden"
        )
    expected = validate_four_foot_leg_order(data_leg_order, name="data_leg_order")
    if value.leg_order != expected:
        raise ValueError(
            "foot lever-arm leg_order must exactly match the paired contact/force "
            "leg order"
        )
    return value


def foot_positions_from_body_origin_B_to_com_lever_arms(
    foot_positions: FootPositionsFromBodyOriginB,
    total_com_C_from_go2_body_origin_B_body_m: object,
    *,
    target_leg_order: Optional[Sequence[str]] = None,
) -> FootLeverArmsFromComBody:
    """Convert ``{}^B r_BF`` to ``{}^B r_CF = r_BF - r_BC``.

    ``ReducedState`` is referenced to total-system CoM ``C``, whereas the Go2
    URDF forward kinematics returns positions relative to body origin ``B``.
    Requiring a typed input prevents an unlabeled B-referenced array from being
    passed directly as a dynamics moment arm.  ``target_leg_order`` performs an
    explicit row permutation when the dynamics/contact order differs.
    """

    if not isinstance(foot_positions, FootPositionsFromBodyOriginB):
        raise TypeError("foot_positions must be FootPositionsFromBodyOriginB")
    offset_BC = _as_finite_array(
        total_com_C_from_go2_body_origin_B_body_m,
        (3,),
        "total_com_C_from_go2_body_origin_B_body_m",
    )
    target = (
        foot_positions.leg_order
        if target_leg_order is None
        else validate_four_foot_leg_order(target_leg_order, name="target_leg_order")
    )
    indices = four_foot_reorder_indices(foot_positions.leg_order, target)
    reordered = foot_positions.values_m[np.asarray(indices, dtype=int)]
    return FootLeverArmsFromComBody(
        values_m=reordered - offset_BC[np.newaxis, :],
        leg_order=target,
    )


@dataclass(frozen=True)
class RotorAerodynamics:
    """Quasi-steady rotor coefficients from paper Eqs. (8)-(9).

    ``spin_directions[i]`` is the paper's ``sigma_i`` and must be exactly -1
    or +1.  Rotor angular speed is treated as a nonnegative magnitude.
    """

    thrust_coefficient_n_per_rad_s_squared: float
    drag_torque_coefficient_nm_per_rad_s_squared: float
    spin_directions: FloatArray

    def __post_init__(self) -> None:
        thrust_coefficient = _finite_scalar(
            self.thrust_coefficient_n_per_rad_s_squared,
            "thrust_coefficient_n_per_rad_s_squared",
            minimum=0.0,
            strictly_greater=True,
        )
        drag_coefficient = _finite_scalar(
            self.drag_torque_coefficient_nm_per_rad_s_squared,
            "drag_torque_coefficient_nm_per_rad_s_squared",
            minimum=0.0,
        )
        directions = _as_finite_array(
            self.spin_directions,
            (ROTOR_COUNT,),
            "spin_directions",
        )
        if not np.all((directions == -1.0) | (directions == 1.0)):
            raise ValueError("spin_directions must contain only -1 or +1")

        object.__setattr__(
            self,
            "thrust_coefficient_n_per_rad_s_squared",
            thrust_coefficient,
        )
        object.__setattr__(
            self,
            "drag_torque_coefficient_nm_per_rad_s_squared",
            drag_coefficient,
        )
        _set_array(self, "spin_directions", directions)

    @property
    def reaction_torque_ratio_m(self) -> float:
        """Return ``k_m / k_f`` from paper Eq. (9), in metres."""

        return (
            self.drag_torque_coefficient_nm_per_rad_s_squared
            / self.thrust_coefficient_n_per_rad_s_squared
        )


@dataclass(frozen=True)
class FixedDeployedRotorGeometry:
    """Measured geometry for the mechanically locked, fully deployed layout.

    ``lever_arms_from_com_body_m[i]`` is the fixed vector from the identified
    reference-configuration total-system centre of mass to rotor ``i``'s
    thrust-axis centre, expressed in the declared body frame. It is a directly
    configured constant; allowed leg/payload motion must keep centre-of-mass
    drift inside a separately commissioned model-error envelope. This model
    deliberately contains no folding angle, linkage, motor, online centre-of-
    mass update, or deployment-position calculation.

    中文：着陆时机架已机械锁定，因此每个力臂直接保存为“整机质心 C 到旋翼轴心”
    的机体系常量。当前不建模折展电机，也不在线重算力臂；若载荷移动导致 C 漂移
    超出辨识包络，必须禁止硬件输出，而不能继续沿用该固定矩阵。
    """

    lever_arms_from_com_body_m: FloatArray
    thrust_directions_body: FloatArray

    def __post_init__(self) -> None:
        lever_arms = _as_finite_array(
            self.lever_arms_from_com_body_m,
            (ROTOR_COUNT, 3),
            "lever_arms_from_com_body_m",
        )
        directions = _as_finite_array(
            self.thrust_directions_body,
            (ROTOR_COUNT, 3),
            "thrust_directions_body",
        )
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("each thrust_directions_body row must be a unit vector")
        _set_array(self, "lever_arms_from_com_body_m", lever_arms)
        _set_array(self, "thrust_directions_body", directions)


@dataclass(frozen=True)
class RotorActuatorConfig:
    """Identified first-order thrust constants and paper Eq. (12) bounds."""

    time_constants_s: FloatArray
    thrust_min_n: FloatArray
    thrust_max_n: FloatArray
    thrust_rate_min_n_per_s: FloatArray
    thrust_rate_max_n_per_s: FloatArray

    def __post_init__(self) -> None:
        time_constants = _as_finite_array(
            self.time_constants_s,
            (ROTOR_COUNT,),
            "time_constants_s",
        )
        thrust_min = _as_finite_array(
            self.thrust_min_n,
            (ROTOR_COUNT,),
            "thrust_min_n",
        )
        thrust_max = _as_finite_array(
            self.thrust_max_n,
            (ROTOR_COUNT,),
            "thrust_max_n",
        )
        rate_min = _as_finite_array(
            self.thrust_rate_min_n_per_s,
            (ROTOR_COUNT,),
            "thrust_rate_min_n_per_s",
        )
        rate_max = _as_finite_array(
            self.thrust_rate_max_n_per_s,
            (ROTOR_COUNT,),
            "thrust_rate_max_n_per_s",
        )

        if np.any(time_constants <= 0.0):
            raise ValueError("time_constants_s must be strictly positive")
        if np.any(thrust_min < 0.0):
            raise ValueError("thrust_min_n cannot be negative")
        if np.any(thrust_min > thrust_max):
            raise ValueError("thrust_min_n must not exceed thrust_max_n")
        if np.any(rate_min > rate_max):
            raise ValueError("thrust_rate_min_n_per_s must not exceed thrust_rate_max_n_per_s")
        if np.any(rate_min > 0.0) or np.any(rate_max < 0.0):
            raise ValueError(
                "thrust-rate bounds must contain zero for a sustainable steady command"
            )

        for field_name, value in (
            ("time_constants_s", time_constants),
            ("thrust_min_n", thrust_min),
            ("thrust_max_n", thrust_max),
            ("thrust_rate_min_n_per_s", rate_min),
            ("thrust_rate_max_n_per_s", rate_max),
        ):
            _set_array(self, field_name, value)


@dataclass(frozen=True)
class ReducedDynamicsConfig:
    """Physical parameters for paper Eqs. (30)-(34).

    ``gravity_world_m_per_s2`` includes direction and magnitude.  The inertia
    is about the CoM and expressed in body frame.  No defaults are provided.
    """

    mass_kg: float
    inertia_body_kg_m2: FloatArray
    gravity_world_m_per_s2: FloatArray
    rotor_allocation_body: FloatArray

    def __post_init__(self) -> None:
        mass = _finite_scalar(
            self.mass_kg,
            "mass_kg",
            minimum=0.0,
            strictly_greater=True,
        )
        inertia = _as_finite_array(
            self.inertia_body_kg_m2,
            (3, 3),
            "inertia_body_kg_m2",
        )
        gravity = _as_finite_array(
            self.gravity_world_m_per_s2,
            (3,),
            "gravity_world_m_per_s2",
        )
        allocation = _as_finite_array(
            self.rotor_allocation_body,
            (6, ROTOR_COUNT),
            "rotor_allocation_body",
        )

        if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-10):
            raise ValueError("inertia_body_kg_m2 must be symmetric")
        if float(np.min(np.linalg.eigvalsh(inertia))) <= 0.0:
            raise ValueError("inertia_body_kg_m2 must be positive definite")

        object.__setattr__(self, "mass_kg", mass)
        _set_array(self, "inertia_body_kg_m2", inertia)
        _set_array(self, "gravity_world_m_per_s2", gravity)
        _set_array(self, "rotor_allocation_body", allocation)


@dataclass(frozen=True)
class ReducedState:
    """Paper Eq. (30) state referenced to total-system centre of mass ``C``.

    ``position_world_m`` and ``linear_velocity_world_m_per_s`` are the world-
    frame position and linear velocity of ``C``; they are not the position or
    velocity of the Go2 body origin ``B`` or of a flight-controller IMU.
    ``rotation_body_to_world`` maps the Go2 body axes into world axes, and
    ``angular_velocity_body_rad_per_s`` is expressed in those body axes.

    中文：旋翼状态保存的是各轴当前实际推力估计，用于一阶执行器动态；它不是油门、
    PWM 或 MPC 下一拍命令。C、B、IMU 三个原点不可互换，必须通过已标定刚体变换换算。
    """

    position_world_m: FloatArray
    linear_velocity_world_m_per_s: FloatArray
    rotation_body_to_world: FloatArray
    angular_velocity_body_rad_per_s: FloatArray
    rotor_thrusts_n: FloatArray

    def __post_init__(self) -> None:
        position = _as_finite_array(
            self.position_world_m,
            (3,),
            "position_world_m",
        )
        linear_velocity = _as_finite_array(
            self.linear_velocity_world_m_per_s,
            (3,),
            "linear_velocity_world_m_per_s",
        )
        rotation = require_rotation_matrix(
            self.rotation_body_to_world,
            name="rotation_body_to_world",
        )
        angular_velocity = _as_finite_array(
            self.angular_velocity_body_rad_per_s,
            (3,),
            "angular_velocity_body_rad_per_s",
        )
        thrusts = _as_finite_array(
            self.rotor_thrusts_n,
            (ROTOR_COUNT,),
            "rotor_thrusts_n",
        )

        for field_name, value in (
            ("position_world_m", position),
            ("linear_velocity_world_m_per_s", linear_velocity),
            ("rotation_body_to_world", rotation),
            ("angular_velocity_body_rad_per_s", angular_velocity),
            ("rotor_thrusts_n", thrusts),
        ):
            _set_array(self, field_name, value)


@dataclass(frozen=True)
class ReducedInput:
    """Paper Eq. (30) input: world contact forces and rotor commands.

    中文：足力为世界系三维期望接触力；未接触脚由 NLP 等式强制为零。旋翼量是
    最终总推力命令，不是飞控 residual，后者必须由基线相减和 κ 安全融合产生。
    """

    contact_forces_world_n: FloatArray
    rotor_thrust_commands_n: FloatArray

    def __post_init__(self) -> None:
        contact_forces = _as_finite_array(
            self.contact_forces_world_n,
            (FOOT_COUNT, 3),
            "contact_forces_world_n",
        )
        rotor_commands = _as_finite_array(
            self.rotor_thrust_commands_n,
            (ROTOR_COUNT,),
            "rotor_thrust_commands_n",
        )
        _set_array(self, "contact_forces_world_n", contact_forces)
        _set_array(self, "rotor_thrust_commands_n", rotor_commands)


@dataclass(frozen=True)
class ReducedStateDerivative:
    """Continuous derivative corresponding to :class:`ReducedState`."""

    position_rate_world_m_per_s: FloatArray
    linear_acceleration_world_m_per_s2: FloatArray
    rotation_rate_body_to_world_per_s: FloatArray
    angular_acceleration_body_rad_per_s2: FloatArray
    rotor_thrust_rates_n_per_s: FloatArray

    def __post_init__(self) -> None:
        specifications = (
            ("position_rate_world_m_per_s", (3,)),
            ("linear_acceleration_world_m_per_s2", (3,)),
            ("rotation_rate_body_to_world_per_s", (3, 3)),
            ("angular_acceleration_body_rad_per_s2", (3,)),
            ("rotor_thrust_rates_n_per_s", (ROTOR_COUNT,)),
        )
        for field_name, shape in specifications:
            value = _as_finite_array(getattr(self, field_name), shape, field_name)
            _set_array(self, field_name, value)


@dataclass(frozen=True)
class RotorConstraintResiduals:
    """Nonnegative-margin form of all actuator constraints in Eq. (12)."""

    thrust_lower_margin_n: FloatArray
    thrust_upper_margin_n: FloatArray
    thrust_rate_lower_margin_n_per_s: FloatArray
    thrust_rate_upper_margin_n_per_s: FloatArray
    command_lower_margin_n: FloatArray
    command_upper_margin_n: FloatArray

    def __post_init__(self) -> None:
        for field_name in (
            "thrust_lower_margin_n",
            "thrust_upper_margin_n",
            "thrust_rate_lower_margin_n_per_s",
            "thrust_rate_upper_margin_n_per_s",
            "command_lower_margin_n",
            "command_upper_margin_n",
        ):
            value = _as_finite_array(
                getattr(self, field_name),
                (ROTOR_COUNT,),
                field_name,
            )
            _set_array(self, field_name, value)

    def is_feasible(self, *, atol: float = 0.0) -> bool:
        """Return whether every Eq. (12) margin is at least ``-atol``."""

        tolerance = _validate_tolerance(atol)
        return all(
            bool(np.all(getattr(self, field_name) >= -tolerance))
            for field_name in (
                "thrust_lower_margin_n",
                "thrust_upper_margin_n",
                "thrust_rate_lower_margin_n_per_s",
                "thrust_rate_upper_margin_n_per_s",
                "command_lower_margin_n",
                "command_upper_margin_n",
            )
        )


@dataclass(frozen=True)
class ImpactLimits:
    """Caller-supplied horizontal-surface limits from paper Eq. (44)."""

    friction_coefficients: FloatArray
    maximum_normal_impulse_ns: float
    impact_duration_s: float
    maximum_average_normal_force_n: float

    def __post_init__(self) -> None:
        friction = _as_finite_array(
            self.friction_coefficients,
            (FOOT_COUNT,),
            "friction_coefficients",
        )
        if np.any(friction < 0.0):
            raise ValueError("friction_coefficients cannot be negative")
        maximum_impulse = _finite_scalar(
            self.maximum_normal_impulse_ns,
            "maximum_normal_impulse_ns",
            minimum=0.0,
        )
        duration = _finite_scalar(
            self.impact_duration_s,
            "impact_duration_s",
            minimum=0.0,
            strictly_greater=True,
        )
        maximum_force = _finite_scalar(
            self.maximum_average_normal_force_n,
            "maximum_average_normal_force_n",
            minimum=0.0,
        )

        _set_array(self, "friction_coefficients", friction)
        object.__setattr__(self, "maximum_normal_impulse_ns", maximum_impulse)
        object.__setattr__(self, "impact_duration_s", duration)
        object.__setattr__(self, "maximum_average_normal_force_n", maximum_force)


@dataclass(frozen=True)
class ImpulseConstraintResiduals:
    """Nonnegative-margin form of every inequality in paper Eq. (44)."""

    normal_lower_margin_ns: FloatArray
    normal_upper_margin_ns: FloatArray
    friction_cone_margin_ns: FloatArray
    average_force_upper_margin_n: FloatArray
    equivalent_average_normal_force_n: FloatArray

    def __post_init__(self) -> None:
        for field_name in (
            "normal_lower_margin_ns",
            "normal_upper_margin_ns",
            "friction_cone_margin_ns",
            "average_force_upper_margin_n",
            "equivalent_average_normal_force_n",
        ):
            value = _as_finite_array(
                getattr(self, field_name),
                (FOOT_COUNT,),
                field_name,
            )
            _set_array(self, field_name, value)

    def is_feasible(self, *, atol: float = 0.0) -> bool:
        """Return whether every inequality margin is at least ``-atol``."""

        tolerance = _validate_tolerance(atol)
        return all(
            bool(np.all(getattr(self, field_name) >= -tolerance))
            for field_name in (
                "normal_lower_margin_ns",
                "normal_upper_margin_ns",
                "friction_cone_margin_ns",
                "average_force_upper_margin_n",
            )
        )


__all__ = [
    "FOOT_COUNT",
    "GO2_SDK_LEG_ORDER",
    "ROTOR_COUNT",
    "FootLeverArmsFromComBody",
    "FootLeverArmsFromComBodyHorizon",
    "FootPositionsFromBodyOriginB",
    "ImpactLimits",
    "ImpulseConstraintResiduals",
    "ReducedDynamicsConfig",
    "ReducedInput",
    "ReducedState",
    "ReducedStateDerivative",
    "RotorActuatorConfig",
    "RotorAerodynamics",
    "RotorConstraintResiduals",
    "FixedDeployedRotorGeometry",
    "foot_positions_from_body_origin_B_to_com_lever_arms",
    "four_foot_reorder_indices",
    "require_com_foot_lever_arms",
    "validate_four_foot_leg_order",
]
