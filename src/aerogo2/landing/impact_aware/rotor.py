"""Rotor aerodynamics, allocation, and first-order thrust dynamics.

This module implements paper Eqs. (8)-(17).  It computes mathematical values
only; it cannot arm, transmit, or otherwise write to rotor hardware.

中文说明：四个旋翼的位置和轴向在展开着陆阶段视为固定，不运行折展机构运动学。
输入顺序由配置固定为 ``[RR, LF, LR, RF]``；分配矩阵将单轴推力映射为机体系六维
合力/力矩，同时包含旋向导致的反扭矩。油门/PWM 到推力的映射不在本文件内，未完成
安装状态标定前，制造商静态曲线只能用于离线量级检查。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from aerogo2.landing.impact_aware.math_utils import (
    _as_finite_array,
    _finite_scalar,
    _readonly,
)
from aerogo2.landing.impact_aware.types import (
    ROTOR_COUNT,
    FixedDeployedRotorGeometry,
    FloatArray,
    RotorActuatorConfig,
    RotorAerodynamics,
    RotorConstraintResiduals,
)


def thrust_and_reaction_torque(
    angular_speeds_rad_per_s: object,
    aerodynamics: RotorAerodynamics,
) -> Tuple[FloatArray, FloatArray]:
    """Return rotor thrust and signed reaction-torque magnitudes (Eq. 8).

    The returned arrays have units N and N*m.  The signed torque magnitude is
    along the corresponding rotor thrust direction; its sign includes
    ``sigma_i``.  Angular speeds are nonnegative magnitudes because rotation
    direction is represented separately by ``spin_directions``.
    """

    angular_speeds = _as_finite_array(
        angular_speeds_rad_per_s,
        (ROTOR_COUNT,),
        "angular_speeds_rad_per_s",
    )
    if np.any(angular_speeds < 0.0):
        raise ValueError("angular_speeds_rad_per_s cannot be negative")
    speed_squared = angular_speeds * angular_speeds
    thrusts = aerodynamics.thrust_coefficient_n_per_rad_s_squared * speed_squared
    reaction_torques = (
        aerodynamics.spin_directions
        * aerodynamics.drag_torque_coefficient_nm_per_rad_s_squared
        * speed_squared
    )
    return _readonly(thrusts), _readonly(reaction_torques)


def build_fixed_deployed_allocation_matrix(
    geometry: FixedDeployedRotorGeometry,
    aerodynamics: RotorAerodynamics,
) -> FloatArray:
    """Build the fixed-layout body-frame 6x4 allocation (Eqs. 13-17).

    Rows 0:3 map thrust to force in N.  Rows 3:6 map thrust to moment
    in N*m. The configured vectors are already the measured, fully deployed
    lever arms from the identified reference-configuration total-system CoM;
    this function does not calculate a rotor position, update CoM online, or
    model the folding mechanism. The cross product is retained because
    ``r_i x f_i`` is the physical thrust moment.
    """

    force_axes = geometry.thrust_directions_body
    thrust_moments = np.cross(geometry.lever_arms_from_com_body_m, force_axes)
    reaction_moments = (
        aerodynamics.spin_directions[:, np.newaxis]
        * aerodynamics.reaction_torque_ratio_m
        * force_axes
    )
    torque_axes = thrust_moments + reaction_moments
    allocation = np.vstack((force_axes.T, torque_axes.T))
    return _readonly(allocation)


def rotor_wrench_body(
    allocation_body: object,
    rotor_thrusts_n: object,
) -> FloatArray:
    """Map four thrust magnitudes to body force/torque wrench (Eq. 16)."""

    allocation = _as_finite_array(
        allocation_body,
        (6, ROTOR_COUNT),
        "allocation_body",
    )
    thrusts = _as_finite_array(
        rotor_thrusts_n,
        (ROTOR_COUNT,),
        "rotor_thrusts_n",
    )
    return _readonly(allocation @ thrusts)


def first_order_thrust_rate(
    actual_thrusts_n: object,
    commanded_thrusts_n: object,
    actuator: RotorActuatorConfig,
) -> FloatArray:
    """Evaluate ``Gamma_T^-1 (u_T - T)`` from paper Eq. (11)."""

    actual = _as_finite_array(
        actual_thrusts_n,
        (ROTOR_COUNT,),
        "actual_thrusts_n",
    )
    commanded = _as_finite_array(
        commanded_thrusts_n,
        (ROTOR_COUNT,),
        "commanded_thrusts_n",
    )
    return _readonly((commanded - actual) / actuator.time_constants_s)


def first_order_thrust_step(
    actual_thrusts_n: object,
    commanded_thrusts_n: object,
    actuator: RotorActuatorConfig,
    dt_s: object,
) -> FloatArray:
    """Exactly discretize Eq. (11) for a zero-order-held command.

    This helper does not clamp the result to Eq. (12); callers can inspect
    constraint residuals explicitly with :func:`evaluate_rotor_constraints`.
    """

    actual = _as_finite_array(
        actual_thrusts_n,
        (ROTOR_COUNT,),
        "actual_thrusts_n",
    )
    commanded = _as_finite_array(
        commanded_thrusts_n,
        (ROTOR_COUNT,),
        "commanded_thrusts_n",
    )
    dt = _finite_scalar(dt_s, "dt_s", minimum=0.0)
    decay = np.exp(-dt / actuator.time_constants_s)
    return _readonly(commanded + (actual - commanded) * decay)


def evaluate_rotor_constraints(
    actual_thrusts_n: object,
    thrust_rates_n_per_s: object,
    commanded_thrusts_n: object,
    actuator: RotorActuatorConfig,
) -> RotorConstraintResiduals:
    """Return nonnegative margins for every bound in paper Eq. (12)."""

    actual = _as_finite_array(
        actual_thrusts_n,
        (ROTOR_COUNT,),
        "actual_thrusts_n",
    )
    rates = _as_finite_array(
        thrust_rates_n_per_s,
        (ROTOR_COUNT,),
        "thrust_rates_n_per_s",
    )
    commanded = _as_finite_array(
        commanded_thrusts_n,
        (ROTOR_COUNT,),
        "commanded_thrusts_n",
    )
    return RotorConstraintResiduals(
        thrust_lower_margin_n=actual - actuator.thrust_min_n,
        thrust_upper_margin_n=actuator.thrust_max_n - actual,
        thrust_rate_lower_margin_n_per_s=(rates - actuator.thrust_rate_min_n_per_s),
        thrust_rate_upper_margin_n_per_s=(actuator.thrust_rate_max_n_per_s - rates),
        command_lower_margin_n=commanded - actuator.thrust_min_n,
        command_upper_margin_n=actuator.thrust_max_n - commanded,
    )


__all__ = [
    "build_fixed_deployed_allocation_matrix",
    "evaluate_rotor_constraints",
    "first_order_thrust_rate",
    "first_order_thrust_step",
    "rotor_wrench_body",
    "thrust_and_reaction_torque",
]
