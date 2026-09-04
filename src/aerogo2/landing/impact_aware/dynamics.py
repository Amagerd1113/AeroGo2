"""Reduced continuous and discrete prediction dynamics from Eqs. (30)-(34).

中文说明：预测状态以整机总质心 C 为平动原点，合成重力、已激活足端接触力和固定
展开旋翼产生的合力/力矩。旋翼推力采用一阶执行器模型，并在离散步中先推进连续
动力学、再正交化姿态矩阵。这里不包含 Go2 内部关节伺服、PX4 姿态基线或传输延迟；
这些影响必须通过执行计划、约束和实机辨识参数体现。
"""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np

from aerogo2.landing.impact_aware.math_utils import (
    _as_binary_vector,
    _as_finite_array,
    _finite_scalar,
    _readonly,
    integrate_body_rotation,
    require_rotation_matrix,
    skew,
)
from aerogo2.landing.impact_aware.rotor import (
    first_order_thrust_rate,
    first_order_thrust_step,
    rotor_wrench_body,
)
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    FloatArray,
    FootLeverArmsFromComBody,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    ReducedStateDerivative,
    RotorActuatorConfig,
    require_com_foot_lever_arms,
)


def aggregate_contact_wrench(
    contact_forces_world_n: object,
    contact_indicators: object,
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBody,
    rotation_body_to_world: object,
    *,
    contact_force_leg_order: object,
) -> Tuple[FloatArray, FloatArray]:
    """Return resultant force and CoM torque from paper Eq. (31).

    The third argument is ``{}^B r_CF``, from total-system CoM ``C`` to each
    foot.  It is not the Go2 FK result ``{}^B r_BF``.  A typed
    :class:`FootLeverArmsFromComBody` must be supplied after the explicit B-to-C
    conversion.  Its row order must exactly match ``contact_force_leg_order``;
    unlabeled arrays are deliberately rejected.
    """

    forces_world = _as_finite_array(
        contact_forces_world_n,
        (FOOT_COUNT, 3),
        "contact_forces_world_n",
    )
    indicators = _as_binary_vector(
        contact_indicators,
        FOOT_COUNT,
        "contact_indicators",
    )
    lever_arms = require_com_foot_lever_arms(
        foot_lever_arms_from_com_body_m,
        data_leg_order=contact_force_leg_order,
    )
    foot_lever_arms = lever_arms.values_m
    rotation = require_rotation_matrix(
        rotation_body_to_world,
        name="rotation_body_to_world",
    )

    active = indicators[:, np.newaxis]
    resultant_force_world = np.sum(active * forces_world, axis=0)
    forces_body = forces_world @ rotation
    resultant_torque_body = np.sum(
        active * np.cross(foot_lever_arms, forces_body),
        axis=0,
    )
    return _readonly(resultant_force_world), _readonly(resultant_torque_body)


def reduced_continuous_dynamics(
    state: ReducedState,
    control: ReducedInput,
    contact_indicators: object,
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBody,
    config: ReducedDynamicsConfig,
    actuator: RotorActuatorConfig,
    *,
    contact_force_leg_order: object,
) -> ReducedStateDerivative:
    """Evaluate the reduced-order ODE in paper Eqs. (30)-(33).

    Contact forces and linear state quantities are world-frame values.  Angular
    velocity, inertia, rotor wrench, CoM-to-foot lever arms, and torques are
    body-frame values.  No contact or actuator constraints are silently
    enforced here.
    """

    if not isinstance(state, ReducedState):
        raise TypeError("state must be a ReducedState")
    if not isinstance(control, ReducedInput):
        raise TypeError("control must be a ReducedInput")
    if not isinstance(config, ReducedDynamicsConfig):
        raise TypeError("config must be a ReducedDynamicsConfig")
    if not isinstance(actuator, RotorActuatorConfig):
        raise TypeError("actuator must be a RotorActuatorConfig")

    resultant_contact_force_world, resultant_contact_torque_body = aggregate_contact_wrench(
        control.contact_forces_world_n,
        contact_indicators,
        foot_lever_arms_from_com_body_m,
        state.rotation_body_to_world,
        contact_force_leg_order=contact_force_leg_order,
    )
    rotor_wrench = rotor_wrench_body(
        config.rotor_allocation_body,
        state.rotor_thrusts_n,
    )
    rotor_force_body = rotor_wrench[:3]
    rotor_torque_body = rotor_wrench[3:]

    linear_acceleration_world = (
        config.gravity_world_m_per_s2
        + (resultant_contact_force_world + state.rotation_body_to_world @ rotor_force_body)
        / config.mass_kg
    )

    angular_momentum_body = config.inertia_body_kg_m2 @ state.angular_velocity_body_rad_per_s
    angular_acceleration_body = cast(
        FloatArray,
        np.linalg.solve(
            config.inertia_body_kg_m2,
            resultant_contact_torque_body
            + rotor_torque_body
            - np.cross(state.angular_velocity_body_rad_per_s, angular_momentum_body),
        ),
    )

    thrust_rates = first_order_thrust_rate(
        state.rotor_thrusts_n,
        control.rotor_thrust_commands_n,
        actuator,
    )
    rotation_rate = state.rotation_body_to_world @ skew(state.angular_velocity_body_rad_per_s)

    return ReducedStateDerivative(
        position_rate_world_m_per_s=state.linear_velocity_world_m_per_s,
        linear_acceleration_world_m_per_s2=linear_acceleration_world,
        rotation_rate_body_to_world_per_s=rotation_rate,
        angular_acceleration_body_rad_per_s2=angular_acceleration_body,
        rotor_thrust_rates_n_per_s=thrust_rates,
    )


def reduced_discrete_step(
    state: ReducedState,
    control: ReducedInput,
    contact_indicators: object,
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBody,
    config: ReducedDynamicsConfig,
    actuator: RotorActuatorConfig,
    dt_s: object,
    *,
    contact_force_leg_order: object,
) -> ReducedState:
    """Advance paper Eq. (34) by one documented Lie-Euler step.

    Position, world linear velocity, and body angular velocity use forward
    Euler.  Attitude uses ``R @ Exp(omega_B * dt)`` so the matrix remains on
    ``SO(3)``.  The independently identified first-order rotor state uses its
    exact zero-order-hold update, avoiding the ``dt / tau`` instability of an
    Euler actuator step.  The method performs no clipping and no impact reset;
    touchdown reset is a separate operation.
    """

    dt = _finite_scalar(dt_s, "dt_s", minimum=0.0, strictly_greater=True)
    derivative = reduced_continuous_dynamics(
        state,
        control,
        contact_indicators,
        foot_lever_arms_from_com_body_m,
        config,
        actuator,
        contact_force_leg_order=contact_force_leg_order,
    )
    return ReducedState(
        position_world_m=(state.position_world_m + dt * derivative.position_rate_world_m_per_s),
        linear_velocity_world_m_per_s=(
            state.linear_velocity_world_m_per_s + dt * derivative.linear_acceleration_world_m_per_s2
        ),
        rotation_body_to_world=integrate_body_rotation(
            state.rotation_body_to_world,
            state.angular_velocity_body_rad_per_s,
            dt,
        ),
        angular_velocity_body_rad_per_s=(
            state.angular_velocity_body_rad_per_s
            + dt * derivative.angular_acceleration_body_rad_per_s2
        ),
        rotor_thrusts_n=first_order_thrust_step(
            state.rotor_thrusts_n,
            control.rotor_thrust_commands_n,
            actuator,
            dt,
        ),
    )


__all__ = [
    "aggregate_contact_wrench",
    "reduced_continuous_dynamics",
    "reduced_discrete_step",
]
