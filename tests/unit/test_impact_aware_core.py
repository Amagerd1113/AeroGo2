"""Focused equation-level tests for the hardware-isolated landing core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Optional

import numpy as np
import pytest

from aerogo2.landing.impact_aware import (
    GO2_SDK_LEG_ORDER,
    FixedDeployedRotorGeometry,
    FootLeverArmsFromComBody,
    FootPositionsFromBodyOriginB,
    ImpactLimits,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
    RotorAerodynamics,
    aggregate_contact_wrench,
    build_fixed_deployed_allocation_matrix,
    contact_transition_indicators,
    evaluate_impulse_constraints,
    evaluate_rotor_constraints,
    first_order_thrust_rate,
    first_order_thrust_step,
    foot_positions_from_body_origin_B_to_com_lever_arms,
    foot_post_impact_velocity,
    impulses_world_to_body,
    integrate_contact_impulse,
    is_rotation_matrix,
    momentum_reset,
    reduced_continuous_dynamics,
    reduced_discrete_step,
    rotor_wrench_body,
    skew,
    so3_exp,
    sticking_constraint_satisfied,
    sticking_velocity_residual,
    thrust_and_reaction_torque,
    total_com_C_linear_velocity_world_from_go2_body_origin_B,
    total_com_C_position_world_from_go2_body_origin_B,
    vee,
)


def _aerodynamics() -> RotorAerodynamics:
    return RotorAerodynamics(
        thrust_coefficient_n_per_rad_s_squared=2.0,
        drag_torque_coefficient_nm_per_rad_s_squared=0.2,
        spin_directions=np.array([1.0, -1.0, 1.0, -1.0]),
    )


def _geometry() -> FixedDeployedRotorGeometry:
    return FixedDeployedRotorGeometry(
        lever_arms_from_com_body_m=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        thrust_directions_body=np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
    )


def _actuator() -> RotorActuatorConfig:
    return RotorActuatorConfig(
        time_constants_s=np.array([1.0, 2.0, 4.0, 8.0]),
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 10.0),
        thrust_rate_min_n_per_s=np.full(4, -2.0),
        thrust_rate_max_n_per_s=np.full(4, 2.0),
    )


def _dynamics_config(
    *,
    gravity_world_m_per_s2: Optional[np.ndarray] = None,
) -> ReducedDynamicsConfig:
    gravity = (
        np.array([0.0, 0.0, -9.81]) if gravity_world_m_per_s2 is None else gravity_world_m_per_s2
    )
    return ReducedDynamicsConfig(
        mass_kg=2.0,
        inertia_body_kg_m2=np.diag([2.0, 3.0, 4.0]),
        gravity_world_m_per_s2=gravity,
        rotor_allocation_body=build_fixed_deployed_allocation_matrix(_geometry(), _aerodynamics()),
    )


def _state(
    *,
    position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
    angular_velocity: Optional[np.ndarray] = None,
    thrusts: Optional[np.ndarray] = None,
) -> ReducedState:
    return ReducedState(
        position_world_m=np.zeros(3) if position is None else position,
        linear_velocity_world_m_per_s=np.zeros(3) if velocity is None else velocity,
        rotation_body_to_world=np.eye(3) if rotation is None else rotation,
        angular_velocity_body_rad_per_s=(
            np.zeros(3) if angular_velocity is None else angular_velocity
        ),
        rotor_thrusts_n=np.ones(4) if thrusts is None else thrusts,
    )


def test_config_and_state_boundaries_are_defensively_immutable() -> None:
    source_position = np.array([1.0, 2.0, 3.0])
    state = _state(position=source_position)
    source_position[0] = 99.0

    assert state.position_world_m[0] == 1.0
    with pytest.raises(ValueError):
        state.position_world_m[0] = 2.0
    with pytest.raises(FrozenInstanceError):
        state.position_world_m = np.zeros(3)

    with pytest.raises(ValueError, match="shape"):
        _state(velocity=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        _state(thrusts=np.array([1.0, 1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        _state(rotation=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="positive definite"):
        ReducedDynamicsConfig(
            mass_kg=1.0,
            inertia_body_kg_m2=np.diag([1.0, 1.0, 0.0]),
            gravity_world_m_per_s2=np.zeros(3),
            rotor_allocation_body=np.zeros((6, 4)),
        )


def test_so3_helpers_preserve_frame_kinematics() -> None:
    vector = np.array([0.2, -0.3, 0.4])
    other = np.array([-1.0, 2.0, 0.5])
    assert np.allclose(skew(vector) @ other, np.cross(vector, other))
    assert np.allclose(vee(skew(vector)), vector)

    quarter_turn = so3_exp(np.array([0.0, 0.0, np.pi / 2.0]))
    assert is_rotation_matrix(quarter_turn)
    assert np.allclose(quarter_turn @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


def test_go2_body_origin_B_to_total_com_C_rigid_point_kinematics() -> None:
    rotation = so3_exp(np.array([0.0, 0.0, np.pi / 2.0]))
    offset_BC_body = np.array([0.1, 0.2, 0.3])

    position_C_world = total_com_C_position_world_from_go2_body_origin_B(
        np.array([1.0, 2.0, 3.0]),
        rotation,
        offset_BC_body,
    )
    velocity_C_world = total_com_C_linear_velocity_world_from_go2_body_origin_B(
        np.array([1.0, -2.0, 0.5]),
        rotation,
        np.array([0.0, 0.0, 2.0]),
        offset_BC_body,
    )

    assert np.allclose(position_C_world, [0.8, 2.1, 3.3], atol=1e-12)
    assert np.allclose(velocity_C_world, [0.8, -2.4, 0.5], atol=1e-12)
    assert not position_C_world.flags.writeable
    assert not velocity_C_world.flags.writeable


def test_go2_body_origin_B_to_total_com_C_kinematics_reject_invalid_frames() -> None:
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        total_com_C_position_world_from_go2_body_origin_B(
            np.zeros(3),
            np.diag([1.0, 1.0, -1.0]),
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="shape"):
        total_com_C_linear_velocity_world_from_go2_body_origin_B(
            np.zeros(3),
            np.eye(3),
            np.zeros(2),
            np.zeros(3),
        )


def test_rotor_equations_and_allocation_match_known_geometry() -> None:
    aerodynamics = _aerodynamics()
    thrusts, reaction_torques = thrust_and_reaction_torque(
        np.array([1.0, 2.0, 3.0, 4.0]),
        aerodynamics,
    )
    assert aerodynamics.reaction_torque_ratio_m == pytest.approx(0.1)
    assert np.allclose(thrusts, [2.0, 8.0, 18.0, 32.0])
    assert np.allclose(reaction_torques, [0.2, -0.8, 1.8, -3.2])

    allocation = build_fixed_deployed_allocation_matrix(_geometry(), aerodynamics)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.1, -0.1, 0.1, -0.1],
        ]
    )
    assert np.allclose(allocation, expected)
    assert np.allclose(rotor_wrench_body(allocation, np.ones(4)), [0, 0, 4, 0, 0, 0])


def test_fixed_deployed_geometry_defensively_copies_direct_constants() -> None:
    lever_arms = np.array(
        [
            [-0.3, -0.2, 0.1],
            [0.3, 0.2, 0.1],
            [-0.3, 0.2, 0.1],
            [0.3, -0.2, 0.1],
        ]
    )
    axes = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    geometry = FixedDeployedRotorGeometry(lever_arms, axes)
    expected = geometry.lever_arms_from_com_body_m.copy()

    lever_arms[:] = 999.0
    axes[:] = -1.0

    assert np.array_equal(geometry.lever_arms_from_com_body_m, expected)
    assert np.array_equal(
        geometry.thrust_directions_body,
        np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
    )
    with pytest.raises(ValueError):
        geometry.lever_arms_from_com_body_m[0, 0] = 0.0


def test_rotor_first_order_dynamics_and_constraint_margins() -> None:
    actuator = _actuator()
    actual = np.ones(4)
    commanded = np.array([2.0, 3.0, 4.0, 5.0])
    rates = first_order_thrust_rate(actual, commanded, actuator)
    assert np.allclose(rates, [1.0, 1.0, 0.75, 0.5])

    exact_step = first_order_thrust_step(actual, commanded, actuator, 0.2)
    expected = commanded + (actual - commanded) * np.exp(-0.2 / actuator.time_constants_s)
    assert np.allclose(exact_step, expected)

    residuals = evaluate_rotor_constraints(actual, rates, commanded, actuator)
    assert residuals.is_feasible()
    violated = evaluate_rotor_constraints(
        actual,
        np.array([3.0, 0.0, 0.0, 0.0]),
        commanded,
        actuator,
    )
    assert not violated.is_feasible()
    assert violated.thrust_rate_upper_margin_n_per_s[0] == -1.0


def test_reduced_continuous_dynamics_use_documented_frames() -> None:
    state = _state()
    control = ReducedInput(
        contact_forces_world_n=np.array(
            [
                [2.0, 0.0, 4.0],
                [100.0, 100.0, 100.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        rotor_thrust_commands_n=np.array([2.0, 3.0, 4.0, 5.0]),
    )
    feet = np.zeros((4, 3))
    feet[0] = [1.0, 0.0, -1.0]

    derivative = reduced_continuous_dynamics(
        state,
        control,
        np.array([1, 0, 0, 0]),
        FootLeverArmsFromComBody(feet, GO2_SDK_LEG_ORDER),
        _dynamics_config(),
        _actuator(),
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    assert np.allclose(derivative.position_rate_world_m_per_s, np.zeros(3))
    assert np.allclose(
        derivative.linear_acceleration_world_m_per_s2,
        [1.0, 0.0, -5.81],
    )
    assert np.allclose(
        derivative.angular_acceleration_body_rad_per_s2,
        [0.0, -2.0, 0.0],
    )
    assert np.allclose(derivative.rotation_rate_body_to_world_per_s, np.zeros((3, 3)))
    assert np.allclose(derivative.rotor_thrust_rates_n_per_s, [1.0, 1.0, 0.75, 0.5])


def test_contact_torque_requires_explicit_B_to_C_foot_lever_arm_conversion() -> None:
    feet_from_B_values = np.zeros((4, 3))
    feet_from_B_values[0] = [1.0, 0.0, 0.0]
    feet_from_B = FootPositionsFromBodyOriginB(
        feet_from_B_values,
        GO2_SDK_LEG_ORDER,
    )
    with pytest.raises(TypeError, match="converted to CoM lever arms"):
        aggregate_contact_wrench(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([1, 0, 0, 0]),
            feet_from_B,
            np.eye(3),
            contact_force_leg_order=GO2_SDK_LEG_ORDER,
        )

    with pytest.raises(TypeError, match="unlabeled arrays are forbidden"):
        aggregate_contact_wrench(
            np.zeros((4, 3)),
            np.zeros(4),
            feet_from_B_values,
            np.eye(3),
            contact_force_leg_order=GO2_SDK_LEG_ORDER,
        )

    lever_arms = foot_positions_from_body_origin_B_to_com_lever_arms(
        feet_from_B,
        np.array([0.0, 0.0, 0.5]),
    )
    force, torque = aggregate_contact_wrench(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([1, 0, 0, 0]),
        lever_arms,
        np.eye(3),
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    np.testing.assert_allclose(force, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(torque, [0.5, 0.0, 1.0])

    with pytest.raises(ValueError, match="leg_order must exactly match"):
        aggregate_contact_wrench(
            np.zeros((4, 3)),
            np.zeros(4),
            lever_arms,
            np.eye(3),
            contact_force_leg_order=("FL", "FR", "RR", "RL"),
        )


def test_impact_entry_points_reject_untyped_B_origin_and_mismatched_order() -> None:
    state = _state()
    raw = np.zeros((4, 3))
    from_B = FootPositionsFromBodyOriginB(raw, GO2_SDK_LEG_ORDER)
    typed = FootLeverArmsFromComBody(raw, GO2_SDK_LEG_ORDER)

    for invalid in (raw, from_B):
        with pytest.raises(TypeError):
            momentum_reset(
                state,
                np.zeros((4, 3)),
                np.zeros(4),
                invalid,
                _dynamics_config(),
                impulse_leg_order=GO2_SDK_LEG_ORDER,
            )
        with pytest.raises(TypeError):
            foot_post_impact_velocity(
                state,
                invalid,
                np.zeros((4, 3, 3)),
                np.zeros((4, 3)),
                leg_order=GO2_SDK_LEG_ORDER,
            )

    mismatched = ("FL", "FR", "RR", "RL")
    with pytest.raises(ValueError, match="leg_order must exactly match"):
        momentum_reset(
            state,
            np.zeros((4, 3)),
            np.zeros(4),
            typed,
            _dynamics_config(),
            impulse_leg_order=mismatched,
        )
    with pytest.raises(ValueError, match="leg_order must exactly match"):
        foot_post_impact_velocity(
            state,
            typed,
            np.zeros((4, 3, 3)),
            np.zeros((4, 3)),
            leg_order=mismatched,
        )


def test_reduced_discrete_step_uses_lie_euler_and_exact_rotor_zoh() -> None:
    config = ReducedDynamicsConfig(
        mass_kg=1.0,
        inertia_body_kg_m2=np.eye(3),
        gravity_world_m_per_s2=np.zeros(3),
        rotor_allocation_body=np.zeros((6, 4)),
    )
    actuator = RotorActuatorConfig(
        time_constants_s=np.ones(4),
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 10.0),
        thrust_rate_min_n_per_s=np.full(4, -100.0),
        thrust_rate_max_n_per_s=np.full(4, 100.0),
    )
    state = _state(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 1.0]),
        thrusts=np.zeros(4),
    )
    control = ReducedInput(np.zeros((4, 3)), np.ones(4))
    after = reduced_discrete_step(
        state,
        control,
        np.zeros(4),
        FootLeverArmsFromComBody(np.zeros((4, 3)), GO2_SDK_LEG_ORDER),
        config,
        actuator,
        0.1,
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    assert np.allclose(after.position_world_m, [1.05, 2.0, 3.0])
    assert np.allclose(after.linear_velocity_world_m_per_s, state.linear_velocity_world_m_per_s)
    assert np.allclose(after.rotation_body_to_world, so3_exp([0.0, 0.0, 0.1]))
    assert np.allclose(after.angular_velocity_body_rad_per_s, [0.0, 0.0, 1.0])
    assert np.allclose(after.rotor_thrusts_n, np.full(4, 1.0 - np.exp(-0.1)))


def test_sampled_impulse_and_contact_transition_equations() -> None:
    forces = np.zeros((3, 4, 3))
    forces[:, 0, :] = [1.0, 2.0, 3.0]
    impulse = integrate_contact_impulse(forces, np.array([0.0, 0.2, 0.5]))
    assert np.allclose(impulse[0], [0.5, 1.0, 1.5])
    assert np.allclose(impulse[1:], 0.0)

    touchdown, participation = contact_transition_indicators(
        np.array([1, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
    )
    assert np.array_equal(touchdown, [0, 1, 0, 0])
    assert np.array_equal(participation, [1, 1, 0, 0])

    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_contact_impulse(forces, np.array([0.0, 0.2, 0.2]))
    with pytest.raises(ValueError, match="0 or 1"):
        contact_transition_indicators([0, 0.5, 0, 0], [0, 1, 0, 0])


def test_impulse_frame_transform_and_momentum_reset() -> None:
    rotation = so3_exp([0.0, 0.0, np.pi / 2.0])
    world_impulses = np.zeros((4, 3))
    world_impulses[0] = [1.0, 0.0, 0.0]
    body_impulses = impulses_world_to_body(world_impulses, rotation)
    assert np.allclose(body_impulses[0], [0.0, -1.0, 0.0], atol=1e-12)

    pre_impact = _state(
        velocity=np.array([0.0, 0.0, -1.0]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
    )
    impulses = np.zeros((4, 3))
    impulses[0] = [2.0, 0.0, 4.0]
    impulses[1] = [100.0, 100.0, 100.0]
    feet = np.zeros((4, 3))
    feet[0] = [1.0, 0.0, 0.0]
    after = momentum_reset(
        pre_impact,
        impulses,
        np.array([1, 0, 0, 0]),
        FootLeverArmsFromComBody(feet, GO2_SDK_LEG_ORDER),
        _dynamics_config(),
        impulse_leg_order=GO2_SDK_LEG_ORDER,
    )
    assert np.allclose(after.linear_velocity_world_m_per_s, [1.0, 0.0, 1.0])
    assert np.allclose(
        after.angular_velocity_body_rad_per_s,
        [0.1, 0.2 - 4.0 / 3.0, 0.3],
    )
    assert np.allclose(after.position_world_m, pre_impact.position_world_m)
    assert np.allclose(after.rotation_body_to_world, pre_impact.rotation_body_to_world)
    assert np.allclose(after.rotor_thrusts_n, pre_impact.rotor_thrusts_n)


def test_foot_velocity_and_sticking_residual() -> None:
    state = _state(
        velocity=np.array([1.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 2.0]),
    )
    feet = np.zeros((4, 3))
    feet[0] = [1.0, 0.0, 0.0]
    jacobians = np.tile(np.eye(3), (4, 1, 1))
    joint_velocities = np.zeros((4, 3))
    joint_velocities[0] = [0.0, 0.0, 1.0]
    velocities = foot_post_impact_velocity(
        state,
        FootLeverArmsFromComBody(feet, GO2_SDK_LEG_ORDER),
        jacobians,
        joint_velocities,
        leg_order=GO2_SDK_LEG_ORDER,
    )
    assert np.allclose(velocities[0], [1.0, 2.0, 1.0])
    assert np.allclose(velocities[1:], [[1.0, 0.0, 0.0]] * 3)

    participation = np.array([1, 0, 0, 0])
    residual = sticking_velocity_residual(velocities, participation)
    assert np.allclose(residual[0], velocities[0])
    assert np.allclose(residual[1:], 0.0)
    assert not sticking_constraint_satisfied(velocities, participation, atol_m_per_s=1.5)
    assert sticking_constraint_satisfied(np.zeros((4, 3)), participation)


def test_impulse_constraint_residuals_cover_inactive_and_active_feet() -> None:
    limits = ImpactLimits(
        friction_coefficients=np.full(4, 0.5),
        maximum_normal_impulse_ns=10.0,
        impact_duration_s=0.1,
        maximum_average_normal_force_n=80.0,
    )
    impulses = np.zeros((4, 3))
    impulses[0] = [3.0, 0.0, 6.0]
    participation = np.array([1, 0, 0, 0])
    residuals = evaluate_impulse_constraints(impulses, participation, limits)
    assert residuals.is_feasible()
    assert residuals.normal_lower_margin_ns[0] == 6.0
    assert residuals.normal_upper_margin_ns[0] == 4.0
    assert residuals.friction_cone_margin_ns[0] == 0.0
    assert residuals.equivalent_average_normal_force_n[0] == 60.0
    assert residuals.average_force_upper_margin_n[0] == 20.0

    inactive_tangential_impulse = impulses.copy()
    inactive_tangential_impulse[1, 0] = 0.1
    assert not evaluate_impulse_constraints(
        inactive_tangential_impulse,
        participation,
        limits,
    ).is_feasible()

    excessive_average_force = impulses.copy()
    excessive_average_force[0, 2] = 9.0
    excessive = evaluate_impulse_constraints(
        excessive_average_force,
        participation,
        limits,
    )
    assert not excessive.is_feasible()
    assert excessive.average_force_upper_margin_n[0] == -10.0


def test_strict_configuration_validation_rejects_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        FixedDeployedRotorGeometry(np.zeros((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match=r"-1 or \+1"):
        RotorAerodynamics(1.0, 0.1, np.array([1, -1, 0, -1]))
    with pytest.raises(ValueError, match="strictly positive"):
        RotorActuatorConfig(
            np.array([1.0, 1.0, 0.0, 1.0]),
            np.zeros(4),
            np.ones(4),
            -np.ones(4),
            np.ones(4),
        )
    with pytest.raises(ValueError, match="contain zero"):
        RotorActuatorConfig(
            np.ones(4),
            np.zeros(4),
            np.ones(4),
            np.ones(4),
            np.full(4, 2.0),
        )
    with pytest.raises(ValueError, match="finite"):
        ReducedInput(np.full((4, 3), np.inf), np.zeros(4))
    with pytest.raises(TypeError, match="real numeric"):
        ReducedInput(np.full((4, 3), "1"), np.zeros(4))
