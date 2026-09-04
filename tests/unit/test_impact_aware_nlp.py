from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
import pytest

import aerogo2.landing.impact_aware.nlp as nlp_module
from aerogo2.landing.impact_aware.dynamics import reduced_discrete_step
from aerogo2.landing.impact_aware.impact import (
    evaluate_impulse_constraints,
    foot_post_impact_velocity,
    momentum_reset,
)
from aerogo2.landing.impact_aware.nlp import (
    CONTROL_DIM,
    STATE_DIM,
    ContactForceLimits,
    ImpactAwareMPCProblem,
    ImpactAwareNLP,
    ImpactEvent,
    LandingContactGeometry,
    MPCReferences,
    MPCWarmStart,
    MPCWeights,
    RotorExecutionPlan,
    SLSQPSettings,
    StateBounds,
    reconstruct_transport_target,
)
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FootLeverArmsFromComBody,
    FootLeverArmsFromComBodyHorizon,
    FootPositionsFromBodyOriginB,
    ImpactLimits,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
)


def _weights(*, input_rate_scale: float = 1.0) -> MPCWeights:
    return MPCWeights(
        tracking=np.eye(12),
        input=np.eye(CONTROL_DIM),
        input_rate=input_rate_scale * np.eye(CONTROL_DIM),
        slack=np.eye(STATE_DIM),
        terminal_tracking=np.eye(12),
        impulse=np.eye(3),
        touchdown_velocity=np.eye(3),
    )


def _problem(
    *,
    gravity_z: float = -9.81,
    initial_velocity: Optional[np.ndarray] = None,
    initial_thrust_per_rotor: float = 4.905,
    command_per_rotor: float = 4.905,
    contact_schedule: Optional[np.ndarray] = None,
    impact_events: Tuple[ImpactEvent, ...] = (),
    previous_command_per_rotor: Optional[float] = None,
    weights: Optional[MPCWeights] = None,
    rotor_execution_plan: Optional[RotorExecutionPlan] = None,
    landing_contact_geometry: Optional[LandingContactGeometry] = None,
    initial_rotation_body_to_world: Optional[np.ndarray] = None,
) -> ImpactAwareMPCProblem:
    horizon = 1
    schedule = (
        np.zeros((horizon + 1, 4), dtype=int) if contact_schedule is None else contact_schedule
    )
    velocity = np.zeros(3) if initial_velocity is None else initial_velocity
    rotation = (
        np.eye(3)
        if initial_rotation_body_to_world is None
        else initial_rotation_body_to_world
    )
    state = ReducedState(
        position_world_m=np.zeros(3),
        linear_velocity_world_m_per_s=velocity,
        rotation_body_to_world=rotation,
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=np.full(4, initial_thrust_per_rotor),
    )
    previous_command = (
        command_per_rotor if previous_command_per_rotor is None else previous_command_per_rotor
    )
    previous_input = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.full(4, previous_command),
    )
    allocation = np.zeros((6, 4))
    allocation[2, :] = 1.0
    dynamics = ReducedDynamicsConfig(
        mass_kg=2.0,
        inertia_body_kg_m2=np.diag([1.0, 1.0, 1.0]),
        gravity_world_m_per_s2=np.array([0.0, 0.0, gravity_z]),
        rotor_allocation_body=allocation,
    )
    actuator = RotorActuatorConfig(
        time_constants_s=np.full(4, 0.2),
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 20.0),
        thrust_rate_min_n_per_s=np.full(4, -100.0),
        thrust_rate_max_n_per_s=np.full(4, 100.0),
    )
    references = MPCReferences(
        position_world_m=np.zeros((horizon + 1, 3)),
        linear_velocity_world_m_per_s=np.repeat(velocity[None, :], horizon + 1, axis=0),
        rotation_body_to_world=np.repeat(rotation[None, :, :], horizon + 1, axis=0),
        angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
        contact_forces_world_n=np.zeros((horizon, 4, 3)),
        rotor_thrust_commands_n=np.full((horizon, 4), command_per_rotor),
    )
    bounds = StateBounds(
        lower=np.full((horizon + 1, STATE_DIM), -np.inf),
        upper=np.full((horizon + 1, STATE_DIM), np.inf),
        soft_mask=np.zeros((horizon, STATE_DIM), dtype=bool),
    )
    geometry = landing_contact_geometry
    if impact_events and geometry is None:
        geometry = LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=float(velocity[2]) * 0.05,
            touchdown_position_tolerance_m=1.0e-8,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        )
    elif geometry is None:
        geometry = LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=-1.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        )
    return ImpactAwareMPCProblem(
        initial_state=state,
        previous_input=previous_input,
        dt_s=0.05,
        contact_schedule=schedule,
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            np.zeros((horizon + 1, 4, 3)),
            GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=references,
        state_bounds=bounds,
        contact_limits=ContactForceLimits(
            friction_coefficients=np.full(4, 0.5),
            maximum_normal_force_n=np.full(4, 100.0),
        ),
        impact_events=impact_events,
        dynamics_config=dynamics,
        rotor_actuator_config=actuator,
        weights=_weights() if weights is None else weights,
        landing_contact_geometry=geometry,
        rotor_execution_plan=rotor_execution_plan,
    )


def _single_foot_impact_event() -> ImpactEvent:
    return ImpactEvent(
        step=1,
        touchdown=np.array([1, 0, 0, 0]),
        participation=np.array([1, 0, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=ImpactLimits(
            friction_coefficients=np.full(4, 0.5),
            maximum_normal_impulse_ns=2.0,
            impact_duration_s=0.1,
            maximum_average_normal_force_n=20.0,
        ),
    )


def test_hover_no_contact_reference_is_a_zero_cost_feasible_solve() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None
    nlp = ImpactAwareNLP(_problem())

    guess = nlp.initial_guess()
    assert nlp.objective(guess) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(nlp.equality_residual(guess))) < 1e-12
    assert np.min(nlp.inequality_residual(guess)) >= 0.0

    result = nlp.solve(
        SLSQPSettings(
            max_iterations=20,
            ftol=1e-10,
            constraint_tolerance=1e-8,
            timeout_s=2.0,
            display=False,
        )
    )

    assert result.success, result.message
    assert result.status == "success"
    assert result.objective == pytest.approx(0.0, abs=1e-10)
    assert result.max_equality_violation < 1e-8
    assert result.first_input is not None


def test_touchdown_reset_impulse_and_sticking_equalities_match_paper() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    event = _single_foot_impact_event()
    problem = _problem(
        gravity_z=0.0,
        initial_velocity=np.array([0.0, 0.0, -1.0]),
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(event,),
    )
    nlp = ImpactAwareNLP(problem)
    control = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.zeros(4),
    )
    pre_impact = reduced_discrete_step(
        problem.initial_state,
        control,
        problem.contact_schedule[0],
        problem.foot_lever_arms_from_com_body_m.at_step(0),
        problem.dynamics_config,
        problem.rotor_actuator_config,
        problem.dt_s,
        contact_force_leg_order=problem.foot_leg_order,
    )
    impulses = np.zeros((4, 3))
    impulses[0, 2] = 2.0  # mass * 1 m/s for the caller-selected 2 kg model
    post_impact = momentum_reset(
        pre_impact,
        impulses,
        event.participation,
        problem.foot_lever_arms_from_com_body_m.at_step(1),
        problem.dynamics_config,
        impulse_leg_order=problem.foot_leg_order,
    )
    candidate = nlp.pack(
        MPCWarmStart(
            states=(problem.initial_state, post_impact),
            controls=(control,),
            slacks=np.zeros((1, STATE_DIM)),
            impulses_by_step={1: impulses},
        )
    )

    assert np.max(np.abs(nlp.equality_residual(candidate))) < 1e-12
    assert np.min(nlp.inequality_residual(candidate)) >= -1e-12
    unpacked = nlp.unpack(candidate)
    assert unpacked.states[1].linear_velocity_world_m_per_s == pytest.approx(np.zeros(3))
    assert unpacked.impulses_by_step[1][0] == pytest.approx([0.0, 0.0, 2.0])


def test_touchdown_cost_uses_preimpact_leg_speed_while_sticking_uses_postimpact() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    event = _single_foot_impact_event()
    zero_weights = MPCWeights(
        tracking=np.zeros((12, 12)),
        input=np.zeros((CONTROL_DIM, CONTROL_DIM)),
        input_rate=np.zeros((CONTROL_DIM, CONTROL_DIM)),
        slack=np.zeros((STATE_DIM, STATE_DIM)),
        terminal_tracking=np.zeros((12, 12)),
        impulse=np.zeros((3, 3)),
        touchdown_velocity=np.eye(3),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(event,),
        weights=zero_weights,
    )
    jacobians = np.zeros((2, 4, 3, 3))
    jacobians[1, 0] = np.eye(3)
    preimpact_joint_velocities = np.zeros((2, 4, 3))
    preimpact_joint_velocities[1, 0] = [0.0, 0.0, -1.0]
    problem = replace(
        problem,
        leg_jacobians_body=jacobians,
        joint_velocities_rad_per_s=preimpact_joint_velocities,
    )
    nlp = ImpactAwareNLP(problem)
    decision = nlp.initial_guess()

    assert np.max(np.abs(nlp.equality_residual(decision))) < 1.0e-12
    assert nlp.objective(decision) == pytest.approx(1.0)
    trajectories = nlp.unpack(decision)
    post_velocity = foot_post_impact_velocity(
        trajectories.states[1],
        problem.foot_lever_arms_from_com_body_m.at_step(1),
        problem.leg_jacobians_body[1],
        event.post_impact_joint_velocities_rad_per_s,
        leg_order=problem.foot_leg_order,
    )
    assert post_velocity[0] == pytest.approx(np.zeros(3))


def test_impact_event_rejects_coerced_postimpact_joint_velocity() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        replace(
            _single_foot_impact_event(),
            post_impact_joint_velocities_rad_per_s=np.full((4, 3), "0"),
        )


def test_slsqp_solves_nonzero_arm_and_planned_leg_motion_touchdown() -> None:
    pytest.importorskip("scipy")
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    event = ImpactEvent(
        step=1,
        touchdown=np.array([1, 0, 0, 0]),
        participation=np.array([1, 0, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=ImpactLimits(
            friction_coefficients=np.ones(4),
            maximum_normal_impulse_ns=5.0,
            impact_duration_s=0.1,
            maximum_average_normal_force_n=50.0,
        ),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_velocity=np.array([0.1, -0.05, -0.8]),
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(event,),
    )
    feet = np.zeros((2, 4, 3))
    feet[:, 0] = [0.25, -0.15, -0.35]
    jacobians = np.zeros((2, 4, 3, 3))
    jacobians[1, 0] = np.diag([0.2, 0.15, 0.1])
    joint_velocities = np.zeros((2, 4, 3))
    joint_velocities[1, 0] = [0.1, -0.1, 0.2]
    problem = replace(
        problem,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            feet,
            GO2_SDK_LEG_ORDER,
        ),
        landing_contact_geometry=LandingContactGeometry(
            np.array([0.0, 0.0, 1.0]),
            -0.39,
            1.0e-8,
            0.0,
            np.deg2rad(30.0),
        ),
        leg_jacobians_body=jacobians,
        joint_velocities_rad_per_s=joint_velocities,
        impact_events=(
            replace(
                event,
                post_impact_joint_velocities_rad_per_s=joint_velocities[1],
            ),
        ),
    )
    nlp = ImpactAwareNLP(problem)

    automatic = nlp.unpack(nlp.initial_guess())
    zero_impulse_warm_start = MPCWarmStart(
        states=automatic.states,
        controls=automatic.controls,
        slacks=automatic.slacks,
        impulses_by_step={1: np.zeros((4, 3))},
    )
    initial_decision = nlp.pack(zero_impulse_warm_start)
    assert np.max(np.abs(nlp.equality_residual(initial_decision))) > 0.5

    result = nlp.solve(
        SLSQPSettings(
            max_iterations=200,
            ftol=1e-10,
            constraint_tolerance=1e-7,
            timeout_s=5.0,
            display=False,
        ),
        warm_start=zero_impulse_warm_start,
    )

    assert result.success, result.message
    assert result.raw_solver_status == 0
    assert result.iterations >= 2
    assert result.max_equality_violation < 1e-7
    assert result.min_inequality_residual >= -1e-7
    impulse = result.impulses_by_step[1]
    assert np.linalg.norm(impulse[0, :2]) > 0.1
    reset = momentum_reset(
        result.pre_impact_states[1],
        impulse,
        event.participation,
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        problem.dynamics_config,
        impulse_leg_order=GO2_SDK_LEG_ORDER,
    )
    assert result.states[1].linear_velocity_world_m_per_s == pytest.approx(
        reset.linear_velocity_world_m_per_s,
        abs=1e-8,
    )
    assert result.states[1].angular_velocity_body_rad_per_s == pytest.approx(
        reset.angular_velocity_body_rad_per_s,
        abs=1e-8,
    )
    foot_velocity = foot_post_impact_velocity(
        result.states[1],
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        jacobians[1],
        joint_velocities[1],
        leg_order=GO2_SDK_LEG_ORDER,
    )
    assert foot_velocity[0] == pytest.approx(np.zeros(3), abs=1e-7)
    impulse_residuals = evaluate_impulse_constraints(
        impulse,
        event.participation,
        event.impulse_limits,
    )
    assert impulse_residuals.is_feasible(atol=1e-7)
    assert impulse_residuals.friction_cone_margin_ns[0] > 0.0
    assert impulse_residuals.average_force_upper_margin_n[0] > 0.0


def test_slsqp_projects_redundant_two_foot_sticking_but_validates_all_rows() -> None:
    pytest.importorskip("scipy")
    schedule = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
    event = ImpactEvent(
        step=1,
        touchdown=np.array([1, 1, 0, 0]),
        participation=np.array([1, 1, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=ImpactLimits(
            friction_coefficients=np.full(4, 0.8),
            maximum_normal_impulse_ns=3.0,
            impact_duration_s=0.1,
            maximum_average_normal_force_n=30.0,
        ),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_velocity=np.array([0.0, 0.0, -0.8]),
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(event,),
    )
    feet = np.zeros((2, 4, 3))
    feet[:, 0] = [0.25, 0.0, -0.3]
    feet[:, 1] = [-0.25, 0.0, -0.3]
    problem = replace(
        problem,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            feet,
            GO2_SDK_LEG_ORDER,
        ),
        landing_contact_geometry=LandingContactGeometry(
            np.array([0.0, 0.0, 1.0]),
            -0.34,
            1.0e-8,
            0.0,
            np.deg2rad(30.0),
        ),
    )
    nlp = ImpactAwareNLP(problem)
    automatic = nlp.unpack(nlp.initial_guess())
    warm_start = MPCWarmStart(
        states=automatic.states,
        controls=automatic.controls,
        slacks=automatic.slacks,
        impulses_by_step={1: np.zeros((4, 3))},
    )
    decision = nlp.pack(warm_start)

    # Two point contacts have six scalar sticking equations but rank five.
    assert nlp.equality_residual(decision).size == nlp._solver_equality_residual(decision).size + 1
    result = nlp.solve(
        SLSQPSettings(200, 1e-10, 1e-7, 5.0, False),
        warm_start=warm_start,
    )

    assert result.success, result.message
    assert result.raw_solver_status == 0
    assert result.iterations >= 2
    assert result.max_equality_violation < 1e-7
    impulse = result.impulses_by_step[1]
    assert impulse[0, 2] == pytest.approx(0.8, abs=1e-7)
    assert impulse[1, 2] == pytest.approx(0.8, abs=1e-7)
    velocities = foot_post_impact_velocity(
        result.states[1],
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        np.zeros((4, 3, 3)),
        np.zeros((4, 3)),
        leg_order=GO2_SDK_LEG_ORDER,
    )
    assert velocities[:2] == pytest.approx(np.zeros((2, 3)), abs=1e-7)
    assert evaluate_impulse_constraints(
        impulse,
        event.participation,
        event.impulse_limits,
    ).is_feasible(atol=1e-7)


def test_three_phase_touchdown_rollout_transfers_support_to_legs() -> None:
    pytest.importorskip("scipy")
    horizon = 3
    dt_s = 0.05
    hover_thrust_per_rotor_n = 2.4525
    initial_state = ReducedState(
        position_world_m=np.array([0.0, 0.0, 0.4]),
        linear_velocity_world_m_per_s=np.array([0.0, 0.0, -0.2]),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=np.full(4, hover_thrust_per_rotor_n),
    )
    allocation = np.zeros((6, 4))
    allocation[2] = 1.0
    dynamics = ReducedDynamicsConfig(
        mass_kg=2.0,
        inertia_body_kg_m2=np.eye(3),
        gravity_world_m_per_s2=np.array([0.0, 0.0, -9.81]),
        rotor_allocation_body=allocation,
    )
    actuator = RotorActuatorConfig(
        time_constants_s=np.full(4, 0.2),
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 20.0),
        thrust_rate_min_n_per_s=np.full(4, -100.0),
        thrust_rate_max_n_per_s=np.full(4, 100.0),
    )
    schedule = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    feet = np.zeros((horizon + 1, 4, 3))
    feet[:, 0] = [0.25, 0.0, -0.3]
    feet[:, 1] = [-0.25, 0.0, -0.3]
    event = ImpactEvent(
        step=1,
        touchdown=np.array([1, 1, 0, 0]),
        participation=np.array([1, 1, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=ImpactLimits(
            friction_coefficients=np.full(4, 0.8),
            maximum_normal_impulse_ns=3.0,
            impact_duration_s=0.1,
            maximum_average_normal_force_n=30.0,
        ),
    )

    flight_control = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.full(4, hover_thrust_per_rotor_n),
    )
    pre_touchdown = reduced_discrete_step(
        initial_state,
        flight_control,
        schedule[0],
        FootLeverArmsFromComBody(feet[0], GO2_SDK_LEG_ORDER),
        dynamics,
        actuator,
        dt_s,
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    touchdown_impulses = np.zeros((4, 3))
    touchdown_impulses[:2, 2] = 0.44525
    touchdown_state = momentum_reset(
        pre_touchdown,
        touchdown_impulses,
        event.participation,
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        dynamics,
        impulse_leg_order=GO2_SDK_LEG_ORDER,
    )
    initial_support = np.zeros((4, 3))
    initial_support[:2, 2] = 4.905
    initial_support_control = ReducedInput(
        contact_forces_world_n=initial_support,
        rotor_thrust_commands_n=np.full(4, hover_thrust_per_rotor_n / 2.0),
    )
    supported_state = reduced_discrete_step(
        touchdown_state,
        initial_support_control,
        schedule[1],
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        dynamics,
        actuator,
        dt_s,
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    transferred_support = np.zeros((4, 3))
    transferred_support[:2, 2] = (19.62 - float(np.sum(supported_state.rotor_thrusts_n))) / 2.0
    transferred_support_control = ReducedInput(
        contact_forces_world_n=transferred_support,
        rotor_thrust_commands_n=np.zeros(4),
    )
    settled_state = reduced_discrete_step(
        supported_state,
        transferred_support_control,
        schedule[2],
        FootLeverArmsFromComBody(feet[2], GO2_SDK_LEG_ORDER),
        dynamics,
        actuator,
        dt_s,
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    reference_states = (initial_state, touchdown_state, supported_state, settled_state)
    reference_controls = (
        flight_control,
        initial_support_control,
        transferred_support_control,
    )
    references = MPCReferences(
        position_world_m=np.stack([state.position_world_m for state in reference_states]),
        linear_velocity_world_m_per_s=np.stack(
            [state.linear_velocity_world_m_per_s for state in reference_states]
        ),
        rotation_body_to_world=np.stack(
            [state.rotation_body_to_world for state in reference_states]
        ),
        angular_velocity_body_rad_per_s=np.stack(
            [state.angular_velocity_body_rad_per_s for state in reference_states]
        ),
        contact_forces_world_n=np.stack(
            [control.contact_forces_world_n for control in reference_controls]
        ),
        rotor_thrust_commands_n=np.stack(
            [control.rotor_thrust_commands_n for control in reference_controls]
        ),
    )
    problem = ImpactAwareMPCProblem(
        initial_state=initial_state,
        previous_input=flight_control,
        dt_s=dt_s,
        contact_schedule=schedule,
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            feet,
            GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=references,
        state_bounds=StateBounds(
            lower=np.full((horizon + 1, STATE_DIM), -np.inf),
            upper=np.full((horizon + 1, STATE_DIM), np.inf),
            soft_mask=np.zeros((horizon, STATE_DIM), dtype=bool),
        ),
        contact_limits=ContactForceLimits(
            friction_coefficients=np.full(4, 0.8),
            maximum_normal_force_n=np.full(4, 30.0),
        ),
        impact_events=(event,),
        dynamics_config=dynamics,
        rotor_actuator_config=actuator,
        weights=MPCWeights(
            tracking=np.eye(12),
            input=0.1 * np.eye(CONTROL_DIM),
            input_rate=0.01 * np.eye(CONTROL_DIM),
            slack=np.eye(STATE_DIM),
            terminal_tracking=10.0 * np.eye(12),
            impulse=0.1 * np.eye(3),
            touchdown_velocity=np.eye(3),
        ),
        landing_contact_geometry=LandingContactGeometry(
            np.array([0.0, 0.0, 1.0]),
            0.09,
            1.0e-8,
            0.0,
            np.deg2rad(30.0),
        ),
    )
    nlp = ImpactAwareNLP(problem)
    automatic = nlp.unpack(nlp.initial_guess())
    warm_start = MPCWarmStart(
        states=automatic.states,
        controls=automatic.controls,
        slacks=automatic.slacks,
        impulses_by_step={1: np.zeros((4, 3))},
    )
    assert np.max(np.abs(nlp.equality_residual(nlp.pack(warm_start)))) > 0.4

    result = nlp.solve(
        SLSQPSettings(300, 1e-9, 1e-6, 8.0, False),
        warm_start=warm_start,
    )

    assert result.success, result.message
    assert result.raw_solver_status == 0
    assert result.iterations >= 2
    assert result.max_equality_violation < 1e-6
    assert result.min_inequality_residual >= -1e-6
    touchdown_velocity = foot_post_impact_velocity(
        result.states[1],
        FootLeverArmsFromComBody(feet[1], GO2_SDK_LEG_ORDER),
        np.zeros((4, 3, 3)),
        np.zeros((4, 3)),
        leg_order=GO2_SDK_LEG_ORDER,
    )
    assert touchdown_velocity[:2] == pytest.approx(np.zeros((2, 3)), abs=1e-6)
    assert evaluate_impulse_constraints(
        result.impulses_by_step[1],
        event.participation,
        event.impulse_limits,
    ).is_feasible(atol=1e-6)
    early_leg_load = result.controls[1].contact_forces_world_n[:2, 2]
    late_leg_load = result.controls[2].contact_forces_world_n[:2, 2]
    assert np.all(late_leg_load > early_leg_load)
    assert np.sum(result.states[3].rotor_thrusts_n) < np.sum(result.states[1].rotor_thrusts_n)
    assert np.sum(result.controls[2].rotor_thrust_commands_n) < np.sum(
        result.controls[0].rotor_thrust_commands_n
    )


def test_problem_rejects_joint_motion_incompatible_with_multi_foot_sticking() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
    event = ImpactEvent(
        step=1,
        touchdown=np.array([1, 1, 0, 0]),
        participation=np.array([1, 1, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=ImpactLimits(np.ones(4), 3.0, 0.1, 30.0),
    )
    problem = _problem(contact_schedule=schedule, impact_events=(event,))
    feet = np.zeros((2, 4, 3))
    # Keep both feet on the configured plane so geometry validation passes
    # and this test reaches the intended multi-foot velocity check.
    feet[:, 0] = [0.25, 0.0, 0.0]
    feet[:, 1] = [-0.25, 0.0, 0.0]
    jacobians = np.zeros((2, 4, 3, 3))
    jacobians[1, 0] = np.eye(3)
    joint_velocities = np.zeros((2, 4, 3))
    joint_velocities[1, 0, 0] = 0.1

    with pytest.raises(ValueError, match="joint velocities are incompatible"):
        replace(
            problem,
            foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
                feet,
                GO2_SDK_LEG_ORDER,
            ),
            leg_jacobians_body=jacobians,
            joint_velocities_rad_per_s=joint_velocities,
            impact_events=(
                replace(
                    event,
                    post_impact_joint_velocities_rad_per_s=joint_velocities[1],
                ),
            ),
        )


def test_impulse_friction_violation_is_exposed_as_negative_margin() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    event = _single_foot_impact_event()
    problem = _problem(
        gravity_z=0.0,
        initial_velocity=np.array([0.0, 0.0, -0.5]),
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(event,),
    )
    nlp = ImpactAwareNLP(problem)
    baseline = nlp.unpack(nlp.initial_guess())
    impulses = np.zeros((4, 3))
    impulses[0] = [1.0, 0.0, 1.0]
    violating = nlp.pack(
        MPCWarmStart(
            states=baseline.states,
            controls=baseline.controls,
            slacks=baseline.slacks,
            impulses_by_step={1: impulses},
        )
    )

    assert np.min(nlp.inequality_residual(violating)) == pytest.approx(-0.5)


def test_k0_input_delta_uses_previous_applied_input_and_warm_start_round_trips() -> None:
    zero_weights = MPCWeights(
        tracking=np.zeros((12, 12)),
        input=np.zeros((CONTROL_DIM, CONTROL_DIM)),
        input_rate=np.eye(CONTROL_DIM),
        slack=np.zeros((STATE_DIM, STATE_DIM)),
        terminal_tracking=np.zeros((12, 12)),
        impulse=np.zeros((3, 3)),
        touchdown_velocity=np.zeros((3, 3)),
    )
    problem = _problem(previous_command_per_rotor=5.905, weights=zero_weights)
    nlp = ImpactAwareNLP(problem)
    guess = nlp.initial_guess()

    assert nlp.objective(guess) == pytest.approx(4.0)
    warm_start = nlp.unpack(guess)
    assert np.array_equal(nlp.initial_guess(warm_start), guess)


def test_rotor_execution_plan_strictly_validates_calibrated_arrays() -> None:
    plan = RotorExecutionPlan(
        baseline_thrusts_n=np.full((1, 4), 10.0),
        correction_gains=np.array([0.05]),
        maximum_raw_correction_n=np.full(4, 4.0),
    )
    assert plan.horizon == 1
    assert not plan.baseline_thrusts_n.flags.writeable
    assert not plan.correction_gains.flags.writeable
    assert not plan.maximum_raw_correction_n.flags.writeable

    invalid_types = (
        (np.full((1, 4), True), np.array([0.05]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([True]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([0.05]), [4.0, 4.0, 4.0, "4"]),
        ([[10.0, 10.0, 10.0, True]], np.array([0.05]), np.full(4, 4.0)),
    )
    for baseline, gains, maximum_raw in invalid_types:
        with pytest.raises(TypeError, match="real numeric"):
            RotorExecutionPlan(baseline, gains, maximum_raw)

    invalid_values = (
        (np.full((1, 4), np.nan), np.array([0.05]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([np.inf]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([-0.01]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([1.01]), np.full(4, 4.0)),
        (np.full((1, 4), 10.0), np.array([0.05]), np.zeros(4)),
    )
    for baseline, gains, maximum_raw in invalid_values:
        with pytest.raises(ValueError):
            RotorExecutionPlan(baseline, gains, maximum_raw)

    with pytest.raises(ValueError, match="zero or at least"):
        RotorExecutionPlan(
            np.full((1, 4), 10.0),
            np.array([1.0e-12]),
            np.full(4, 4.0),
        )


def test_problem_validates_execution_horizon_and_baseline_actuator_bounds() -> None:
    two_stage_plan = RotorExecutionPlan(
        np.full((2, 4), 10.0),
        np.full(2, 0.05),
        np.full(4, 4.0),
    )
    with pytest.raises(ValueError, match="same horizon"):
        replace(_problem(), rotor_execution_plan=two_stage_plan)

    out_of_bounds_plan = RotorExecutionPlan(
        np.full((1, 4), 21.0),
        np.array([0.05]),
        np.full(4, 4.0),
    )
    with pytest.raises(ValueError, match="actuator bounds"):
        replace(_problem(), rotor_execution_plan=out_of_bounds_plan)
    with pytest.raises(TypeError, match="RotorExecutionPlan or None"):
        replace(_problem(), rotor_execution_plan="invalid")


def test_kappa_zero_fixes_applied_command_to_baseline_and_returns_no_target() -> None:
    pytest.importorskip("scipy")
    plan = RotorExecutionPlan(
        baseline_thrusts_n=np.full((1, 4), 10.0),
        correction_gains=np.zeros(1),
        maximum_raw_correction_n=np.full(4, 4.0),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_thrust_per_rotor=0.0,
        command_per_rotor=20.0,
        previous_command_per_rotor=10.0,
        rotor_execution_plan=plan,
    )
    nlp = ImpactAwareNLP(problem)
    guess = nlp.unpack(nlp.initial_guess())

    assert guess.controls[0].rotor_thrust_commands_n == pytest.approx(np.full(4, 10.0))
    exact_thrust = 10.0 * (
        1.0 - np.exp(-problem.dt_s / problem.rotor_actuator_config.time_constants_s)
    )
    assert guess.states[1].rotor_thrusts_n == pytest.approx(exact_thrust)
    assert nlp.rotor_execution_residual(nlp.pack(guess)) == pytest.approx(np.zeros(8))
    assert reconstruct_transport_target(plan, 0, np.full(4, 10.0)) is None

    result = nlp.solve(SLSQPSettings(50, 1e-10, 1e-7, 3.0, False))
    assert result.success, result.message
    assert result.first_input is not None
    assert result.first_input.rotor_thrust_commands_n == pytest.approx(np.full(4, 10.0))


def test_small_kappa_uses_applied_rollout_and_audits_raw_correction_envelope() -> None:
    plan = RotorExecutionPlan(
        baseline_thrusts_n=np.full((1, 4), 10.0),
        correction_gains=np.array([0.05]),
        maximum_raw_correction_n=np.full(4, 4.0),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_thrust_per_rotor=0.0,
        command_per_rotor=20.0,
        previous_command_per_rotor=10.0,
        rotor_execution_plan=plan,
    )
    nlp = ImpactAwareNLP(problem)
    guess = nlp.unpack(nlp.initial_guess())
    decision = nlp.pack(guess)

    # 10 + 0.05 * 4 = 10.2 N is the applied command and drives Eq. (31).
    assert guess.controls[0].rotor_thrust_commands_n == pytest.approx(np.full(4, 10.2))
    exact_thrust = 10.2 * (
        1.0 - np.exp(-problem.dt_s / problem.rotor_actuator_config.time_constants_s)
    )
    assert guess.states[1].rotor_thrusts_n == pytest.approx(exact_thrust)
    execution = nlp.rotor_execution_residual(decision)
    assert execution[:4] == pytest.approx(np.full(4, 0.4))
    assert execution[4:] == pytest.approx(np.zeros(4), abs=1e-12)
    assert nlp.inequality_residual(decision).size == (
        nlp._physical_inequality_residual(decision).size + 8
    )

    transport = reconstruct_transport_target(plan, 0, np.full(4, 10.2))
    assert transport is not None
    assert transport.target_thrusts_n == pytest.approx(np.full(4, 14.0))
    assert transport.is_gain_limited_reconstruction
    assert np.abs(transport.target_thrusts_n - plan.baseline_thrusts_n[0]) == pytest.approx(
        plan.maximum_raw_correction_n
    )

    violating_control = replace(
        guess.controls[0],
        rotor_thrust_commands_n=np.full(4, 10.3),
    )
    violating = nlp.pack(
        MPCWarmStart(
            states=guess.states,
            controls=(violating_control,),
            slacks=guess.slacks,
            impulses_by_step={},
        )
    )
    assert np.min(nlp.rotor_execution_residual(violating)) == pytest.approx(-0.1)
    assert np.min(nlp.inequality_residual(violating)) == pytest.approx(-0.1)
    assert np.min(nlp._physical_inequality_residual(violating)) >= 0.0
    with pytest.raises(ValueError, match="execution-plan envelope"):
        reconstruct_transport_target(plan, 0, np.full(4, 10.3))


def test_kappa_one_transport_target_equals_applied_and_intersects_actuator_bounds() -> None:
    plan = RotorExecutionPlan(
        baseline_thrusts_n=np.full((1, 4), 19.0),
        correction_gains=np.ones(1),
        maximum_raw_correction_n=np.full(4, 4.0),
    )
    problem = _problem(
        command_per_rotor=25.0,
        previous_command_per_rotor=19.0,
        rotor_execution_plan=plan,
    )
    nlp = ImpactAwareNLP(problem)
    guess = nlp.unpack(nlp.initial_guess())

    # The raw envelope reaches 23 N, but the actuator upper bound is 20 N.
    assert guess.controls[0].rotor_thrust_commands_n == pytest.approx(np.full(4, 20.0))
    transport = reconstruct_transport_target(plan, 0, np.full(4, 20.0))
    assert transport is not None
    assert transport.target_thrusts_n == pytest.approx(np.full(4, 20.0))
    assert not transport.is_gain_limited_reconstruction

    with pytest.raises(TypeError, match="step must be an integer"):
        reconstruct_transport_target(plan, True, np.full(4, 20.0))
    with pytest.raises(TypeError, match="real numeric"):
        reconstruct_transport_target(plan, 0, [20.0, 20.0, 20.0, False])


def test_timeout_returns_trajectory_and_explicit_failure_diagnostics() -> None:
    pytest.importorskip("scipy")
    nlp = ImpactAwareNLP(_problem())

    started = time.perf_counter()
    result = nlp.solve(
        SLSQPSettings(
            max_iterations=5,
            ftol=1e-8,
            constraint_tolerance=1e-6,
            timeout_s=0.05,
            display=False,
        )
    )
    elapsed = time.perf_counter() - started

    assert not result.success
    assert result.status == "timeout"
    assert len(result.states) == 2
    assert len(result.controls) == 1
    assert result.first_input is None
    assert result.diagnostic_first_candidate is not None
    assert "timeout" in result.message.lower()
    assert elapsed < 1.0


def test_problem_rejects_unmodeled_touchdown_in_known_contact_schedule() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    base = _problem()

    with pytest.raises(ValueError, match="without an ImpactEvent"):
        replace(base, contact_schedule=schedule)


def test_problem_requires_explicit_ground_and_monotonic_landing_schedule() -> None:
    event = _single_foot_impact_event()
    touchdown = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    with pytest.raises(ValueError, match="landing_contact_geometry is required"):
        replace(
            _problem(),
            contact_schedule=touchdown,
            impact_events=(event,),
            landing_contact_geometry=None,
        )

    with pytest.raises(ValueError, match="required for every impact-aware"):
        replace(_problem(), landing_contact_geometry=None)

    release = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="must be monotonic"):
        replace(_problem(), contact_schedule=release)

    with pytest.raises(ValueError, match="four unique"):
        replace(_problem(), foot_leg_order=("FR", "FR", "RR", "RL"))


def test_problem_requires_typed_com_horizon_and_matching_leg_order() -> None:
    base = _problem()
    raw = np.zeros((base.horizon + 1, 4, 3))

    with pytest.raises(TypeError, match="FootLeverArmsFromComBodyHorizon"):
        replace(base, foot_lever_arms_from_com_body_m=raw)

    with pytest.raises(TypeError, match="FootLeverArmsFromComBodyHorizon"):
        replace(
            base,
            foot_lever_arms_from_com_body_m=FootPositionsFromBodyOriginB(
                raw[0],
                GO2_SDK_LEG_ORDER,
            ),
        )

    with pytest.raises(ValueError, match="leg_order must exactly match"):
        replace(
            base,
            foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
                raw,
                ("FL", "FR", "RR", "RL"),
            ),
        )


def test_general_ground_plane_enforces_nonpenetration_touchdown_and_descent_guards() -> None:
    schedule = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    geometry = LandingContactGeometry(
        ground_normal_world=np.array([1.0, 0.0, 0.0]),
        ground_plane_offset_m=-0.05,
        touchdown_position_tolerance_m=1.0e-8,
        minimum_downward_speed_m_per_s=0.1,
        maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_velocity=np.array([-1.0, 0.0, 0.0]),
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=schedule,
        impact_events=(_single_foot_impact_event(),),
        landing_contact_geometry=geometry,
        initial_rotation_body_to_world=np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
    )
    nlp = ImpactAwareNLP(problem)
    decision = nlp.initial_guess()
    trajectories = nlp.unpack(decision)

    assert trajectories.impulses_by_step[1][0] == pytest.approx([2.0, 0.0, 0.0])
    assert np.min(nlp.inequality_residual(decision)) >= -1.0e-12

    too_far = ImpactAwareNLP(
        replace(
            problem,
            landing_contact_geometry=replace(geometry, ground_plane_offset_m=-0.10),
        )
    )
    assert np.min(too_far.inequality_residual(decision)) < -0.049

    too_slow = ImpactAwareNLP(
        replace(
            problem,
            landing_contact_geometry=replace(
                geometry,
                minimum_downward_speed_m_per_s=1.1,
            ),
        )
    )
    assert np.min(too_slow.inequality_residual(decision)) == pytest.approx(-0.1)

    with pytest.raises(ValueError, match="initial foot geometry penetrates"):
        replace(
            problem,
            landing_contact_geometry=replace(geometry, ground_plane_offset_m=0.01),
        )


def test_general_ground_normal_is_used_for_continuous_contact_force_cone() -> None:
    geometry = LandingContactGeometry(
        np.array([1.0, 0.0, 0.0]),
        0.0,
        0.01,
        0.0,
        np.deg2rad(30.0),
    )
    problem = _problem(
        gravity_z=0.0,
        initial_thrust_per_rotor=0.0,
        command_per_rotor=0.0,
        contact_schedule=np.ones((2, 4), dtype=int),
        landing_contact_geometry=geometry,
        initial_rotation_body_to_world=np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
    )
    nlp = ImpactAwareNLP(problem)
    trajectory = nlp.unpack(nlp.initial_guess())
    forces = np.zeros((4, 3))
    forces[0, 0] = 3.0
    aligned = nlp.pack(
        MPCWarmStart(
            trajectory.states,
            (
                ReducedInput(
                    contact_forces_world_n=forces,
                    rotor_thrust_commands_n=np.zeros(4),
                ),
            ),
            trajectory.slacks,
            {},
        )
    )
    assert np.min(nlp._physical_inequality_residual(aligned)) >= 0.0

    forces[0] = [0.0, 0.0, 3.0]
    tangential = np.array(aligned, copy=True)
    control_rows = tangential[nlp._layout.control_slice].reshape(1, CONTROL_DIM)
    control_rows[0, :12] = forces.reshape(12)
    assert np.min(nlp._physical_inequality_residual(tangential)) == pytest.approx(-3.0)

    separated_states = list(trajectory.states)
    separated_states[1] = replace(
        separated_states[1],
        position_world_m=np.array([0.1, 0.0, 0.0]),
    )
    separated = nlp.pack(
        MPCWarmStart(
            tuple(separated_states),
            trajectory.controls,
            trajectory.slacks,
            {},
        )
    )
    assert np.min(nlp._physical_inequality_residual(separated)) <= -0.09


def test_initial_scheduled_contact_cannot_start_above_ground_band() -> None:
    geometry = LandingContactGeometry(
        ground_normal_world=np.array([0.0, 0.0, 1.0]),
        ground_plane_offset_m=-0.1,
        touchdown_position_tolerance_m=0.01,
        minimum_downward_speed_m_per_s=0.0,
        maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
    )

    with pytest.raises(ValueError, match="initial scheduled contact"):
        _problem(
            contact_schedule=np.ones((2, 4), dtype=int),
            landing_contact_geometry=geometry,
        )


def test_so3_geodesic_tracking_error_does_not_vanish_at_pi() -> None:
    problem = _problem()
    rotations = np.array(problem.references.rotation_body_to_world, copy=True)
    rotations[0] = np.diag([1.0, -1.0, -1.0])
    references = replace(problem.references, rotation_body_to_world=rotations)

    error = nlp_module._tracking_error(problem.initial_state, references, 0)

    assert np.linalg.norm(error[6:9]) == pytest.approx(np.pi)
    assert np.linalg.norm(error) == pytest.approx(np.pi)


@pytest.mark.parametrize(
    "rotation_axis",
    (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    ),
)
@pytest.mark.parametrize("angle", (np.pi - 1.0e-7, np.pi + 1.0e-7))
def test_so3_geodesic_error_remains_nonzero_on_both_sides_of_pi(
    rotation_axis: np.ndarray,
    angle: float,
) -> None:
    problem = _problem()
    rotations = np.array(problem.references.rotation_body_to_world, copy=True)
    rotations[0] = nlp_module.so3_exp(rotation_axis * angle)
    references = replace(problem.references, rotation_body_to_world=rotations)

    error = nlp_module._tracking_error(problem.initial_state, references, 0)

    assert np.linalg.norm(error[6:9]) == pytest.approx(np.pi - 1.0e-7, abs=1.0e-9)


def test_landing_geometry_enforces_hard_tilt_cone() -> None:
    geometry = LandingContactGeometry(
        ground_normal_world=np.array([0.0, 0.0, 1.0]),
        ground_plane_offset_m=-1.0,
        touchdown_position_tolerance_m=0.01,
        minimum_downward_speed_m_per_s=0.0,
        maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
    )
    nlp = ImpactAwareNLP(replace(_problem(), landing_contact_geometry=geometry))
    decision = nlp.initial_guess()
    state_rows = decision[nlp._layout.state_slice].reshape(-1, STATE_DIM)
    state_rows[0, 6] = np.deg2rad(40.0)

    assert np.min(nlp._physical_inequality_residual(decision)) < 0.0


def test_landing_geometry_rejects_tilt_limit_at_ninety_degrees() -> None:
    with pytest.raises(ValueError, match="strictly below pi/2"):
        LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=0.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=0.5 * np.pi,
        )


def test_postsolve_audit_rejects_solver_success_that_violates_variable_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nlp = ImpactAwareNLP(_problem())

    def fake_solver(
        solver: ImpactAwareNLP,
        settings: SLSQPSettings,
        candidate: np.ndarray,
        deadline: float,
    ) -> object:
        del settings, deadline
        violating = np.array(candidate, copy=True)
        violating[solver._layout.slack_slice.start] = 1.0
        return nlp_module._SLSQPOutcome(
            "completed",
            violating,
            True,
            "simulated success",
            1,
            0,
        )

    monkeypatch.setattr(ImpactAwareNLP, "_solve_with_hard_timeout", fake_solver)
    result = nlp.solve(SLSQPSettings(5, 1.0e-8, 1.0e-8, 1.0, False))

    assert not result.success
    assert result.status == "constraint_violation"
    assert result.min_inequality_residual >= 0.0
    assert result.min_variable_bound_residual == pytest.approx(-1.0)
    assert "min(bounds)" in result.message


def test_postsolve_elapsed_time_overrun_cannot_be_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nlp = ImpactAwareNLP(_problem())

    def delayed_solver(
        solver: ImpactAwareNLP,
        settings: SLSQPSettings,
        candidate: np.ndarray,
        deadline: float,
    ) -> object:
        del solver, settings, deadline
        time.sleep(0.08)
        return nlp_module._SLSQPOutcome(
            "completed",
            np.array(candidate, copy=True),
            True,
            "simulated late feasible result",
            1,
            0,
        )

    monkeypatch.setattr(ImpactAwareNLP, "_solve_with_hard_timeout", delayed_solver)
    result = nlp.solve(SLSQPSettings(5, 1.0e-8, 1.0e-8, 0.05, False))

    assert not result.success
    assert result.status == "timeout"
    assert result.solve_time_s >= 0.05
    assert result.first_input is None


@pytest.mark.parametrize("failure_kind", ("timeout", "termination_failure"))
def test_isolated_solver_failure_never_runs_unbounded_parent_audit(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    nlp = ImpactAwareNLP(_problem())

    def failed_solver(
        solver: ImpactAwareNLP,
        settings: SLSQPSettings,
        candidate: np.ndarray,
        deadline: float,
    ) -> object:
        del solver, settings, deadline
        return nlp_module._SLSQPOutcome(
            failure_kind,
            np.array(candidate, copy=True),
            False,
            "synthetic isolated-worker failure",
            0,
            None,
        )

    def forbidden_post_audit(decision: object) -> object:
        del decision
        raise AssertionError("post-timeout objective/constraint audit must not run")

    monkeypatch.setattr(ImpactAwareNLP, "_solve_with_hard_timeout", failed_solver)
    monkeypatch.setattr(nlp, "objective", forbidden_post_audit)
    monkeypatch.setattr(nlp, "equality_residual", forbidden_post_audit)
    monkeypatch.setattr(nlp, "inequality_residual", forbidden_post_audit)
    monkeypatch.setattr(nlp, "variable_bound_residual", forbidden_post_audit)

    result = nlp.solve(SLSQPSettings(5, 1.0e-8, 1.0e-8, 0.05, False))

    assert not result.success
    assert result.status == failure_kind
    assert result.first_input is None
    assert result.diagnostic_first_candidate is not None
    assert result.max_equality_violation == np.inf
    assert result.min_inequality_residual == -np.inf
    assert result.min_variable_bound_residual == -np.inf
