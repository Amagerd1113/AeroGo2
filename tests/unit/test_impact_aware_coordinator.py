from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pytest

from aerogo2.common.enums import SystemState
from aerogo2.landing.impact_aware.admittance import (
    AdmittanceConfig,
    AxisAlignedWorkspace,
    LegAdmittanceController,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetectorConfig,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.coordinator import (
    ImpactAwareLandingCoordinator,
    LandingCycleInput,
    LandingInputFreshness,
)
from aerogo2.landing.impact_aware.integration import ImpactLandingPhase
from aerogo2.landing.impact_aware.nlp import (
    CONTROL_DIM,
    STATE_DIM,
    ContactForceLimits,
    ImpactAwareMPCProblem,
    LandingContactGeometry,
    MPCReferences,
    MPCWeights,
    RotorExecutionPlan,
    SLSQPSettings,
    StateBounds,
)
from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FootLeverArmsFromComBodyHorizon,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
)


def _problem() -> ImpactAwareMPCProblem:
    horizon = 1
    thrust = 4.905
    state = ReducedState(
        position_world_m=np.zeros(3),
        linear_velocity_world_m_per_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=np.full(4, thrust),
    )
    previous = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.full(4, thrust),
    )
    allocation = np.zeros((6, 4))
    allocation[2, :] = 1.0
    return ImpactAwareMPCProblem(
        initial_state=state,
        previous_input=previous,
        dt_s=1.0,
        contact_schedule=np.zeros((horizon + 1, 4), dtype=int),
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            np.zeros((horizon + 1, 4, 3)),
            GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=MPCReferences(
            position_world_m=np.zeros((horizon + 1, 3)),
            linear_velocity_world_m_per_s=np.zeros((horizon + 1, 3)),
            rotation_body_to_world=np.repeat(np.eye(3)[None, :, :], horizon + 1, axis=0),
            angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
            contact_forces_world_n=np.zeros((horizon, 4, 3)),
            rotor_thrust_commands_n=np.full((horizon, 4), thrust),
        ),
        state_bounds=StateBounds(
            lower=np.full((horizon + 1, STATE_DIM), -np.inf),
            upper=np.full((horizon + 1, STATE_DIM), np.inf),
            soft_mask=np.zeros((horizon, STATE_DIM), dtype=bool),
        ),
        contact_limits=ContactForceLimits(
            friction_coefficients=np.full(4, 0.5),
            maximum_normal_force_n=np.full(4, 100.0),
        ),
        impact_events=(),
        dynamics_config=ReducedDynamicsConfig(
            mass_kg=2.0,
            inertia_body_kg_m2=np.eye(3),
            gravity_world_m_per_s2=np.array([0.0, 0.0, -9.81]),
            rotor_allocation_body=allocation,
        ),
        rotor_actuator_config=RotorActuatorConfig(
            time_constants_s=np.full(4, 0.2),
            thrust_min_n=np.zeros(4),
            thrust_max_n=np.full(4, 20.0),
            thrust_rate_min_n_per_s=np.full(4, -100.0),
            thrust_rate_max_n_per_s=np.full(4, 100.0),
        ),
        weights=MPCWeights(
            tracking=np.eye(12),
            input=np.eye(CONTROL_DIM),
            input_rate=np.eye(CONTROL_DIM),
            slack=np.eye(STATE_DIM),
            terminal_tracking=np.eye(12),
            impulse=np.eye(3),
            touchdown_velocity=np.eye(3),
        ),
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=-1.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
    )


def _coordinator(
    *,
    monotonic_time_s: float = 1.01,
    target_gain: float = 0.25,
    monotonic_clock: Optional[Callable[[], float]] = None,
    total_com_C_from_go2_body_origin_B_body_m: object = (0.0, 0.0, 0.0),
) -> ImpactAwareLandingCoordinator:
    admittance = AdmittanceConfig(
        transition_duration_s=0.2,
        touchdown_inertia=np.ones(3),
        stance_inertia=np.ones(3),
        touchdown_damping=np.ones(3),
        stance_damping=np.ones(3),
        restoring_stiffness=np.ones(3),
        joint_lower=np.full(3, -2.0),
        joint_upper=np.full(3, 2.0),
        joint_rate_limit=np.full(3, 10.0),
    )
    legs = tuple(
        LegAdmittanceController(
            admittance,
            lambda foot: foot,
            AxisAlignedWorkspace(np.full(3, -1.0), np.full(3, 1.0)),
            np.zeros(3),
        )
        for _ in range(4)
    )
    return ImpactAwareLandingCoordinator(
        solver_settings=SLSQPSettings(
            max_iterations=20,
            ftol=1e-10,
            constraint_tolerance=1e-8,
            timeout_s=0.8,
        ),
        contact_detector=FootContactDetector(
            ContactDetectorConfig(
                contact_on_threshold_n=np.full(4, 20.0),
                contact_off_threshold_n=np.full(4, 10.0),
                filter_time_constant_s=0.01,
                contact_confirm_s=0.05,
                release_confirm_s=0.05,
            )
        ),
        leg_controllers=legs,
        rotor_blender=RotorCorrectionBlender(
            RotorCorrectionSafetyConfig(
                target_gain=target_gain,
                thrust_min_n=np.zeros(4),
                thrust_max_n=np.full(4, 20.0),
                maximum_correction_n=np.full(4, 5.0),
                maximum_gain_rise_per_s=10.0,
            )
        ),
        fixed_rotor_allocation_body=_problem().dynamics_config.rotor_allocation_body,
        total_com_C_from_go2_body_origin_B_body_m=(total_com_C_from_go2_body_origin_B_body_m),
        monotonic_clock=(
            monotonic_clock if monotonic_clock is not None else lambda: monotonic_time_s
        ),
    )


def _cycle(state: SystemState, timestamp_s: float = 1.0) -> LandingCycleInput:
    return LandingCycleInput(
        sequence=7,
        timestamp_s=timestamp_s,
        dt_s=1.0,
        command_ttl_s=1.0,
        system_state=state,
        freshness=LandingInputFreshness(
            state_estimate_timestamp_s=timestamp_s,
            contact_forces_timestamp_s=timestamp_s,
            kinematics_timestamp_s=timestamp_s,
            foot_plan_timestamp_s=timestamp_s,
            flight_controller_baseline_timestamp_s=timestamp_s,
            maximum_source_age_s=0.5,
            all_sources_healthy=True,
        ),
        problem=_problem(),
        flight_controller_session_id=11,
        flight_controller_target_tick=102,
        flight_controller_baseline_version=7,
        flight_controller_baseline_thrusts_n=np.full(4, 4.905),
        flight_controller_baseline_prediction_thrusts_n=np.full((1, 4), 4.905),
        measured_normal_forces_n=np.zeros(4),
        estimated_contact_forces_world_n=np.zeros((4, 3)),
        nominal_foot_positions_world_m=np.zeros((4, 3)),
    )


def test_successful_cycle_builds_coherent_leg_and_fc_residual_bundle() -> None:
    pytest.importorskip("scipy")
    result = _coordinator().compute(_cycle(SystemState.AUTO_LANDING))

    assert result.success, result.message
    assert result.phase is ImpactLandingPhase.PRE_TOUCHDOWN
    assert result.command is not None
    assert result.command.leg.sequence == result.command.rotor.sequence == 7
    assert result.command.rotor.fc_session_id == 11
    assert result.command.rotor.target_fc_tick == 102
    assert result.command.rotor.baseline_version == 7
    assert result.command.rotor.baseline_timestamp_s == 1.0
    assert len(result.command.leg.joint_positions_rad) == 12
    assert result.command.rotor.applied_residual_thrusts_n == pytest.approx((0.0,) * 4)
    assert result.command.rotor.applied_total_thrusts_n == pytest.approx((4.905,) * 4)
    assert len(result.leg_outputs) == 4


def test_admittance_uses_go2_body_origin_B_not_total_com_C() -> None:
    pytest.importorskip("scipy")
    offset = np.array([0.0, 0.0, 0.05])
    result = _coordinator(
        total_com_C_from_go2_body_origin_B_body_m=offset,
    ).compute(_cycle(SystemState.AUTO_LANDING))

    assert result.success, result.message
    assert len(result.leg_outputs) == 4
    for output in result.leg_outputs:
        # The synthetic cycle places C and each nominal foot at world zero;
        # therefore B=-offset and the B->foot nominal vector is +offset.
        assert output.nominal_foot_position_body == pytest.approx(offset)


def test_inactive_state_withholds_every_command_without_running_measurement_pipeline() -> None:
    result = _coordinator().compute(_cycle(SystemState.FLIGHT_MANUAL))

    assert not result.success
    assert result.status == "inactive_state"
    assert result.phase is ImpactLandingPhase.INACTIVE
    assert result.command is None
    assert result.solver_result is None
    assert result.contact_detection is None
    assert result.rotor_output is not None
    assert result.rotor_output.applied_gain == 0.0


def test_nonincreasing_contact_timestamp_fails_closed_and_removes_residual() -> None:
    pytest.importorskip("scipy")
    coordinator = _coordinator(monotonic_time_s=2.01)
    assert coordinator.compute(_cycle(SystemState.AUTO_LANDING, 2.0)).success

    result = coordinator.compute(_cycle(SystemState.AUTO_LANDING, 2.0))

    assert not result.success
    assert result.status == "invalid_contact_measurement"
    assert result.command is None
    assert result.rotor_output is not None
    assert result.rotor_output.applied_gain == 0.0
    assert result.rotor_output.applied_total_thrusts_n == pytest.approx((4.905,) * 4)

    blocked = coordinator.compute(_cycle(SystemState.AUTO_LANDING, 2.0))
    assert blocked.status == "rotor_correction_fault_latched"
    assert blocked.command is None


def test_stale_source_is_rejected_before_contact_or_solver_state_changes() -> None:
    cycle = _cycle(SystemState.AUTO_LANDING)
    stale = LandingCycleInput(
        sequence=cycle.sequence,
        timestamp_s=cycle.timestamp_s,
        dt_s=cycle.dt_s,
        command_ttl_s=cycle.command_ttl_s,
        system_state=cycle.system_state,
        freshness=LandingInputFreshness(
            state_estimate_timestamp_s=0.0,
            contact_forces_timestamp_s=cycle.timestamp_s,
            kinematics_timestamp_s=cycle.timestamp_s,
            foot_plan_timestamp_s=cycle.timestamp_s,
            flight_controller_baseline_timestamp_s=cycle.timestamp_s,
            maximum_source_age_s=0.5,
            all_sources_healthy=True,
        ),
        problem=cycle.problem,
        flight_controller_session_id=cycle.flight_controller_session_id,
        flight_controller_target_tick=cycle.flight_controller_target_tick,
        flight_controller_baseline_version=cycle.flight_controller_baseline_version,
        flight_controller_baseline_thrusts_n=cycle.flight_controller_baseline_thrusts_n,
        flight_controller_baseline_prediction_thrusts_n=(
            cycle.flight_controller_baseline_prediction_thrusts_n
        ),
        measured_normal_forces_n=cycle.measured_normal_forces_n,
        estimated_contact_forces_world_n=cycle.estimated_contact_forces_world_n,
        nominal_foot_positions_world_m=cycle.nominal_foot_positions_world_m,
    )

    result = _coordinator().compute(stale)

    assert not result.success
    assert result.status == "input_source_unhealthy"
    assert "stale" in result.message
    assert result.command is None


def test_successful_solver_result_is_withheld_after_command_deadline() -> None:
    pytest.importorskip("scipy")
    times = iter((1.01, 100.0))
    result = _coordinator(monotonic_clock=lambda: next(times)).compute(
        _cycle(SystemState.AUTO_LANDING)
    )

    assert not result.success
    assert result.status == "command_deadline_missed"
    assert result.solver_result is not None
    assert result.solver_result.success


def test_serial_reference_uses_contact_source_time_not_snapshot_time() -> None:
    pytest.importorskip("scipy")
    cycle = _cycle(SystemState.AUTO_LANDING)
    source_time = 0.99
    cycle = replace(
        cycle,
        freshness=replace(
            cycle.freshness,
            contact_forces_timestamp_s=source_time,
        ),
    )

    result = _coordinator().compute(cycle)

    assert result.success, result.message
    assert result.contact_detection is not None
    assert result.contact_detection.timestamp_s == source_time


def test_serial_reference_rechecks_source_age_after_solver_completion() -> None:
    pytest.importorskip("scipy")
    times = iter((1.01, 1.6))

    result = _coordinator(monotonic_clock=lambda: next(times)).compute(
        _cycle(SystemState.AUTO_LANDING)
    )

    assert not result.success
    assert result.status == "input_source_became_stale"
    assert result.command is None
    assert result.solver_result is not None and result.solver_result.success
    assert result.command is None
    assert result.rotor_output is not None
    assert result.rotor_output.applied_gain == 0.0


def test_confirmed_contact_must_match_the_current_mpc_hybrid_mode() -> None:
    pytest.importorskip("scipy")
    now = [3.01]
    coordinator = _coordinator(monotonic_clock=lambda: now[0])
    first = replace(
        _cycle(SystemState.AUTO_LANDING, 3.0),
        measured_normal_forces_n=np.full(4, 30.0),
    )
    assert coordinator.compute(first).success
    now[0] = 3.07
    second = replace(
        _cycle(SystemState.AUTO_LANDING, 3.06),
        measured_normal_forces_n=np.full(4, 30.0),
    )

    result = coordinator.compute(second)

    assert not result.success
    assert result.status == "contact_schedule_mismatch"
    assert result.phase is ImpactLandingPhase.TOUCHDOWN
    assert result.contact_detection is not None
    assert result.contact_detection.contacts == (True, True, True, True)
    assert result.command is None
    assert result.rotor_output is not None
    assert result.rotor_output.applied_gain == 0.0


def test_coordinator_executes_the_mpc_applied_command_without_second_gain() -> None:
    pytest.importorskip("scipy")
    base = _problem()
    input_diagonal = np.zeros(CONTROL_DIM)
    input_diagonal[12:] = 100.0
    problem = replace(
        base,
        references=replace(
            base.references,
            rotor_thrust_commands_n=np.full((1, 4), 6.0),
        ),
        weights=MPCWeights(
            tracking=np.zeros((12, 12)),
            input=np.diag(input_diagonal),
            input_rate=np.zeros((CONTROL_DIM, CONTROL_DIM)),
            slack=np.zeros((STATE_DIM, STATE_DIM)),
            terminal_tracking=np.zeros((12, 12)),
            impulse=np.zeros((3, 3)),
            touchdown_velocity=np.zeros((3, 3)),
        ),
    )
    cycle = replace(_cycle(SystemState.AUTO_LANDING), problem=problem)

    result = _coordinator(target_gain=0.25).compute(cycle)

    assert result.success, result.message
    assert result.solver_result is not None
    assert result.solver_result.first_input is not None
    assert result.command is not None
    applied = result.solver_result.first_input.rotor_thrust_commands_n
    baseline = np.asarray(result.command.rotor.baseline_thrusts_n)
    payload = np.asarray(result.command.rotor.applied_residual_thrusts_n)
    assert np.asarray(result.command.rotor.applied_total_thrusts_n) == pytest.approx(applied)
    assert payload == pytest.approx(applied - baseline)
    assert result.command.rotor.correction_gain == pytest.approx(0.25)
    assert result.command.rotor.transport_raw_residual_thrusts_n is not None
    assert np.asarray(result.command.rotor.transport_raw_residual_thrusts_n) == pytest.approx(
        payload / 0.25
    )
    assert result.command.rotor.transport_target_semantics == (
        "gain_limited_algebraic_reconstruction"
    )


def test_zero_gain_models_and_sends_only_the_fc_baseline() -> None:
    pytest.importorskip("scipy")
    result = _coordinator(target_gain=0.0).compute(_cycle(SystemState.AUTO_LANDING))

    assert result.success, result.message
    assert result.command is not None
    assert result.command.rotor.applied_total_thrusts_n == pytest.approx((4.905,) * 4)
    assert result.command.rotor.applied_residual_thrusts_n == pytest.approx((0.0,) * 4)
    assert result.command.rotor.transport_raw_residual_thrusts_n is None
    assert result.command.rotor.transport_target_semantics == "zero_gain_no_transport_target"


def test_live_execution_plan_mismatch_is_rejected_before_solve() -> None:
    supplied = RotorExecutionPlan(
        baseline_thrusts_n=np.full((1, 4), 4.905),
        correction_gains=np.array([0.9]),
        maximum_raw_correction_n=np.full(4, 5.0),
    )
    cycle = _cycle(SystemState.AUTO_LANDING)
    cycle = replace(
        cycle,
        problem=replace(cycle.problem, rotor_execution_plan=supplied),
    )

    result = _coordinator(target_gain=0.25).compute(cycle)

    assert not result.success
    assert result.status == "rotor_execution_plan_invalid"
    assert result.solver_result is None
    assert result.command is None


def test_cycle_timing_contract_rejects_coercion_and_multi_cycle_ttl() -> None:
    with pytest.raises(TypeError, match="real number"):
        LandingInputFreshness(
            state_estimate_timestamp_s=True,  # type: ignore[arg-type]
            contact_forces_timestamp_s=1.0,
            kinematics_timestamp_s=1.0,
            foot_plan_timestamp_s=1.0,
            flight_controller_baseline_timestamp_s=1.0,
            maximum_source_age_s=0.1,
            all_sources_healthy=True,
        )

    with pytest.raises(ValueError, match="cannot exceed one MPC cycle"):
        replace(_cycle(SystemState.AUTO_LANDING), command_ttl_s=1.01)


def test_actual_mpc_geometry_and_nominal_admittance_plan_may_differ() -> None:
    cycle = _cycle(SystemState.AUTO_LANDING)
    nominal = np.asarray(cycle.nominal_foot_positions_world_m).copy()
    nominal[0, 0] = 0.01
    distinct = replace(cycle, nominal_foot_positions_world_m=nominal)
    assert distinct.nominal_foot_positions_world_m[0, 0] == pytest.approx(0.01)
    assert distinct.problem.foot_lever_arms_from_com_body_m.values_m[0, 0, 0] == pytest.approx(
        0.0
    )

    stale_plan = replace(
        distinct.freshness,
        foot_plan_timestamp_s=distinct.timestamp_s - 0.6,
    )
    result = _coordinator().compute(replace(distinct, freshness=stale_plan))
    assert result.status == "input_source_unhealthy"
    assert "foot_plan_timestamp_s is stale" in result.message


def test_rotor_allocation_is_pinned_for_the_entire_landing_session() -> None:
    cycle = _cycle(SystemState.AUTO_LANDING)
    changed_allocation = np.asarray(cycle.problem.dynamics_config.rotor_allocation_body).copy()
    changed_allocation[3, 0] = 0.01
    changed_dynamics = replace(
        cycle.problem.dynamics_config,
        rotor_allocation_body=changed_allocation,
    )
    changed_cycle = replace(
        cycle,
        problem=replace(cycle.problem, dynamics_config=changed_dynamics),
    )
    coordinator = _coordinator()

    rejected = coordinator.compute(changed_cycle)

    assert not rejected.success
    assert rejected.status == "static_rotor_geometry_mismatch"
    assert rejected.command is None
    assert rejected.rotor_output is not None
    assert rejected.rotor_output.applied_gain == 0.0

    coordinator.reset()
    rejected_after_reset = coordinator.compute(changed_cycle)
    assert rejected_after_reset.status == "static_rotor_geometry_mismatch"


def test_touchdown_replan_preserves_detector_state_and_can_continue() -> None:
    pytest.importorskip("scipy")
    now = [3.01]
    coordinator = _coordinator(monotonic_clock=lambda: now[0])
    first = replace(
        _cycle(SystemState.AUTO_LANDING, 3.0),
        measured_normal_forces_n=np.full(4, 30.0),
    )
    assert coordinator.compute(first).success

    now[0] = 3.07
    touchdown = replace(
        _cycle(SystemState.AUTO_LANDING, 3.06),
        measured_normal_forces_n=np.full(4, 30.0),
    )
    replan = coordinator.compute(touchdown)
    assert replan.status == "contact_schedule_mismatch"
    assert replan.contact_detection is not None
    assert all(replan.contact_detection.contacts)

    contact_problem = replace(
        _problem(),
        contact_schedule=np.ones((2, 4), dtype=int),
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=0.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
    )
    now[0] = 3.13
    replanned_cycle = replace(
        _cycle(SystemState.AUTO_LANDING, 3.12),
        problem=contact_problem,
        measured_normal_forces_n=np.full(4, 30.0),
    )
    continued = coordinator.compute(replanned_cycle)

    assert continued.success, continued.message
    assert continued.phase is ImpactLandingPhase.POST_TOUCHDOWN_RECOVERY


def test_unplanned_post_touchdown_release_and_prior_faults_remain_latched() -> None:
    pytest.importorskip("scipy")
    now = [4.01]
    coordinator = _coordinator(monotonic_clock=lambda: now[0])
    high = replace(
        _cycle(SystemState.AUTO_LANDING, 4.0),
        measured_normal_forces_n=np.full(4, 30.0),
    )
    assert coordinator.compute(high).success
    now[0] = 4.07
    high_confirm = replace(
        high,
        timestamp_s=4.06,
        freshness=replace(
            high.freshness,
            state_estimate_timestamp_s=4.06,
            contact_forces_timestamp_s=4.06,
            kinematics_timestamp_s=4.06,
            foot_plan_timestamp_s=4.06,
            flight_controller_baseline_timestamp_s=4.06,
        ),
    )
    assert coordinator.compute(high_confirm).status == "contact_schedule_mismatch"

    contact_problem = replace(
        _problem(),
        contact_schedule=np.ones((2, 4), dtype=int),
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=0.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
    )
    now[0] = 4.31
    first_low = replace(
        _cycle(SystemState.AUTO_LANDING, 4.30),
        problem=contact_problem,
        measured_normal_forces_n=np.zeros(4),
    )
    assert coordinator.compute(first_low).success
    now[0] = 4.37
    second_low = replace(
        _cycle(SystemState.AUTO_LANDING, 4.36),
        problem=contact_problem,
        measured_normal_forces_n=np.zeros(4),
    )
    lost = coordinator.compute(second_low)
    assert lost.status == "contact_loss_latched"

    now[0] = 4.43
    again = coordinator.compute(
        replace(_cycle(SystemState.AUTO_LANDING, 4.42), problem=contact_problem)
    )
    assert again.status == "contact_loss_latched"

    coordinator.reset()
    now[0] = 4.51
    assert coordinator.compute(_cycle(SystemState.AUTO_LANDING, 4.50)).success
