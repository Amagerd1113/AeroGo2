from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.enums import SystemState
from aerogo2.landing.impact_aware.integration import (
    CoordinatedLandingCommand,
    FlightControllerRotorResidualCommand,
    Go2JointPositionCommand,
    ImpactLandingPhase,
    UnavailableFlightControllerResidualSink,
    UnavailableGo2LowLevelSink,
    phase_for_system_state,
)


def _leg(sequence: int = 7) -> Go2JointPositionCommand:
    return Go2JointPositionCommand(
        sequence=sequence,
        timestamp_s=1.0,
        valid_until_s=1.05,
        joint_positions_rad=(0.0,) * 12,
        desired_contact_forces_world_n=(0.0,) * 12,
    )


def _rotor(sequence: int = 7) -> FlightControllerRotorResidualCommand:
    return FlightControllerRotorResidualCommand(
        sequence=sequence,
        timestamp_s=1.0,
        valid_until_s=1.05,
        fc_session_id=11,
        target_fc_tick=102,
        baseline_version=7,
        baseline_timestamp_s=1.0,
        baseline_thrusts_n=(4.0,) * 4,
        transport_raw_residual_thrusts_n=(1.0,) * 4,
        applied_residual_thrusts_n=(0.1,) * 4,
        applied_total_thrusts_n=(4.1,) * 4,
        correction_gain=0.1,
        transport_target_semantics="gain_limited_algebraic_reconstruction",
    )


def test_fsm_maps_to_paper_landing_phases() -> None:
    assert phase_for_system_state(SystemState.AUTO_LANDING) is ImpactLandingPhase.PRE_TOUCHDOWN
    assert phase_for_system_state(SystemState.TOUCHDOWN_VERIFY) is ImpactLandingPhase.INACTIVE
    assert phase_for_system_state(SystemState.LANDING_COMPLIANT) is ImpactLandingPhase.INACTIVE
    assert phase_for_system_state(SystemState.FLIGHT_MANUAL) is ImpactLandingPhase.INACTIVE


def test_coherent_bundle_requires_shared_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        CoordinatedLandingCommand(
            phase=ImpactLandingPhase.PRE_TOUCHDOWN,
            leg=_leg(sequence=1),
            rotor=_rotor(sequence=2),
            solver_succeeded=True,
            solver_status="ok",
            solver_time_s=0.001,
        )


def test_coherent_bundle_requires_a_typed_active_phase() -> None:
    with pytest.raises(TypeError, match="ImpactLandingPhase"):
        CoordinatedLandingCommand(
            phase="not-a-phase",  # type: ignore[arg-type]
            leg=_leg(),
            rotor=_rotor(),
            solver_succeeded=True,
            solver_status="ok",
            solver_time_s=0.001,
        )

    with pytest.raises(TypeError, match="Go2JointPositionCommand"):
        CoordinatedLandingCommand(
            phase=ImpactLandingPhase.PRE_TOUCHDOWN,
            leg="not-a-leg-command",  # type: ignore[arg-type]
            rotor=_rotor(),
            solver_succeeded=True,
            solver_status="ok",
            solver_time_s=0.001,
        )

    with pytest.raises(ValueError, match="solver_status"):
        CoordinatedLandingCommand(
            phase=ImpactLandingPhase.PRE_TOUCHDOWN,
            leg=_leg(),
            rotor=_rotor(),
            solver_succeeded=True,
            solver_status=1,  # type: ignore[arg-type]
            solver_time_s=0.001,
        )


def test_rotor_residual_payload_enforces_the_gain_and_sum_identities() -> None:
    with pytest.raises(ValueError, match="correction_gain"):
        FlightControllerRotorResidualCommand(
            sequence=7,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=(1.0,) * 4,
            applied_residual_thrusts_n=(0.2,) * 4,
            applied_total_thrusts_n=(4.2,) * 4,
            correction_gain=0.1,
            transport_target_semantics="gain_limited_algebraic_reconstruction",
        )

    with pytest.raises(ValueError, match="baseline_thrusts_n"):
        FlightControllerRotorResidualCommand(
            sequence=7,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=(1.0,) * 4,
            applied_residual_thrusts_n=(0.1,) * 4,
            applied_total_thrusts_n=(4.2,) * 4,
            correction_gain=0.1,
            transport_target_semantics="gain_limited_algebraic_reconstruction",
        )


def test_zero_gain_payload_has_no_invented_full_gain_target() -> None:
    command = FlightControllerRotorResidualCommand(
        sequence=7,
        timestamp_s=1.0,
        valid_until_s=1.05,
        fc_session_id=11,
        target_fc_tick=102,
        baseline_version=7,
        baseline_timestamp_s=1.0,
        baseline_thrusts_n=(4.0,) * 4,
        transport_raw_residual_thrusts_n=None,
        applied_residual_thrusts_n=(0.0,) * 4,
        applied_total_thrusts_n=(4.0,) * 4,
        correction_gain=0.0,
        transport_target_semantics="zero_gain_no_transport_target",
    )
    assert command.transport_raw_residual_thrusts_n is None

    with pytest.raises(ValueError, match="positive correction_gain"):
        FlightControllerRotorResidualCommand(
            sequence=7,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=None,
            applied_residual_thrusts_n=(0.0,) * 4,
            applied_total_thrusts_n=(4.0,) * 4,
            correction_gain=0.1,
            transport_target_semantics="invalid",
        )

    with pytest.raises(ValueError, match="zero correction_gain"):
        FlightControllerRotorResidualCommand(
            sequence=7,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=(0.0,) * 4,
            applied_residual_thrusts_n=(0.0,) * 4,
            applied_total_thrusts_n=(4.0,) * 4,
            correction_gain=0.0,
            transport_target_semantics="zero_gain_no_transport_target",
        )


def test_tiny_positive_and_near_one_gains_keep_exact_transport_modes() -> None:
    tiny = 5.0e-10
    command = FlightControllerRotorResidualCommand(
        sequence=8,
        timestamp_s=1.0,
        valid_until_s=1.05,
        fc_session_id=11,
        target_fc_tick=102,
        baseline_version=7,
        baseline_timestamp_s=1.0,
        baseline_thrusts_n=(4.0,) * 4,
        transport_raw_residual_thrusts_n=(1.0,) * 4,
        applied_residual_thrusts_n=(tiny,) * 4,
        applied_total_thrusts_n=(4.0 + tiny,) * 4,
        correction_gain=tiny,
        transport_target_semantics="gain_limited_algebraic_reconstruction",
    )
    assert command.transport_raw_residual_thrusts_n == (1.0,) * 4

    near_one = 0.99999999995
    with pytest.raises(ValueError, match="exactly match"):
        FlightControllerRotorResidualCommand(
            sequence=9,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=(1.0,) * 4,
            applied_residual_thrusts_n=(near_one,) * 4,
            applied_total_thrusts_n=(4.0 + near_one,) * 4,
            correction_gain=near_one,
            transport_target_semantics="active_gain_one_transport_target",
        )


def test_active_bundle_rejects_an_unhealthy_solver_flag() -> None:
    with pytest.raises(ValueError, match="solver_succeeded"):
        CoordinatedLandingCommand(
            phase=ImpactLandingPhase.PRE_TOUCHDOWN,
            leg=_leg(),
            rotor=_rotor(),
            solver_succeeded=False,
            solver_status="failed",
            solver_time_s=0.001,
        )


def test_commands_have_explicit_ttl() -> None:
    assert _leg().is_fresh(1.03)
    assert not _leg().is_fresh(1.05)
    assert not _leg().is_fresh(1.06)
    assert _rotor().is_fresh(1.03)
    assert not _rotor().is_fresh(1.05)


def test_rotor_command_freezes_fc_identity_units_and_order() -> None:
    command = _rotor()
    assert command.fc_session_id == 11
    assert command.target_fc_tick == 102
    assert command.baseline_version == 7
    assert command.thrust_unit == "N"
    assert command.rotor_order == ("RR", "LF", "LR", "RF")


def test_rotor_command_rejects_future_baseline_and_uint64_overflow() -> None:
    with pytest.raises(ValueError, match="baseline_timestamp_s"):
        replace(_rotor(), baseline_timestamp_s=1.01)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        replace(_rotor(), sequence=1 << 64)


@pytest.mark.parametrize("sequence", [False, 1.5])
def test_command_sequences_are_strict_integers(sequence: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        Go2JointPositionCommand(
            sequence=sequence,  # type: ignore[arg-type]
            timestamp_s=1.0,
            valid_until_s=1.05,
            joint_positions_rad=(0.0,) * 12,
            desired_contact_forces_world_n=(0.0,) * 12,
        )


def test_lowcmd_frame_keeps_slow_policy_identity_separate() -> None:
    command = replace(
        _leg(sequence=101),
        source_policy_sequence=7,
        source_policy_generation=3,
        source_contact_epoch=2,
    )

    assert command.sequence == 101
    assert command.source_policy_sequence == 7
    assert command.source_policy_generation == 3
    assert command.source_contact_epoch == 2

    with pytest.raises(ValueError, match="source_policy_generation"):
        replace(command, source_policy_generation=None)
    with pytest.raises(ValueError, match="source_contact_epoch"):
        replace(command, source_contact_epoch=None)


def test_command_scalars_and_vectors_reject_boolean_or_string_coercion() -> None:
    with pytest.raises(TypeError, match="real number"):
        Go2JointPositionCommand(
            sequence=1,
            timestamp_s=True,  # type: ignore[arg-type]
            valid_until_s=1.05,
            joint_positions_rad=(0.0,) * 12,
            desired_contact_forces_world_n=(0.0,) * 12,
        )
    with pytest.raises(TypeError, match="real numeric"):
        Go2JointPositionCommand(
            sequence=1,
            timestamp_s=1.0,
            valid_until_s=1.05,
            joint_positions_rad=("0",) * 12,  # type: ignore[arg-type]
            desired_contact_forces_world_n=(0.0,) * 12,
        )
    with pytest.raises(TypeError, match="real number"):
        FlightControllerRotorResidualCommand(
            sequence=1,
            timestamp_s=1.0,
            valid_until_s=1.05,
            fc_session_id=11,
            target_fc_tick=102,
            baseline_version=7,
            baseline_timestamp_s=1.0,
            baseline_thrusts_n=(4.0,) * 4,
            transport_raw_residual_thrusts_n=(1.0,) * 4,
            applied_residual_thrusts_n=(0.1,) * 4,
            applied_total_thrusts_n=(4.1,) * 4,
            correction_gain=True,  # type: ignore[arg-type]
            transport_target_semantics="invalid",
        )


@pytest.mark.asyncio
async def test_missing_real_adapters_fail_closed() -> None:
    go2_result = await UnavailableGo2LowLevelSink().send_joint_position_command(_leg())
    fc_result = await UnavailableFlightControllerResidualSink().send_rotor_residual(_rotor())
    assert not go2_result.ok
    assert go2_result.code == "GO2_LOW_LEVEL_NOT_CONFIGURED"
    assert not fc_result.ok
    assert fc_result.code == "FC_ROTOR_RESIDUAL_NOT_CONFIGURED"
