from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Go2ControlAuthorityState, SystemState
from aerogo2.common.models import (
    Go2ControlAuthorityStatus,
    Go2LowLevelStatus,
    Go2MotorFeedback,
    LandingEstimate,
    LowCmdOwnershipState,
    SystemSnapshot,
)
from aerogo2.landing.impact_aware.integration import (
    FlightControllerResidualSinkStatus,
    FlightControllerResidualState,
    attest_post_touchdown_recovery,
)
from aerogo2.manager.transition_guards import TransitionGuards
from aerogo2.safety.safety_monitor import SafetyMonitor
from aerogo2.simulation.world import SimulationWorld

_MAPPING_HASH = "a" * 64
_TRACKING_LIMIT_RAD = 0.2


def _enabled_lowcmd_config(config: AppConfig) -> AppConfig:
    """Enable only the commissioned facts exercised by these snapshot tests."""

    return replace(
        config,
        go2=replace(
            config.go2,
            low_level=replace(
                config.go2.low_level,
                enabled=True,
                low_state_max_age_s=0.02,
                mapping_hash=_MAPPING_HASH,
                tracking_position_error_limit_rad=(_TRACKING_LIMIT_RAD,) * 12,
            ),
        ),
    )


def _active_lowcmd_snapshot(
    base: SystemSnapshot,
    *,
    position_errors_rad: tuple[float, ...] = (0.0,) * 12,
) -> SystemSnapshot:
    """Build a causal LowCmd-write/LowState pair with no SportMode sample."""

    now = base.timestamp
    references = (0.0,) * 12
    motors = tuple(
        Go2MotorFeedback(
            motor_id=index,
            joint_name=f"joint_{index}",
            q_rad=references[index] + position_errors_rad[index],
            dq_rad_s=0.0,
            tau_est_nm=0.0,
            temperature_c=25.0,
            lost=False,
            timestamp=now,
        )
        for index in range(12)
    )
    low_level = Go2LowLevelStatus(
        timestamp=now,
        connected=True,
        ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
        owner_epoch=99,
        healthy=True,
        low_state_timestamp=now,
        low_state_age_s=0.0,
        target_sequence=4,
        target_age_s=0.0,
        target_deadline=now + 0.02,
        publisher_active=True,
        writer_alive=True,
        last_write_timestamp=now - 0.001,
        watchdog_healthy=True,
        safe_hold_active=False,
        safe_hold_settled=False,
        high_level_released=True,
        network_exclusivity_verified=True,
        mapping_hash_verified=True,
        active_mapping_hash=_MAPPING_HASH,
        fault_reason=None,
        motors=motors,
        tracking_error_timestamp=now,
        tracking_reference_write_timestamp=now - 0.001,
        tracking_reference_write_generation=1,
        tracking_reference_q_rad=references,
        position_error_rad=position_errors_rad,
    )
    authority = Go2ControlAuthorityStatus(
        state=Go2ControlAuthorityState.LOWCMD_ACTIVE,
        timestamp=now,
        generation=1,
        ownership_epoch=low_level.owner_epoch,
        reason="test LowCmd owner is active",
    )
    return replace(
        base,
        state=SystemState.AUTO_LANDING,
        pixhawk=replace(base.pixhawk, armed=True, landed=False),
        go2=replace(
            base.go2,
            # SportModeState can disappear after ReleaseMode().  These fields
            # intentionally look unavailable while LowState remains healthy.
            connected=False,
            timestamp=0.0,
            message_age_s=float("inf"),
            locomotion_mode="UNKNOWN",
            joints_locked=False,
            controller_active=False,
            low_level_status=low_level,
            control_authority=authority,
        ),
    )


def _retained_lowcmd_status() -> Go2LowLevelStatus:
    return Go2LowLevelStatus(
        timestamp=10.0,
        connected=True,
        ownership_state=LowCmdOwnershipState.SAFE_HOLD,
        owner_epoch=17,
        healthy=True,
        low_state_timestamp=10.0,
        low_state_age_s=0.0,
        publisher_active=True,
        writer_alive=True,
        watchdog_healthy=True,
        safe_hold_active=True,
        safe_hold_settled=True,
        high_level_released=True,
        network_exclusivity_verified=True,
        mapping_hash_verified=True,
        active_mapping_hash=_MAPPING_HASH,
        fault_reason=None,
    )


def _confirmed_zero_residual_status() -> FlightControllerResidualSinkStatus:
    return FlightControllerResidualSinkStatus(
        timestamp_s=9.99,
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.CONFIRMED_ZERO,
        residual_active=False,
        clear_confirmed=True,
        fc_session_id=23,
        last_sequence=41,
        active_valid_until_s=None,
        last_error="",
        control_epoch=5,
        transport_generation=7,
        clear_through_command_sequence=41,
        residual_register_inactive=True,
        baseline_controller_retained=True,
        clear_ack_timestamp_s=9.97,
        clear_execution_timestamp_s=9.98,
    )


def _attest(
    residual_status: FlightControllerResidualSinkStatus,
):
    return attest_post_touchdown_recovery(
        timestamp_s=10.0,
        valid_until_s=10.1,
        maximum_residual_status_age_s=0.02,
        landing_session_id=3,
        sequence=11,
        contact_epoch=2,
        contacts=(True, True, True, True),
        admittance_blends=(1.0, 1.0, 1.0, 1.0),
        controller_quiesced=True,
        recovery_complete=True,
        load_transfer_complete=True,
        body_state_stable=True,
        go2_status=_retained_lowcmd_status(),
        residual_status=residual_status,
    )


def test_lowcmd_authority_uses_lowstate_when_sportmode_is_unreadable(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    snapshot = _active_lowcmd_snapshot(safe_walk_snapshot)

    codes = {violation.code for violation in SafetyMonitor(config).evaluate(snapshot)}

    assert "GO2_TIMEOUT" not in codes
    assert "GO2_LOWSTATE_TIMEOUT" not in codes
    assert "GO2_JOINT_LOCK_LOST" not in codes
    assert "GO2_JOINT_TRACKING_ERROR" not in codes
    assert "GO2_CONTROL_AUTHORITY_INCONSISTENT" not in codes
    assert "GO2_LOWCMD_OWNER_UNHEALTHY" not in codes


def test_bounded_high_level_reacquisition_does_not_require_sportmode_sample_yet(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    now = safe_walk_snapshot.timestamp
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.GO2_GROUND_HANDOVER,
        go2=replace(
            safe_walk_snapshot.go2,
            connected=False,
            timestamp=0.0,
            joints_locked=False,
            low_level_status=Go2LowLevelStatus(
                timestamp=now,
                ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
            ),
            control_authority=Go2ControlAuthorityStatus(
                state=Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
                timestamp=now,
                transition_started_at=now - 0.01,
                transition_deadline=now + 0.5,
                generation=4,
                ownership_epoch=0,
                reason="waiting for causal post-release JOINT_LOCK",
            ),
        ),
    )

    codes = {violation.code for violation in SafetyMonitor(config).evaluate(snapshot)}

    assert "GO2_TIMEOUT" not in codes
    assert "GO2_CONTROL_AUTHORITY_TRANSITION_TIMEOUT" not in codes
    assert "GO2_CONTROL_AUTHORITY_INCONSISTENT" not in codes


def test_joint_tracking_limit_is_inclusive_but_epsilon_overrun_is_rejected(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    boundary_errors = (_TRACKING_LIMIT_RAD,) + (0.0,) * 11
    overrun_errors = (_TRACKING_LIMIT_RAD + 1.0e-8,) + (0.0,) * 11

    boundary_codes = {
        violation.code
        for violation in SafetyMonitor(config).evaluate(
            _active_lowcmd_snapshot(
                safe_walk_snapshot,
                position_errors_rad=boundary_errors,
            )
        )
    }
    overrun_codes = {
        violation.code
        for violation in SafetyMonitor(config).evaluate(
            _active_lowcmd_snapshot(
                safe_walk_snapshot,
                position_errors_rad=overrun_errors,
            )
        )
    }

    assert "GO2_JOINT_TRACKING_ERROR" not in boundary_codes
    assert "GO2_JOINT_TRACKING_ERROR" in overrun_codes


def test_explicit_authority_cannot_contradict_raw_lowcmd_ownership(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    snapshot = _active_lowcmd_snapshot(safe_walk_snapshot)
    contradictory = replace(
        snapshot,
        go2=replace(
            snapshot.go2,
            control_authority=Go2ControlAuthorityStatus(
                state=Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK,
                timestamp=snapshot.timestamp,
                generation=2,
                ownership_epoch=0,
                reason="contradictory test state",
            ),
        ),
    )

    codes = {violation.code for violation in SafetyMonitor(config).evaluate(contradictory)}

    assert "GO2_CONTROL_AUTHORITY_INCONSISTENT" in codes


def test_post_touchdown_attestation_accepts_exact_fc_clear_barrier() -> None:
    evidence = _attest(_confirmed_zero_residual_status())

    assert evidence.confirmed
    assert evidence.go2_ownership_epoch == 17
    assert evidence.fc_session_id == 23
    assert evidence.clear_through_command_sequence == evidence.last_residual_command_sequence
    assert evidence.baseline_controller_retained
    assert evidence.residual_register_inactive


def test_post_touchdown_attestation_rejects_clear_watermark_behind_command() -> None:
    status = replace(
        _confirmed_zero_residual_status(),
        clear_through_command_sequence=40,
    )

    with pytest.raises(ValueError, match="persistent zero residual"):
        _attest(status)


def test_post_touchdown_attestation_rejects_lost_fc_baseline() -> None:
    status = replace(
        _confirmed_zero_residual_status(),
        baseline_controller_retained=False,
    )

    with pytest.raises(ValueError, match="baseline"):
        _attest(status)


def test_post_touchdown_attestation_rejects_pending_residual_stage() -> None:
    status = FlightControllerResidualSinkStatus(
        timestamp_s=9.99,
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.STAGE_PENDING,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=23,
        last_sequence=42,
        active_valid_until_s=None,
        last_error="",
        pending_command_sequence=42,
        pending_started_s=9.98,
        pending_valid_until_s=10.02,
        control_epoch=5,
        transport_generation=7,
        clear_through_command_sequence=41,
        residual_register_inactive=False,
        baseline_controller_retained=True,
    )

    with pytest.raises(ValueError, match="persistent zero residual"):
        _attest(status)


def test_post_touchdown_attestation_rejects_stale_fc_clear_status() -> None:
    status = replace(
        _confirmed_zero_residual_status(),
        timestamp_s=9.96,
        clear_ack_timestamp_s=9.94,
        clear_execution_timestamp_s=9.95,
    )

    with pytest.raises(ValueError, match="persistent zero residual"):
        _attest(status)


def test_manager_rejects_retimestamped_recovery_evidence_replay(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    manager = world.manager
    manager._impact_landing_session_id = 3
    original = _attest(_confirmed_zero_residual_status())

    manager.accept_impact_landing_recovery_evidence(original)
    assert manager._impact_recovery.confirmed

    replay = replace(original, timestamp=10.01, valid_until=10.11)
    manager.accept_impact_landing_recovery_evidence(replay)

    assert not manager._impact_recovery.confirmed
    assert "replayed" in manager._impact_recovery.reason


def test_manager_requires_immutable_clear_times_and_a_new_persistent_zero_sample(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    manager = world.manager
    manager._impact_landing_session_id = 3
    original = _attest(_confirmed_zero_residual_status())
    manager.accept_impact_landing_recovery_evidence(original)

    world.clock.advance(0.01)
    refreshed = replace(
        original,
        sequence=original.sequence + 1,
        timestamp=10.01,
        valid_until=10.11,
        residual_zero_status_timestamp=10.01,
    )
    manager.accept_impact_landing_recovery_evidence(refreshed)
    assert manager._impact_recovery == refreshed

    world.clock.advance(0.01)
    reused_status = replace(
        refreshed,
        sequence=refreshed.sequence + 1,
        timestamp=10.02,
        valid_until=10.12,
    )
    manager.accept_impact_landing_recovery_evidence(reused_status)
    assert not manager._impact_recovery.confirmed

    manager._last_confirmed_impact_recovery = refreshed
    restamped_clear = replace(
        refreshed,
        sequence=refreshed.sequence + 2,
        timestamp=10.02,
        valid_until=10.12,
        residual_zero_ack_timestamp=9.98,
        residual_zero_status_timestamp=10.02,
    )
    manager.accept_impact_landing_recovery_evidence(restamped_clear)
    assert not manager._impact_recovery.confirmed


def test_only_auto_landing_requires_recovery_gate_before_touchdown_verify(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    landed = replace(
        safe_walk_snapshot,
        state=SystemState.AUTO_LANDING,
        pixhawk=replace(safe_walk_snapshot.pixhawk, armed=True, landed=True),
        landing_estimate=LandingEstimate(
            valid=True,
            ground_detected=True,
            height_m=0.0,
            vertical_velocity_mps=0.0,
            horizontal_velocity_mps=0.0,
            timestamp=safe_walk_snapshot.timestamp,
            reason="fresh touchdown estimate",
        ),
        impact_landing_session_id=8,
    )
    guards = TransitionGuards(config)

    automatic = guards.evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        landed,
    )
    manual = guards.evaluate(
        SystemState.FLIGHT_MANUAL,
        SystemState.TOUCHDOWN_VERIFY,
        replace(landed, state=SystemState.FLIGHT_MANUAL),
    )

    assert not automatic.permitted
    assert "IMPACT_RECOVERY_EXIT_NOT_CONFIRMED" in automatic.codes
    assert manual.permitted
    assert "IMPACT_RECOVERY_EXIT_NOT_CONFIRMED" not in manual.codes


def test_automatic_touchdown_exit_requires_same_owner_safe_hold(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    evidence = _attest(_confirmed_zero_residual_status())
    active = _active_lowcmd_snapshot(safe_walk_snapshot)
    active = replace(
        active,
        pixhawk=replace(active.pixhawk, landed=True),
        landing_estimate=LandingEstimate(
            valid=True,
            ground_detected=True,
            height_m=0.0,
            vertical_velocity_mps=0.0,
            horizontal_velocity_mps=0.0,
            timestamp=10.0,
            reason="fresh touchdown estimate",
        ),
        impact_landing_session_id=evidence.landing_session_id,
        impact_recovery=replace(evidence, go2_ownership_epoch=99),
        post_touchdown_stable_since=9.0,
        post_touchdown_last_stability_check_at=10.0,
        post_touchdown_stable_dwell_complete=True,
        impact_landing_exit_ready=True,
    )

    active_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        active,
    )

    safe_status = _retained_lowcmd_status()
    safe = replace(
        active,
        go2=replace(
            active.go2,
            low_level_status=safe_status,
            control_authority=Go2ControlAuthorityStatus(
                state=Go2ControlAuthorityState.LOWCMD_SAFE_HOLD,
                timestamp=10.0,
                generation=2,
                ownership_epoch=safe_status.owner_epoch,
                reason="same owner retained safe hold",
            ),
        ),
        impact_recovery=evidence,
    )
    safe_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        safe,
    )
    stale_source_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        replace(
            safe,
            pixhawk=replace(safe.pixhawk, landed_state_timestamp=9.69),
        ),
    )
    incoherent_source_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        replace(
            safe,
            pixhawk=replace(safe.pixhawk, attitude_timestamp=9.749),
        ),
    )
    invalid_payload_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        replace(
            safe,
            pixhawk=replace(safe.pixhawk, vertical_velocity_mps=float("nan")),
        ),
    )
    invalid_estimate_results = tuple(
        TransitionGuards(config).evaluate(
            SystemState.AUTO_LANDING,
            SystemState.TOUCHDOWN_VERIFY,
            replace(
                safe,
                landing_estimate=replace(safe.landing_estimate, **changes),
            ),
        )
        for changes in (
            {"valid": 1},
            {"ground_detected": "yes"},
            {"height_m": float("nan")},
            {"vertical_velocity_mps": float("inf")},
            {"horizontal_velocity_mps": float("nan")},
        )
    )
    active_setpoint_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        replace(safe, external_setpoint_active=True),
    )
    expired_at = 10.06
    expired_status = replace(
        safe_status,
        timestamp=expired_at,
        low_state_timestamp=expired_at,
    )
    expired_result = TransitionGuards(config).evaluate(
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        replace(
            safe,
            timestamp=expired_at,
            go2=replace(safe.go2, low_level_status=expired_status),
            impact_recovery=replace(evidence, valid_until=10.05),
        ),
    )

    assert not active_result.permitted
    assert "IMPACT_RECOVERY_EXIT_NOT_CONFIRMED" in active_result.codes
    assert safe_result.permitted
    assert not stale_source_result.permitted
    assert "PIXHAWK_TOUCHDOWN_EVIDENCE_STALE" in stale_source_result.codes
    assert not incoherent_source_result.permitted
    assert "PIXHAWK_TOUCHDOWN_EVIDENCE_INCOHERENT" in incoherent_source_result.codes
    assert not invalid_payload_result.permitted
    assert "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID" in invalid_payload_result.codes
    assert all(not result.permitted for result in invalid_estimate_results)
    assert all("AUTOLAND_ESTIMATOR_INVALID" in result.codes for result in invalid_estimate_results)
    assert not active_setpoint_result.permitted
    assert not expired_result.permitted
