from __future__ import annotations

from dataclasses import replace
from typing import Set

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446State,
    SystemState,
)
from aerogo2.common.models import (
    EscTelemetry,
    LandingEstimate,
    SystemSnapshot,
)
from aerogo2.safety.safety_monitor import SafetyMonitor

REQUIRED_VIOLATION_CODES = (
    "PIXHAWK_TIMEOUT",
    "F446_TIMEOUT",
    "GO2_TIMEOUT",
    "RC_TIMEOUT",
    "RC_FAILSAFE",
    "PIXHAWK_ARMED_DURING_TRANSFORM",
    "ESC_RPM_NONZERO_DURING_TRANSFORM",
    "GO2_MOVING_DURING_TRANSFORM",
    "F446_FAULT",
    "F446_OVERCURRENT",
    "F446_STATE_UNKNOWN",
    "F446_CONFIGURATION_CONFLICT",
    "AUTOLAND_CONTROLLER_TIMEOUT",
    "AUTOLAND_ESTIMATOR_INVALID",
    "MANUAL_OVERRIDE_REQUESTED",
    "INVALID_STATE_TRANSITION",
    "EXTERNAL_SETPOINT_OUTSIDE_AUTOLAND",
)


def violation_codes(
    monitor: SafetyMonitor,
    snapshot: SystemSnapshot,
) -> Set[str]:
    return {item.code for item in monitor.evaluate(snapshot)}


def flight_snapshot(
    app_config: AppConfig,
    base: SystemSnapshot,
    state: SystemState = SystemState.AUTO_LANDING,
) -> SystemSnapshot:
    now = base.timestamp
    return replace(
        base,
        state=state,
        pixhawk=replace(
            base.pixhawk,
            connected=True,
            armed=True,
            landed=False,
            failsafe=False,
            heartbeat_timestamp=now,
        ),
        f446=replace(
            base.f446,
            connected=True,
            state=app_config.f446.expected_flight_state,
            duty=0,
            fault_message=None,
            timestamp=now,
        ),
        go2=replace(
            base.go2,
            connected=True,
            velocity_mps=0.0,
            stable=True,
            joints_locked=True,
            locomotion_mode="JOINT_LOCK",
            timestamp=now,
        ),
        rc=replace(
            base.rc,
            connected=True,
            failsafe=False,
            auto_landing_request=AutoLandingRequest.AUTO_EXECUTE,
            manual_override=False,
            timestamp=now,
        ),
        configuration=Configuration.FLIGHT,
        landing_estimate=LandingEstimate(
            valid=True,
            ground_detected=True,
            height_m=1.0,
            vertical_velocity_mps=0.0,
            horizontal_velocity_mps=0.0,
            timestamp=now,
            reason="valid simulated estimate",
        ),
        autoland_active=state is SystemState.AUTO_LANDING,
        external_setpoint_active=False,
        active_fault_codes=(),
    )


def scenario_for_code(
    app_config: AppConfig,
    base: SystemSnapshot,
    code: str,
) -> SystemSnapshot:
    now = base.timestamp
    if code == "PIXHAWK_TIMEOUT":
        return replace(
            base,
            pixhawk=replace(
                base.pixhawk,
                heartbeat_timestamp=now - app_config.safety.pixhawk_timeout_s - 0.001,
            ),
        )
    if code == "F446_TIMEOUT":
        return replace(
            base,
            f446=replace(
                base.f446,
                timestamp=now - app_config.safety.f446_timeout_s - 0.001,
            ),
        )
    if code == "GO2_TIMEOUT":
        return replace(
            base,
            go2=replace(
                base.go2,
                timestamp=now - app_config.safety.go2_timeout_s - 0.001,
            ),
        )
    if code == "RC_TIMEOUT":
        return replace(
            base,
            rc=replace(
                base.rc,
                timestamp=now - app_config.safety.rc_timeout_s - 0.001,
            ),
        )
    if code == "RC_FAILSAFE":
        return replace(base, rc=replace(base.rc, failsafe=True))
    if code == "PIXHAWK_ARMED_DURING_TRANSFORM":
        return replace(
            base,
            state=SystemState.TRANSFORM_TO_FLIGHT,
            pixhawk=replace(base.pixhawk, armed=True),
        )
    if code == "ESC_RPM_NONZERO_DURING_TRANSFORM":
        esc = list(base.pixhawk.esc)
        esc[0] = replace(
            esc[0],
            rpm=app_config.safety.maximum_safe_esc_rpm_for_transform,
            timestamp=now,
        )
        return replace(
            base,
            state=SystemState.TRANSFORM_TO_FLIGHT,
            pixhawk=replace(base.pixhawk, esc=tuple(esc)),
        )
    if code == "GO2_MOVING_DURING_TRANSFORM":
        return replace(
            base,
            state=SystemState.TRANSFORM_TO_FLIGHT,
            go2=replace(
                base.go2,
                velocity_mps=app_config.safety.stationary_velocity_mps,
            ),
        )
    if code == "F446_FAULT":
        return replace(
            base,
            configuration=Configuration.UNKNOWN,
            f446=replace(
                base.f446,
                state=F446State.FAULT,
                fault_message="simulated fault",
            ),
        )
    if code == "F446_OVERCURRENT":
        return replace(
            base,
            f446=replace(
                base.f446,
                used_current_adc=app_config.safety.maximum_transform_current_adc,
            ),
        )
    if code == "F446_STATE_UNKNOWN":
        return replace(
            base,
            configuration=Configuration.UNKNOWN,
            f446=replace(base.f446, state=F446State.UNKNOWN),
        )
    if code == "F446_CONFIGURATION_CONFLICT":
        return replace(
            base,
            f446=replace(
                base.f446,
                state=app_config.f446.expected_flight_state,
            ),
        )
    if code == "AUTOLAND_ESTIMATOR_INVALID":
        snapshot = flight_snapshot(app_config, base)
        return replace(
            snapshot,
            landing_estimate=replace(
                snapshot.landing_estimate,
                valid=False,
                reason="simulated estimator failure",
            ),
        )
    if code == "MANUAL_OVERRIDE_REQUESTED":
        snapshot = flight_snapshot(app_config, base)
        return replace(
            snapshot,
            rc=replace(snapshot.rc, manual_override=True),
        )
    if code == "AUTOLAND_CONTROLLER_TIMEOUT":
        return replace(base, active_fault_codes=(code,))
    if code == "INVALID_STATE_TRANSITION":
        return replace(base, active_fault_codes=(code,))
    if code == "EXTERNAL_SETPOINT_OUTSIDE_AUTOLAND":
        return replace(base, external_setpoint_active=True)
    raise AssertionError(f"missing SafetyMonitor scenario for {code}")


@pytest.mark.parametrize("expected_code", REQUIRED_VIOLATION_CODES)
def test_each_safety_violation_code_is_detected(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    expected_code: str,
) -> None:
    monitor = SafetyMonitor(app_config)
    snapshot = scenario_for_code(app_config, safe_walk_snapshot, expected_code)
    assert expected_code in violation_codes(monitor, snapshot)


def test_safe_walk_has_no_safety_violations(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    assert SafetyMonitor(app_config).evaluate(safe_walk_snapshot) == []


@pytest.mark.parametrize(
    "code,timeout_name",
    [
        ("PIXHAWK_TIMEOUT", "pixhawk_timeout_s"),
        ("F446_TIMEOUT", "f446_timeout_s"),
        ("GO2_TIMEOUT", "go2_timeout_s"),
        ("RC_TIMEOUT", "rc_timeout_s"),
    ],
)
def test_exact_timeout_is_fresh_and_epsilon_over_is_stale(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    code: str,
    timeout_name: str,
) -> None:
    timeout = float(getattr(app_config.safety, timeout_name))
    now = safe_walk_snapshot.timestamp

    def with_timestamp(timestamp: float) -> SystemSnapshot:
        if code == "PIXHAWK_TIMEOUT":
            return replace(
                safe_walk_snapshot,
                pixhawk=replace(
                    safe_walk_snapshot.pixhawk,
                    heartbeat_timestamp=timestamp,
                ),
            )
        if code == "F446_TIMEOUT":
            return replace(
                safe_walk_snapshot,
                f446=replace(safe_walk_snapshot.f446, timestamp=timestamp),
            )
        if code == "GO2_TIMEOUT":
            return replace(
                safe_walk_snapshot,
                go2=replace(safe_walk_snapshot.go2, timestamp=timestamp),
            )
        return replace(
            safe_walk_snapshot,
            rc=replace(safe_walk_snapshot.rc, timestamp=timestamp),
        )

    monitor = SafetyMonitor(app_config)
    exact = with_timestamp(now - timeout)
    stale = with_timestamp(now - timeout - 0.000001)
    assert code not in violation_codes(monitor, exact)
    assert code in violation_codes(monitor, stale)


@pytest.mark.parametrize("bad_timestamp", [float("nan"), float("inf"), 10.001])
def test_invalid_or_future_heartbeat_is_stale(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    bad_timestamp: float,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(
            safe_walk_snapshot.pixhawk,
            heartbeat_timestamp=bad_timestamp,
        ),
    )
    assert "PIXHAWK_TIMEOUT" in violation_codes(SafetyMonitor(app_config), snapshot)


def test_active_transform_requires_exact_zero_rpm(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    threshold = app_config.safety.maximum_safe_esc_rpm_for_transform
    base = replace(safe_walk_snapshot, state=SystemState.TRANSFORM_TO_FLIGHT)
    below = (
        EscTelemetry(1, "RR", rpm=threshold - 0.001, timestamp=base.timestamp),
        EscTelemetry(2, "LF", timestamp=base.timestamp),
        EscTelemetry(3, "LR", timestamp=base.timestamp),
        EscTelemetry(4, "RF", timestamp=base.timestamp),
    )
    equal = (replace(below[0], rpm=threshold),) + below[1:]
    monitor = SafetyMonitor(app_config)
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in violation_codes(
        monitor,
        replace(
            base,
            pixhawk=replace(
                base.pixhawk, esc=below, esc_rpm={item.slot: item.rpm for item in below}
            ),
        ),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in violation_codes(
        monitor,
        replace(
            base,
            pixhawk=replace(
                base.pixhawk, esc=equal, esc_rpm={item.slot: item.rpm for item in equal}
            ),
        ),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" not in violation_codes(
        monitor,
        replace(base, pixhawk=replace(base.pixhawk, esc=safe_walk_snapshot.pixhawk.esc)),
    )


def test_nonfinite_or_unhealthy_esc_is_unsafe_during_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    base = replace(safe_walk_snapshot, state=SystemState.TRANSFORM_TO_FLIGHT)
    for replacement in (
        replace(base.pixhawk.esc[0], rpm=float("nan")),
        replace(base.pixhawk.esc[0], healthy=False),
    ):
        esc = (replacement,) + base.pixhawk.esc[1:]
        snapshot = replace(base, pixhawk=replace(base.pixhawk, esc=esc))
        assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in violation_codes(
            SafetyMonitor(app_config),
            snapshot,
        )


def test_go2_velocity_and_current_equal_to_limits_are_unsafe(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    monitor = SafetyMonitor(app_config)
    transform = replace(
        safe_walk_snapshot,
        state=SystemState.TRANSFORM_TO_FLIGHT,
        go2=replace(
            safe_walk_snapshot.go2,
            velocity_mps=app_config.safety.stationary_velocity_mps,
        ),
    )
    current = replace(
        safe_walk_snapshot,
        f446=replace(
            safe_walk_snapshot.f446,
            used_current_adc=app_config.safety.maximum_transform_current_adc,
        ),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" in violation_codes(monitor, transform)
    assert "F446_OVERCURRENT" in violation_codes(monitor, current)


def test_values_just_below_motion_and_current_limits_are_safe(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.TRANSFORM_TO_FLIGHT,
        go2=replace(
            safe_walk_snapshot.go2,
            velocity_mps=app_config.safety.stationary_velocity_mps - 0.000001,
        ),
        f446=replace(
            safe_walk_snapshot.f446,
            used_current_adc=app_config.safety.maximum_transform_current_adc - 1,
        ),
    )
    codes = violation_codes(SafetyMonitor(app_config), snapshot)
    assert "GO2_MOVING_DURING_TRANSFORM" not in codes
    assert "F446_OVERCURRENT" not in codes


def test_configuration_conflict_is_suppressed_only_during_active_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    moving = replace(
        safe_walk_snapshot,
        state=SystemState.TRANSFORM_TO_FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=F446State.LIMIT_FWD,
            duty=app_config.f446.flight_duty,
        ),
    )
    assert "F446_CONFIGURATION_CONFLICT" not in violation_codes(
        SafetyMonitor(app_config),
        moving,
    )


def test_homing_monitor_stays_fail_closed_when_esc_telemetry_is_lost(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    homing = replace(
        safe_walk_snapshot,
        state=SystemState.HOMING_TO_WALK,
        configuration=Configuration.UNKNOWN,
        f446=replace(
            safe_walk_snapshot.f446,
            state=(
                F446State.LIMIT_FWD
                if app_config.f446.direction_for(Configuration.WALK.value) == "forward"
                else F446State.LIMIT_REV
            ),
            duty=app_config.f446.walk_duty,
        ),
    )
    monitor = SafetyMonitor(app_config)
    safe_codes = violation_codes(monitor, homing)
    assert "F446_CONFIGURATION_CONFLICT" not in safe_codes
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" not in safe_codes

    missing_esc = replace(
        homing,
        pixhawk=replace(homing.pixhawk, esc=(), esc_rpm={}),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in violation_codes(
        monitor,
        missing_esc,
    )


def test_monitor_is_deterministic_and_does_not_mutate_snapshot(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    monitor = SafetyMonitor(app_config)
    before = safe_walk_snapshot
    first = monitor.evaluate(safe_walk_snapshot)
    second = monitor.evaluate(safe_walk_snapshot)
    assert first == second
    assert safe_walk_snapshot == before


def test_violation_factories_preserve_timestamp_and_code(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    monitor = SafetyMonitor(app_config)
    invalid = monitor.invalid_transition_violation(
        safe_walk_snapshot,
        SystemState.FLIGHT_READY,
        "test rejection",
    )
    timeout = monitor.controller_timeout_violation(safe_walk_snapshot)
    assert invalid.code == "INVALID_STATE_TRANSITION"
    assert timeout.code == "AUTOLAND_CONTROLLER_TIMEOUT"
    assert invalid.timestamp == timeout.timestamp == safe_walk_snapshot.timestamp
