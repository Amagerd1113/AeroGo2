from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    Configuration,
    F446State,
    MorphologyRequest,
    SystemState,
)
from aerogo2.common.models import EscTelemetry, SystemSnapshot
from aerogo2.manager.transition_guards import TransitionGuards


def evaluate(app_config: AppConfig, snapshot: SystemSnapshot):
    return TransitionGuards(app_config).evaluate(
        SystemState.WALK,
        SystemState.WALK_TO_FLIGHT_PRECHECK,
        snapshot,
    )


def home_snapshot(snapshot: SystemSnapshot) -> SystemSnapshot:
    return replace(
        snapshot,
        state=SystemState.BOOT_SAFE,
        configuration=Configuration.UNKNOWN,
        f446=replace(snapshot.f446, state=F446State.IDLE, duty=0),
    )


def evaluate_home(app_config: AppConfig, snapshot: SystemSnapshot):
    return TransitionGuards(app_config).evaluate(
        SystemState.BOOT_SAFE,
        SystemState.HOMING_TO_WALK,
        snapshot,
    )


def test_safe_unknown_home_to_walk_is_allowed(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = home_snapshot(safe_walk_snapshot)
    assert evaluate_home(app_config, snapshot).permitted


def test_home_rejects_known_configuration(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        home_snapshot(safe_walk_snapshot),
        configuration=Configuration.WALK,
    )
    assert "F446_HOME_NOT_REQUIRED" in evaluate_home(app_config, snapshot).codes


def test_home_rejects_non_idle_f446(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    initial = home_snapshot(safe_walk_snapshot)
    snapshot = replace(
        initial,
        f446=replace(initial.f446, state=F446State.LIMIT_REV),
    )
    assert "F446_HOME_START_STATE_INVALID" in evaluate_home(app_config, snapshot).codes


def test_home_rejects_pixhawk_failsafe(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    initial = home_snapshot(safe_walk_snapshot)
    snapshot = replace(
        initial,
        pixhawk=replace(initial.pixhawk, failsafe=True),
    )
    assert "PIXHAWK_FAILSAFE" in evaluate_home(app_config, snapshot).codes


def test_home_rejects_unpowered_or_missing_esc_telemetry(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    initial = home_snapshot(safe_walk_snapshot)
    snapshot = replace(
        initial,
        pixhawk=replace(initial.pixhawk, esc=(), esc_rpm={}),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in evaluate_home(app_config, snapshot).codes


def test_home_rejects_unstable_go2(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    initial = home_snapshot(safe_walk_snapshot)
    snapshot = replace(
        initial,
        go2=replace(initial.go2, stable=False),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" in evaluate_home(app_config, snapshot).codes


def test_safe_flight_precheck_is_allowed(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    assert evaluate(app_config, safe_walk_snapshot).permitted


def test_pixhawk_armed_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(safe_walk_snapshot.pixhawk, armed=True),
    )
    assert "PIXHAWK_ARMED_DURING_TRANSFORM" in evaluate(app_config, snapshot).codes


@pytest.mark.parametrize("rpm", [0.01, 1.0, 49.0, -49.0])
def test_initial_precheck_accepts_finite_online_rpm_strictly_below_limit(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    rpm: float,
) -> None:
    esc = (
        EscTelemetry(1, "RR", rpm=rpm, timestamp=10.0),
        EscTelemetry(2, "LF", timestamp=10.0),
        EscTelemetry(3, "LR", timestamp=10.0),
        EscTelemetry(4, "RF", timestamp=10.0),
    )
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(
            safe_walk_snapshot.pixhawk,
            esc=esc,
            esc_rpm={item.slot: item.rpm for item in esc},
        ),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" not in evaluate(app_config, snapshot).codes


@pytest.mark.parametrize("rpm", [50.0, 51.0, 1000.0, -50.0, -1000.0])
def test_initial_precheck_rejects_rpm_at_or_above_absolute_limit(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    rpm: float,
) -> None:
    esc = (
        EscTelemetry(1, "RR", rpm=rpm, timestamp=10.0),
        EscTelemetry(2, "LF", timestamp=10.0),
        EscTelemetry(3, "LR", timestamp=10.0),
        EscTelemetry(4, "RF", timestamp=10.0),
    )
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(
            safe_walk_snapshot.pixhawk,
            esc=esc,
            esc_rpm={item.slot: item.rpm for item in esc},
        ),
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in evaluate(app_config, snapshot).codes


def test_active_transform_entry_requires_exact_zero_rpm(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    esc = (
        EscTelemetry(1, "RR", rpm=0.001, timestamp=10.0),
        EscTelemetry(2, "LF", timestamp=10.0),
        EscTelemetry(3, "LR", timestamp=10.0),
        EscTelemetry(4, "RF", timestamp=10.0),
    )
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.WALK_TO_FLIGHT_PRECHECK,
        pixhawk=replace(
            safe_walk_snapshot.pixhawk,
            esc=esc,
            esc_rpm={item.slot: item.rpm for item in esc},
        ),
    )
    result = TransitionGuards(app_config).evaluate(
        SystemState.WALK_TO_FLIGHT_PRECHECK,
        SystemState.TRANSFORM_TO_FLIGHT,
        snapshot,
    )
    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in result.codes


@pytest.mark.parametrize("velocity", [-0.1, 0.05, 0.2])
def test_go2_motion_rejects_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    velocity: float,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        go2=replace(safe_walk_snapshot.go2, velocity_mps=velocity),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" in evaluate(app_config, snapshot).codes


def test_go2_velocity_strictly_below_configured_limit_is_allowed(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        go2=replace(
            safe_walk_snapshot.go2,
            velocity_mps=app_config.safety.stationary_velocity_mps - 0.000001,
        ),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" not in evaluate(app_config, snapshot).codes


def test_active_go2_controller_rejects_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        go2=replace(
            safe_walk_snapshot.go2,
            controller_active=True,
        ),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" in evaluate(app_config, snapshot).codes


def test_unstable_go2_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        go2=replace(safe_walk_snapshot.go2, stable=False),
    )
    assert "GO2_MOVING_DURING_TRANSFORM" in evaluate(app_config, snapshot).codes


def test_flight_enable_high_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        rc=replace(
            safe_walk_snapshot.rc,
            channels={
                **safe_walk_snapshot.rc.channels,
                app_config.rc.flight_enable_channel: 1900,
            },
            flight_enable=True,
        ),
    )
    assert "FLIGHT_ENABLE_HIGH" in evaluate(app_config, snapshot).codes


@pytest.mark.parametrize("case", ["missing", "ambiguous", "mismatch"])
def test_transform_rejects_invalid_or_inconsistent_raw_ch5(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    case: str,
) -> None:
    channels = dict(safe_walk_snapshot.rc.channels)
    if case == "missing":
        channels.pop(app_config.rc.flight_enable_channel, None)
    elif case == "ambiguous":
        channels[app_config.rc.flight_enable_channel] = app_config.rc.low_max + 1
    else:
        channels[app_config.rc.flight_enable_channel] = app_config.rc.high_min
    snapshot = replace(
        safe_walk_snapshot,
        rc=replace(
            safe_walk_snapshot.rc,
            channels=channels,
            flight_enable=False,
        ),
    )

    result = evaluate(app_config, snapshot)

    assert not result.permitted
    assert "FLIGHT_ENABLE_INVALID" in result.codes


def test_f446_fault_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(safe_walk_snapshot.f446, state=F446State.FAULT),
    )
    assert "F446_FAULT" in evaluate(app_config, snapshot).codes


@pytest.mark.parametrize("current", [None, float("nan"), -1])
def test_invalid_or_missing_f446_current_rejects_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    current: float,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(
            safe_walk_snapshot.f446,
            used_current_adc=current,
        ),
    )
    assert "F446_OVERCURRENT" in evaluate(app_config, snapshot).codes


def test_f446_current_equal_to_threshold_minus_margin_is_allowed(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    threshold = safe_walk_snapshot.f446.threshold_adc
    assert threshold is not None
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(
            safe_walk_snapshot.f446,
            used_current_adc=threshold - app_config.f446.current_safe_margin_adc,
        ),
    )
    assert "F446_OVERCURRENT" not in evaluate(app_config, snapshot).codes


def test_stale_f446_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(safe_walk_snapshot.f446, timestamp=1.0),
    )
    assert "F446_TIMEOUT" in evaluate(app_config, snapshot).codes


def test_stale_pixhawk_rejects_transform(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(safe_walk_snapshot.pixhawk, heartbeat_timestamp=1.0),
    )
    assert "PIXHAWK_TIMEOUT" in evaluate(app_config, snapshot).codes


def test_rc_failsafe_rejects_transform_and_request_is_not_retained(
    app_config: AppConfig, safe_walk_snapshot: SystemSnapshot
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        rc=replace(
            safe_walk_snapshot.rc,
            failsafe=True,
            morphology_request=MorphologyRequest.HOLD,
        ),
    )
    result = evaluate(app_config, snapshot)
    assert "RC_TIMEOUT" in result.codes
    assert "FLIGHT_REQUEST_NOT_ACTIVE" in result.codes
