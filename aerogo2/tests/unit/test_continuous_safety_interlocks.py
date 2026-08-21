from __future__ import annotations

from dataclasses import replace
from typing import Set

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State, SystemState
from aerogo2.common.models import SystemSnapshot
from aerogo2.safety.interlocks import SafetyInterlocks
from aerogo2.safety.safety_monitor import SafetyMonitor


def _codes(config: AppConfig, snapshot: SystemSnapshot) -> Set[str]:
    return {item.code for item in SafetyMonitor(config).evaluate(snapshot)}


def _moving_state(config: AppConfig, target: Configuration) -> F446State:
    direction = config.f446.direction_for(target.value)
    return F446State.LIMIT_FWD if direction == "forward" else F446State.LIMIT_REV


def _active_transform(
    config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> SystemSnapshot:
    return replace(
        safe_walk_snapshot,
        state=SystemState.TRANSFORM_TO_FLIGHT,
        configuration=Configuration.UNKNOWN,
        f446=replace(
            safe_walk_snapshot.f446,
            state=_moving_state(config, Configuration.FLIGHT),
            duty=config.f446.flight_duty,
        ),
    )


def test_nonfinite_f446_current_fails_closed(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(
            safe_walk_snapshot.f446,
            used_current_adc=float("nan"),
        ),
    )

    assert "F446_OVERCURRENT" in _codes(app_config, snapshot)


@pytest.mark.parametrize("field", ["failsafe", "rc_failsafe"])
def test_pixhawk_failsafe_stops_active_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    field: str,
) -> None:
    transform = _active_transform(app_config, safe_walk_snapshot)
    snapshot = replace(
        transform,
        pixhawk=replace(transform.pixhawk, **{field: True}),
    )

    assert "PIXHAWK_FAILSAFE" in _codes(app_config, snapshot)


def test_flight_enable_high_stops_active_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    transform = _active_transform(app_config, safe_walk_snapshot)
    snapshot = replace(
        transform,
        rc=replace(
            transform.rc,
            channels={
                **transform.rc.channels,
                app_config.rc.flight_enable_channel: 1900,
            },
            flight_enable=True,
        ),
    )

    assert "FLIGHT_ENABLE_HIGH" in _codes(app_config, snapshot)


def test_flight_readiness_requires_fresh_go2_even_when_go2_config_is_disabled(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    disabled_config = replace(
        app_config,
        go2=replace(app_config.go2, enabled=False),
    )
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.FLIGHT_READY,
        configuration=Configuration.FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=app_config.f446.expected_flight_state,
        ),
        go2=replace(
            safe_walk_snapshot.go2,
            connected=False,
            timestamp=0.0,
        ),
    )

    result = SafetyInterlocks(disabled_config).can_enter_flight_ready(snapshot)

    assert not result.permitted
    assert "GO2_TIMEOUT" in result.codes


def test_malformed_esc_types_fail_closed_during_active_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    transform = _active_transform(app_config, safe_walk_snapshot)
    slot = min(app_config.esc.slots)
    tuple_items = tuple(
        replace(item, rpm="0", healthy=1) if item.slot == slot else item
        for item in transform.pixhawk.esc
    )
    rpm_by_slot = dict(transform.pixhawk.esc_rpm)
    online_by_slot = dict(transform.pixhawk.esc_online)
    rpm_by_slot[slot] = "0"
    online_by_slot[slot] = 1
    snapshot = replace(
        transform,
        pixhawk=replace(
            transform.pixhawk,
            esc=tuple_items,
            esc_rpm=rpm_by_slot,
            esc_online=online_by_slot,
        ),
    )

    assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in _codes(app_config, snapshot)


def test_unknown_configuration_is_a_conflict_outside_transform(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        configuration=Configuration.UNKNOWN,
    )

    assert "F446_CONFIGURATION_CONFLICT" in _codes(app_config, snapshot)


def test_expected_active_transform_duty_and_unknown_configuration_are_allowed(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = _active_transform(app_config, safe_walk_snapshot)

    assert "F446_CONFIGURATION_CONFLICT" not in _codes(app_config, snapshot)


def test_wrong_f446_direction_during_active_transform_is_a_conflict(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    transform = _active_transform(app_config, safe_walk_snapshot)
    wrong_state = (
        F446State.LIMIT_REV if transform.f446.state is F446State.LIMIT_FWD else F446State.LIMIT_FWD
    )
    snapshot = replace(
        transform,
        f446=replace(transform.f446, state=wrong_state),
    )

    assert "F446_CONFIGURATION_CONFLICT" in _codes(app_config, snapshot)
