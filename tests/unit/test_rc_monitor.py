from __future__ import annotations

from typing import Dict

import pytest

from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import AutoLandingRequest, MorphologyRequest, RCPosition


def safe_channels() -> Dict[int, int]:
    return {
        1: 1500,
        2: 1500,
        3: 1500,
        4: 1500,
        5: 1000,
        6: 1500,
        7: 1000,
        8: 1000,
        9: 1500,
        10: 1000,
        11: 1000,
        12: 1000,
    }


def stabilize(
    monitor: RCMonitor,
    clock: ManualClock,
    channels: Dict[int, int],
    debounce_s: float,
) -> None:
    monitor.update(channels)
    clock.advance(debounce_s)
    monitor.update(channels)


@pytest.mark.parametrize(
    "pwm,expected",
    [
        (800, RCPosition.LOW),
        (1200, RCPosition.LOW),
        (1300, RCPosition.MIDDLE),
        (1700, RCPosition.MIDDLE),
        (1800, RCPosition.HIGH),
        (2200, RCPosition.HIGH),
    ],
)
def test_threshold_boundaries_are_inclusive(
    app_config: AppConfig,
    clock: ManualClock,
    pwm: int,
    expected: RCPosition,
) -> None:
    assert RCMonitor(app_config.rc, clock).classify(pwm) is expected


@pytest.mark.parametrize("pwm", [0, 799, 1201, 1299, 1701, 1799, 2201])
def test_invalid_and_threshold_gap_values_are_unknown(
    app_config: AppConfig,
    clock: ManualClock,
    pwm: int,
) -> None:
    assert RCMonitor(app_config.rc, clock).classify(pwm) is RCPosition.UNKNOWN


def test_request_does_not_change_before_debounce(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 2000
    first = monitor.update(channels)
    clock.advance(app_config.rc.debounce_s - 0.01)
    second = monitor.update(channels)
    assert first.morphology_request is MorphologyRequest.HOLD
    assert second.morphology_request is MorphologyRequest.HOLD


def test_request_changes_at_stable_debounce_boundary(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 2000
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    assert monitor.get_status().morphology_request is MorphologyRequest.FLIGHT_REQUEST


def test_candidate_change_restarts_debounce_timer(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 2000
    monitor.update(channels)
    clock.advance(app_config.rc.debounce_s / 2)
    channels[9] = 1000
    monitor.update(channels)
    clock.advance(app_config.rc.debounce_s / 2)
    status = monitor.update(channels)
    assert status.morphology_request is MorphologyRequest.HOLD


def test_ch9_low_middle_high_mapping(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    expected = {
        1000: MorphologyRequest.WALK,
        1500: MorphologyRequest.HOLD,
        2000: MorphologyRequest.FLIGHT_REQUEST,
    }
    for pwm, request in expected.items():
        monitor = RCMonitor(app_config.rc, clock)
        channels = safe_channels()
        channels[9] = pwm
        stabilize(monitor, clock, channels, app_config.rc.debounce_s)
        assert monitor.get_status().morphology_request is request


def test_ch10_low_middle_high_mapping(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    expected = {
        1000: AutoLandingRequest.MANUAL,
        1500: AutoLandingRequest.AUTO_READY,
        2000: AutoLandingRequest.AUTO_EXECUTE,
    }
    for pwm, request in expected.items():
        monitor = RCMonitor(app_config.rc, clock)
        channels = safe_channels()
        channels[10] = pwm
        stabilize(monitor, clock, channels, app_config.rc.debounce_s)
        assert monitor.get_status().auto_landing_request is request


def test_flight_enable_requires_debounced_high(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[5] = 2000
    assert not monitor.update(channels).flight_enable
    clock.advance(app_config.rc.debounce_s)
    assert monitor.update(channels).flight_enable


def test_rc_timeout_restores_all_high_level_requests_to_safe_values(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels.update({5: 2000, 9: 2000, 10: 2000})
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    assert monitor.get_status().flight_enable
    clock.advance(app_config.rc.timeout_s)
    status = monitor.get_status()
    assert status.failsafe
    assert not status.connected
    assert not status.flight_enable
    assert status.morphology_request is MorphologyRequest.HOLD
    assert status.auto_landing_request is AutoLandingRequest.MANUAL


def test_explicit_failsafe_immediately_discards_latched_requests(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels.update({5: 2000, 9: 2000, 10: 2000})
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    status = monitor.update(channels, failsafe=True)
    assert status.failsafe
    assert not status.flight_enable
    assert status.morphology_request is MorphologyRequest.HOLD
    assert status.auto_landing_request is AutoLandingRequest.MANUAL


def test_recovery_from_failsafe_requires_fresh_debounce(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 2000
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    monitor.update(channels, failsafe=True)
    recovered = monitor.update(channels, failsafe=False)
    assert recovered.morphology_request is MorphologyRequest.HOLD


def test_ambiguous_pwm_immediately_removes_flight_request(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 2000
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    channels[9] = 1750
    status = monitor.update(channels)
    assert status.morphology_request is MorphologyRequest.HOLD
    assert monitor.position(9) is RCPosition.UNKNOWN


def test_raw_channels_remain_visible(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[9] = 1876
    monitor.update(channels)
    assert monitor.raw_channels[9] == 1876


def test_ch10_manual_position_requests_manual_override(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    assert monitor.get_status().manual_override


def test_centered_sticks_do_not_request_stick_override_in_auto(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[10] = 2000
    stabilize(monitor, clock, channels, app_config.rc.debounce_s)
    assert not monitor.get_status().manual_override


@pytest.mark.parametrize("channel", [1, 2, 3, 4])
def test_each_stick_axis_offset_requests_manual_override(
    app_config: AppConfig,
    clock: ManualClock,
    channel: int,
) -> None:
    monitor = RCMonitor(app_config.rc, clock)
    channels = safe_channels()
    channels[10] = 2000
    channels[channel] = 1500 + app_config.rc.manual_override_deadband_us + 1
    status = monitor.update(channels)
    assert status.manual_override


def test_explicit_airframe_stick_calibration_overrides_simulation_defaults(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    monitor = RCMonitor(
        app_config.rc,
        clock,
        stick_neutral_us={1: 1510, 2: 1490, 3: 1000, 4: 1520},
    )
    channels = safe_channels()
    channels.update({1: 1510, 2: 1490, 3: 1000, 4: 1520, 10: 2000})

    assert not monitor.update(channels).manual_override
