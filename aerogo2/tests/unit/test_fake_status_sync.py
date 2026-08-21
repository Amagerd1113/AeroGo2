from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aerogo2.app import AeroGo2Application
from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig


def test_manual_clock_accepts_wall_epoch_and_advances_both_views() -> None:
    clock = ManualClock(10.0, wall_initial=2_000_000_000.0)
    clock.advance(0.25)
    assert clock.monotonic() == 10.25
    assert clock.wall_time() == 2_000_000_000.25


def test_application_shares_one_clock_with_world_and_event_sink(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    config = replace(
        app_config,
        system=replace(app_config.system, log_directory=tmp_path),
    )
    app = AeroGo2Application(config)
    try:
        app.clock.advance(0.5)
        record = app.event_sink.emit(event_type="CLOCK_TEST", system_state="BOOT_SAFE")
        assert app.world.clock is app.clock
        assert record["monotonic_timestamp"] == app.world.clock.monotonic()
    finally:
        app.event_sink.stop()


@pytest.mark.asyncio
async def test_fake_pixhawk_keeps_public_status_fields_synchronized() -> None:
    clock = ManualClock(10.0)
    pixhawk = FakePixhawk(clock=clock)
    await pixhawk.connect()
    pixhawk.inject_attitude(0.1, -0.2, 0.3)
    pixhawk.inject_landed_state(
        False,
        vertical_velocity_mps=-0.4,
        relative_altitude_m=1.5,
    )
    status = pixhawk.inject_esc_rpm(1, 123.0, healthy=False)

    assert status.timestamp == status.heartbeat_timestamp == clock.monotonic()
    assert status.message_age_s == 0.0
    assert status.attitude_rpy == (0.1, -0.2, 0.3)
    assert status.local_position[2] == -1.5
    assert status.local_velocity[2] == -0.4
    assert status.esc_rpm[1] == 123.0
    assert status.esc_online[1] is False

    status = pixhawk.inject_failsafe(True)
    assert status.failsafe is True
    assert status.rc_failsafe is True

    status = pixhawk.inject_status(
        esc_rpm={1: 45.0},
        esc_online={1: True},
    )
    assert status.esc[0].rpm == 45.0
    assert status.esc[0].healthy is True


@pytest.mark.asyncio
async def test_fake_go2_keeps_motion_and_mode_views_synchronized() -> None:
    clock = ManualClock(10.0)
    go2 = FakeGo2(clock=clock)
    await go2.connect()

    moving = go2.inject_motion(0.2, stable=False)
    assert moving.body_velocity == (0.2, 0.0, 0.0)
    assert moving.velocity_mps == 0.2
    assert moving.moving is True
    assert moving.locomotion_mode == "WALK"
    assert moving.message_age_s == 0.0

    assert await go2.request_stop()
    stopped = go2.get_status()
    assert stopped.body_velocity == (0.0, 0.0, 0.0)
    assert stopped.moving is False
    assert stopped.locomotion_mode == "STOPPED"

    vector_status = go2.inject_status(body_velocity=(0.1, -0.2, 0.0))
    assert vector_status.body_velocity == (0.1, -0.2, 0.0)
    assert vector_status.velocity_mps == pytest.approx(5**0.5 / 10)
    assert vector_status.moving is True
