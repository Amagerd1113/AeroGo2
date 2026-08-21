from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.simulation.world import SimulationWorld

NONFINITE = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("value", NONFINITE)
def test_manual_clock_rejects_nonfinite_initial_time(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ManualClock(value)


@pytest.mark.parametrize("value", NONFINITE)
def test_manual_clock_rejects_nonfinite_wall_time(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ManualClock(10.0, wall_initial=value)


@pytest.mark.parametrize("value", NONFINITE)
def test_manual_clock_rejects_nonfinite_advance_without_mutation(value: float) -> None:
    clock = ManualClock(10.0, wall_initial=2_000_000_000.0)
    with pytest.raises(ValueError, match="finite"):
        clock.advance(value)
    assert clock.monotonic() == 10.0
    assert clock.wall_time() == 2_000_000_000.0


def test_manual_clock_rejects_finite_overflow_without_mutation() -> None:
    clock = ManualClock(1e308, wall_initial=1e308)

    with pytest.raises(ValueError, match="keep both clocks finite"):
        clock.advance(1e308)

    assert clock.monotonic() == 1e308
    assert clock.wall_time() == 1e308


@pytest.mark.parametrize("value", NONFINITE)
@pytest.mark.asyncio
async def test_simulation_step_rejects_nonfinite_duration_without_advancing(
    app_config: AppConfig,
    value: float,
) -> None:
    world = SimulationWorld(app_config)
    before = world.clock.monotonic()

    result = await world.step(value)

    assert not result.ok
    assert result.code == "INVALID_STEP"
    assert world.clock.monotonic() == before


@pytest.mark.parametrize("value", NONFINITE)
def test_normal_fake_motion_helpers_reject_nonfinite_values(value: float) -> None:
    go2 = FakeGo2(clock=ManualClock(10.0))
    pixhawk = FakePixhawk(clock=ManualClock(10.0))

    with pytest.raises(ValueError, match="finite"):
        go2.inject_motion(value)
    with pytest.raises(ValueError, match="finite"):
        pixhawk.inject_attitude(value, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        pixhawk.inject_landed_state(False, vertical_velocity_mps=value)
    with pytest.raises(ValueError, match="finite"):
        pixhawk.inject_landed_state(False, relative_altitude_m=value)


@pytest.mark.parametrize("value", NONFINITE)
def test_rc_monitor_rejects_nonfinite_timestamp(
    app_config: AppConfig,
    value: float,
) -> None:
    monitor = RCMonitor(app_config.rc, ManualClock(10.0))
    with pytest.raises(ValueError, match="finite"):
        monitor.update({}, timestamp=value)


def test_low_level_inject_status_remains_available_for_fault_injection() -> None:
    go2 = FakeGo2(clock=ManualClock(10.0))
    pixhawk = FakePixhawk(clock=ManualClock(10.0))

    assert math.isnan(go2.inject_status(velocity_mps=float("nan")).velocity_mps)
    assert math.isnan(pixhawk.inject_status(roll_rad=float("nan")).attitude_rpy[0])


def test_packaged_default_configs_match_project_configs(project_root: Path) -> None:
    packaged = Path(__file__).resolve().parents[2] / "src" / "aerogo2" / "default_configs"
    source = project_root / "configs"
    names = (
        "system.yaml",
        "serial.yaml",
        "rc_channels.yaml",
        "safety_limits.yaml",
        "f446.yaml",
        "landing.yaml",
    )

    for name in names:
        assert yaml.safe_load((packaged / name).read_text(encoding="utf-8")) == yaml.safe_load(
            (source / name).read_text(encoding="utf-8")
        )
