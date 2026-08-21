from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.asyncio
async def test_nominal_dry_run_completes_full_mission(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        result = await world.run_scenario("nominal")
        assert result.ok
        assert result.states == (
            SystemState.BOOT_SAFE,
            SystemState.WALK,
            SystemState.WALK_TO_FLIGHT_PRECHECK,
            SystemState.TRANSFORM_TO_FLIGHT,
            SystemState.FLIGHT_READY,
            SystemState.FLIGHT_MANUAL,
            SystemState.AUTO_LANDING_READY,
            SystemState.AUTO_LANDING,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.FLIGHT_TO_WALK_PRECHECK,
            SystemState.TRANSFORM_TO_WALK,
            SystemState.WALK,
        )
        assert result.final_state is SystemState.WALK
        assert result.details["setpoints"] > 0
        assert world.pixhawk.get_status().armed is False
        assert world.pixhawk.external_setpoints_active is False
    finally:
        await world.shutdown()


@pytest.mark.parametrize(
    ("scenario", "expected_state"),
    [
        ("transform-failure", SystemState.FAULT),
        ("rc-loss", SystemState.FLIGHT_MANUAL),
        ("pixhawk-timeout", SystemState.FAULT),
        ("f446-overcurrent", SystemState.FAULT),
        ("landing", SystemState.FLIGHT_MANUAL),
    ],
)
@pytest.mark.asyncio
async def test_fault_and_override_scenarios_fail_safe(
    app_config: AppConfig,
    scenario: str,
    expected_state: SystemState,
) -> None:
    world = SimulationWorld(app_config)
    try:
        result = await world.run_scenario(scenario)
        assert result.ok
        assert result.final_state is expected_state
        assert world.pixhawk.external_setpoints_active is False
        if world.pixhawk.get_status().armed:
            assert expected_state is SystemState.FLIGHT_MANUAL
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_unknown_scenario_never_mutates_the_running_state(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        initial = world.manager.state
        result = await world.run_scenario("not-a-scenario")
        assert not result.ok
        assert result.states == (initial,)
        assert world.manager.state is initial
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_overcurrent_scenario_reports_specific_safety_code(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        result = await world.run_scenario("f446-overcurrent")
        assert result.ok
        assert result.messages[0].startswith("F446_OVERCURRENT:")
        assert "F446_OVERCURRENT" in world.manager.snapshot.active_fault_codes
        assert world.manager.snapshot.f446.over_active
    finally:
        await world.shutdown()
