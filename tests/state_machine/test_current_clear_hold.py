from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.simulation.world import SimulationWorld


def _movement_commands(world: SimulationWorld) -> tuple[str, ...]:
    return tuple(
        record.command
        for record in world.f446.command_history
        if record.command.startswith(("limf ", "limr "))
    )


@pytest.mark.asyncio
async def test_transform_waits_for_configured_f446_current_clear_hold(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)

        first = await world.manager.request_transform_flight(operator_confirmed=True)

        assert not first.ok
        assert first.code == "F446_CURRENT_CLEAR_HOLD_REQUIRED"
        assert world.manager.state is SystemState.WALK
        assert _movement_commands(world) == ()

        await world.step(
            max(
                app_config.f446.current_clear_hold_s,
                app_config.safety.stationary_confirm_s,
            )
            + 0.001
        )
        second = await world.manager.request_transform_flight(operator_confirmed=True)
        assert second.ok
        assert world.manager.state is SystemState.FLIGHT_READY
        assert len(_movement_commands(world)) == 1
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_current_violation_resets_clear_hold_without_actuator_motion(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        await world.step(
            max(
                app_config.f446.current_clear_hold_s,
                app_config.safety.stationary_confirm_s,
            )
            + 0.001
        )
        threshold = world.f446.get_status().threshold_adc
        assert threshold is not None
        world.f446.inject_status(
            used_current_adc=threshold - app_config.f446.current_safe_margin_adc + 1
        )
        await world.manager.refresh_snapshot()
        world.f446.inject_status(used_current_adc=0)
        await world.manager.refresh_snapshot()
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)

        result = await world.manager.request_transform_flight(operator_confirmed=True)

        assert not result.ok
        assert result.code == "F446_CURRENT_CLEAR_HOLD_REQUIRED"
        assert world.manager.state is SystemState.WALK
        assert _movement_commands(world) == ()
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_missing_threshold_fails_preflight_closed(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        world.f446.inject_status(threshold_adc=None)

        result = await world.manager.preflight("transform-flight")

        assert not result.ok
        assert result.code == "PREFLIGHT_FAILED"
        assert any(
            check["code"] in {"F446_OVERCURRENT", "F446_CURRENT_MARGIN_UNSAFE"}
            for check in result.data["checks"]
            if not check["passed"]
        )
        assert _movement_commands(world) == ()
    finally:
        await world.shutdown()
