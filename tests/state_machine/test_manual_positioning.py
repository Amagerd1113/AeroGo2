from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State, SystemState
from aerogo2.simulation.world import SimulationWorld


async def _settle_interlocks(world: SimulationWorld) -> None:
    await world.step(
        max(
            world.config.safety.stationary_confirm_s,
            world.config.f446.current_clear_hold_s,
        )
        + 0.01
    )


@pytest.mark.asyncio
async def test_manual_reverse_stop_and_operator_walk_confirmation(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        await _settle_interlocks(world)
        assert (await world.manager.enter_manual_positioning(operator_confirmed=True)).ok
        assert world.manager.state is SystemState.MANUAL_POSITIONING

        threshold = await world.manager.set_f446_current_threshold(1200)
        assert threshold.ok
        await _settle_interlocks(world)

        reverse = await world.manager.start_f446_maintenance_motion("mr", 500)
        assert reverse.ok
        assert reverse.data["signed_duty"] == -500
        assert world.manager.snapshot.f446.state is F446State.MANUAL_REV
        assert world.manager.snapshot.f446.duty == -500
        assert any(record.command == "mr 500" for record in world.f446.command_history)

        stopped = await world.manager.stop_supervised()
        assert stopped.ok
        assert world.manager.state is SystemState.MANUAL_POSITIONING
        assert world.manager.snapshot.f446.state is F446State.IDLE
        assert world.manager.snapshot.f446.duty == 0

        await _settle_interlocks(world)
        wrong_target = await world.manager.confirm_manual_configuration(
            Configuration.FLIGHT,
            operator_confirmed=True,
        )
        assert not wrong_target.ok
        assert wrong_target.code == "F446_DIRECTION_TARGET_MISMATCH"

        confirmed = await world.manager.confirm_manual_configuration(
            Configuration.WALK,
            operator_confirmed=True,
        )
        assert confirmed.ok
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.configuration is Configuration.WALK
        assert world.manager.snapshot.configuration_source == "operator"
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_manual_forward_stop_and_operator_flight_confirmation(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        await _settle_interlocks(world)
        assert (await world.manager.enter_manual_positioning(operator_confirmed=True)).ok

        forward = await world.manager.start_f446_maintenance_motion("mf", 500)
        assert forward.ok
        assert forward.data["signed_duty"] == 500
        assert world.manager.snapshot.f446.state is F446State.MANUAL_FWD
        assert world.manager.snapshot.f446.duty == 500

        assert (await world.manager.stop_transform_motion()).ok
        assert world.manager.state is SystemState.MANUAL_POSITIONING
        await _settle_interlocks(world)

        confirmed = await world.manager.confirm_manual_configuration(
            Configuration.FLIGHT,
            operator_confirmed=True,
        )
        assert confirmed.ok
        assert world.manager.state is SystemState.FLIGHT_READY
        assert world.manager.snapshot.configuration is Configuration.FLIGHT
        assert world.manager.snapshot.configuration_source == "operator"
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_limit_motion_requires_safe_threshold_and_accepts_limit_stop(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        await _settle_interlocks(world)
        assert (await world.manager.enter_manual_positioning(operator_confirmed=True)).ok

        unsafe = await world.manager.start_f446_maintenance_motion("limr", 500)
        assert not unsafe.ok
        assert unsafe.code == "F446_LIMIT_THRESHOLD_UNSAFE"

        assert (await world.manager.set_f446_current_threshold(1200)).ok
        await _settle_interlocks(world)
        automatic = await world.manager.start_f446_maintenance_motion("limr", 500)
        assert automatic.ok
        assert automatic.data["automatic_limit_stop"] is True
        assert world.manager.snapshot.f446.state is F446State.LIMIT_REV
        assert world.manager.snapshot.f446.duty == -500

        world.f446.inject_status(
            state=F446State.LIMIT_REACHED_REV,
            duty=0,
            used_current_adc=0,
            used_raw=0,
            used_mv=0,
        )
        await world.manager.refresh_snapshot()
        await _settle_interlocks(world)
        confirmed = await world.manager.confirm_manual_configuration(
            Configuration.WALK,
            operator_confirmed=True,
        )

        assert confirmed.ok
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.configuration_source == "f446_limit"
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_manual_motion_host_timeout_stops_without_leaving_session(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        await _settle_interlocks(world)
        assert (await world.manager.enter_manual_positioning(operator_confirmed=True)).ok
        assert (await world.manager.start_f446_maintenance_motion("mf", 200)).ok

        await world.step(app_config.f446.transform_timeout_s + 0.01)

        assert world.manager.state is SystemState.MANUAL_POSITIONING
        assert world.manager.snapshot.f446.state is F446State.IDLE
        assert world.manager.snapshot.f446.duty == 0
        assert world.f446.command_history[-1].command == "stop"
    finally:
        await world.shutdown()
