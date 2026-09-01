from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State, SystemState
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.asyncio
async def test_nominal_transition_history_contains_every_guarded_stage(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        result = await world.run_scenario("nominal")
        assert result.ok
        history = world.manager._state_machine.history
        assert tuple(record.new_state for record in history) == (
            SystemState.WALK,
            SystemState.WALK_TO_FLIGHT_PRECHECK,
            SystemState.TRANSFORM_TO_FLIGHT,
            SystemState.GO2_JOINT_LOCK_WAIT,
            SystemState.FLIGHT_READY,
            SystemState.FLIGHT_MANUAL,
            SystemState.AUTO_LANDING_READY,
            SystemState.AUTO_LANDING,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.FLIGHT_TO_WALK_PRECHECK,
            SystemState.TRANSFORM_TO_WALK,
            SystemState.WALK,
        )
        assert all(record.permitted for record in history)
        assert all(record.entry_action_error is None for record in history)
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_unknown_f446_can_home_to_verified_walk(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    world.f446._configuration = Configuration.UNKNOWN
    world.f446.inject_status(state=F446State.IDLE, duty=0)

    try:
        started = await world.start()
        assert started.ok
        assert world.manager.state is SystemState.BOOT_SAFE
        assert world.manager.snapshot.configuration is Configuration.UNKNOWN

        assert await world.go2.request_stop()
        await world.manager.refresh_snapshot()
        world.clock.advance(
            max(
                app_config.safety.stationary_confirm_s,
                app_config.f446.current_clear_hold_s,
            )
            + 0.01
        )
        world._heartbeat_all()
        world._feed_rc(debounce=False)
        for slot in app_config.esc.slots:
            world.pixhawk.inject_esc_rpm(slot, 0.0)
        await world.manager.refresh_snapshot()

        preflight = await world.manager.preflight("home-walk")
        assert preflight.ok

        result = await world.manager.request_home_walk(operator_confirmed=True)

        assert result.ok
        assert result.code == "F446_HOME_WALK_VERIFIED"
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.configuration is Configuration.WALK
        assert world.manager.snapshot.f446.state is app_config.f446.expected_walk_state
        assert world.manager.snapshot.f446.duty == 0
        assert any(record.command.startswith("limr ") for record in world.f446.command_history)
    finally:
        await world.shutdown()
