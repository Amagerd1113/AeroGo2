from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.asyncio
async def test_transform_timeout_latches_fault_until_explicit_clear(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        scenario = await world.run_scenario("transform-failure")
        assert scenario.ok
        assert world.manager.state is SystemState.FAULT
        assert "F446_TRANSFORM_TIMEOUT" in world.manager.snapshot.active_fault_codes
        assert tuple(record.command for record in world.f446.command_history)[-1] == "stop"

        cleared = await world.manager.clear_fault()
        assert cleared.ok
        assert world.manager.state is SystemState.BOOT_SAFE
        assert world.manager.snapshot.active_fault_codes == ()
        assert "clear" not in tuple(record.command for record in world.f446.command_history)
    finally:
        await world.shutdown()
