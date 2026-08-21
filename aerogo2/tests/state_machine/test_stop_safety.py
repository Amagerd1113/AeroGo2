from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import F446State, SystemState
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.asyncio
async def test_stop_supervised_stops_owned_motion_without_disarming_pixhawk(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        started = await world.start()
        assert started.ok
        assert world.manager.state is SystemState.WALK

        # Arrange active supervised outputs and an independently armed Pixhawk.
        world.pixhawk.inject_armed_state(True)
        setpoint = await world.pixhawk.send_velocity_setpoint(0.1, -0.1, 0.2, 0.0)
        assert setpoint.ok
        world.f446.inject_status(state=F446State.LIMIT_FWD, duty=120)
        world.go2.inject_motion(0.4, stable=False)
        await world.manager.refresh_snapshot()

        assert world.manager.snapshot.pixhawk.armed
        assert world.pixhawk.external_setpoints_active
        assert world.manager.snapshot.f446.duty == 120
        assert world.manager.snapshot.go2.controller_active

        result = await world.manager.stop_supervised()

        assert result.ok
        assert "Rotor shutdown is not performed" in result.message
        assert "RadioMaster" in result.message
        assert not world.pixhawk.external_setpoints_active
        assert world.f446.get_status().duty == 0
        assert world.f446.command_history[-1].command == "stop"
        assert world.go2.get_status().velocity_mps == 0.0
        assert world.go2.get_status().stable
        assert not world.go2.get_status().controller_active

        # The manager owns no arm/disarm operation: stopping must preserve the
        # independent RadioMaster-controlled armed telemetry bit.
        assert world.pixhawk.get_status().armed
        assert world.manager.snapshot.pixhawk.armed
        assert world.manager.state is SystemState.WALK
    finally:
        await world.shutdown()
