from __future__ import annotations

from typing import List

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.common.models import LandingEstimate
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.asyncio
async def test_manual_landing_reaches_touchdown_verify_without_auto_disarm(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        states: List[SystemState] = []
        assert (await world._reach_flight_manual(states)).ok
        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert world.pixhawk.get_status().armed

        world._set_landing_estimate(
            LandingEstimate(
                valid=False,
                ground_detected=False,
                height_m=None,
                timestamp=world.clock.monotonic(),
                reason="manual landing does not depend on Ubuntu ground perception",
            )
        )
        world.pixhawk.inject_landed_state(
            True,
            vertical_velocity_mps=0.0,
            relative_altitude_m=0.0,
        )
        world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
        for slot in app_config.esc.slots:
            world.pixhawk.inject_esc_rpm(slot, 0.0)

        steps = int(app_config.safety.touchdown_confirm_s / 0.05) + 3
        for _ in range(steps):
            await world.step(0.05)
            if world.manager.state is SystemState.TOUCHDOWN_VERIFY:
                break

        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
        assert world.manager.snapshot.pixhawk.armed
        assert not world.manager.snapshot.external_setpoint_active
    finally:
        await world.shutdown()
