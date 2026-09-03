from __future__ import annotations

from typing import List

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, SystemState
from aerogo2.common.models import LandingEstimate
from aerogo2.simulation.world import SimulationWorld


async def _reach_touchdown_verify(world: SimulationWorld) -> None:
    states: List[SystemState] = []
    assert (await world.start()).ok
    assert (await world._reach_flight_manual(states)).ok
    world._set_landing_estimate(
        LandingEstimate(
            valid=False,
            ground_detected=False,
            height_m=None,
            timestamp=world.clock.monotonic(),
            reason="manual landing",
        )
    )
    world.pixhawk.inject_landed_state(
        True,
        vertical_velocity_mps=0.0,
        relative_altitude_m=0.0,
    )
    world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
    for slot in world.config.esc.slots:
        world.pixhawk.inject_esc_rpm(slot, 0.0)
    steps = int(world.config.safety.touchdown_confirm_s / 0.05) + 3
    for _ in range(steps):
        await world.step(0.05)
        if world.manager.state is SystemState.TOUCHDOWN_VERIFY:
            break
    assert world.manager.state is SystemState.TOUCHDOWN_VERIFY


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


@pytest.mark.asyncio
async def test_ground_wait_cannot_trigger_touchdown_before_confirmed_airborne_phase(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        assert (await world._settle_for_transform()).ok
        assert (
            await world.manager.request_transform_flight(operator_confirmed=True)
        ).ok
        assert (await world.manager.authorize_ground_arm()).ok

        world._set_switches(morphology=1900, autoland=1000, flight_enable=1900)
        world.pixhawk.inject_armed_state(True)
        world.pixhawk.inject_landed_state(
            True,
            vertical_velocity_mps=0.0,
            relative_altitude_m=0.0,
        )
        world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
        for slot in app_config.esc.slots:
            world.pixhawk.inject_esc_rpm(slot, 0.0)

        await world.manager.tick()
        assert world.manager.state is SystemState.FLIGHT_MANUAL

        wait_s = app_config.safety.airborne_confirm_s + app_config.safety.touchdown_confirm_s
        for _ in range(int(wait_s / 0.05) + 5):
            await world.step(0.05)

        report = world.manager.query("touchdown status")
        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert report["airborne_confirmed"] is False
        assert report["touchdown_detection_enabled"] is False
        assert report["touchdown_candidate_elapsed_s"] == 0.0
        assert report["pixhawk_armed"] is True
        assert report["pixhawk_landed"] is True
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_touchdown_can_use_guarded_manual_positioning_to_reach_walk(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_touchdown_verify(world)
        world.pixhawk.inject_armed_state(False)
        world._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
        await world.step(
            max(
                world.config.safety.stationary_confirm_s,
                world.config.f446.current_clear_hold_s,
            )
            + 0.01
        )

        entered = await world.manager.enter_manual_positioning(operator_confirmed=True)
        assert entered.ok
        assert entered.data["entry_state"] == "TOUCHDOWN_VERIFY"
        assert entered.data["post_touchdown_recovery"] is True
        assert world.manager.state is SystemState.MANUAL_POSITIONING

        assert (await world.manager.start_f446_maintenance_motion("mr", 500)).ok
        assert (await world.manager.stop_transform_motion()).ok
        await world.step(
            max(
                world.config.safety.stationary_confirm_s,
                world.config.f446.current_clear_hold_s,
            )
            + 0.01
        )
        assert (
            await world.manager.mark_manual_configuration(
                Configuration.WALK,
                operator_confirmed=True,
            )
        ).ok
        await world.step(
            max(
                world.config.safety.stationary_confirm_s,
                world.config.f446.current_clear_hold_s,
            )
            + 0.01
        )
        confirmed = await world.manager.confirm_manual_configuration(
            Configuration.WALK,
            operator_confirmed=True,
        )

        assert confirmed.ok
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.configuration is Configuration.WALK
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_touchdown_manual_positioning_rejects_armed_pixhawk(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_touchdown_verify(world)

        rejected = await world.manager.enter_manual_positioning(operator_confirmed=True)

        assert not rejected.ok
        assert "Pixhawk is armed" in rejected.message
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_touchdown_manual_positioning_requires_landed_state(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_touchdown_verify(world)
        world.pixhawk.inject_armed_state(False)
        world.pixhawk.inject_landed_state(False)
        world._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
        await world.step(
            max(
                world.config.safety.stationary_confirm_s,
                world.config.f446.current_clear_hold_s,
            )
            + 0.01
        )

        rejected = await world.manager.enter_manual_positioning(operator_confirmed=True)

        assert not rejected.ok
        assert "Post-touchdown manual positioning requires Pixhawk landed=true" in rejected.message
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
    finally:
        await world.shutdown()
