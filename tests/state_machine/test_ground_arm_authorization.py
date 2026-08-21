from __future__ import annotations

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import CommandStatus, SystemState
from aerogo2.manager.command_service import CommandService
from aerogo2.simulation.world import SimulationWorld


async def _reach_flight_ready(world: SimulationWorld) -> None:
    assert (await world.start()).ok
    world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
    assert (await world._settle_for_transform()).ok
    assert (await world.manager.request_transform_flight(operator_confirmed=True)).ok
    assert world.manager.state is SystemState.FLIGHT_READY


@pytest.mark.asyncio
async def test_ground_authorization_requires_flight_ready(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok

        result = await world.manager.authorize_ground_arm()

        assert not result.ok
        assert result.code == "NOT_IN_FLIGHT_READY"
        assert not world.manager.snapshot.ground_arm_authorized
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_offline_esc_rejects_ground_authorization(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_flight_ready(world)
        world.pixhawk.inject_esc_rpm(2, 0.0, healthy=False)
        await world.manager.refresh_snapshot()

        result = await world.manager.authorize_ground_arm()

        assert not result.ok
        assert result.code == "ESC_TELEMETRY_UNSAFE"
        assert not world.manager.snapshot.ground_arm_authorized
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_ground_authorization_expires_after_thirty_seconds(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_flight_ready(world)
        result = await world.manager.authorize_ground_arm()
        assert result.ok
        assert world.manager.snapshot.ground_arm_authorized

        world.clock.advance(30.01)
        world._heartbeat_all()
        await world.manager.refresh_snapshot()

        assert not world.manager.snapshot.ground_arm_authorized
        assert world.manager.query("flight auth-status")["remaining_s"] == 0.0
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_command_service_routes_authorize_and_revoke(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_flight_ready(world)
        service = CommandService(world.manager)

        authorized = await service.run("ground_arm_authorize", ())
        assert authorized.status is CommandStatus.SUCCESS
        assert world.manager.snapshot.ground_arm_authorized

        revoked = await service.run("ground_arm_revoke", ())
        assert revoked.status is CommandStatus.SUCCESS
        assert not world.pixhawk.ground_arm_authorization_active()
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_authorized_radio_arm_consumes_one_shot_gate(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_flight_ready(world)
        assert (await world.manager.authorize_ground_arm()).ok

        world._set_switches(morphology=1900, autoland=1000, flight_enable=1900)
        world.pixhawk.inject_armed_state(True)
        await world.manager.tick()

        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert world.manager.snapshot.pixhawk.armed
        assert not world.manager.snapshot.ground_arm_authorized
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_unauthorized_radio_arm_fails_closed_without_auto_disarm(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await _reach_flight_ready(world)
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1900)
        world.pixhawk.inject_armed_state(True)

        violations = await world.manager.tick()

        assert any(item.code == "UNAUTHORIZED_PIXHAWK_ARM" for item in violations)
        assert world.manager.state is SystemState.FAULT
        assert world.manager.snapshot.pixhawk.armed
    finally:
        await world.shutdown()
