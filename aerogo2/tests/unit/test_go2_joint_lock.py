from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from aerogo2.bridges.go2_sdk_bridge import UnitreeGo2Bridge
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, SafetySeverity, SystemState
from aerogo2.common.models import SystemSnapshot
from aerogo2.safety.safety_monitor import SafetyMonitor
from aerogo2.simulation.world import SimulationWorld


def _joint_lock_message() -> SimpleNamespace:
    return SimpleNamespace(
        velocity=(0.0, 0.0, 0.0),
        imu_state=SimpleNamespace(rpy=(0.0, 0.0, 0.0)),
        mode=6,
        error_code=100,
    )


def _balance_stand_message() -> SimpleNamespace:
    return SimpleNamespace(
        velocity=(0.0, 0.0, 0.0),
        imu_state=SimpleNamespace(rpy=(0.0, 0.0, 0.0)),
        mode=1,
        error_code=100,
        foot_force=(101, 202, 303, 404),
    )


def test_go2_mode_six_is_authoritative_joint_lock(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    bridge._on_state(_joint_lock_message())
    bridge._connected = True

    status = bridge.get_status()

    assert status.joints_locked
    assert status.locomotion_mode == "JOINT_LOCK"
    assert status.standing
    assert status.stable


@pytest.mark.asyncio
async def test_flight_pose_disables_unitree_joystick_then_confirms_lock(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock, allow_control=True)
    calls: list[object] = []

    class Client:
        def StopMove(self) -> int:
            calls.append("StopMove")
            return 0

        def SwitchJoystick(self, enabled: bool) -> int:
            calls.append(("SwitchJoystick", enabled))
            return 0

        def StandUp(self) -> int:
            calls.append("StandUp")
            bridge._on_state(_joint_lock_message())
            return 0

        def BalanceStand(self) -> int:
            calls.append("BalanceStand")
            return 0

    bridge._client = Client()
    bridge._connected = True

    assert await bridge.request_flight_pose()
    assert bridge.get_status().joints_locked
    assert calls == ["StopMove", ("SwitchJoystick", False), "StandUp"]

    assert await bridge.request_stop()
    assert calls == ["StopMove", ("SwitchJoystick", False), "StandUp"]

    assert await bridge.request_stand()
    assert calls[-2:] == [("SwitchJoystick", True), "BalanceStand"]


def test_armed_flight_reports_emergency_if_joint_lock_is_lost(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    locked = replace(
        safe_walk_snapshot,
        state=SystemState.FLIGHT_MANUAL,
        configuration=Configuration.FLIGHT,
        pixhawk=replace(safe_walk_snapshot.pixhawk, armed=True),
        f446=replace(
            safe_walk_snapshot.f446,
            state=app_config.f446.expected_flight_state,
        ),
        go2=replace(
            safe_walk_snapshot.go2,
            joints_locked=True,
            locomotion_mode="JOINT_LOCK",
        ),
    )
    monitor = SafetyMonitor(app_config)
    assert all(item.code != "GO2_JOINT_LOCK_LOST" for item in monitor.evaluate(locked))

    unlocked = replace(
        locked,
        go2=replace(locked.go2, joints_locked=False, locomotion_mode="IDLE_STAND"),
    )
    violation = next(
        item for item in monitor.evaluate(unlocked) if item.code == "GO2_JOINT_LOCK_LOST"
    )

    assert violation.severity is SafetySeverity.EMERGENCY


@pytest.mark.asyncio
async def test_transform_fails_closed_if_go2_rejects_joint_lock(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        assert (await world._settle_for_transform()).ok
        world.go2.inject_flight_lock_failure()

        result = await world.manager.request_transform_flight(operator_confirmed=True)

        assert not result.ok
        assert result.code == "GO2_JOINT_LOCK_FAILED"
        assert world.manager.state is SystemState.FAULT
        assert world.manager.snapshot.f446.duty == 0
        assert not world.manager.snapshot.go2.joints_locked
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_prelocked_go2_still_disables_original_remote(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock, allow_control=True)
    bridge._on_state(_joint_lock_message())
    bridge._connected = True
    calls: list[object] = []

    class Client:
        def SwitchJoystick(self, enabled: bool) -> int:
            calls.append(("SwitchJoystick", enabled))
            return 0

        def StopMove(self) -> int:
            calls.append("StopMove")
            return 0

        def StandUp(self) -> int:
            calls.append("StandUp")
            return 0

    bridge._client = Client()

    assert await bridge.request_flight_pose()

    assert bridge.get_status().joints_locked


@pytest.mark.asyncio
async def test_landing_balance_keeps_original_remote_disabled_then_relocks(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock, allow_control=True)
    bridge._on_state(_joint_lock_message())
    bridge._connected = True
    calls: list[object] = []

    class Client:
        def SwitchJoystick(self, enabled: bool) -> int:
            calls.append(("SwitchJoystick", enabled))
            return 0

        def BalanceStand(self) -> int:
            calls.append("BalanceStand")
            bridge._on_state(_balance_stand_message())
            return 0

        def StopMove(self) -> int:
            calls.append("StopMove")
            return 0

        def StandUp(self) -> int:
            calls.append("StandUp")
            bridge._on_state(_joint_lock_message())
            return 0

    bridge._client = Client()
    assert await bridge.request_flight_pose()
    calls.clear()

    assert await bridge.request_landing_pose()
    status = bridge.get_status()
    assert calls == ["BalanceStand"]
    assert status.locomotion_mode == "BALANCE_STAND"
    assert status.foot_force == (101, 202, 303, 404)
    assert status.foot_force_valid
    assert not status.joints_locked

    calls.clear()
    assert await bridge.request_flight_pose()
    assert calls == ["StopMove", "StandUp"]
    assert bridge.get_status().joints_locked


def test_invalid_foot_force_shape_fails_closed(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    message = _balance_stand_message()
    message.foot_force = (1, 2, 3)

    bridge._on_state(message)

    assert bridge.get_status().foot_force == (0, 0, 0, 0)
    assert not bridge.get_status().foot_force_valid
