from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from aerogo2.bridges.go2_sdk_bridge import UnitreeGo2Bridge
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, RuntimeMode, SafetySeverity, SystemState
from aerogo2.common.models import SystemSnapshot
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.manager.system_manager import SystemManager
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
async def test_unlocked_flight_pose_waits_for_operator_then_disables_joystick(
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

    assert not await bridge.request_flight_pose()
    assert not bridge.get_status().joints_locked
    assert calls == ["StopMove"]

    bridge._on_state(_joint_lock_message())
    assert await bridge.request_flight_pose()
    assert bridge.get_status().joints_locked
    assert calls == ["StopMove", ("SwitchJoystick", False)]

    assert await bridge.request_stop()
    assert calls == ["StopMove", ("SwitchJoystick", False)]

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
    assert not await bridge.request_flight_pose()
    assert calls == ["StopMove"]
    assert not bridge.get_status().joints_locked

    bridge._on_state(_joint_lock_message())
    assert await bridge.request_flight_pose()
    assert calls == ["StopMove"]
    assert bridge.get_status().joints_locked


def _hardware_manager(world: SimulationWorld) -> SystemManager:
    config = replace(
        world.config,
        system=replace(
            world.config.system,
            dry_run=False,
            hardware_write_enabled=True,
        ),
    )
    manager = SystemManager(
        config=config,
        pixhawk=world.pixhawk,
        f446=world.f446,
        go2=world.go2,
        landing_controller=SafeDescentController(config),
        clock=world.clock,
        runtime_mode=RuntimeMode.HARDWARE,
    )
    return manager


@pytest.mark.asyncio
async def test_hardware_transform_waits_for_phone_mode_six_without_false_fault(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    manager = _hardware_manager(world)
    try:
        assert (await manager.start()).ok
        world._feed_rc(debounce=True)
        assert (await manager.connect_all()).ok
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        manager.accept_rc_status(
            world.rc_monitor.update(
                world._channels,
                connected=True,
                failsafe=False,
                timestamp=world.clock.monotonic(),
            )
        )
        world.clock.advance(
            max(
                app_config.safety.stationary_confirm_s,
                app_config.f446.current_clear_hold_s,
            )
            + 0.01
        )
        world._heartbeat_all()
        manager.accept_rc_status(world.rc_monitor.update(world._channels, connected=True, failsafe=False, timestamp=world.clock.monotonic()))
        await manager.refresh_snapshot()

        result = await manager.request_transform_flight(operator_confirmed=True)

        assert result.ok
        assert result.code == "GO2_JOINT_LOCK_OPERATOR_REQUIRED"
        assert manager.state is SystemState.GO2_JOINT_LOCK_WAIT
        assert manager.snapshot.configuration is Configuration.FLIGHT
        assert manager.snapshot.f446.duty == 0

        world.go2.inject_status(
            locomotion_mode="BALANCE_STAND",
            body_velocity=(-0.005, -0.001, -0.021),
            stable=False,
            controller_active=True,
            joints_locked=False,
        )
        await manager.tick()
        assert manager.state is SystemState.GO2_JOINT_LOCK_WAIT
        assert all(item.code != "GO2_MOVING_DURING_TRANSFORM" for item in manager.violations)

        world.go2.inject_status(
            locomotion_mode="JOINT_LOCK",
            body_velocity=(0.0, 0.0, 0.0),
            stable=True,
            standing=True,
            controller_active=False,
            joints_locked=True,
        )
        await manager.tick()

        assert manager.state is SystemState.FLIGHT_READY
        assert manager.snapshot.go2.joints_locked
    finally:
        await manager.shutdown()


def test_joint_lock_wait_rejects_locomotion_but_allows_small_posture_motion(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    base = replace(
        safe_walk_snapshot,
        state=SystemState.GO2_JOINT_LOCK_WAIT,
        configuration=Configuration.FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=app_config.f446.expected_flight_state,
            duty=0,
        ),
        go2=replace(
            safe_walk_snapshot.go2,
            locomotion_mode="BALANCE_STAND",
            body_velocity=(0.005, 0.001, 0.021),
            velocity_mps=0.0051,
            stable=False,
            moving=True,
            controller_active=True,
            joints_locked=False,
        ),
    )
    monitor = SafetyMonitor(app_config)

    safe_codes = {item.code for item in monitor.evaluate(base)}
    assert "GO2_MOVING_DURING_TRANSFORM" not in safe_codes
    assert "GO2_UNSAFE_DURING_JOINT_LOCK" not in safe_codes

    locomotion = replace(
        base,
        go2=replace(base.go2, locomotion_mode="LOCOMOTION"),
    )
    unsafe_codes = {item.code for item in monitor.evaluate(locomotion)}
    assert "GO2_UNSAFE_DURING_JOINT_LOCK" in unsafe_codes


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
