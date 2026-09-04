from __future__ import annotations

import math
from dataclasses import replace
from typing import Tuple
from unittest.mock import AsyncMock

import pytest

from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    MorphologyRequest,
    RuntimeMode,
    SystemState,
)
from aerogo2.common.models import (
    F446Status,
    Go2Status,
    LandingCommand,
    LandingEstimate,
    PixhawkStatus,
    RCStatus,
    SystemSnapshot,
)
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.landing.safety_filter import LandingSafetyFilter
from aerogo2.manager.system_manager import SystemManager


def landing_snapshot(
    app_config: AppConfig,
    timestamp: float = 10.0,
) -> SystemSnapshot:
    return SystemSnapshot(
        timestamp=timestamp,
        state=SystemState.AUTO_LANDING,
        pixhawk=PixhawkStatus(
            connected=True,
            armed=True,
            landed=False,
            failsafe=False,
            heartbeat_timestamp=timestamp,
            attitude_timestamp=timestamp,
            kinematics_timestamp=timestamp,
            landed_state_timestamp=timestamp,
        ),
        f446=F446Status(
            connected=True,
            state=app_config.f446.expected_flight_state,
            duty=0,
            timestamp=timestamp,
        ),
        go2=Go2Status(
            connected=True,
            velocity_mps=0.0,
            stable=True,
            timestamp=timestamp,
        ),
        rc=RCStatus(
            connected=True,
            failsafe=False,
            morphology_request=MorphologyRequest.HOLD,
            auto_landing_request=AutoLandingRequest.AUTO_EXECUTE,
            manual_override=False,
            timestamp=timestamp,
        ),
        configuration=Configuration.FLIGHT,
        landing_estimate=LandingEstimate(
            valid=True,
            ground_detected=True,
            height_m=1.0,
            vertical_velocity_mps=0.0,
            horizontal_velocity_mps=0.0,
            timestamp=timestamp,
            reason="valid simulated estimate",
        ),
        autoland_active=True,
        external_setpoint_active=False,
    )


def snapshot_at(snapshot: SystemSnapshot, timestamp: float) -> SystemSnapshot:
    return replace(
        snapshot,
        timestamp=timestamp,
        pixhawk=replace(
            snapshot.pixhawk,
            heartbeat_timestamp=timestamp,
            attitude_timestamp=timestamp,
            kinematics_timestamp=timestamp,
            landed_state_timestamp=timestamp,
        ),
        f446=replace(snapshot.f446, timestamp=timestamp),
        go2=replace(snapshot.go2, timestamp=timestamp),
        rc=replace(snapshot.rc, timestamp=timestamp),
        landing_estimate=replace(snapshot.landing_estimate, timestamp=timestamp),
    )


def assert_zero_invalid(command: LandingCommand) -> None:
    assert command.valid is False
    assert (
        command.vx_des,
        command.vy_des,
        command.vz_des,
        command.yaw_rate_des,
    ) == (0.0, 0.0, 0.0, 0.0)


def test_safe_descent_valid_command_uses_local_ned_limit(
    app_config: AppConfig,
) -> None:
    command = SafeDescentController(app_config).update(
        landing_snapshot(app_config),
        1.0 / app_config.landing.controller_hz,
    )
    assert command.valid
    assert command.vx_des == 0.0
    assert command.vy_des == 0.0
    assert command.vz_des == app_config.landing.maximum_descent_speed_mps
    assert command.yaw_rate_des == 0.0


def test_safe_descent_is_invalid_by_default(app_config: AppConfig) -> None:
    snapshot = SystemSnapshot(timestamp=10.0, state=SystemState.BOOT_SAFE)
    command = SafeDescentController(app_config).update(snapshot, 0.01)
    assert_zero_invalid(command)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_state",
        "inactive",
        "unknown_configuration",
        "pixhawk_disconnected",
        "pixhawk_disarmed",
        "pixhawk_failsafe",
        "pixhawk_stale",
        "rc_disconnected",
        "rc_failsafe",
        "rc_stale",
        "manual_override",
        "ch10_manual",
        "estimate_invalid",
        "ground_invalid",
        "estimate_stale",
        "active_fault",
    ],
)
def test_every_landing_gate_fails_closed(
    app_config: AppConfig,
    case: str,
) -> None:
    snapshot = landing_snapshot(app_config)
    if case == "wrong_state":
        snapshot = replace(snapshot, state=SystemState.AUTO_LANDING_READY)
    elif case == "inactive":
        snapshot = replace(snapshot, autoland_active=False)
    elif case == "unknown_configuration":
        snapshot = replace(snapshot, configuration=Configuration.UNKNOWN)
    elif case == "pixhawk_disconnected":
        snapshot = replace(
            snapshot,
            pixhawk=replace(snapshot.pixhawk, connected=False),
        )
    elif case == "pixhawk_disarmed":
        snapshot = replace(snapshot, pixhawk=replace(snapshot.pixhawk, armed=False))
    elif case == "pixhawk_failsafe":
        snapshot = replace(snapshot, pixhawk=replace(snapshot.pixhawk, failsafe=True))
    elif case == "pixhawk_stale":
        snapshot = replace(
            snapshot,
            pixhawk=replace(
                snapshot.pixhawk,
                heartbeat_timestamp=(
                    snapshot.timestamp - app_config.safety.pixhawk_timeout_s - 0.001
                ),
            ),
        )
    elif case == "rc_disconnected":
        snapshot = replace(snapshot, rc=replace(snapshot.rc, connected=False))
    elif case == "rc_failsafe":
        snapshot = replace(snapshot, rc=replace(snapshot.rc, failsafe=True))
    elif case == "rc_stale":
        snapshot = replace(
            snapshot,
            rc=replace(
                snapshot.rc,
                timestamp=snapshot.timestamp - app_config.safety.rc_timeout_s - 0.001,
            ),
        )
    elif case == "manual_override":
        snapshot = replace(snapshot, rc=replace(snapshot.rc, manual_override=True))
    elif case == "ch10_manual":
        snapshot = replace(
            snapshot,
            rc=replace(
                snapshot.rc,
                auto_landing_request=AutoLandingRequest.MANUAL,
            ),
        )
    elif case == "estimate_invalid":
        snapshot = replace(
            snapshot,
            landing_estimate=replace(snapshot.landing_estimate, valid=False),
        )
    elif case == "ground_invalid":
        snapshot = replace(
            snapshot,
            landing_estimate=replace(
                snapshot.landing_estimate,
                ground_detected=False,
            ),
        )
    elif case == "estimate_stale":
        snapshot = replace(
            snapshot,
            landing_estimate=replace(
                snapshot.landing_estimate,
                timestamp=(snapshot.timestamp - app_config.landing.controller_timeout_s - 0.001),
            ),
        )
    elif case == "active_fault":
        snapshot = replace(snapshot, active_fault_codes=("SIMULATED_FAULT",))
    else:
        raise AssertionError(f"unhandled case {case}")
    assert_zero_invalid(SafeDescentController(app_config).update(snapshot, 0.01))


def test_landing_filter_clamps_vector_descent_and_yaw(
    app_config: AppConfig,
) -> None:
    snapshot = landing_snapshot(app_config)
    candidate = LandingCommand(
        vx_des=3.0,
        vy_des=4.0,
        vz_des=3.0,
        yaw_rate_des=2.0,
        valid=True,
        reason="limit test",
        timestamp=snapshot.timestamp,
    )
    command = LandingSafetyFilter(app_config).apply(candidate, snapshot, 0.01)
    assert command.valid
    assert math.hypot(command.vx_des, command.vy_des) == pytest.approx(
        app_config.landing.maximum_horizontal_speed_mps
    )
    assert command.vx_des == pytest.approx(0.3)
    assert command.vy_des == pytest.approx(0.4)
    assert command.vz_des == app_config.landing.maximum_descent_speed_mps
    assert command.yaw_rate_des == app_config.landing.maximum_yaw_rate_rad_s


def test_landing_filter_clamps_climb_candidate_to_zero(
    app_config: AppConfig,
) -> None:
    snapshot = landing_snapshot(app_config)
    candidate = LandingCommand(
        vz_des=-3.0,
        yaw_rate_des=-3.0,
        valid=True,
        reason="negative limit test",
        timestamp=snapshot.timestamp,
    )
    command = LandingSafetyFilter(app_config).apply(candidate, snapshot, 0.01)
    assert command.valid
    assert command.vz_des == 0.0
    assert command.yaw_rate_des == -app_config.landing.maximum_yaw_rate_rad_s


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nonfinite_landing_output_is_invalid(
    app_config: AppConfig,
    value: float,
) -> None:
    snapshot = landing_snapshot(app_config)
    candidate = LandingCommand(
        vx_des=value,
        valid=True,
        reason="nonfinite test",
        timestamp=snapshot.timestamp,
    )
    assert_zero_invalid(LandingSafetyFilter(app_config).apply(candidate, snapshot, 0.01))


@pytest.mark.parametrize(
    "dt",
    [0.0, -0.01, float("nan"), float("inf"), 0.100001],
)
def test_invalid_or_timed_out_controller_dt_fails_closed(
    app_config: AppConfig,
    dt: float,
) -> None:
    assert_zero_invalid(SafeDescentController(app_config).update(landing_snapshot(app_config), dt))


def test_controller_timeout_boundary_is_inclusive(
    app_config: AppConfig,
) -> None:
    initial_timestamp = app_config.landing.controller_timeout_s
    snapshot = landing_snapshot(app_config, timestamp=initial_timestamp)
    controller = SafeDescentController(app_config)
    assert controller.update(snapshot, 0.01).valid
    exact = snapshot_at(snapshot, initial_timestamp + app_config.landing.controller_timeout_s)
    assert controller.update(
        exact,
        app_config.landing.controller_timeout_s,
    ).valid


def test_elapsed_time_above_timeout_fails_even_if_reported_dt_is_small(
    app_config: AppConfig,
) -> None:
    initial_timestamp = 1.0
    snapshot = landing_snapshot(app_config, timestamp=initial_timestamp)
    controller = SafeDescentController(app_config)
    assert controller.update(snapshot, 0.01).valid
    late = snapshot_at(
        snapshot,
        initial_timestamp + app_config.landing.controller_timeout_s + 0.000001,
    )
    assert_zero_invalid(controller.update(late, 0.01))


def test_manual_override_immediately_invalidates_previous_valid_output(
    app_config: AppConfig,
) -> None:
    snapshot = landing_snapshot(app_config)
    controller = SafeDescentController(app_config)
    assert controller.update(snapshot, 0.01).valid
    overridden = snapshot_at(snapshot, snapshot.timestamp + 0.01)
    overridden = replace(
        overridden,
        rc=replace(overridden.rc, manual_override=True),
    )
    command = controller.update(overridden, 0.01)
    assert_zero_invalid(command)
    assert "manual override" in command.reason


async def manager_in_autoland(
    app_config: AppConfig,
    clock: ManualClock,
) -> Tuple[SystemManager, FakePixhawk, FakeF446, FakeGo2]:
    pixhawk = FakePixhawk(clock=clock)
    f446 = FakeF446(config=app_config.f446, clock=clock)
    go2 = FakeGo2(clock=clock)
    manager = SystemManager(
        config=app_config,
        pixhawk=pixhawk,
        f446=f446,
        go2=go2,
        landing_controller=SafeDescentController(app_config),
        clock=clock,
    )
    assert (await manager.start()).ok
    assert (await manager.connect_all()).ok
    manager.accept_rc_status(
        RCStatus(
            connected=True,
            failsafe=False,
            channels={app_config.rc.flight_enable_channel: 1000},
            flight_enable=False,
            morphology_request=MorphologyRequest.FLIGHT_REQUEST,
            auto_landing_request=AutoLandingRequest.MANUAL,
            timestamp=clock.monotonic(),
        )
    )
    await manager.refresh_snapshot()
    clock.advance(app_config.safety.stationary_confirm_s)
    pixhawk.inject_telemetry_cycle()
    f446.inject_status()
    go2.inject_status()
    manager.accept_rc_status(
        RCStatus(
            connected=True,
            failsafe=False,
            channels={app_config.rc.flight_enable_channel: 1000},
            flight_enable=False,
            morphology_request=MorphologyRequest.FLIGHT_REQUEST,
            auto_landing_request=AutoLandingRequest.MANUAL,
            timestamp=clock.monotonic(),
        )
    )
    assert (await manager.request_transform_flight(operator_confirmed=True)).ok
    assert (await manager.authorize_ground_arm()).ok
    manager.accept_rc_status(
        RCStatus(
            connected=True,
            failsafe=False,
            channels={app_config.rc.flight_enable_channel: 1900},
            flight_enable=True,
            morphology_request=MorphologyRequest.FLIGHT_REQUEST,
            auto_landing_request=AutoLandingRequest.MANUAL,
            timestamp=clock.monotonic(),
        )
    )
    pixhawk.inject_armed_state(True)
    await manager.tick()
    assert manager.state is SystemState.FLIGHT_MANUAL

    manager.accept_landing_estimate(
        LandingEstimate(
            valid=True,
            ground_detected=True,
            height_m=1.0,
            vertical_velocity_mps=0.0,
            horizontal_velocity_mps=0.0,
            timestamp=clock.monotonic(),
            reason="valid simulated estimate",
        )
    )
    manager.accept_rc_status(
        RCStatus(
            connected=True,
            failsafe=False,
            channels={app_config.rc.flight_enable_channel: 1900},
            flight_enable=True,
            morphology_request=MorphologyRequest.HOLD,
            auto_landing_request=AutoLandingRequest.AUTO_READY,
            timestamp=clock.monotonic(),
        )
    )
    assert (await manager.prepare_autoland()).ok
    assert manager.state is SystemState.AUTO_LANDING_READY
    assert pixhawk.setpoint_history == ()

    manager.accept_rc_status(
        replace(
            manager.snapshot.rc,
            auto_landing_request=AutoLandingRequest.AUTO_EXECUTE,
            timestamp=clock.monotonic(),
        )
    )
    assert (await manager.start_autoland()).ok
    assert manager.state is SystemState.AUTO_LANDING
    assert pixhawk.external_setpoints_active
    assert len(pixhawk.setpoint_history) == 1
    return manager, pixhawk, f446, go2


@pytest.mark.asyncio
async def test_manual_override_aborts_setpoints_without_disarming(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    manager, pixhawk, _, _ = await manager_in_autoland(app_config, clock)
    clock.advance(0.02)
    pixhawk.inject_telemetry_cycle()
    manager.accept_landing_estimate(
        replace(manager.snapshot.landing_estimate, timestamp=clock.monotonic())
    )
    manager.accept_rc_status(
        replace(
            manager.snapshot.rc,
            manual_override=True,
            auto_landing_request=AutoLandingRequest.MANUAL,
            timestamp=clock.monotonic(),
        )
    )
    await manager.tick()
    assert manager.state is SystemState.FLIGHT_MANUAL
    assert pixhawk.external_setpoints_active is False
    assert pixhawk.get_status().armed is True
    assert manager.last_landing_command.valid is False


@pytest.mark.asyncio
async def test_controller_timeout_aborts_and_stops_existing_setpoint(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    manager, pixhawk, f446, go2 = await manager_in_autoland(app_config, clock)
    clock.advance(app_config.landing.controller_timeout_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    f446.inject_status()
    go2.inject_status()
    manager.accept_rc_status(replace(manager.snapshot.rc, timestamp=clock.monotonic()))
    manager.accept_landing_estimate(
        replace(manager.snapshot.landing_estimate, timestamp=clock.monotonic())
    )
    result = await manager.update_autoland()
    assert result.ok
    assert manager.state is SystemState.FLIGHT_MANUAL
    assert pixhawk.external_setpoints_active is False
    assert pixhawk.get_status().armed is True
    assert manager.last_landing_command.valid is False
    assert "timeout" in manager.last_landing_command.reason


class _FixedLandingController:
    def __init__(self, command: LandingCommand) -> None:
        self._command = command

    def reset(self) -> None:
        return None

    def update(self, snapshot: SystemSnapshot, dt: float) -> LandingCommand:
        del dt
        return replace(self._command, timestamp=snapshot.timestamp)


def _refresh_autoland_inputs(
    manager: SystemManager,
    pixhawk: FakePixhawk,
    f446: FakeF446,
    go2: FakeGo2,
    clock: ManualClock,
) -> None:
    pixhawk.inject_telemetry_cycle()
    f446.inject_status()
    go2.inject_status()
    manager.accept_rc_status(replace(manager.snapshot.rc, timestamp=clock.monotonic()))
    manager.accept_landing_estimate(
        replace(manager.snapshot.landing_estimate, timestamp=clock.monotonic())
    )


@pytest.mark.asyncio
async def test_manager_final_filter_blocks_rogue_nonfinite_controller_output(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    manager, pixhawk, f446, go2 = await manager_in_autoland(app_config, clock)
    before = len(pixhawk.setpoint_history)
    manager._landing_controller = _FixedLandingController(
        LandingCommand(valid=True, vx_des=float("nan"), reason="rogue")
    )
    clock.advance(1.0 / app_config.landing.controller_hz)
    _refresh_autoland_inputs(manager, pixhawk, f446, go2, clock)

    result = await manager.update_autoland()

    assert result.ok
    assert manager.state is SystemState.FLIGHT_MANUAL
    assert len(pixhawk.setpoint_history) == before
    assert not manager.last_landing_command.valid


@pytest.mark.asyncio
async def test_manager_final_filter_clamps_rogue_overlimit_controller_output(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    manager, pixhawk, f446, go2 = await manager_in_autoland(app_config, clock)
    before = len(pixhawk.setpoint_history)
    manager._landing_controller = _FixedLandingController(
        LandingCommand(
            valid=True,
            vx_des=3.0,
            vy_des=4.0,
            vz_des=3.0,
            yaw_rate_des=2.0,
            reason="rogue",
        )
    )
    clock.advance(1.0 / app_config.landing.controller_hz)
    _refresh_autoland_inputs(manager, pixhawk, f446, go2, clock)

    result = await manager.update_autoland()

    assert result.ok
    assert manager.state is SystemState.AUTO_LANDING
    assert len(pixhawk.setpoint_history) == before + 1
    sent = pixhawk.setpoint_history[-1]
    assert math.hypot(sent.vx, sent.vy) == pytest.approx(
        app_config.landing.maximum_horizontal_speed_mps
    )
    assert sent.vz == app_config.landing.maximum_descent_speed_mps
    assert sent.yaw_rate == app_config.landing.maximum_yaw_rate_rad_s


@pytest.mark.parametrize(
    "field",
    ["height_m", "vertical_velocity_mps", "horizontal_velocity_mps"],
)
def test_landing_filter_rejects_missing_required_estimate_values(
    app_config: AppConfig,
    field: str,
) -> None:
    snapshot = landing_snapshot(app_config)
    snapshot = replace(
        snapshot,
        landing_estimate=replace(snapshot.landing_estimate, **{field: None}),
    )
    command = SafeDescentController(app_config).update(snapshot, 0.01)
    assert_zero_invalid(command)


@pytest.mark.asyncio
async def test_non_dry_run_never_issues_any_control_bridge_write(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    pixhawk = FakePixhawk(clock=clock)
    f446 = FakeF446(config=app_config.f446, clock=clock)
    go2 = FakeGo2(clock=clock)
    control_calls = (
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    (
        pixhawk.send_velocity_setpoint,
        pixhawk.stop_external_setpoints,
        f446.move_to_configuration,
        f446.stop,
        go2.request_stop,
        go2.request_stand,
        go2.request_flight_pose,
    ) = control_calls
    manager = SystemManager(
        config=app_config,
        pixhawk=pixhawk,
        f446=f446,
        go2=go2,
        landing_controller=SafeDescentController(app_config),
        clock=clock,
        runtime_mode=RuntimeMode.HARDWARE_READONLY,
    )
    assert (await manager.start()).ok

    for result in (
        await manager.walk_stop(),
        await manager.walk_stand(),
        await manager.request_transform_flight(operator_confirmed=True),
        await manager.request_transform_walk(operator_confirmed=True),
        await manager.prepare_autoland(),
        await manager.start_autoland(),
        await manager.update_autoland(),
    ):
        assert result.code == "PHASE_NOT_AVAILABLE"
    assert (await manager.stop_transform_motion()).ok
    assert (await manager.stop_supervised()).ok
    await manager._enter_boot_safe(manager.snapshot)
    await manager._enter_fault_state(manager.snapshot)
    await manager._enter_emergency_stop(manager.snapshot)
    assert (await manager.shutdown()).ok

    for call in control_calls:
        call.assert_not_awaited()
