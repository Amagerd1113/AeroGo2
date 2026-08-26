from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import aerogo2.bridges.pixhawk_mavlink_bridge as pixhawk_bridge_module
from aerogo2.bridges.go2_sdk_bridge import UnitreeGo2Bridge
from aerogo2.bridges.pixhawk_mavlink_bridge import MavlinkPixhawkBridge
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import RuntimeMode
from aerogo2.common.exceptions import BridgeError
from aerogo2.hardware.runtime import HardwareWorld
from aerogo2.main import async_main, build_parser


def test_go2_state_callback_maps_three_axis_motion_and_stability(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    message = SimpleNamespace(
        velocity=(0.01, -0.01, 0.0),
        imu_state=SimpleNamespace(rpy=(0.02, -0.03, 0.4)),
        mode=1,
        error_code=0,
    )

    bridge._on_state(message)
    bridge._connected = True
    status = bridge.get_status()

    assert status.body_velocity == (0.01, -0.01, 0.0)
    assert status.body_rpy == (0.02, -0.03, 0.4)
    assert status.locomotion_mode == "BALANCE_STAND"
    assert status.stable
    assert not status.moving


@pytest.mark.parametrize(
    ("mode", "state_code", "expected_standing", "expected_stable"),
    (
        (0, 100, True, True),
        (0, 1001, False, False),
        (5, 100, False, False),
        (0, 12, False, False),
    ),
)
def test_go2_state_codes_fail_closed_except_verified_upright_codes(
    app_config: AppConfig,
    clock: ManualClock,
    mode: int,
    state_code: int,
    expected_standing: bool,
    expected_stable: bool,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    bridge._on_state(
        SimpleNamespace(
            velocity=(0.0, 0.0, 0.0),
            imu_state=SimpleNamespace(rpy=(0.0, 0.0, 0.0)),
            mode=mode,
            error_code=state_code,
        )
    )
    bridge._connected = True
    status = bridge.get_status()

    assert status.fault_code == state_code
    assert status.standing is expected_standing
    assert status.stable is expected_stable


def test_go2_state_callback_fails_closed_on_invalid_vector(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    bridge._on_state(
        SimpleNamespace(
            velocity=(float("nan"),),
            imu_state=SimpleNamespace(rpy=()),
            mode=3,
            error_code=12,
        )
    )
    bridge._connected = True

    status = bridge.get_status()

    assert status.body_velocity == (0.0, 0.0, 0.0)
    assert status.body_rpy == (0.0, 0.0, 0.0)
    assert status.fault_code == 12
    assert not status.stable


def test_go2_unknown_and_locomotion_modes_fail_closed(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)
    stationary = SimpleNamespace(
        velocity=(0.0, 0.0, 0.0),
        imu_state=SimpleNamespace(rpy=(0.0, 0.0, 0.0)),
        error_code=0,
    )

    bridge._on_state(SimpleNamespace(**vars(stationary), mode=99))
    bridge._connected = True
    unknown = bridge.get_status()

    assert unknown.locomotion_mode == "MODE_99"
    assert not unknown.standing
    assert not unknown.stable

    bridge._on_state(SimpleNamespace(**vars(stationary), mode=3))
    locomotion = bridge.get_status()

    assert locomotion.locomotion_mode == "LOCOMOTION"
    assert locomotion.moving
    assert locomotion.controller_active
    assert not locomotion.stable


@pytest.mark.asyncio
async def test_go2_control_is_locked_by_default(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = UnitreeGo2Bridge(app_config.go2, clock=clock)

    with pytest.raises(BridgeError, match="locked"):
        await bridge.request_stop()


def test_pixhawk_esc_telemetry_mapping_and_units(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    message = SimpleNamespace(
        rpm=(0, 100, 200, 300),
        voltage=(2400, 2410, 2420, 2430),
        current=(100, 200, 300, 400),
        temperature=(30, 31, 32, 33),
    )

    items = bridge._parse_esc(message, clock.monotonic())

    assert [(item.slot, item.physical_position) for item in items] == [
        (1, "RR"),
        (2, "LF"),
        (3, "LR"),
        (4, "RF"),
    ]
    assert items[0].rpm == 0.0
    assert items[0].voltage_v == 24.0
    assert items[0].current_a == 1.0
    assert all(item.healthy for item in items)


def test_pixhawk_explicit_legacy_shift_merges_groups_and_marks_esc2_offline(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(
        app_config.pixhawk,
        app_config.esc.slots,
        esc_mavlink_display_shift=1,
        clock=clock,
    )
    bridge._connected = True
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_1_TO_4",
            rpm=(0, 0, 0, 0),
            voltage=(0, 4487, 0, 4437),
            current=(0, 0, 0, 0),
            temperature=(0, 13, 0, 13),
            count=(0, 100, 0, 100),
        )
    )
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_5_TO_8",
            rpm=(0, 0, 0, 0),
            voltage=(4468, 0, 0, 0),
            current=(0, 0, 0, 0),
            temperature=(14, 0, 0, 0),
            count=(100, 0, 0, 0),
        )
    )

    status = bridge.get_status()
    items = {item.slot: item for item in status.esc}

    assert status.esc_raw_present_slots == (2, 4, 5)
    assert status.esc_mavlink_display_shift == 1
    assert status.esc_online == {1: True, 2: False, 3: True, 4: True}
    assert items[1].voltage_v == 44.87
    assert items[2].voltage_v == 0.0
    assert math.isnan(items[2].rpm)
    assert items[3].voltage_v == 44.37
    assert items[4].voltage_v == 44.68


def test_pixhawk_default_zero_shift_maps_partial_new_firmware_telemetry(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    bridge._connected = True
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_1_TO_4",
            rpm=(0, 0, 0, 0),
            voltage=(0, 4487, 0, 4437),
            current=(0, 0, 0, 0),
            temperature=(0, 13, 0, 13),
            count=(0, 100, 0, 100),
        )
    )

    status = bridge.get_status()

    assert status.esc_raw_present_slots == (2, 4)
    assert status.esc_mavlink_display_shift == 0
    assert status.esc_online == {1: False, 2: True, 3: False, 4: True}


def test_pixhawk_wrong_configured_shift_fails_closed_on_unexpected_raw_slot(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    bridge._connected = True
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_1_TO_4",
            rpm=(0, 0, 0, 0),
            voltage=(0, 4487, 0, 4437),
            current=(0, 0, 0, 0),
            temperature=(0, 13, 0, 13),
            count=(0, 100, 0, 100),
        )
    )
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_5_TO_8",
            rpm=(0, 0, 0, 0),
            voltage=(4468, 0, 0, 0),
            current=(0, 0, 0, 0),
            temperature=(14, 0, 0, 0),
            count=(100, 0, 0, 0),
        )
    )

    status = bridge.get_status()

    assert status.esc_raw_present_slots == (2, 4, 5)
    assert status.esc_mavlink_display_shift == 0
    assert not any(status.esc_online.values())
    assert all(math.isnan(item.rpm) for item in status.esc)


def test_pixhawk_esc_telemetry_staleness_fails_closed(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    bridge._connected = True
    bridge._handle_message(
        SimpleNamespace(
            get_type=lambda: "ESC_TELEMETRY_1_TO_4",
            rpm=(0, 0, 0, 0),
            voltage=(2400, 2410, 2420, 2430),
            current=(0, 0, 0, 0),
            temperature=(30, 31, 32, 33),
            count=(1, 1, 1, 1),
        )
    )
    clock.advance(1.01)

    status = bridge.get_status()

    assert status.esc_raw_present_slots == ()
    assert status.esc_mavlink_display_shift == 0
    assert not any(status.esc_online.values())


@pytest.mark.asyncio
async def test_pixhawk_critical_state_does_not_claim_rc_failsafe(
    app_config: AppConfig,
) -> None:
    world = HardwareWorld(app_config, runtime_mode=RuntimeMode.HARDWARE_READONLY)
    now = world.clock.monotonic()
    world.pixhawk._connected = True
    world.pixhawk._last_rc_timestamp = now
    world.pixhawk._status = replace(
        world.pixhawk._status,
        connected=True,
        failsafe=True,
        rc_failsafe=False,
        rc_channels={channel: 1500 for channel in range(1, 17)},
        heartbeat_timestamp=now,
        timestamp=now,
    )

    snapshot = await world.manager.refresh_snapshot()

    assert snapshot.pixhawk.failsafe
    assert not snapshot.pixhawk.rc_failsafe
    assert not snapshot.rc.failsafe


@pytest.mark.asyncio
async def test_pixhawk_outputs_remain_locked(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)

    result = await bridge.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)

    assert not result.ok
    assert result.code == "PIXHAWK_SETPOINT_LOCKED"
    with pytest.raises(BridgeError, match="RadioMaster"):
        await bridge.request_mode("GUIDED")


@pytest.mark.asyncio
async def test_pixhawk_ground_arm_authorization_requires_matching_ack(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    sent: list[tuple[object, ...]] = []

    def command_long_send(*args: object) -> None:
        sent.append(args)
        bridge._handle_message(
            SimpleNamespace(
                get_type=lambda: "COMMAND_ACK",
                command=31000,
                result=0,
                result_param2=int(args[7]),
            )
        )

    bridge._connection = SimpleNamespace(mav=SimpleNamespace(command_long_send=command_long_send))
    bridge._mavlink = SimpleNamespace(mavlink=SimpleNamespace(MAV_RESULT_ACCEPTED=0))
    bridge._connected = True

    enabled = await bridge.set_ground_arm_authorization(True, 30.0)

    assert enabled.ok
    assert bridge.ground_arm_authorization_active()
    assert sent[0][2] == 31000
    assert sent[0][4:9] == (1.0, 6202.0, 1.0, 1.0, 30.0)

    revoked = await bridge.set_ground_arm_authorization(False, 0.0)
    assert revoked.ok
    assert not bridge.ground_arm_authorization_active()
    assert sent[-1][4] == 0.0


@pytest.mark.asyncio
async def test_pixhawk_ground_arm_authorization_rejects_negative_ack(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)

    def command_long_send(*args: object) -> None:
        bridge._handle_message(
            SimpleNamespace(
                get_type=lambda: "COMMAND_ACK",
                command=31000,
                result=2,
                result_param2=int(args[7]),
            )
        )

    bridge._connection = SimpleNamespace(mav=SimpleNamespace(command_long_send=command_long_send))
    bridge._mavlink = SimpleNamespace(mavlink=SimpleNamespace(MAV_RESULT_ACCEPTED=0))
    bridge._connected = True

    result = await bridge.set_ground_arm_authorization(True, 30.0)

    assert not result.ok
    assert result.code == "PIXHAWK_ARM_AUTH_GATE_REJECTED"
    assert not bridge.ground_arm_authorization_active()


@pytest.mark.asyncio
async def test_pixhawk_ground_arm_authorization_times_out_without_lua_ack(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = MavlinkPixhawkBridge(app_config.pixhawk, app_config.esc.slots, clock=clock)
    bridge._connection = SimpleNamespace(mav=SimpleNamespace(command_long_send=lambda *args: None))
    bridge._mavlink = SimpleNamespace(mavlink=SimpleNamespace(MAV_RESULT_ACCEPTED=0))
    bridge._connected = True
    monkeypatch.setattr(pixhawk_bridge_module, "_GROUND_ARM_AUTH_ACK_TIMEOUT_S", 0.01)

    result = await bridge.set_ground_arm_authorization(True, 30.0)

    assert not result.ok
    assert result.code == "PIXHAWK_ARM_AUTH_GATE_TIMEOUT"
    assert not bridge.ground_arm_authorization_active()


def test_hardware_world_never_unlocks_pixhawk_outputs(app_config: AppConfig) -> None:
    unlocked = replace(
        app_config,
        system=replace(
            app_config.system,
            dry_run=False,
            hardware_write_enabled=True,
        ),
    )

    world = HardwareWorld(unlocked, runtime_mode=RuntimeMode.HARDWARE)

    assert world.f446._allow_motion
    assert world.go2._allow_control
    assert not world.pixhawk._allow_setpoints


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        ["shell", "--hardware"],
        ["shell", "--hardware", "--confirm-hardware", "I_UNDERSTAND_HARDWARE_RISK"],
        ["monitor"],
    ],
)
async def test_hardware_cli_refuses_missing_unlocks(
    arguments: list[str],
    project_root: Path,
) -> None:
    arguments.extend(["--config", str(project_root / "configs" / "system.yaml")])
    args = build_parser().parse_args(arguments)

    assert await async_main(args) == 2
