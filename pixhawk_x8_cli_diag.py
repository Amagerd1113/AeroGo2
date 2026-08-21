#!/usr/bin/env python3
# pixhawk_x8_cli_diag.py
# Ubuntu 22.04 MAVLink diagnostic CLI for Pixhawk 6X + Hobbywing X8 G2 DroneCAN testing.
#
# Purpose:
#   This script can passively decode DroneCAN through ArduPilot's MAVLink CAN-forwarding
#   transport. It also drives MAV_CMD_DO_MOTOR_TEST, requests useful MAVLink telemetry
#   streams, and audits the parameters that route motor output to DroneCAN.
#
# Safety:
#   - REMOVE PROPELLERS.
#   - Use a current-limited supply.
#   - Keep the vehicle restrained.
#   - Do not run Mission Planner/QGC/MAVProxy on the same serial port at the same time.
#   - This is a bench diagnostic tool, not flight-control software.

from __future__ import annotations

import argparse
import glob
import os
import select
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MAVLINK20", "1")

try:
    from pymavlink import mavutil
except ModuleNotFoundError as exc:
    if exc.name != "pymavlink":
        raise
    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_venv_dir = os.path.join(project_dir, ".venv")
    local_venv_python = os.path.join(local_venv_dir, "bin", "python")
    running_as_script = os.path.abspath(sys.argv[0]) == os.path.abspath(__file__)
    if running_as_script and os.path.exists(local_venv_python) and os.path.abspath(sys.prefix) != os.path.abspath(local_venv_dir):
        os.execv(local_venv_python, [local_venv_python, __file__, *sys.argv[1:]])
    print("Missing dependency: pymavlink. Install it with: python3 -m pip install pymavlink", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import serial
except ModuleNotFoundError:
    SERIAL_EXCEPTIONS = (OSError, PermissionError)
else:
    SERIAL_EXCEPTIONS = (serial.SerialException, OSError, PermissionError)


RESULT_TEXT = {
    0: "ACCEPTED",
    1: "TEMPORARILY_REJECTED",
    2: "DENIED",
    3: "UNSUPPORTED",
    4: "FAILED",
    5: "IN_PROGRESS",
    6: "CANCELLED",
    7: "COMMAND_LONG_ONLY",
    8: "COMMAND_INT_ONLY",
    9: "UNSUPPORTED_MAV_FRAME",
}

MODE_PWM = "pwm"
MODE_PERCENT = "percent"

# Public defaults consumed by the AeroGo2 X8 bench adapter. Keep the
# command-line parser and runtime tied to these constants so the validated
# diagnostic remains the single source of truth.
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_PWM_MIN = 1000
DEFAULT_PWM_LIMIT = 1450
DEFAULT_PERCENT_LIMIT = 50.0
DEFAULT_REFRESH_PERIOD = 0.8
DEFAULT_HOLD_DURATION = 3.0

SERIAL_PATTERNS = (
    "/dev/serial/by-id/*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
)

COMMON_MODE_ALIASES = {
    "STABLE": "STABILIZE",
    "STAB": "STABILIZE",
    "STABILIZED": "STABILIZE",
    "ALT": "ALT_HOLD",
    "ALTHOLD": "ALT_HOLD",
    "ALTITUDE": "ALT_HOLD",
    "ALTITUDEHOLD": "ALT_HOLD",
    "POSHOLD": "POSHOLD",
    "POS_HOLD": "POSHOLD",
    "POSITIONHOLD": "POSHOLD",
    "POSITION_HOLD": "POSHOLD",
    "RETURN": "RTL",
    "RETURNHOME": "RTL",
    "RETURN_HOME": "RTL",
    "RETURN_TO_LAUNCH": "RTL",
    "RTH": "RTL",
    "LANDING": "LAND",
    "HOLD": "LOITER",
    "GPSHOLD": "LOITER",
    "GPS_HOLD": "LOITER",
    "GUIDEDNOGPS": "GUIDED_NOGPS",
    "GUIDED_NO_GPS": "GUIDED_NOGPS",
}

GUIDED_VELOCITY_TYPE_MASK = (
    getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_X_IGNORE", 1)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_Y_IGNORE", 2)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_Z_IGNORE", 4)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_AX_IGNORE", 64)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_AY_IGNORE", 128)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_AZ_IGNORE", 256)
    | getattr(mavutil.mavlink, "POSITION_TARGET_TYPEMASK_YAW_IGNORE", 1024)
)

# This aircraft's Hobbywing NodeID and ThrottleID are identical:
#   ID1 rear-right, ID2 front-left, ID3 rear-left, ID4 front-right.
# DroneCAN RawCommand slots follow the SERVO output channel, so the motor
# functions must be assigned to outputs in that exact ID order.
#
# ArduCopter Quad X motor functions:
#   Motor1 = front-right, Motor2 = rear-left, Motor3 = front-left, Motor4 = rear-right
# Mission Planner/MAV_CMD_DO_MOTOR_TEST test order starts front-right and proceeds clockwise:
#   motor-test id 1 -> front-right -> ThrottleID/output 4
#   motor-test id 2 -> rear-right  -> ThrottleID/output 1
#   motor-test id 3 -> rear-left   -> ThrottleID/output 3
#   motor-test id 4 -> front-left  -> ThrottleID/output 2
POSITION_TO_OUTPUT_CHANNEL = {"rr": 1, "fl": 2, "rl": 3, "fr": 4}
POSITION_ALIASES_TO_OUTPUT_CHANNEL = {
    "fl": 2,
    "lf": 2,
    "front_left": 2,
    "left_front": 2,
    "fr": 4,
    "rf": 4,
    "front_right": 4,
    "right_front": 4,
    "rl": 3,
    "lr": 3,
    "rear_left": 3,
    "left_rear": 3,
    "bl": 3,
    "back_left": 3,
    "left_back": 3,
    "rr": 1,
    "rear_right": 1,
    "right_rear": 1,
    "br": 1,
    "back_right": 1,
    "right_back": 1,
}
OUTPUT_CHANNEL_TO_POSITION = {v: k.upper() for k, v in POSITION_TO_OUTPUT_CHANNEL.items()}
MOTOR_TEST_TO_POSITION = {1: "FR", 2: "RR", 3: "RL", 4: "FL"}
POSITION_TO_MOTOR_TEST = {v.lower(): k for k, v in MOTOR_TEST_TO_POSITION.items()}
POSITION_TO_MOTOR_FUNCTION = {"fr": 33, "rl": 34, "fl": 35, "rr": 36}
MOTOR_FUNCTION_TO_POSITION = {v: k.upper() for k, v in POSITION_TO_MOTOR_FUNCTION.items()}
MOTOR_TEST_TO_MOTOR_FUNCTION = {
    motor_test_id: POSITION_TO_MOTOR_FUNCTION[position.lower()]
    for motor_test_id, position in MOTOR_TEST_TO_POSITION.items()
}
MOTOR_FUNCTION_TO_MOTOR_TEST = {v: k for k, v in MOTOR_TEST_TO_MOTOR_FUNCTION.items()}
FALLBACK_MOTOR_FUNCTION_TO_OUTPUT_CHANNEL = {
    POSITION_TO_MOTOR_FUNCTION[position]: channel
    for position, channel in POSITION_TO_OUTPUT_CHANNEL.items()
}
MOTOR_TEST_TO_OUTPUT_CHANNEL = {
    motor_test_id: FALLBACK_MOTOR_FUNCTION_TO_OUTPUT_CHANNEL[motor_function]
    for motor_test_id, motor_function in MOTOR_TEST_TO_MOTOR_FUNCTION.items()
}
POSITION_ALIASES_TO_POSITION = {
    alias: OUTPUT_CHANNEL_TO_POSITION[channel].lower()
    for alias, channel in POSITION_ALIASES_TO_OUTPUT_CHANNEL.items()
}
SERVO_FUNCTION_PARAM_NAMES = tuple(f"SERVO{i}_FUNCTION" for i in range(1, 5))
EXPECTED_NODE_TO_POSITION = {1: "RR", 2: "FL", 3: "RL", 4: "FR"}

# MAVLink message IDs. Keep fallbacks so the script works with older pymavlink packages.
MSG_SERVO_OUTPUT_RAW = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_SERVO_OUTPUT_RAW", 36)
MSG_RC_CHANNELS = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_RC_CHANNELS", 65)
MSG_SYS_STATUS = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_SYS_STATUS", 1)
MSG_ESC_STATUS = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_STATUS", 291)
MSG_ESC_TELEMETRY_1_TO_4 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_1_TO_4", 11030)
MSG_ESC_TELEMETRY_5_TO_8 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_5_TO_8", 11031)
MSG_ESC_TELEMETRY_9_TO_12 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_9_TO_12", 11032)
MSG_ESC_TELEMETRY_13_TO_16 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_13_TO_16", 11040)
MSG_ESC_TELEMETRY_17_TO_20 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_17_TO_20", 11041)
MSG_ESC_TELEMETRY_21_TO_24 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_21_TO_24", 11042)
MSG_ESC_TELEMETRY_25_TO_28 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_25_TO_28", 11043)
MSG_ESC_TELEMETRY_29_TO_32 = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ESC_TELEMETRY_29_TO_32", 11044)
MSG_ATTITUDE = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_ATTITUDE", 30)
MSG_GLOBAL_POSITION_INT = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_GLOBAL_POSITION_INT", 33)
MSG_GPS_RAW_INT = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_GPS_RAW_INT", 24)
MSG_VFR_HUD = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_VFR_HUD", 74)
MSG_EKF_STATUS_REPORT = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_EKF_STATUS_REPORT", 193)
MSG_AUTOPILOT_VERSION = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_AUTOPILOT_VERSION", 148)
MSG_UAVCAN_NODE_STATUS = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_UAVCAN_NODE_STATUS", 310)
MSG_UAVCAN_NODE_INFO = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_UAVCAN_NODE_INFO", 311)
MAV_CMD_RUN_PREARM_CHECKS = getattr(mavutil.mavlink, "MAV_CMD_RUN_PREARM_CHECKS", 401)
MAV_CMD_REQUEST_MESSAGE = getattr(mavutil.mavlink, "MAV_CMD_REQUEST_MESSAGE", 512)
MAV_CMD_UAVCAN_GET_NODE_INFO = getattr(mavutil.mavlink, "MAV_CMD_UAVCAN_GET_NODE_INFO", 5200)
MAV_RESULT_ACCEPTED = getattr(mavutil.mavlink, "MAV_RESULT_ACCEPTED", 0)
MAV_RESULT_IN_PROGRESS = getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5)
SAFETY_SWITCH_STATE_SAFE = getattr(mavutil.mavlink, "SAFETY_SWITCH_STATE_SAFE", 0)
SAFETY_SWITCH_STATE_DANGEROUS = getattr(mavutil.mavlink, "SAFETY_SWITCH_STATE_DANGEROUS", 1)
ESC_TELEMETRY_MAX_AGE = 2.0
CAN_ESC_OFFSET_PARAM = "CAN_D1_UC_ESC_OF"

ESC_TELEMETRY_GROUPS = (
    ("ESC_TELEMETRY_1_TO_4", MSG_ESC_TELEMETRY_1_TO_4, 1),
    ("ESC_TELEMETRY_5_TO_8", MSG_ESC_TELEMETRY_5_TO_8, 5),
    ("ESC_TELEMETRY_9_TO_12", MSG_ESC_TELEMETRY_9_TO_12, 9),
    ("ESC_TELEMETRY_13_TO_16", MSG_ESC_TELEMETRY_13_TO_16, 13),
    ("ESC_TELEMETRY_17_TO_20", MSG_ESC_TELEMETRY_17_TO_20, 17),
    ("ESC_TELEMETRY_21_TO_24", MSG_ESC_TELEMETRY_21_TO_24, 21),
    ("ESC_TELEMETRY_25_TO_28", MSG_ESC_TELEMETRY_25_TO_28, 25),
    ("ESC_TELEMETRY_29_TO_32", MSG_ESC_TELEMETRY_29_TO_32, 29),
)
ESC_TELEMETRY_FIRST_ID_BY_TYPE = {name: first_id for name, _, first_id in ESC_TELEMETRY_GROUPS}

UAVCAN_HEALTH_TEXT = {
    getattr(mavutil.mavlink, "UAVCAN_NODE_HEALTH_OK", 0): "OK",
    getattr(mavutil.mavlink, "UAVCAN_NODE_HEALTH_WARNING", 1): "WARNING",
    getattr(mavutil.mavlink, "UAVCAN_NODE_HEALTH_ERROR", 2): "ERROR",
    getattr(mavutil.mavlink, "UAVCAN_NODE_HEALTH_CRITICAL", 3): "CRITICAL",
}

UAVCAN_MODE_TEXT = {
    getattr(mavutil.mavlink, "UAVCAN_NODE_MODE_OPERATIONAL", 0): "OPERATIONAL",
    getattr(mavutil.mavlink, "UAVCAN_NODE_MODE_INITIALIZATION", 1): "INITIALIZATION",
    getattr(mavutil.mavlink, "UAVCAN_NODE_MODE_MAINTENANCE", 2): "MAINTENANCE",
    getattr(mavutil.mavlink, "UAVCAN_NODE_MODE_SOFTWARE_UPDATE", 3): "SOFTWARE_UPDATE",
    getattr(mavutil.mavlink, "UAVCAN_NODE_MODE_OFFLINE", 7): "OFFLINE",
}


PARAM_EXPECT_HOBBYWING: List[Tuple[str, Optional[float], str]] = [
    ("FRAME_CLASS", 1, "Quad"),
    ("FRAME_TYPE", 1, "Quad X"),
    ("SERVO1_FUNCTION", 36, "ThrottleID 1 / rear-right / Motor4"),
    ("SERVO2_FUNCTION", 35, "ThrottleID 2 / front-left / Motor3"),
    ("SERVO3_FUNCTION", 34, "ThrottleID 3 / rear-left / Motor2"),
    ("SERVO4_FUNCTION", 33, "ThrottleID 4 / front-right / Motor1"),
    ("CAN_P1_DRIVER", 1, "CAN1 uses driver 1"),
    ("CAN_P1_BITRATE", 1000000, "CAN1 bitrate 1 Mbps"),
    ("CAN_D1_PROTOCOL", 1, "Driver 1 protocol DroneCAN"),
    ("CAN_D1_UC_ESC_BM", 15, "Enable ESC outputs 1..4"),
    ("CAN_D1_UC_ESC_OF", 0, "No ESC offset for first test"),
    ("CAN_D1_UC_SRV_BM", 0, "Do not send servo outputs as DroneCAN servo"),
    ("CAN_D1_UC_ESC_RV", 0, "No DroneCAN ESC reverse bitmask for first test"),
    ("CAN_D1_UC_OPTION", 128, "Hobbywing ESC option enabled"),
    ("ESC_TLM_MAV_OFS", 0, "Keep MAVLink offset neutral; detect firmware display shift separately"),
    ("MOT_PWM_MIN", 1000, "Motor output minimum"),
    ("MOT_PWM_MAX", 2000, "Motor output maximum"),
]

PARAM_EXPECT_STANDARD: List[Tuple[str, Optional[float], str]] = [
    (name, (0 if name == "CAN_D1_UC_OPTION" else expected), note.replace("Hobbywing ESC option enabled", "Standard DroneCAN RawCommand path"))
    for name, expected, note in PARAM_EXPECT_HOBBYWING
]

PARAM_EXTRA_BENCH: List[str] = [
    "BRD_SAFETY_DEFLT",
    "ARMING_CHECK",
    "BATT_MONITOR",
    "MOT_SAFE_DISARM",
    "MOT_SPIN_ARM",
    "MOT_SPIN_MIN",
    "LOG_DISARMED",
]


@dataclass
class ActiveOutput:
    mode: str
    value: float


def find_serial_ports():
    ports = []
    for pattern in SERIAL_PATTERNS:
        ports.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(ports))


def print_serial_ports():
    ports = find_serial_ports()
    if not ports:
        print("No Pixhawk-like serial ports found. Checked: " + ", ".join(SERIAL_PATTERNS))
        return
    print("Available serial ports:")
    for port in ports:
        print(f"  {port}")


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    ports = find_serial_ports()
    if not ports:
        raise FileNotFoundError("No serial ports found for --port auto")
    return ports[0]


def wait_for_port(port: str, timeout: float) -> str:
    if timeout <= 0:
        return resolve_port(port)
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            resolved = resolve_port(port)
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(0.25)
            continue
        if not resolved.startswith("/dev/") or os.path.exists(resolved):
            return resolved
        last_error = FileNotFoundError(f"serial port does not exist: {resolved}")
        time.sleep(0.25)
    if last_error is not None:
        raise last_error
    return resolve_port(port)


def print_serial_error(port: str, exc: BaseException):
    print(f"ERROR: cannot open/connect serial port {port}: {exc}", file=sys.stderr)
    print("Check these first:", file=sys.stderr)
    print("  1. Close QGroundControl, Mission Planner, MAVProxy, or any other script using the same port.", file=sys.stderr)
    print("  2. Replug Pixhawk and run: python3 pixhawk_x8_cli_diag.py --list-ports", file=sys.stderr)
    print("  3. If it is a permission issue, add the user to dialout and log in again.", file=sys.stderr)
    print("  4. To wait after closing another program, add: --wait-port 10 --connect-retries 5", file=sys.stderr)


def decode_hobbywing_get_esc_id(source_node_id: int, payload: Iterable[int]) -> Optional[Tuple[int, int]]:
    """Decode Hobbywing's broadcast GetEscID response: [NodeID, ThrottleID]."""
    try:
        values = [int(value) for value in payload]
    except (TypeError, ValueError):
        return None
    if len(values) < 2 or values[0] != int(source_node_id):
        return None
    node_id, throttle_id = values[:2]
    if not (1 <= node_id <= 127 and 1 <= throttle_id <= 8):
        return None
    return node_id, throttle_id


def run_dronecan_probe(port: str, baud: int, duration: float, bus_number: int = 1) -> bool:
    """Passively inspect Hobbywing DroneCAN traffic forwarded by ArduPilot."""
    try:
        import dronecan
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--can-probe requires pydronecan; install it in .venv with: .venv/bin/pip install dronecan"
        ) from exc

    get_esc_id_type = dronecan.TYPENAMES["com.hobbywing.esc.GetEscID"]
    status1_type = dronecan.TYPENAMES["com.hobbywing.esc.StatusMsg1"]
    status2_type = dronecan.TYPENAMES["com.hobbywing.esc.StatusMsg2"]
    node_status_type = dronecan.TYPENAMES["uavcan.protocol.NodeStatus"]
    standard_raw_command_type = dronecan.TYPENAMES["uavcan.equipment.esc.RawCommand"]
    hobbywing_raw_command_type = dronecan.TYPENAMES["com.hobbywing.esc.RawCommand"]

    throttle_ids_by_node: Dict[int, set] = {}
    status1_by_node = {}
    status2_by_node = {}
    node_status_by_node = {}
    raw_command_sources: Dict[str, set] = {"standard_raw": set(), "hobbywing_raw": set()}
    last_raw_commands: Dict[str, List[int]] = {}
    nonzero_raw_command_counts = {"standard_raw": 0, "hobbywing_raw": 0}
    max_abs_raw_commands: Dict[str, List[int]] = {}
    message_counts = {
        "get_esc_id": 0,
        "status1": 0,
        "status2": 0,
        "node_status": 0,
        "standard_raw": 0,
        "hobbywing_raw": 0,
    }
    decode_errors: Dict[str, int] = {}

    def on_get_esc_id(event):
        message_counts["get_esc_id"] += 1
        source_node_id = int(event.transfer.source_node_id)
        decoded = decode_hobbywing_get_esc_id(source_node_id, event.message.payload)
        if decoded is not None:
            node_id, throttle_id = decoded
            throttle_ids_by_node.setdefault(node_id, set()).add(throttle_id)

    def on_status1(event):
        message_counts["status1"] += 1
        status1_by_node[int(event.transfer.source_node_id)] = event.message

    def on_status2(event):
        message_counts["status2"] += 1
        status2_by_node[int(event.transfer.source_node_id)] = event.message

    def on_node_status(event):
        message_counts["node_status"] += 1
        node_status_by_node[int(event.transfer.source_node_id)] = event.message

    def record_raw_command(name: str, event, field: str):
        message_counts[name] += 1
        raw_command_sources[name].add(int(event.transfer.source_node_id))
        values = [int(value) for value in getattr(event.message, field)]
        last_raw_commands[name] = values
        if any(values):
            nonzero_raw_command_counts[name] += 1
        maxima = max_abs_raw_commands.setdefault(name, [0] * len(values))
        if len(maxima) < len(values):
            maxima.extend([0] * (len(values) - len(maxima)))
        for index, value in enumerate(values):
            maxima[index] = max(maxima[index], abs(value))

    def on_standard_raw_command(event):
        record_raw_command("standard_raw", event, "cmd")

    def on_hobbywing_raw_command(event):
        record_raw_command("hobbywing_raw", event, "command")

    node = None
    uri = f"mavcan:{port}"
    print(
        f"Passive DroneCAN probe: port={port} baud={baud} CAN{bus_number} "
        f"duration={duration:.1f}s"
    )
    print("No DroneCAN motor command or parameter write will be sent.")
    try:
        node = dronecan.make_node(
            uri,
            node_id=None,
            bus_number=bus_number,
            mavlink_target_system=1,
            baudrate=baud,
            catch_handler_exceptions=False,
        )
        node.add_handler(get_esc_id_type, on_get_esc_id)
        node.add_handler(status1_type, on_status1)
        node.add_handler(status2_type, on_status2)
        node.add_handler(node_status_type, on_node_status)
        node.add_handler(standard_raw_command_type, on_standard_raw_command)
        node.add_handler(hobbywing_raw_command_type, on_hobbywing_raw_command)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            try:
                node.spin(min(0.25, max(0.0, deadline - time.monotonic())))
            except dronecan.UAVCANException as exc:
                # MAVLink forwarding can lose part of an unrelated multi-frame
                # transfer. Keep collecting the single-frame Hobbywing traffic.
                key = str(exc)
                decode_errors[key] = decode_errors.get(key, 0) + 1
    except Exception as exc:
        raise RuntimeError(f"DroneCAN probe failed on {uri}: {exc}") from exc
    finally:
        if node is not None:
            node.close()

    print("\nDroneCAN probe summary")
    print(
        "  frames: "
        + " ".join(f"{name}={count}" for name, count in message_counts.items())
    )
    print(f"  skipped incomplete/invalid transfers: {sum(decode_errors.values())}")
    for error, count in list(decode_errors.items())[:3]:
        print(f"    {count}x {error}")
    print(f"  NodeStatus sources: {sorted(node_status_by_node) or 'none'}")
    esc_nodes = sorted(set(status1_by_node) | set(status2_by_node) | set(throttle_ids_by_node))
    print(f"  Hobbywing ESC sources: {esc_nodes or 'none'}")
    for name in ("standard_raw", "hobbywing_raw"):
        print(
            f"  {name}: count={message_counts[name]} "
            f"nonzero={nonzero_raw_command_counts[name]} "
            f"sources={sorted(raw_command_sources[name]) or 'none'} "
            f"last={last_raw_commands.get(name, 'none')} "
            f"max_abs={max_abs_raw_commands.get(name, 'none')}"
        )

    for node_id in esc_nodes:
        position = EXPECTED_NODE_TO_POSITION.get(node_id, "OTHER")
        throttles = sorted(throttle_ids_by_node.get(node_id, set()))
        details = [
            f"node_id={node_id}",
            f"position={position}",
            f"throttle_id={throttles if throttles else 'not observed'}",
        ]
        status1 = status1_by_node.get(node_id)
        if status1 is not None:
            details.extend((f"rpm={int(status1.rpm)}", f"pwm={int(status1.pwm)}"))
        status2 = status2_by_node.get(node_id)
        if status2 is not None:
            details.extend(
                (
                    f"voltage={int(status2.input_voltage) * 0.1:.1f}V",
                    f"current={int(status2.current) * 0.1:.1f}A",
                    f"temperature={int(status2.temperature)}C",
                )
            )
        print("  " + " ".join(details))

    expected_nodes = set(EXPECTED_NODE_TO_POSITION)
    observed_nodes = set(esc_nodes)
    missing_nodes = sorted(expected_nodes - observed_nodes)
    extra_nodes = sorted(observed_nodes - expected_nodes)
    mismatches = []
    for node_id in sorted(expected_nodes):
        throttles = throttle_ids_by_node.get(node_id, set())
        if throttles != {node_id}:
            mismatches.append((node_id, sorted(throttles)))

    if missing_nodes:
        print(f"  FAIL: missing expected ESC NodeID(s): {missing_nodes}")
    if extra_nodes:
        print(f"  FAIL: unexpected Hobbywing ESC NodeID(s): {extra_nodes}")
    if mismatches:
        print("  FAIL: NodeID/ThrottleID mismatch or GetEscID response missing:")
        for node_id, throttles in mismatches:
            print(f"    NodeID {node_id}: expected ThrottleID {node_id}, observed {throttles or 'none'}")

    passed = not missing_nodes and not extra_nodes and not mismatches
    if passed:
        print("  PASS: NodeID and ThrottleID are 1:1 for all four expected positions.")
    return passed


def run_hobbywing_config_probe(port: str, baud: int, timeout: float, bus_number: int = 1) -> bool:
    """Read each Hobbywing ESC's major configuration without changing it."""
    try:
        import dronecan
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--can-config-probe requires pydronecan; install it in .venv with: .venv/bin/pip install dronecan"
        ) from exc

    config_type = dronecan.TYPENAMES["com.hobbywing.esc.GetMajorConfig"]
    responses = {}
    node = None
    uri = f"mavcan:{port}"
    print(
        f"Read-only Hobbywing config probe: port={port} baud={baud} CAN{bus_number} "
        f"timeout={timeout:.1f}s"
    )
    print("Only GetMajorConfig requests will be sent; no ESC parameter or motor command will be written.")

    def make_callback(node_id: int):
        def callback(event):
            if event is not None:
                responses[node_id] = event.response
        return callback

    try:
        node = dronecan.make_node(
            uri,
            node_id=127,
            bus_number=bus_number,
            mavlink_target_system=1,
            baudrate=baud,
            catch_handler_exceptions=False,
        )
        for node_id in sorted(EXPECTED_NODE_TO_POSITION):
            node.request(config_type.Request(option=0), node_id, make_callback(node_id), timeout=timeout)
        deadline = time.monotonic() + timeout + 0.5
        while time.monotonic() < deadline and len(responses) < len(EXPECTED_NODE_TO_POSITION):
            node.spin(min(0.25, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        raise RuntimeError(f"Hobbywing config probe failed on {uri}: {exc}") from exc
    finally:
        if node is not None:
            node.close()

    print("\nHobbywing major configuration")
    for node_id in sorted(EXPECTED_NODE_TO_POSITION):
        response = responses.get(node_id)
        if response is None:
            print(f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} response=TIMEOUT")
            continue
        throttle_source = int(response.throttle_source)
        source_name = {0: "CAN_DIGITAL", 1: "PWM"}.get(throttle_source, f"UNKNOWN({throttle_source})")
        print(
            f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} "
            f"throttle_source={source_name} throttle_channel={int(response.throttle_channel)} "
            f"direction={int(response.direction)} led_status={int(response.led_status)}"
        )

    missing = sorted(set(EXPECTED_NODE_TO_POSITION) - set(responses))
    wrong_source = sorted(
        node_id for node_id, response in responses.items() if int(response.throttle_source) != 0
    )
    if missing:
        print(f"  FAIL: GetMajorConfig timed out for NodeID(s): {missing}")
    if wrong_source:
        print(f"  FAIL: ESC NodeID(s) not configured for CAN_DIGITAL throttle: {wrong_source}")
    passed = not missing and not wrong_source
    if passed:
        print("  PASS: all four ESCs use CAN_DIGITAL throttle.")
    return passed


def set_hobbywing_can_throttle(port: str, baud: int, timeout: float, bus_number: int = 1) -> bool:
    """Set all expected Hobbywing ESCs to CAN digital throttle."""
    try:
        import dronecan
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--set-can-throttle requires pydronecan; install it in .venv with: .venv/bin/pip install dronecan"
        ) from exc

    service_type = dronecan.TYPENAMES["com.hobbywing.esc.SetThrottleSource"]
    responses = {}
    node = None
    uri = f"mavcan:{port}"
    print(
        f"Hobbywing throttle-source setup: port={port} baud={baud} CAN{bus_number} "
        f"timeout={timeout:.1f}s"
    )
    print("Writing source=0 (CAN_DIGITAL) to expected ESC NodeIDs 1..4; no motor command will be sent.")

    def make_callback(node_id: int):
        def callback(event):
            if event is not None:
                responses[node_id] = event.response
        return callback

    try:
        node = dronecan.make_node(
            uri,
            node_id=127,
            bus_number=bus_number,
            mavlink_target_system=1,
            baudrate=baud,
            catch_handler_exceptions=False,
        )
        for node_id in sorted(EXPECTED_NODE_TO_POSITION):
            node.request(service_type.Request(source=0), node_id, make_callback(node_id), timeout=timeout)
        deadline = time.monotonic() + timeout + 0.5
        while time.monotonic() < deadline and len(responses) < len(EXPECTED_NODE_TO_POSITION):
            node.spin(min(0.25, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        raise RuntimeError(f"Hobbywing throttle-source setup failed on {uri}: {exc}") from exc
    finally:
        if node is not None:
            node.close()

    print("\nHobbywing throttle-source responses")
    failed = []
    for node_id in sorted(EXPECTED_NODE_TO_POSITION):
        response = responses.get(node_id)
        if response is None:
            print(f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} response=TIMEOUT")
            failed.append(node_id)
            continue
        source = int(response.source)
        source_name = {0: "CAN_DIGITAL", 1: "PWM"}.get(source, f"UNKNOWN({source})")
        print(f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} source={source_name}")
        if source != 0:
            failed.append(node_id)
    if failed:
        print(f"  FAIL: CAN_DIGITAL was not acknowledged by NodeID(s): {failed}")
        return False
    print("  PASS: all four ESCs acknowledged CAN_DIGITAL throttle.")
    return True


def run_dronecan_node_info_probe(port: str, baud: int, timeout: float, bus_number: int = 1) -> bool:
    """Read standard DroneCAN node identity and firmware information."""
    try:
        import dronecan
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--can-node-info requires pydronecan; install it in .venv with: .venv/bin/pip install dronecan"
        ) from exc

    responses = {}
    node = None
    uri = f"mavcan:{port}"
    print(f"Read-only DroneCAN node-info probe: port={port} baud={baud} CAN{bus_number} timeout={timeout:.1f}s")

    def make_callback(node_id: int):
        def callback(event):
            if event is not None:
                responses[node_id] = event.response
        return callback

    try:
        node = dronecan.make_node(
            uri,
            node_id=127,
            bus_number=bus_number,
            mavlink_target_system=1,
            baudrate=baud,
            catch_handler_exceptions=False,
        )
        for node_id in sorted(EXPECTED_NODE_TO_POSITION):
            request = dronecan.uavcan.protocol.GetNodeInfo.Request()
            node.request(request, node_id, make_callback(node_id), timeout=timeout)
        deadline = time.monotonic() + timeout + 0.5
        while time.monotonic() < deadline and len(responses) < len(EXPECTED_NODE_TO_POSITION):
            node.spin(min(0.25, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        raise RuntimeError(f"DroneCAN node-info probe failed on {uri}: {exc}") from exc
    finally:
        if node is not None:
            node.close()

    print("\nDroneCAN node information")
    for node_id in sorted(EXPECTED_NODE_TO_POSITION):
        response = responses.get(node_id)
        if response is None:
            print(f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} response=TIMEOUT")
            continue
        name = bytes(int(value) for value in response.name).rstrip(b"\0").decode("ascii", errors="replace")
        software = response.software_version
        hardware = response.hardware_version
        print(
            f"  node_id={node_id} position={EXPECTED_NODE_TO_POSITION[node_id]} name={name or 'UNKNOWN'} "
            f"sw={int(software.major)}.{int(software.minor)} vcs_commit={int(software.vcs_commit):08x} "
            f"hw={int(hardware.major)}.{int(hardware.minor)}"
        )
    missing = sorted(set(EXPECTED_NODE_TO_POSITION) - set(responses))
    if missing:
        print(f"  FAIL: GetNodeInfo timed out for NodeID(s): {missing}")
        return False
    print("  PASS: all four ESCs returned standard DroneCAN node information.")
    return True


def connect_with_retries(cli, show_help: bool, retries: int, delay: float):
    attempts = max(1, retries + 1)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            cli.connect(show_help=show_help)
            return
        except SERIAL_EXCEPTIONS as exc:
            last_exc = exc
            cli.close()
            if attempt >= attempts:
                raise
            print(f"Serial open/connect failed on attempt {attempt}/{attempts}: {exc}", file=sys.stderr)
            print(f"Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc


class PixhawkX8CLI:
    def __init__(self, port: str, baud: int, connect_timeout: float = 30.0):
        self.port = port
        self.baud = baud
        self.connect_timeout = connect_timeout
        self.master = None

        self.target_system = 1
        self.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        self.tx_enabled = False
        self.active: Dict[int, ActiveOutput] = {}

        # Bench-safe defaults. Increase manually only when needed.
        self.pwm_min = DEFAULT_PWM_MIN
        self.pwm_limit = DEFAULT_PWM_LIMIT
        self.percent_limit = DEFAULT_PERCENT_LIMIT

        # ArduPilot motor test requires a timeout. We emulate "continuous output"
        # by refreshing commands periodically before the timeout expires.
        self.refresh_period = DEFAULT_REFRESH_PERIOD
        self.hold_duration = DEFAULT_HOLD_DURATION
        self.safety_retry_period = 1.0
        self.esc_gate = "all"

        self.last_refresh = 0.0
        self.safety_off_confirmed = False
        self.motor_function_to_output_channel: Dict[int, int] = dict(FALLBACK_MOTOR_FUNCTION_TO_OUTPUT_CHANNEL)
        self.last_heartbeat = None
        self.last_sys_status = None
        self.last_gps = None
        self.last_esc = None
        self.last_esc_status = None
        self.last_esc_telemetry = None
        self.last_autopilot_version = None
        self.last_esc_telemetry_by_type = {}
        self.last_servo_output = None
        self.last_rc_channels = None
        self.last_attitude = None
        self.last_global_position = None
        self.last_vfr_hud = None
        self.last_ekf_status = None
        self.last_esc_time = 0.0
        self.last_esc_status_time = 0.0
        self.last_esc_telemetry_time = 0.0
        self.last_esc_telemetry_time_by_type = {}
        self.esc_status_frames = 0
        self.esc_telemetry_frames = 0
        self.esc_telemetry_frames_by_type = {name: 0 for name, _, _ in ESC_TELEMETRY_GROUPS}
        self.uavcan_node_statuses = {}
        self.uavcan_node_infos = {}
        self.command_acks = []
        self.statustexts: List[str] = []
        self.last_motor_test_error = ""

        self.verbose = True
        self.outmon = True
        self.ack_required = True
        self.param_cache: Dict[str, float] = {}
        self.start_monotonic = time.monotonic()
        self.last_command_ok = True

    # ---------- generic helpers ----------

    def ts(self) -> str:
        wall = time.strftime("%H:%M:%S", time.localtime())
        millis = int((time.time() % 1.0) * 1000)
        elapsed = time.monotonic() - self.start_monotonic
        return f"{wall}.{millis:03d} +{elapsed:8.3f}s"

    def log(self, text: str):
        print(f"[{self.ts()}] {text}")

    def mark(self, text: str):
        print("\n" + "#" * 72)
        self.log(f"MARK: {text}")
        print("#" * 72 + "\n")

    def result_name(self, result: int) -> str:
        return RESULT_TEXT.get(int(result), f"UNKNOWN({result})")

    def connect(self, show_help: bool = True):
        self.close()
        print(f"Connecting to {self.port} at {self.baud}...")
        self.master = mavutil.mavlink_connection(
            self.port,
            baud=self.baud,
            source_system=255,
            source_component=0,
        )

        print(f"Waiting for Pixhawk heartbeat, timeout={self.connect_timeout:.1f}s...")
        heartbeat = self.master.wait_heartbeat(timeout=self.connect_timeout)
        if heartbeat is None:
            raise TimeoutError(f"No Pixhawk heartbeat received on {self.port} within {self.connect_timeout:.1f}s")

        self.target_system = self.master.target_system
        self.target_component = self.master.target_component or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        print("Connected.")
        print(f"target_system={self.target_system}, target_component={self.target_component}")
        self.mark("CLI connected. For X8 G2 use 'streams on', 'audit std', 'safety off', then run the test.")
        if show_help:
            self.print_help()

    def close(self):
        master = self.master
        if master is not None:
            if self.safety_off_confirmed:
                try:
                    self.mark("FAIL-SAFE CLOSE: stop motor-test outputs and restore safety ON")
                    self.trigger_off()
                    self.set_safety_switch(dangerous=False)
                except Exception as exc:
                    print(f"WARNING: fail-safe X8 cleanup failed: {exc}", file=sys.stderr)
            self.master = None
            try:
                master.close()
            except Exception:
                pass

    def require_disarmed_motor_test(self) -> bool:
        armed = self.is_armed()
        if armed is False:
            return True
        print(f"Rejected: X8 motor test requires a confirmed DISARMED heartbeat; armed={armed}.")
        self.last_command_ok = False
        return False

    def send_command_long(self, command: int, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            command,
            0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
        )

    def wait_for_command_ack(
        self,
        command: int,
        timeout: float = 2.0,
        start_index: Optional[int] = None,
        accepted_results: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        if start_index is None:
            start_index = len(self.command_acks)
        if accepted_results is None:
            accepted_results = (MAV_RESULT_ACCEPTED, MAV_RESULT_IN_PROGRESS)
        accepted_results = tuple(int(result) for result in accepted_results)
        end = time.time() + timeout
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            for ack in self.command_acks[start_index:]:
                if int(ack.command) != int(command):
                    continue
                ok = int(ack.result) in accepted_results
                if ok:
                    self.last_command_ok = True
                    self.log(f"ACK OK: command={ack.command}, result={ack.result} ({self.result_name(ack.result)})")
                else:
                    self.log(f"ACK REJECTED: command={ack.command}, result={ack.result} ({self.result_name(ack.result)})")
                    self.last_command_ok = False
                    self.collect_statustext(0.25)
                    if self.last_motor_test_error:
                        self.log(f"MOTOR TEST ERROR: {self.last_motor_test_error}")
                return ok
            time.sleep(0.03)
        self.log(f"ACK TIMEOUT: command={command} after {timeout:.1f}s")
        self.last_command_ok = False
        return False

    def set_message_interval(self, msg_id: int, hz: float):
        if hz <= 0:
            interval_us = -1
        else:
            interval_us = int(1_000_000 / hz)
        self.send_command_long(
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            msg_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def request_streams(self, enable: bool):
        if enable:
            self.log("Requesting MAVLink diagnostic streams: SERVO_OUTPUT_RAW, RC_CHANNELS, SYS_STATUS, ESC telemetry, GPS, attitude/position")
            self.set_message_interval(MSG_SERVO_OUTPUT_RAW, 10)
            self.set_message_interval(MSG_RC_CHANNELS, 2)
            self.set_message_interval(MSG_SYS_STATUS, 2)
            self.set_message_interval(MSG_ESC_STATUS, 5)
            for _, msg_id, _ in ESC_TELEMETRY_GROUPS:
                self.set_message_interval(msg_id, 5)
            self.set_message_interval(MSG_ATTITUDE, 10)
            self.set_message_interval(MSG_GLOBAL_POSITION_INT, 5)
            self.set_message_interval(MSG_GPS_RAW_INT, 2)
            self.set_message_interval(MSG_VFR_HUD, 2)
            self.set_message_interval(MSG_EKF_STATUS_REPORT, 2)
            self.send_command_long(MAV_CMD_REQUEST_MESSAGE, MSG_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
        else:
            self.log("Stopping requested diagnostic streams")
            self.set_message_interval(MSG_SERVO_OUTPUT_RAW, 0)
            self.set_message_interval(MSG_RC_CHANNELS, 0)
            self.set_message_interval(MSG_SYS_STATUS, 0)
            self.set_message_interval(MSG_ESC_STATUS, 0)
            for _, msg_id, _ in ESC_TELEMETRY_GROUPS:
                self.set_message_interval(msg_id, 0)
            self.set_message_interval(MSG_ATTITUDE, 0)
            self.set_message_interval(MSG_GLOBAL_POSITION_INT, 0)
            self.set_message_interval(MSG_GPS_RAW_INT, 0)
            self.set_message_interval(MSG_VFR_HUD, 0)
            self.set_message_interval(MSG_EKF_STATUS_REPORT, 0)

    # ---------- MAVLink polling ----------

    def poll_mavlink(self, print_messages: bool = True):
        for _ in range(100):
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                continue
            if msg_type == "HEARTBEAT":
                self.last_heartbeat = msg
            elif msg_type == "SYS_STATUS":
                self.last_sys_status = msg
            elif msg_type == "GPS_RAW_INT":
                self.last_gps = msg
            elif msg_type == "RC_CHANNELS":
                self.last_rc_channels = msg
                if print_messages and self.verbose:
                    self.print_rc_channels(msg)
            elif msg_type == "SERVO_OUTPUT_RAW":
                self.last_servo_output = msg
                if print_messages and self.outmon:
                    self.print_servo_output(msg)
            elif msg_type == "ATTITUDE":
                self.last_attitude = msg
            elif msg_type == "GLOBAL_POSITION_INT":
                self.last_global_position = msg
            elif msg_type == "VFR_HUD":
                self.last_vfr_hud = msg
            elif msg_type == "EKF_STATUS_REPORT":
                self.last_ekf_status = msg
            elif msg_type == "AUTOPILOT_VERSION":
                self.last_autopilot_version = msg
                if print_messages and self.verbose:
                    print(f"\n[{self.ts()}] AUTOPILOT_VERSION: {self.autopilot_version_text(msg)}")
            elif msg_type == "ESC_STATUS":
                self.last_esc = msg
                self.last_esc_time = time.time()
                self.last_esc_status = msg
                self.last_esc_status_time = self.last_esc_time
                self.esc_status_frames += 1
                if print_messages and self.verbose:
                    print(f"\n[{self.ts()}] {msg_type}: {msg}")
            elif msg_type in ESC_TELEMETRY_FIRST_ID_BY_TYPE:
                self.last_esc = msg
                self.last_esc_time = time.time()
                self.last_esc_telemetry = msg
                self.last_esc_telemetry_time = self.last_esc_time
                self.last_esc_telemetry_by_type[msg_type] = msg
                self.last_esc_telemetry_time_by_type[msg_type] = self.last_esc_time
                self.esc_telemetry_frames += 1
                self.esc_telemetry_frames_by_type[msg_type] = self.esc_telemetry_frames_by_type.get(msg_type, 0) + 1
                if print_messages and self.verbose:
                    print(f"\n[{self.ts()}] {msg_type}: {msg}")
            elif msg_type == "UAVCAN_NODE_STATUS":
                key = self.mavlink_source_key(msg)
                self.uavcan_node_statuses[key] = msg
                if print_messages and self.verbose:
                    self.print_uavcan_node_status(key, msg)
            elif msg_type == "UAVCAN_NODE_INFO":
                key = self.mavlink_source_key(msg)
                self.uavcan_node_infos[key] = msg
                if print_messages and self.verbose:
                    self.print_uavcan_node_info(key, msg)
            elif msg_type == "STATUSTEXT":
                text = str(msg.text)
                self.statustexts.append(text)
                if "Motor Test:" in text:
                    self.last_motor_test_error = text
                if print_messages:
                    print(f"\n[{self.ts()}] STATUSTEXT: {msg.text}")
            elif msg_type == "COMMAND_ACK":
                self.command_acks.append(msg)
                if print_messages:
                    print(f"\n[{self.ts()}] ACK: command={msg.command}, result={msg.result} ({self.result_name(msg.result)})")
            elif msg_type == "PARAM_VALUE":
                pid = msg.param_id
                if isinstance(pid, bytes):
                    pid = pid.decode(errors="ignore").strip("\x00")
                self.param_cache[str(pid)] = float(msg.param_value)
                if print_messages:
                    print(f"\n[{self.ts()}] PARAM: {pid} = {msg.param_value}")

    def mavlink_source_key(self, msg) -> Tuple[int, int]:
        src_system = msg.get_srcSystem() if hasattr(msg, "get_srcSystem") else 0
        src_component = msg.get_srcComponent() if hasattr(msg, "get_srcComponent") else 0
        return int(src_system or 0), int(src_component or 0)

    def autopilot_version_text(self, msg=None) -> str:
        msg = msg or self.last_autopilot_version
        if msg is None:
            return "UNKNOWN"
        packed = int(getattr(msg, "flight_sw_version", 0))
        major = (packed >> 24) & 0xFF
        minor = (packed >> 16) & 0xFF
        patch = (packed >> 8) & 0xFF
        release_type = packed & 0xFF
        release_names = {0: "dev", 64: "alpha", 128: "beta", 192: "rc", 255: "official"}
        release = release_names.get(release_type, str(release_type))
        custom = bytes(getattr(msg, "flight_custom_version", b"") or b"").hex()
        return f"{major}.{minor}.{patch} {release} git={custom or 'unknown'}"

    def uavcan_health_name(self, value: int) -> str:
        return UAVCAN_HEALTH_TEXT.get(int(value), f"UNKNOWN({value})")

    def uavcan_mode_name(self, value: int) -> str:
        return UAVCAN_MODE_TEXT.get(int(value), f"UNKNOWN({value})")

    def print_uavcan_node_status(self, key: Tuple[int, int], msg):
        sysid, compid = key
        position = EXPECTED_NODE_TO_POSITION.get(compid, "OTHER")
        print(
            f"\n[{self.ts()}] UAVCAN_NODE_STATUS: node_id={compid} "
            f"position={position} bridge_system={sysid} "
            f"uptime={msg.uptime_sec}s health={self.uavcan_health_name(msg.health)} "
            f"mode={self.uavcan_mode_name(msg.mode)} vendor_status={msg.vendor_specific_status_code}"
        )

    def print_uavcan_node_info(self, key: Tuple[int, int], msg):
        sysid, compid = key
        name = str(getattr(msg, "name", "")).strip()
        position = EXPECTED_NODE_TO_POSITION.get(compid, "OTHER")
        print(
            f"\n[{self.ts()}] UAVCAN_NODE_INFO: node_id={compid} "
            f"position={position} bridge_system={sysid} "
            f"name={name or 'UNKNOWN'} hw={msg.hw_version_major}.{msg.hw_version_minor} "
            f"sw={msg.sw_version_major}.{msg.sw_version_minor}"
        )

    def print_servo_output(self, msg):
        fields = []
        for i in range(1, 9):
            name = f"servo{i}_raw"
            if hasattr(msg, name):
                fields.append(f"ch{i}={getattr(msg, name)}")
        if fields:
            print(f"\n[{self.ts()}] SERVO_OUTPUT_RAW: " + " ".join(fields[:8]))

    def print_rc_channels(self, msg):
        fields = []
        for i in range(1, 9):
            name = f"chan{i}_raw"
            if hasattr(msg, name):
                fields.append(f"rc{i}={getattr(msg, name)}")
        if fields:
            print(f"\n[{self.ts()}] RC_CHANNELS: " + " ".join(fields[:8]))

    def esc_telemetry_presence(self, msg) -> Optional[Tuple[List[int], List[int], List[str]]]:
        if msg is None:
            return None
        msg_type = msg.get_type()
        first_id = ESC_TELEMETRY_FIRST_ID_BY_TYPE.get(msg_type)
        if first_id is None:
            return None
        voltages = list(getattr(msg, "voltage", []) or [])
        currents = list(getattr(msg, "current", []) or [])
        rpms = list(getattr(msg, "rpm", []) or [])
        counts = list(getattr(msg, "count", []) or [])
        max_len = max(len(voltages), len(currents), len(rpms), len(counts), 0)
        if max_len == 0:
            return None
        present = []
        missing = []
        voltage_text = []
        for i in range(max_len):
            voltage = voltages[i] if i < len(voltages) else 0
            current = currents[i] if i < len(currents) else 0
            rpm = rpms[i] if i < len(rpms) else 0
            count = counts[i] if i < len(counts) else 0
            esc_id = first_id + i
            if voltage or current or rpm or count:
                present.append(esc_id)
            else:
                missing.append(esc_id)
            voltage_text.append(f"esc{esc_id}={voltage / 100.0:.2f}V" if voltage else f"esc{esc_id}=0")
        return present, missing, voltage_text

    def esc_telemetry_summary(self, msg) -> Optional[str]:
        presence = self.esc_telemetry_presence(msg)
        if presence is None:
            return None
        present, missing, voltage_text = presence
        present_text = ",".join(str(x) for x in present) if present else "none"
        missing_text = ",".join(str(x) for x in missing) if missing else "none"
        return f"esc_telemetry present={present_text}, missing={missing_text}, " + " ".join(voltage_text)

    def esc_telemetry_age(self) -> Optional[float]:
        if not self.last_esc_telemetry_time:
            return None
        return time.time() - self.last_esc_telemetry_time

    def esc_telemetry_items(self):
        if self.last_esc_telemetry_by_type:
            for msg_type, _, first_id in ESC_TELEMETRY_GROUPS:
                msg = self.last_esc_telemetry_by_type.get(msg_type)
                msg_time = self.last_esc_telemetry_time_by_type.get(msg_type)
                if msg is not None and msg_time is not None:
                    yield msg_type, first_id, msg, msg_time
        elif self.last_esc_telemetry is not None and self.last_esc_telemetry_time:
            msg_type = self.last_esc_telemetry.get_type()
            first_id = ESC_TELEMETRY_FIRST_ID_BY_TYPE.get(msg_type)
            if first_id is not None:
                yield msg_type, first_id, self.last_esc_telemetry, self.last_esc_telemetry_time

    def esc_telemetry_is_fresh(self) -> bool:
        now = time.time()
        return any(now - msg_time <= ESC_TELEMETRY_MAX_AGE for _, _, _, msg_time in self.esc_telemetry_items())

    def esc_telemetry_aggregate_presence(self, fresh_only: bool = True) -> Tuple[List[int], List[str], List[str]]:
        now = time.time()
        present = []
        groups = []
        voltage_text = []
        for msg_type, _, msg, msg_time in self.esc_telemetry_items():
            if fresh_only and now - msg_time > ESC_TELEMETRY_MAX_AGE:
                continue
            presence = self.esc_telemetry_presence(msg)
            if presence is None:
                continue
            msg_present, _, msg_voltage_text = presence
            present.extend(msg_present)
            groups.append(msg_type)
            voltage_text.extend(msg_voltage_text)
        return sorted(dict.fromkeys(present)), groups, voltage_text

    def esc_telemetry_aggregate_summary(self, fresh_only: bool = True) -> Optional[str]:
        present, groups, voltage_text = self.esc_telemetry_aggregate_presence(fresh_only=fresh_only)
        if not groups:
            return None
        present_text = ",".join(str(x) for x in present) if present else "none"
        groups_text = ",".join(groups)
        return f"esc_telemetry present={present_text}, groups={groups_text}, " + " ".join(voltage_text)

    def battery_voltage_current(self) -> Tuple[Optional[float], Optional[float]]:
        if self.last_sys_status is None:
            return None, None
        return self.last_sys_status.voltage_battery / 1000.0, self.last_sys_status.current_battery / 100.0

    def collect_statustext(self, sec: float):
        end = time.time() + sec
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            time.sleep(0.03)

    def listen(self, sec: float):
        self.log(f"Listening for {sec:.1f}s")
        end = time.time() + sec
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            self.refresh_active_outputs()
            time.sleep(0.03)

    # ---------- parameter helpers ----------

    def set_param(self, name: str, value: float):
        self.log(f"Setting param {name} = {value}")
        self.master.mav.param_set_send(
            self.target_system,
            self.target_component,
            name.encode("ascii"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

    def get_param(self, name: str):
        self.log(f"Requesting param {name}")
        self.master.mav.param_request_read_send(
            self.target_system,
            self.target_component,
            name.encode("ascii"),
            -1,
        )

    def get_params(self, names: Iterable[str], wait: float = 5.0) -> Dict[str, Optional[float]]:
        names = list(dict.fromkeys(names))
        for name in names:
            self.get_param(name)
            time.sleep(0.03)
        deadline = time.time() + wait
        while time.time() < deadline:
            self.poll_mavlink(print_messages=False)
            if all(name in self.param_cache for name in names):
                break
            time.sleep(0.03)
        return {name: self.param_cache.get(name) for name in names}

    def update_motor_output_mapping_from_values(self, values: Dict[str, Optional[float]]) -> Dict[int, int]:
        mapping = dict(self.motor_function_to_output_channel or FALLBACK_MOTOR_FUNCTION_TO_OUTPUT_CHANNEL)
        for channel in range(1, 5):
            value = values.get(f"SERVO{channel}_FUNCTION")
            if value is None:
                continue
            motor_function = int(round(float(value)))
            if motor_function in MOTOR_FUNCTION_TO_POSITION:
                mapping[motor_function] = channel
        self.motor_function_to_output_channel = mapping
        return mapping

    def ensure_motor_output_mapping(self):
        if all(name in self.param_cache for name in SERVO_FUNCTION_PARAM_NAMES):
            self.update_motor_output_mapping_from_values(self.param_cache)
            return
        values = self.get_params(SERVO_FUNCTION_PARAM_NAMES, wait=1.5)
        self.update_motor_output_mapping_from_values(values)

    def current_can_esc_offset(self) -> int:
        value = self.param_cache.get(CAN_ESC_OFFSET_PARAM)
        if value is None:
            return 0
        return max(0, int(round(float(value))))

    def ensure_can_esc_offset(self) -> int:
        if CAN_ESC_OFFSET_PARAM not in self.param_cache:
            self.get_params([CAN_ESC_OFFSET_PARAM], wait=1.0)
        return self.current_can_esc_offset()

    def dronecan_esc_id_for_output_channel(self, channel: int) -> int:
        return int(channel) + self.ensure_can_esc_offset()

    def required_all_dronecan_esc_ids(self) -> List[int]:
        offset = self.ensure_can_esc_offset()
        return [offset + channel for channel in (1, 2, 3, 4)]

    def output_channel_for_motor_test_id(self, motor_id: int) -> Optional[int]:
        motor_function = MOTOR_TEST_TO_MOTOR_FUNCTION.get(motor_id)
        if motor_function is None:
            return None
        return self.motor_function_to_output_channel.get(motor_function)

    def output_channel_for_position(self, position: str) -> Optional[int]:
        motor_function = POSITION_TO_MOTOR_FUNCTION.get(position.lower())
        if motor_function is None:
            return None
        return self.motor_function_to_output_channel.get(motor_function)

    def print_motor_output_mapping(self):
        mapping = self.motor_function_to_output_channel or FALLBACK_MOTOR_FUNCTION_TO_OUTPUT_CHANNEL
        esc_offset = self.current_can_esc_offset()
        print("\nMotor output mapping from SERVOx_FUNCTION:")
        for position in ("fr", "rr", "rl", "fl"):
            motor_function = POSITION_TO_MOTOR_FUNCTION[position]
            motor_test_id = MOTOR_FUNCTION_TO_MOTOR_TEST[motor_function]
            channel = mapping.get(motor_function)
            channel_text = (
                f"ch{channel}/ThrottleID{channel + esc_offset}/NodeID{channel + esc_offset}"
                if channel is not None else "UNKNOWN"
            )
            print(f"  {position.upper():<2} -> motor-test M{motor_test_id} -> {channel_text} (SERVO function {motor_function})")

    def audit_params(self, mode: str):
        if mode == "hw":
            expected = PARAM_EXPECT_HOBBYWING
            title = "Legacy Hobbywing vendor RawCommand path, CAN_D1_UC_OPTION=128"
        elif mode == "std":
            expected = PARAM_EXPECT_STANDARD
            title = "X8 G2 standard uavcan.equipment.esc.RawCommand path, CAN_D1_UC_OPTION=0"
        else:
            print("Usage: audit hw|std")
            return
        names = [x[0] for x in expected] + PARAM_EXTRA_BENCH
        self.mark(f"PARAM AUDIT START: {title}")
        values = self.get_params(names, wait=6.0)
        self.update_motor_output_mapping_from_values(values)
        print("\nParameter audit:")
        print(f"{'PARAM':<22} {'VALUE':>12} {'EXPECT':>12}  RESULT  NOTE")
        print("-" * 82)
        for name, expect, note in expected:
            value = values.get(name)
            result = "?"
            if value is None:
                result = "MISS"
            elif expect is None:
                result = "INFO"
            elif abs(float(value) - float(expect)) < 0.5:
                result = "OK"
            else:
                result = "CHECK"
            print(f"{name:<22} {str(value):>12} {str(expect):>12}  {result:<6} {note}")
        print("\nBench/safety related params, not strict pass/fail:")
        for name in PARAM_EXTRA_BENCH:
            print(f"  {name:<22} {values.get(name)}")
        self.print_motor_output_mapping()
        self.mark("PARAM AUDIT END")

    def apply_setup(self, mode: str):
        if mode == "hw":
            expected = PARAM_EXPECT_HOBBYWING
            title = "Legacy Hobbywing vendor RawCommand setup: CAN_D1_UC_OPTION=128"
        elif mode == "std":
            expected = PARAM_EXPECT_STANDARD
            title = "X8 G2 standard RawCommand setup: CAN_D1_UC_OPTION=0"
        else:
            print("Usage: setup hw|std")
            return
        self.mark(f"APPLY PARAM SETUP START: {title}")
        for name, value, _ in expected:
            if value is not None:
                self.set_param(name, value)
                time.sleep(0.08)
        self.log("Setup commands sent. Run 'audit hw' or 'audit std'. Then run 'reboot' before testing.")
        self.mark("APPLY PARAM SETUP END")

    # ---------- motor-test helpers ----------

    def require_safety_off(self) -> bool:
        if self.safety_off_confirmed:
            return True
        print("Rejected: safety off is not confirmed. Type: safety off, wait for ACK ACCEPTED, then confirm X8 LEDs are solid.")
        self.last_command_ok = False
        return False

    def motor_ids_to_esc_ids(self, motor_ids: Iterable[int]) -> Optional[List[int]]:
        self.ensure_motor_output_mapping()
        esc_ids = []
        for motor_id in motor_ids:
            channel = self.output_channel_for_motor_test_id(motor_id)
            if channel is None:
                print(f"Cannot map motor-test M{motor_id} to a DroneCAN ESC/output channel.")
                self.last_command_ok = False
                return None
            esc_ids.append(self.dronecan_esc_id_for_output_channel(channel))
        return sorted(dict.fromkeys(esc_ids))

    def parse_trigger_target_motor_ids(self, target: str) -> Optional[List[int]]:
        target = target.lower()
        if target == "all":
            return list(range(1, 5))
        motor_id = self.parse_motor_or_output_target(target)
        if motor_id is None:
            self.last_command_ok = False
            return None
        return [motor_id]

    def required_esc_ids_for_gate(self, motor_ids: Iterable[int]) -> Optional[List[int]]:
        mode = self.esc_gate.lower()
        target_esc_ids = self.motor_ids_to_esc_ids(motor_ids)
        if target_esc_ids is None:
            return None
        if mode == "all":
            return self.required_all_dronecan_esc_ids()
        if mode == "target":
            return target_esc_ids
        if mode == "off":
            return []
        print("Internal error: esc_gate must be off, target, or all")
        self.last_command_ok = False
        return None

    def require_dronecan_ready(self, motor_ids: Iterable[int]) -> bool:
        required = self.required_esc_ids_for_gate(motor_ids)
        if required is None:
            return False
        if not required:
            return True

        if not self.esc_telemetry_is_fresh():
            self.log("Checking DroneCAN ESC telemetry before motor-test trigger")
            self.request_streams(True)
            self.listen(1.2)

        if not self.esc_telemetry_is_fresh():
            age = self.esc_telemetry_age()
            if age is None:
                print("Rejected: no DroneCAN ESC telemetry received. Run escdiag 3 and check CAN wiring/bitrate/ESC power.")
            else:
                print(
                    "Rejected: DroneCAN ESC telemetry is stale. "
                    f"last_age={age:.2f}s, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s. "
                    "Run escdiag 3 and check CAN wiring/bitrate/ESC power."
                )
            self.last_command_ok = False
            return False

        present, groups, _ = self.esc_telemetry_aggregate_presence(fresh_only=True)
        if not groups:
            print("Rejected: no DroneCAN ESC telemetry received. Run escdiag 3 and check CAN wiring/bitrate/ESC power.")
            self.last_command_ok = False
            return False

        required_mavlink_slots, display_shift = self.mavlink_slots_for_dronecan_ids(required, present)
        present_set = set(present)
        missing_required = [slot for slot in required_mavlink_slots if slot not in present_set]
        if missing_required:
            print(
                "Rejected: DroneCAN ESC telemetry is not ready for trigger. "
                f"required_dronecan_ids={required}, required_mavlink_slots={required_mavlink_slots}, "
                f"present={present}, missing_mavlink_slots={missing_required}. "
                "For your X8 test keep escgate=all until all LEDs are solid; use escgate target/off only for isolated bench debug."
            )
            self.last_command_ok = False
            return False
        if display_shift:
            print(
                f"INFO: fresh telemetry uses MAVLink display shift {display_shift:+d}; "
                "DroneCAN NodeID/ThrottleID and CAN_D1_UC_ESC_OF are unchanged."
            )
        return True

    def trigger_check(self, target: str, wait: float = 1.5):
        if wait < 0 or wait > 30:
            print("triggercheck wait must be 0..30 seconds")
            self.last_command_ok = False
            return
        motor_ids = self.parse_trigger_target_motor_ids(target)
        if motor_ids is None:
            return
        self.ensure_motor_output_mapping()
        target_esc_ids = self.motor_ids_to_esc_ids(motor_ids)
        required = self.required_esc_ids_for_gate(motor_ids)
        if target_esc_ids is None or required is None:
            return
        print("========== TRIGGER CHECK ==========")
        print(f"target={target}, motor_ids={motor_ids}, target_esc_ids={target_esc_ids}")
        print(f"safety_off_confirmed={self.safety_off_confirmed}, escgate={self.esc_gate}, required_esc_ids={required}")
        self.print_motor_output_mapping()
        self.request_streams(True)
        if wait > 0:
            self.listen(wait)
        summary = self.esc_telemetry_aggregate_summary(fresh_only=False)
        print(summary or "esc_telemetry=NONE")
        fresh = self.esc_telemetry_is_fresh()
        if self.last_esc_telemetry_time:
            age = time.time() - self.last_esc_telemetry_time
            print(f"esc_telemetry_frames={self.esc_telemetry_frames}, last_age={age:.2f}s")
            frame_groups = [
                f"{msg_type}={self.esc_telemetry_frames_by_type.get(msg_type, 0)}"
                for msg_type, _, _ in ESC_TELEMETRY_GROUPS
                if self.esc_telemetry_frames_by_type.get(msg_type, 0)
            ]
            if frame_groups:
                print("esc_telemetry_frame_groups=" + ", ".join(frame_groups))
        print(f"esc_telemetry_fresh={fresh}, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s")
        present, groups, _ = self.esc_telemetry_aggregate_presence(fresh_only=True)
        if required and groups:
            required_slots, display_shift = self.mavlink_slots_for_dronecan_ids(required, present)
            missing_required = [slot for slot in required_slots if slot not in set(present)]
            print(f"trigger_ready={fresh and not missing_required and self.safety_off_confirmed}")
            print(f"required_mavlink_slots={required_slots}, mavlink_display_shift={display_shift}")
            if missing_required:
                print(f"missing_mavlink_slots={missing_required}")
            if not fresh:
                print("not_ready_reason=stale_or_missing_esc_telemetry")
        elif required:
            print("trigger_ready=False")
            if not fresh:
                print("not_ready_reason=stale_or_missing_esc_telemetry")
        else:
            print(f"trigger_ready={self.safety_off_confirmed}")
        observed_present, _, _ = self.esc_telemetry_aggregate_presence(fresh_only=False)
        display_shifts = self.infer_mavlink_slot_shifts(observed_present)
        if display_shifts:
            print(f"mavlink_display_shift_candidates={display_shifts}")
        print("===================================")

    def require_motor_test_ready(self, motor_ids: Optional[Iterable[int]] = None) -> bool:
        if not self.require_disarmed_motor_test():
            return False
        if not self.tx_enabled:
            print("Rejected: tx is OFF. Type: tx on")
            self.last_command_ok = False
            return False
        if not self.require_safety_off():
            return False
        if motor_ids is None:
            motor_ids = range(1, 5)
        return self.require_dronecan_ready(motor_ids)

    def send_motor_test_once(self, motor_id: int, mode: str, value: float):
        if mode == MODE_PWM:
            throttle_type = 1       # param2 = 1: PWM
            throttle_value = int(value)
        elif mode == MODE_PERCENT:
            throttle_type = 0       # param2 = 0: percent
            throttle_value = float(value)
        else:
            print(f"Invalid motor test mode: {mode}")
            return

        self.send_command_long(
            mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
            motor_id,              # param1: motor instance, 1-based
            throttle_type,          # param2: throttle type
            throttle_value,         # param3: throttle value
            self.hold_duration,     # param4: timeout seconds
            1,                      # param5: motor count
            0,                      # param6: test order
            0,                      # param7: unused
        )

    def refresh_active_outputs(self):
        if not self.tx_enabled:
            return
        if not self.safety_off_confirmed:
            return
        if not self.active:
            return
        now = time.time()
        if now - self.last_refresh < self.refresh_period:
            return
        self.last_refresh = now
        for motor_id, out in sorted(self.active.items()):
            self.send_motor_test_once(motor_id, out.mode, out.value)
            time.sleep(0.03)

    def check_motor_id(self, motor_id: int) -> bool:
        if motor_id < 1 or motor_id > 4:
            print("Motor ID must be 1..4")
            return False
        return True

    def motor_label(self, motor_id: int) -> str:
        position = MOTOR_TEST_TO_POSITION.get(motor_id)
        channel = self.output_channel_for_motor_test_id(motor_id)
        if channel is None:
            return f"M{motor_id}/{position}" if position else f"M{motor_id}"
        if position:
            return f"M{motor_id}/{position}/ch{channel}"
        return f"M{motor_id}/ch{channel}"

    def output_channel_to_motor_id(self, channel: int) -> Optional[int]:
        self.ensure_motor_output_mapping()
        motor_id = self.motor_id_for_output_channel(channel)
        if motor_id is not None:
            return motor_id
        print("Output channel must be ch1..ch4 with SERVOx_FUNCTION set to Motor1..Motor4")
        return None

    def motor_id_for_output_channel(self, channel: int) -> Optional[int]:
        for motor_function, output_channel in self.motor_function_to_output_channel.items():
            if output_channel != channel:
                continue
            motor_id = MOTOR_FUNCTION_TO_MOTOR_TEST.get(motor_function)
            if motor_id is not None:
                return motor_id
        return None

    def esc_id_label(self, esc_id: int) -> str:
        self.ensure_motor_output_mapping()
        offset = self.ensure_can_esc_offset()
        channel = int(esc_id) - offset
        if channel < 1 or channel > 4:
            return f"ESC{esc_id}/no-output-map(offset={offset})"
        motor_id = self.motor_id_for_output_channel(channel)
        if motor_id is None:
            return f"ESC{esc_id}/ch{channel}/UNKNOWN"
        position = MOTOR_TEST_TO_POSITION.get(motor_id, "UNKNOWN")
        return f"ESC{esc_id}/ch{channel}/{position}/M{motor_id}"

    def infer_mavlink_slot_shifts(self, present_ids: Iterable[int]) -> List[int]:
        """Infer display-only shift between DroneCAN ESC IDs and MAVLink slots."""
        present_set = set(int(esc_id) for esc_id in present_ids)
        dronecan_ids = set(self.required_all_dronecan_esc_ids())
        shifts = []
        for shift in range(-31, 32):
            shifted = {esc_id + shift for esc_id in dronecan_ids}
            if min(shifted) >= 1 and shifted.issubset(present_set):
                shifts.append(shift)
        return shifts

    def select_mavlink_slot_shift(self, present_ids: Iterable[int]) -> Optional[int]:
        shifts = self.infer_mavlink_slot_shifts(present_ids)
        if len(shifts) == 1:
            return shifts[0]
        if 0 in shifts:
            return 0
        return None

    def mavlink_slots_for_dronecan_ids(
        self, dronecan_ids: Iterable[int], present_ids: Iterable[int]
    ) -> Tuple[List[int], Optional[int]]:
        shift = self.select_mavlink_slot_shift(present_ids)
        if shift is None:
            shift = 0
            selected_shift = None
        else:
            selected_shift = shift
        return [int(esc_id) + shift for esc_id in dronecan_ids], selected_shift

    def position_to_output_channel(self, position: str) -> Optional[int]:
        self.ensure_motor_output_mapping()
        channel = self.output_channel_for_position(position)
        if channel is None:
            print(f"No output channel is assigned to {position.upper()} in SERVO1..4_FUNCTION")
            self.last_command_ok = False
            return None
        return channel

    def position_to_motor_id(self, position: str) -> Optional[int]:
        motor_id = POSITION_TO_MOTOR_TEST.get(position.lower())
        if motor_id is None:
            print("position must be fl/fr/rl/rr")
            self.last_command_ok = False
            return None
        return motor_id

    def set_active_pwm(self, motor_id: int, pwm: int):
        if not self.require_motor_test_ready([motor_id]):
            return False
        if not self.check_motor_id(motor_id):
            self.last_command_ok = False
            return False
        self.ensure_motor_output_mapping()
        if pwm < self.pwm_min:
            pwm = self.pwm_min
        if pwm > self.pwm_limit:
            print(f"Rejected: PWM {pwm} > limit {self.pwm_limit}. Use: limit <value>")
            self.last_command_ok = False
            return False
        self.active[motor_id] = ActiveOutput(MODE_PWM, pwm)
        self.log(f"HOLD: {self.motor_label(motor_id)} PWM={pwm}. Use zero/stop to stop.")
        self.send_motor_test_once(motor_id, MODE_PWM, pwm)
        return True

    def set_active_percent(self, motor_id: int, percent: float):
        if not self.require_motor_test_ready([motor_id]):
            return False
        if not self.check_motor_id(motor_id):
            self.last_command_ok = False
            return False
        self.ensure_motor_output_mapping()
        if percent < 0:
            percent = 0
        if percent > self.percent_limit:
            print(f"Rejected: percent {percent} > limit {self.percent_limit}. Use: plimit <value>")
            self.last_command_ok = False
            return False
        self.active[motor_id] = ActiveOutput(MODE_PERCENT, percent)
        self.log(f"HOLD: {self.motor_label(motor_id)} percent={percent}%. Use zero/stop to stop.")
        self.send_motor_test_once(motor_id, MODE_PERCENT, percent)
        return True

    def set_output_pwm(self, channel: int, pwm: int):
        motor_id = self.output_channel_to_motor_id(channel)
        if motor_id is not None:
            return self.set_active_pwm(motor_id, pwm)
        return False

    def set_output_percent(self, channel: int, percent: float):
        motor_id = self.output_channel_to_motor_id(channel)
        if motor_id is not None:
            return self.set_active_percent(motor_id, percent)
        return False

    def set_all_pwm(self, pwm: int):
        if not self.require_motor_test_ready(range(1, 5)):
            return False
        self.ensure_motor_output_mapping()
        if pwm < self.pwm_min:
            pwm = self.pwm_min
        if pwm > self.pwm_limit:
            print(f"Rejected: PWM {pwm} > limit {self.pwm_limit}. Use: limit <value>")
            self.last_command_ok = False
            return False
        for m in range(1, 5):
            self.active[m] = ActiveOutput(MODE_PWM, pwm)
        self.log(f"HOLD: ALL PWM={pwm}. Use zero/stop to stop.")
        for m in range(1, 5):
            self.send_motor_test_once(m, MODE_PWM, pwm)
            time.sleep(0.05)
        return True

    def set_all_percent(self, percent: float):
        if not self.require_motor_test_ready(range(1, 5)):
            return False
        self.ensure_motor_output_mapping()
        if percent < 0:
            percent = 0
        if percent > self.percent_limit:
            print(f"Rejected: percent {percent} > limit {self.percent_limit}. Use: plimit <value>")
            self.last_command_ok = False
            return False
        for m in range(1, 5):
            self.active[m] = ActiveOutput(MODE_PERCENT, percent)
        self.log(f"HOLD: ALL percent={percent}%. Use zero/stop to stop.")
        for m in range(1, 5):
            self.send_motor_test_once(m, MODE_PERCENT, percent)
            time.sleep(0.05)
        return True

    def zero(self):
        self.log("ZERO: clearing active outputs and sending stop commands to M1-M4")
        should_send_stop = self.safety_off_confirmed and (self.tx_enabled or bool(self.active))
        self.active.clear()
        if not should_send_stop:
            self.log("ZERO: no safety-off/tx/active output state, so no motor-test stop command was sent")
            return
        for _ in range(2):
            for m in range(1, 5):
                self.send_motor_test_once(m, MODE_PWM, 1000)
                time.sleep(0.04)

    def trigger_off(self):
        self.mark("TRIGGER OFF: zero outputs and disable local motor-test tx")
        self.zero()
        self.tx_enabled = False

    def trigger_pwm(self, target: str, pwm: int):
        if not self.require_disarmed_motor_test():
            return
        motor_id = None if target == "all" else self.parse_motor_or_output_target(target)
        if motor_id is None:
            if target == "all":
                if not self.require_safety_off():
                    return
                motor_ids = list(range(1, 5))
                if not self.require_dronecan_ready(motor_ids):
                    return
                self.tx_enabled = True
                self.mark(f"TRIGGER PWM START: target={target}, pwm={pwm}")
                self.poll_mavlink(print_messages=True)
                ack_start = len(self.command_acks)
                self.set_all_pwm(pwm)
                self.confirm_motor_test_ack(ack_start)
                return
            self.last_command_ok = False
            return
        if not self.require_safety_off():
            return
        if not self.require_dronecan_ready([motor_id]):
            return
        self.tx_enabled = True
        self.mark(f"TRIGGER PWM START: target={target}, pwm={pwm}")
        self.poll_mavlink(print_messages=True)
        ack_start = len(self.command_acks)
        self.set_active_pwm(motor_id, pwm)
        self.confirm_motor_test_ack(ack_start)

    def trigger_percent(self, target: str, percent: float):
        if not self.require_disarmed_motor_test():
            return
        motor_id = None if target == "all" else self.parse_motor_or_output_target(target)
        if motor_id is None:
            if target == "all":
                if not self.require_safety_off():
                    return
                motor_ids = list(range(1, 5))
                if not self.require_dronecan_ready(motor_ids):
                    return
                self.tx_enabled = True
                self.mark(f"TRIGGER PERCENT START: target={target}, percent={percent}")
                self.poll_mavlink(print_messages=True)
                ack_start = len(self.command_acks)
                self.set_all_percent(percent)
                self.confirm_motor_test_ack(ack_start)
                return
            self.last_command_ok = False
            return
        if not self.require_safety_off():
            return
        if not self.require_dronecan_ready([motor_id]):
            return
        self.tx_enabled = True
        self.mark(f"TRIGGER PERCENT START: target={target}, percent={percent}")
        self.poll_mavlink(print_messages=True)
        ack_start = len(self.command_acks)
        self.set_active_percent(motor_id, percent)
        self.confirm_motor_test_ack(ack_start)

    def trigger_window_pwm(self, target: str, pwm: int, duration: float):
        if not self.require_disarmed_motor_test():
            return
        motor_id = None if target == "all" else self.parse_motor_or_output_target(target)
        if motor_id is None and target != "all":
            self.last_command_ok = False
            return
        if not self.require_safety_off():
            return
        motor_ids = list(range(1, 5)) if target == "all" else [motor_id]
        if not self.require_dronecan_ready(motor_ids):
            return
        self.tx_enabled = True
        self.mark(f"TRIGGER WINDOW START: target={target}, pwm={pwm}, duration={duration}s")
        try:
            self.poll_mavlink(print_messages=True)
            ack_start = len(self.command_acks)
            if target == "all":
                self.set_all_pwm(pwm)
            else:
                self.set_active_pwm(motor_id, pwm)
            self.confirm_motor_test_ack(ack_start)
            if not self.last_command_ok:
                return
            end = time.time() + duration
            while time.time() < end:
                self.poll_mavlink(print_messages=True)
                self.refresh_active_outputs()
                time.sleep(0.03)
        finally:
            self.trigger_off()

    def confirm_motor_test_ack(self, start_index: Optional[int] = None):
        if self.ack_required:
            self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, start_index=start_index)

    def run_motor_window(self, motor_id: int, mode: str, value: float, duration: float, label: str):
        if not self.require_motor_test_ready([motor_id]):
            return
        if not self.check_motor_id(motor_id):
            self.last_command_ok = False
            return
        if mode == MODE_PWM:
            if value > self.pwm_limit:
                print(f"Rejected: PWM {value} > limit {self.pwm_limit}. Use: limit <value>")
                self.last_command_ok = False
                return
        elif mode == MODE_PERCENT:
            if value > self.percent_limit:
                print(f"Rejected: percent {value} > limit {self.percent_limit}. Use: plimit <value>")
                self.last_command_ok = False
                return
        else:
            print("mode must be pwm or percent")
            self.last_command_ok = False
            return

        self.mark(f"MOTOR TEST START: {label}; motor={self.motor_label(motor_id)}; mode={mode}; value={value}; duration={duration}s")
        next_send = 0.0
        end = time.time() + duration
        while time.time() < end:
            now = time.time()
            if now >= next_send:
                self.send_motor_test_once(motor_id, mode, value)
                next_send = now + self.refresh_period
            self.poll_mavlink(print_messages=True)
            time.sleep(0.03)
        self.mark(f"MOTOR TEST END: {label}")

    def sniffseq_pwm(self, motor_id: int, low_pwm: int, high_pwm: int, dwell: float):
        self.mark("SNIFFER SEQUENCE PWM: idle baseline")
        self.listen(4.0)
        self.run_motor_window(motor_id, MODE_PWM, low_pwm, dwell, f"LOW PWM {low_pwm}")
        self.zero()
        self.listen(2.0)
        self.run_motor_window(motor_id, MODE_PWM, high_pwm, dwell, f"HIGH PWM {high_pwm}")
        self.zero()
        self.listen(3.0)
        self.mark("SNIFFER SEQUENCE PWM COMPLETE")

    def sniffseq_percent(self, motor_id: int, low_pct: float, high_pct: float, dwell: float):
        self.mark("SNIFFER SEQUENCE PERCENT: idle baseline")
        self.listen(4.0)
        self.run_motor_window(motor_id, MODE_PERCENT, low_pct, dwell, f"LOW percent {low_pct}")
        self.zero()
        self.listen(2.0)
        self.run_motor_window(motor_id, MODE_PERCENT, high_pct, dwell, f"HIGH percent {high_pct}")
        self.zero()
        self.listen(3.0)
        self.mark("SNIFFER SEQUENCE PERCENT COMPLETE")

    def scan_pwm(self, motor_id: int, start: int, stop: int, step: int, dwell: float):
        if not self.require_motor_test_ready([motor_id]):
            return
        if not self.check_motor_id(motor_id):
            self.last_command_ok = False
            return
        if step == 0:
            print("step cannot be 0")
            self.last_command_ok = False
            return
        self.log(f"Scanning {self.motor_label(motor_id)}: {start}->{stop} step {step}, dwell {dwell}s")
        val = start
        cond = (lambda x: x <= stop) if step > 0 else (lambda x: x >= stop)
        try:
            while cond(val):
                if not self.set_active_pwm(motor_id, val):
                    return
                end = time.time() + dwell
                while time.time() < end:
                    self.poll_mavlink(print_messages=True)
                    self.refresh_active_outputs()
                    time.sleep(0.03)
                val += step
        finally:
            self.zero()

    def maptest(self, pwm: int, dwell: float):
        if not self.require_motor_test_ready(range(1, 5)):
            return
        self.log(f"Motor mapping test at PWM={pwm}, dwell={dwell}s")
        try:
            for m in range(1, 5):
                self.mark(f"MAPTEST {self.motor_label(m)} PWM={pwm}")
                if not self.set_active_pwm(m, pwm):
                    return
                end = time.time() + dwell
                while time.time() < end:
                    self.poll_mavlink(print_messages=True)
                    self.refresh_active_outputs()
                    time.sleep(0.03)
                self.zero()
                time.sleep(0.5)
        finally:
            self.zero()

    # ---------- Pixhawk helpers ----------

    def current_flight_mode(self) -> str:
        if self.last_heartbeat is None:
            return "UNKNOWN"
        return mavutil.mode_string_v10(self.last_heartbeat)

    def is_armed(self) -> Optional[bool]:
        if self.last_heartbeat is None:
            return None
        return bool(self.last_heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def wait_for_armed(self, armed: bool, timeout: float = 4.0) -> bool:
        expected = "armed" if armed else "disarmed"
        end = time.time() + timeout
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            state = self.is_armed()
            if state is armed:
                self.log(f"Arm state confirmed: {expected}")
                return True
            time.sleep(0.05)
        self.log(f"Arm state not confirmed within {timeout:.1f}s. Current armed={self.is_armed()}")
        return False

    def normalize_mode_name(self, name: str) -> str:
        token = name.strip().upper().replace("-", "_")
        return COMMON_MODE_ALIASES.get(token, token)

    def available_modes(self) -> Dict[str, int]:
        try:
            mapping = self.master.mode_mapping()
        except Exception:
            mapping = None
        return dict(mapping or {})

    def print_modes(self):
        mapping = self.available_modes()
        if not mapping:
            print("No mode mapping from autopilot yet. Wait for heartbeat and try again.")
            self.last_command_ok = False
            return
        names = sorted(mapping.keys())
        print("Available autopilot modes:")
        print("  " + " ".join(names))
        print("Common: STABILIZE ALT_HOLD LOITER POSHOLD GUIDED GUIDED_NOGPS AUTO RTL LAND BRAKE ACRO")

    def set_flight_mode(self, mode_name: str, wait: bool = True) -> bool:
        mapping = self.available_modes()
        mode_name = self.normalize_mode_name(mode_name)
        if not mapping:
            print("Rejected: no autopilot mode mapping available yet.")
            self.last_command_ok = False
            return False
        if mode_name not in mapping:
            print(f"Rejected: mode {mode_name} not available on this firmware. Use: modes")
            self.last_command_ok = False
            return False
        self.master.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mapping[mode_name],
        )
        self.log(f"Requested flight mode: {mode_name}")
        if wait and not self.wait_for_mode(mode_name):
            self.last_command_ok = False
            return False
        return True

    def wait_for_mode(self, mode_name: str, timeout: float = 4.0) -> bool:
        target = self.normalize_mode_name(mode_name)
        end = time.time() + timeout
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            current = self.normalize_mode_name(self.current_flight_mode())
            if current == target:
                self.log(f"Mode confirmed: {target}")
                return True
            time.sleep(0.05)
        self.log(f"Mode not confirmed within {timeout:.1f}s. Current mode: {self.current_flight_mode()}")
        return False

    def arm(self, force: bool = False):
        if not self.require_safety_off():
            return
        if force:
            self.log("Sending FORCE ARM. Bench only. Remove propellers.")
            ack_start = len(self.command_acks)
            self.send_command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 21196, 0, 0, 0, 0, 0)
        else:
            self.log("Sending ARM")
            ack_start = len(self.command_acks)
            self.send_command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 0, 0, 0, 0, 0, 0)
        if self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=4.0, start_index=ack_start):
            if not self.wait_for_armed(True, timeout=4.0):
                self.last_command_ok = False

    def set_safety_switch(self, dangerous: bool, retry_until_success: bool = True) -> bool:
        if not dangerous:
            if self.active or self.tx_enabled:
                self.log("SAFETY ON: zero outputs and disable local motor-test tx first")
                self.zero()
                self.tx_enabled = False
            self.safety_off_confirmed = False

        state = SAFETY_SWITCH_STATE_DANGEROUS if dangerous else SAFETY_SWITCH_STATE_SAFE
        label = "OFF/DANGEROUS" if dangerous else "ON/SAFE"
        attempt = 1
        self.mark(f"SAFETY {label}: retry until Pixhawk ACK ACCEPTED")
        while True:
            self.log(f"Sending safety switch state {label}, attempt {attempt}")
            ack_start = len(self.command_acks)
            self.send_command_long(
                mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE,
                state,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            ok = self.wait_for_command_ack(
                mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE,
                timeout=3.0,
                start_index=ack_start,
                accepted_results=(MAV_RESULT_ACCEPTED,),
            )
            if ok:
                self.safety_off_confirmed = dangerous
                if dangerous:
                    self.log("SAFETY OFF confirmed by Pixhawk ACK. Wait until all Hobbywing X8 LEDs are solid before trigger.")
                else:
                    self.log("SAFETY ON confirmed by Pixhawk ACK. Motor-test activation is locked until safety off is entered again.")
                return True
            if not retry_until_success:
                return False
            self.log(
                f"Safety {label} not confirmed; retrying in {self.safety_retry_period:.1f}s. "
                "Press Ctrl+C to stop."
            )
            time.sleep(self.safety_retry_period)
            attempt += 1

    def run_prearm_checks(self):
        self.mark("PREARM CHECKS START")
        ack_start = len(self.command_acks)
        statustext_start = len(self.statustexts)
        self.send_command_long(MAV_CMD_RUN_PREARM_CHECKS, 0, 0, 0, 0, 0, 0, 0)
        self.wait_for_command_ack(MAV_CMD_RUN_PREARM_CHECKS, timeout=3.0, start_index=ack_start)
        self.collect_statustext(1.2)
        new_text = self.statustexts[statustext_start:]
        if new_text:
            print("\nPrearm messages:")
            for text in new_text:
                print(f"  {text}")
        else:
            print("\nPrearm messages: none received")
        self.mark("PREARM CHECKS END")

    def disarm(self):
        self.log("DISARM: zero first, then disarm")
        self.zero()
        ack_start = len(self.command_acks)
        self.send_command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0)
        if self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=4.0, start_index=ack_start):
            if not self.wait_for_armed(False, timeout=4.0):
                self.last_command_ok = False

    def reboot(self):
        self.log("Sending Pixhawk reboot command")
        self.send_command_long(mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 1, 0, 0, 0, 0, 0, 0)

    def takeoff(self, altitude_m: float):
        if altitude_m <= 0 or altitude_m > 100:
            print("takeoff altitude must be 0..100 meters")
            self.last_command_ok = False
            return
        if self.is_armed() is False:
            print("Rejected: vehicle is disarmed. Use: launch ALT, or safety off; mode guided; arm; takeoff ALT")
            self.last_command_ok = False
            return
        self.log(f"Requesting takeoff to {altitude_m:.1f}m")
        ack_start = len(self.command_acks)
        self.send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            altitude_m,
        )
        self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=5.0, start_index=ack_start)

    def land(self):
        self.set_flight_mode("LAND")

    def rtl(self):
        self.set_flight_mode("RTL")

    def guided_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.target_system,
            self.target_component,
            getattr(mavutil.mavlink, "MAV_FRAME_BODY_NED", 8),
            GUIDED_VELOCITY_TYPE_MASK,
            0,
            0,
            0,
            vx,
            vy,
            vz,
            0,
            0,
            0,
            0,
            yaw_rate,
        )

    def guided_move(self, name: str, vx: float, vy: float, vz: float, yaw_rate: float, duration: float):
        if duration <= 0 or duration > 60:
            print("duration must be 0..60 seconds")
            self.last_command_ok = False
            return
        if self.is_armed() is False:
            print("Rejected: vehicle is disarmed. Use launch/takeoff first, then send movement commands.")
            self.last_command_ok = False
            return
        if self.normalize_mode_name(self.current_flight_mode()) != "GUIDED":
            self.log("Warning: vehicle is not confirmed in GUIDED mode; sending velocity command anyway.")
        self.mark(f"GUIDED MOVE START: {name}; vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}, duration={duration:.1f}s")
        end = time.time() + duration
        while time.time() < end:
            self.guided_velocity(vx, vy, vz, yaw_rate=yaw_rate)
            self.poll_mavlink(print_messages=True)
            time.sleep(0.1)
        self.guided_stop()
        self.mark(f"GUIDED MOVE END: {name}")

    def guided_stop(self):
        self.log("GUIDED STOP: zero body-frame velocity")
        for _ in range(3):
            self.guided_velocity(0.0, 0.0, 0.0, yaw_rate=0.0)
            time.sleep(0.05)

    def flight_test(self, altitude_m: float = 1.0, forward_speed: float = 0.2, forward_duration: float = 1.0, land_wait: float = 8.0):
        if altitude_m <= 0 or altitude_m > 10:
            print("flighttest altitude must be 0..10 meters")
            self.last_command_ok = False
            return
        if forward_speed < 0 or forward_speed > 2:
            print("flighttest speed must be 0..2 m/s")
            self.last_command_ok = False
            return
        if forward_duration < 0 or forward_duration > 10:
            print("flighttest duration must be 0..10 seconds")
            self.last_command_ok = False
            return
        if land_wait < 0 or land_wait > 60:
            print("flighttest land_wait must be 0..60 seconds")
            self.last_command_ok = False
            return

        self.mark(
            f"FLIGHT TEST START: takeoff={altitude_m:.1f}m, forward={forward_speed:.2f}m/s for {forward_duration:.1f}s"
        )
        if self.tx_enabled or self.active:
            self.log("Flight test setup: disabling local motor-test tx")
            self.zero()
            self.tx_enabled = False
        took_off = False
        try:
            self.preflight_check(wait=1.0)
            if not self.last_command_ok:
                return
            if not self.require_safety_off():
                return
            self.set_flight_mode("GUIDED")
            if not self.last_command_ok:
                return
            self.arm()
            if not self.last_command_ok:
                return
            self.takeoff(altitude_m)
            if not self.last_command_ok:
                return
            took_off = True
            self.listen(max(1.0, min(3.0, altitude_m + 1.0)))
            if forward_duration > 0 and forward_speed > 0:
                self.guided_move("flighttest-forward", forward_speed, 0.0, 0.0, 0.0, forward_duration)
                if not self.last_command_ok:
                    return
            self.guided_stop()
            self.land()
            if land_wait > 0:
                self.listen(land_wait)
        finally:
            if self.is_armed() is True:
                if took_off:
                    self.log("Flight test cleanup: LAND then DISARM")
                    self.land()
                    if land_wait > 0:
                        self.listen(min(land_wait, 10.0))
                self.disarm()
            self.status()
            self.mark("FLIGHT TEST END")

    def status(self):
        # Poll once first so status is not based on a stale heartbeat.
        self.poll_mavlink(print_messages=False)
        self.ensure_motor_output_mapping()
        self.ensure_can_esc_offset()
        hb = self.last_heartbeat
        sys_status = self.last_sys_status
        gps = self.last_gps
        if hb is not None:
            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode = mavutil.mode_string_v10(hb)
        else:
            armed = None
            mode = "UNKNOWN"
        print("========== STATUS ==========")
        print(f"time={self.ts()}")
        print(
            f"mode={mode}, armed={armed}, tx={'ON' if self.tx_enabled else 'OFF'}, "
            f"safety_off_confirmed={self.safety_off_confirmed}, escgate={self.esc_gate}, "
            f"esc_offset={self.current_can_esc_offset()}"
        )
        print(f"autopilot_version={self.autopilot_version_text()}")
        active_str = ", ".join([f"{self.motor_label(k)}:{v.mode}={v.value}" for k, v in sorted(self.active.items())])
        print(f"active={{{active_str}}}")
        print(f"pwm_limit={self.pwm_limit}, percent_limit={self.percent_limit}, refresh={self.refresh_period}, hold={self.hold_duration}")
        if sys_status is not None:
            voltage = sys_status.voltage_battery / 1000.0
            current = sys_status.current_battery / 100.0
            print(f"battery={voltage:.2f}V, current={current:.2f}A")
        if gps is not None:
            print(f"gps_fix={gps.fix_type}, sats={gps.satellites_visible}")
        if self.last_vfr_hud is not None:
            print(
                f"vfr=airspeed {self.last_vfr_hud.airspeed:.2f}m/s, "
                f"groundspeed {self.last_vfr_hud.groundspeed:.2f}m/s, "
                f"alt {self.last_vfr_hud.alt:.2f}m, climb {self.last_vfr_hud.climb:.2f}m/s"
            )
        if self.last_global_position is not None:
            rel_alt = self.last_global_position.relative_alt / 1000.0
            vx = self.last_global_position.vx / 100.0
            vy = self.last_global_position.vy / 100.0
            vz = self.last_global_position.vz / 100.0
            print(f"position=relative_alt {rel_alt:.2f}m, velocity N/E/D {vx:.2f}/{vy:.2f}/{vz:.2f}m/s")
        if self.last_attitude is not None:
            roll = self.last_attitude.roll * 57.2957795
            pitch = self.last_attitude.pitch * 57.2957795
            yaw = self.last_attitude.yaw * 57.2957795
            print(f"attitude=roll {roll:+.1f}deg, pitch {pitch:+.1f}deg, yaw {yaw:+.1f}deg")
        if self.last_ekf_status is not None:
            flags = getattr(self.last_ekf_status, "flags", None)
            print(f"ekf_flags={flags}")
        if self.last_servo_output is not None:
            fields = []
            for i in range(1, 5):
                name = f"servo{i}_raw"
                if hasattr(self.last_servo_output, name):
                    fields.append(f"ch{i}={getattr(self.last_servo_output, name)}")
            print("last_servo_output=" + " ".join(fields))
        if self.esc_status_frames or self.esc_telemetry_frames:
            fields = [
                f"ESC_STATUS frames={self.esc_status_frames}",
                f"ESC_TELEMETRY total_frames={self.esc_telemetry_frames}",
            ]
            frame_groups = [
                f"{msg_type}={self.esc_telemetry_frames_by_type.get(msg_type, 0)}"
                for msg_type, _, _ in ESC_TELEMETRY_GROUPS
                if self.esc_telemetry_frames_by_type.get(msg_type, 0)
            ]
            if frame_groups:
                fields.append("groups[" + ", ".join(frame_groups) + "]")
            if self.last_esc_status_time:
                fields.append(f"esc_status_age={time.time() - self.last_esc_status_time:.2f}s")
            if self.last_esc_telemetry_time:
                fields.append(f"esc_telemetry_age={time.time() - self.last_esc_telemetry_time:.2f}s")
            print("esc_messages " + ", ".join(fields))
            print(f"esc_telemetry_fresh={self.esc_telemetry_is_fresh()}, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s")
        esc_summary = self.esc_telemetry_aggregate_summary(fresh_only=False)
        if esc_summary:
            print(esc_summary)
        if self.uavcan_node_statuses or self.uavcan_node_infos:
            keys = sorted(set(self.uavcan_node_statuses) | set(self.uavcan_node_infos))
            print("uavcan_node_sources=" + ",".join(f"{sysid}/{compid}" for sysid, compid in keys))
        self.print_motor_output_mapping()
        print("============================")

    def escdiag(self, wait: float = 3.0):
        if wait < 0 or wait > 30:
            print("escdiag wait must be 0..30 seconds")
            self.last_command_ok = False
            return
        self.mark(f"ESC DIAG START: listen {wait:.1f}s")
        self.request_streams(True)
        if wait > 0:
            self.listen(wait)
        self.status()
        fresh = self.esc_telemetry_is_fresh()
        age = self.esc_telemetry_age()
        present, groups, _ = self.esc_telemetry_aggregate_presence(fresh_only=True)
        if self.last_esc_telemetry is None:
            print("ESC telemetry: none received. Check CAN power, CAN_H/CAN_L, terminators, bitrate, and the ESC RawCommand mode.")
        elif not fresh:
            age_text = "unknown" if age is None else f"{age:.2f}s"
            print(
                "ESC telemetry: stale. "
                f"last_age={age_text}, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s. "
                "Check CAN stream delivery before treating ESC IDs as present."
            )
            observed_present, _, _ = self.esc_telemetry_aggregate_presence(fresh_only=False)
            display_shifts = self.infer_mavlink_slot_shifts(observed_present)
            if display_shifts:
                print(
                    "Stale MAVLink display-shift evidence: "
                    f"observed_slots={observed_present}, shift_candidates={display_shifts}. "
                    "Do not change CAN_D1_UC_ESC_OF from stale MAVLink evidence."
                )
        else:
            if not groups:
                print("ESC telemetry: no fresh readable ESC_TELEMETRY payloads. Check MAVLink/pymavlink message decoding.")
                self.last_command_ok = False
                return
            offset = self.current_can_esc_offset()
            required = [offset + channel for channel in (1, 2, 3, 4)]
            required_slots, display_shift = self.mavlink_slots_for_dronecan_ids(required, present)
            missing_required = [slot for slot in required_slots if slot not in set(present)]
            if missing_required:
                missing_text = ",".join(str(x) for x in missing_required)
                print(
                    "ESC telemetry missing required MAVLink slots: "
                    f"{missing_text}. "
                    "For Hobbywing X8 DroneCAN, check each ESC ThrottleID/NodeID, CAN wiring, and power."
                )
            else:
                print(
                    f"ESC telemetry: DroneCAN IDs {required} are all represented by "
                    f"MAVLink slots {required_slots}."
                )
            if display_shift:
                print(
                    f"MAVLink display shift detected: {display_shift:+d}. "
                    "This does not imply a CAN_D1_UC_ESC_OF change."
                )
        self.mark("ESC DIAG END")

    def offsetdiag(self, wait: float = 3.0):
        if wait < 0 or wait > 30:
            print("offsetdiag wait must be 0..30 seconds")
            self.last_command_ok = False
            return
        self.mark(f"ESC OFFSET DIAG START: listen {wait:.1f}s")
        self.ensure_motor_output_mapping()
        self.ensure_can_esc_offset()
        self.request_streams(True)
        if wait > 0:
            self.listen(wait)

        current_offset = self.current_can_esc_offset()
        required = self.required_all_dronecan_esc_ids()
        observed_present, observed_groups, observed_voltage = self.esc_telemetry_aggregate_presence(fresh_only=False)
        fresh_present, fresh_groups, _ = self.esc_telemetry_aggregate_presence(fresh_only=True)
        observed_shifts = self.infer_mavlink_slot_shifts(observed_present)
        fresh_shifts = self.infer_mavlink_slot_shifts(fresh_present)
        required_slots, selected_shift = self.mavlink_slots_for_dronecan_ids(required, fresh_present)
        missing_current = [slot for slot in required_slots if slot not in set(fresh_present)]

        print("========== ESC OFFSET DIAG ==========")
        print(f"current_CAN_D1_UC_ESC_OF={current_offset}")
        print(f"current_required_ids={required}")
        print(f"fresh={self.esc_telemetry_is_fresh()}, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s")
        print(f"fresh_present_ids={fresh_present if fresh_groups else 'none'}")
        print(f"observed_present_ids={observed_present if observed_groups else 'none'}")
        if observed_groups:
            print("observed_groups=" + ",".join(observed_groups))
        if observed_voltage:
            print("observed_voltage=" + " ".join(observed_voltage))
        frame_groups = [
            f"{msg_type}={self.esc_telemetry_frames_by_type.get(msg_type, 0)}"
            for msg_type, _, _ in ESC_TELEMETRY_GROUPS
            if self.esc_telemetry_frames_by_type.get(msg_type, 0)
        ]
        if frame_groups:
            print("frames_by_group=" + ", ".join(frame_groups))
        if missing_current:
            print(f"missing_required_mavlink_slots={missing_current}")
        else:
            print("missing_required_mavlink_slots=none")
        print(f"required_mavlink_slots={required_slots}")
        print(f"selected_mavlink_display_shift={selected_shift}")
        print(f"fresh_mavlink_shift_candidates={fresh_shifts if fresh_shifts else 'none'}")
        print(f"observed_mavlink_shift_candidates={observed_shifts if observed_shifts else 'none'}")
        print("CAN_D1_UC_ESC_OF controls RawCommand packing and must not be changed to fix a MAVLink-only display shift.")
        print("=====================================")
        self.mark("ESC OFFSET DIAG END")

    def x8diag(self, wait: float = 3.0):
        if wait < 0 or wait > 30:
            print("x8diag wait must be 0..30 seconds")
            self.last_command_ok = False
            return
        self.mark(f"X8 READINESS DIAG START: listen {wait:.1f}s")
        self.ensure_motor_output_mapping()
        self.ensure_can_esc_offset()
        self.request_streams(True)
        if wait > 0:
            self.listen(wait)
        self.poll_mavlink(print_messages=False)

        hb = self.last_heartbeat
        if hb is not None:
            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode = mavutil.mode_string_v10(hb)
        else:
            armed = None
            mode = "UNKNOWN"

        voltage, current = self.battery_voltage_current()
        main_power_ready = voltage is not None and voltage > 5.0
        fresh_present, fresh_groups, _ = self.esc_telemetry_aggregate_presence(fresh_only=True)
        observed_present, observed_groups, observed_voltage = self.esc_telemetry_aggregate_presence(fresh_only=False)
        required = self.required_all_dronecan_esc_ids()
        required_slots, display_shift = self.mavlink_slots_for_dronecan_ids(required, fresh_present)
        missing_required = [slot for slot in required_slots if slot not in set(fresh_present)]
        fresh = self.esc_telemetry_is_fresh()
        current_offset = self.current_can_esc_offset()
        fresh_candidates = self.infer_mavlink_slot_shifts(fresh_present)
        observed_candidates = self.infer_mavlink_slot_shifts(observed_present)
        dronecan_ready = bool(fresh_groups) and not missing_required
        can_attempt_safety_off = main_power_ready and dronecan_ready
        can_attempt_trigger_after_safety_off = can_attempt_safety_off and self.safety_off_confirmed

        print("========== X8 READINESS DIAG ==========")
        print(f"mode={mode}, armed={armed}, tx={'ON' if self.tx_enabled else 'OFF'}, safety_off_confirmed={self.safety_off_confirmed}")
        if voltage is None:
            print("battery=UNKNOWN, main_power_ready=UNKNOWN")
        else:
            print(f"battery={voltage:.2f}V, current={current:.2f}A, main_power_ready={main_power_ready}")
        print(
            f"escgate={self.esc_gate}, current_CAN_D1_UC_ESC_OF={current_offset}, "
            f"required_dronecan_ids={required}, required_mavlink_slots={required_slots}"
        )
        print(f"esc_telemetry_fresh={fresh}, max_age={ESC_TELEMETRY_MAX_AGE:.1f}s")
        print(f"fresh_present_ids={fresh_present if fresh_groups else 'none'}")
        print(f"observed_present_ids={observed_present if observed_groups else 'none'}")
        if observed_groups:
            print("observed_groups=" + ",".join(observed_groups))
        if observed_voltage:
            print("observed_voltage=" + " ".join(observed_voltage))
        frame_groups = [
            f"{msg_type}={self.esc_telemetry_frames_by_type.get(msg_type, 0)}"
            for msg_type, _, _ in ESC_TELEMETRY_GROUPS
            if self.esc_telemetry_frames_by_type.get(msg_type, 0)
        ]
        if frame_groups:
            print("frames_by_group=" + ", ".join(frame_groups))
        if missing_required:
            print(f"missing_required_mavlink_slots={missing_required}")
        else:
            print("missing_required_mavlink_slots=none")
        print(f"mavlink_display_shift={display_shift}")
        print(f"fresh_mavlink_shift_candidates={fresh_candidates if fresh_candidates else 'none'}")
        print(f"observed_mavlink_shift_candidates={observed_candidates if observed_candidates else 'none'}")
        print(f"dronecan_ready={dronecan_ready}")
        print(f"can_attempt_safety_off={can_attempt_safety_off}")
        print(f"can_attempt_trigger_after_safety_off={can_attempt_trigger_after_safety_off}")

        if not main_power_ready:
            print("next_action=restore X8/main battery power before safety off or trigger")
        elif not fresh:
            print("next_action=restore fresh DroneCAN ESC telemetry before trigger")
        elif missing_required:
            print("next_action=run --can-probe to verify ESC NodeID/ThrottleID and inspect missing telemetry")
        elif not self.safety_off_confirmed:
            print("next_action=type safety off, then wait for X8 LEDs to become solid before trigger")
        else:
            print("next_action=triggercheck all, then trigger only if LEDs are solid")
        self.print_motor_output_mapping()
        print("========================================")
        self.mark("X8 READINESS DIAG END")

    def print_uavcan_node_summary(self):
        keys = sorted(set(self.uavcan_node_statuses) | set(self.uavcan_node_infos))
        print("========== UAVCAN NODE SUMMARY ==========")
        if not keys:
            print("No UAVCAN_NODE_STATUS or UAVCAN_NODE_INFO MAVLink messages received.")
            print("This does not prove the CAN bus is empty; it means this MAVLink link did not report node messages during the listen window.")
        for key in keys:
            sysid, compid = key
            status = self.uavcan_node_statuses.get(key)
            info = self.uavcan_node_infos.get(key)
            name = str(getattr(info, "name", "")).strip() if info is not None else ""
            if status is not None:
                health = self.uavcan_health_name(status.health)
                mode = self.uavcan_mode_name(status.mode)
                uptime = f"{status.uptime_sec}s"
                vendor = getattr(status, "vendor_specific_status_code", None)
            else:
                health = mode = uptime = "UNKNOWN"
                vendor = None
            if info is not None:
                hw = f"{info.hw_version_major}.{info.hw_version_minor}"
                sw = f"{info.sw_version_major}.{info.sw_version_minor}"
            else:
                hw = sw = "UNKNOWN"
            print(
                f"node_id={compid} position={EXPECTED_NODE_TO_POSITION.get(compid, 'OTHER')} "
                f"bridge_system={sysid} name={name or 'UNKNOWN'} "
                f"health={health} mode={mode} uptime={uptime} hw={hw} sw={sw} vendor_status={vendor}"
            )
        observed = {compid for _, compid in keys}
        missing = sorted(set(EXPECTED_NODE_TO_POSITION) - observed)
        print(f"expected_esc_node_ids={sorted(EXPECTED_NODE_TO_POSITION)}")
        print(f"missing_esc_node_ids={missing if missing else 'none'}")
        print("=========================================")

    def nodediag(self, wait: float = 5.0):
        if wait < 0 or wait > 30:
            print("nodediag wait must be 0..30 seconds")
            self.last_command_ok = False
            return
        self.mark(f"UAVCAN NODE DIAG START: listen {wait:.1f}s")
        self.log("Requesting UAVCAN node status/info MAVLink messages")
        self.set_message_interval(MSG_UAVCAN_NODE_STATUS, 2)
        self.set_message_interval(MSG_UAVCAN_NODE_INFO, 1)
        ack_start = len(self.command_acks)
        self.send_command_long(MAV_CMD_UAVCAN_GET_NODE_INFO, 0, 0, 0, 0, 0, 0, 0)
        node_info_supported = self.wait_for_command_ack(MAV_CMD_UAVCAN_GET_NODE_INFO, timeout=2.0, start_index=ack_start)
        if not node_info_supported:
            print("Pixhawk did not accept MAV_CMD_UAVCAN_GET_NODE_INFO; continuing with passive MAVLink node-message listen.")
        if wait > 0:
            self.listen(wait)
        self.print_uavcan_node_summary()
        self.last_command_ok = True
        self.mark("UAVCAN NODE DIAG END")

    def preflight_check(self, wait: float = 2.0):
        self.mark("PREFLIGHT CHECK START")
        self.request_streams(True)
        end = time.time() + max(0.0, wait)
        while time.time() < end:
            self.poll_mavlink(print_messages=True)
            time.sleep(0.03)

        params = self.get_params(
            [
                "ARMING_CHECK",
                "BRD_SAFETY_DEFLT",
                "BRD_SAFETYOPTION",
                "BATT_MONITOR",
                "GPS_TYPE",
                "EK3_ENABLE",
                "AHRS_EKF_TYPE",
                "FENCE_ENABLE",
                "FS_THR_ENABLE",
            ],
            wait=3.0 if wait > 0 else 1.2,
        )

        hb = self.last_heartbeat
        armed = None
        mode = "UNKNOWN"
        if hb is not None:
            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode = mavutil.mode_string_v10(hb)

        print("\nPreflight summary:")
        print(f"  mode={mode}, armed={armed}")
        if self.last_sys_status is not None:
            voltage = self.last_sys_status.voltage_battery / 1000.0
            current = self.last_sys_status.current_battery / 100.0
            print(f"  battery={voltage:.2f}V, current={current:.2f}A")
        else:
            print("  battery=UNKNOWN")

        if self.last_gps is not None:
            print(f"  gps_fix={self.last_gps.fix_type}, sats={self.last_gps.satellites_visible}")
        else:
            print("  gps=UNKNOWN")

        if self.last_global_position is not None:
            print(f"  relative_alt={self.last_global_position.relative_alt / 1000.0:.2f}m")
        else:
            print("  position=UNKNOWN")

        if self.last_ekf_status is not None:
            print(f"  ekf_flags={self.last_ekf_status.flags}")
        else:
            print("  ekf=UNKNOWN")

        print("\nParameter snapshot:")
        for name, value in params.items():
            print(f"  {name:<18} {value}")

        print("\nFlight command availability:")
        mapping = self.available_modes()
        for mode_name in ("STABILIZE", "ALT_HOLD", "LOITER", "GUIDED", "RTL", "LAND", "BRAKE"):
            result = "OK" if mode_name in mapping else "MISSING"
            print(f"  {mode_name:<10} {result}")
        gps_fix = getattr(self.last_gps, "fix_type", None) if self.last_gps is not None else None
        sats = getattr(self.last_gps, "satellites_visible", None) if self.last_gps is not None else None
        has_position = self.last_global_position is not None
        has_ekf = self.last_ekf_status is not None
        guided_ready = bool(gps_fix is not None and gps_fix >= 3 and sats is not None and sats > 0 and has_position and has_ekf)
        print("\nMovement readiness:")
        print("  takeoff     requires GUIDED + armed + passing ArduPilot pre-arm checks")
        print("  forward     requires GUIDED + armed + active position/EKF estimate")
        print(f"  guided_ready={guided_ready}")
        if not guided_ready:
            reasons = []
            if gps_fix is None or gps_fix < 3:
                reasons.append(f"GPS fix insufficient ({gps_fix})")
            if sats is None or sats <= 0:
                reasons.append(f"satellites insufficient ({sats})")
            if not has_position:
                reasons.append("no GLOBAL_POSITION_INT")
            if not has_ekf:
                reasons.append("no EKF_STATUS_REPORT")
            print("  reason=" + "; ".join(reasons))
        self.mark("PREFLIGHT CHECK END")

    # ---------- command parser ----------

    def print_help(self):
        print("""
================ Pixhawk 6X + X8 G2 DroneCAN DIAGNOSTIC CLI ================

Safety:
  tx on                         Enable local motor-test commands
  tx off                        Stop all motors and disable local motor-test commands
  zero / stop                   Stop all active outputs
  status                        Show mode/armed/safety gate/battery/active outputs
  listen 5                      Print MAVLink messages for 5 seconds
  verbose on/off                Print/hide RC and ESC telemetry messages
  outmon on/off                 Print/hide SERVO_OUTPUT_RAW messages
  ack on/off                    Require COMMAND_ACK for direct trigger commands
  escgate all/target/off        DroneCAN telemetry gate before trigger, default all
  mark TEXT                     Print timestamp marker for aligning with CAN sniffer logs

MAVLink streams:
  streams on                    Request SERVO_OUTPUT_RAW, RC_CHANNELS, ESC telemetry
  streams off                   Stop requested streams
  preflight 2                   Summarize armed/mode/battery/GPS/EKF/flight modes
  escdiag 3                     Read-only DroneCAN ESC telemetry presence check
  offsetdiag 3                  Separate RawCommand offset from MAVLink display shift
  x8diag 3                      Read-only X8 readiness: power, ESC IDs, offset, trigger gate
  nodediag 5                    Read-only UAVCAN node status/info check
  triggercheck fl 2             Read-only check: mapping + safety + ESC telemetry gate

Parameter audit / setup:
  audit std                     Check X8 G2 standard RawCommand path, OPTION=0
  audit hw                      Check legacy Hobbywing vendor path, OPTION=128
  setup std                     Set X8 G2 standard RawCommand params, then reboot manually
  setup hw                      Set legacy Hobbywing vendor params, then reboot manually
  getparam NAME                 Request one parameter
  param NAME VALUE              Set one parameter
  reboot                        Reboot Pixhawk after CAN/DroneCAN param changes

Motor test - persistent PWM:
  fl 1200                       Hold front-left output from current SERVOx_FUNCTION mapping
  fr 1200                       Hold front-right output from current SERVOx_FUNCTION mapping
  rl 1200                       Hold rear-left output from current SERVOx_FUNCTION mapping
  rr 1200                       Hold rear-right output from current SERVOx_FUNCTION mapping
  m1 1200                       Hold motor-test M1 at PWM=1200 until zero/stop
  m2 1200
  m3 1200
  m4 1200
  ch4 1200 / slot4 1200         Hold ESC/output channel 4
  all 1200                      Hold M1-M4 at PWM=1200

Motor test - persistent percent:
  flp 5                         Hold front-left output at 5%
  frp 5                         Hold front-right output at 5%
  rlp 5                         Hold rear-left output at 5%
  rrp 5                         Hold rear-right output at 5%
  m1p 5                         Hold motor-test M1 at 5% until zero/stop
  m2p 5
  m3p 5
  m4p 5
  ch4p 5 / slot4p 5             Hold ESC/output channel 4 at 5%
  allp 5                        Hold all motors at 5%

Sniffer-aligned experiments:
  sniffpwm m4 1150 1450 6       Idle -> M4/front-left 1150 -> stop -> 1450, with markers
  sniffch fl 1150 1450 6        Same test, addressed by frame position
  sniffch ch4 1150 1450 6       Same test, addressed by ESC/output channel
  sniffpct m4 20 50 6           Idle -> M4/front-left 20% -> stop -> 50%, with markers
  maptest 1200 3                Test M1-M4 one by one, 3s each, with markers
  scan m4 1100 1450 50 2        Sweep M4 PWM from 1100 to 1450

Quad X mapping:
  Position commands fl/fr/rl/rr use ArduCopter motor-test IDs, then read
  SERVO1..4_FUNCTION to show which DroneCAN ESC/output channel is active.
  Run status or audit std to print the current X8 G2 mapping before testing.

Limits and timing:
  limit 1500                    Set max allowed PWM
  plimit 50                     Set max allowed percent
  refresh 0.8                   Refresh period, seconds
  hold 3.0                      Motor-test timeout per command, seconds

Pixhawk arming:
  modes                         List flight modes supported by firmware
  mode guided                   Set flight mode by name: stabilize/alt_hold/loiter/guided/rtl/land/brake
  stabilize / stable            Set STABILIZE mode
  alt_hold / loiter / poshold   Common hold/control modes
  guided / guided_nogps         Guided control modes
  acro / auto / brake           Common manual/mission/safety modes
  rtl / land                    Return-to-launch or land mode
  safety off                    Retry until Pixhawk safety switch OFF/DANGEROUS ACK is ACCEPTED
  safety on                     Zero outputs, lock motor-test activation, then request ON/SAFE
  prearm                        Run ArduPilot pre-arm checks without arming
  arm                           Try to arm Pixhawk
  arm force                     Force arm, bench only, remove propellers
  disarm                        Zero + disarm
  takeoff 2                     Guided takeoff to 2 meters; requires GUIDED + armed
  launch 2                      GUIDED -> arm -> takeoff 2; requires prior safety off
  flighttest 1 0.2 1            Full test after prior safety off: preflight -> arm -> takeoff -> land

Guided movement:
  forward 0.5 2                 Move forward at 0.5 m/s for 2 seconds in GUIDED
  backward 0.5 2                Move backward
  left 0.5 2 / right 0.5 2      Move laterally
  up 0.3 1 / down 0.3 1         Move vertically
  yawleft 20 1                  Yaw left at 20 deg/s for 1 second
  yawright 20 1                 Yaw right
  hover                         Send zero velocity in GUIDED

Direct trigger motor-test:
  trigger fl 1200               Requires prior safety off + escgate, then hold front-left
  trigger ch4 1200              Requires prior safety off + escgate, then hold output channel 4
  trigger m4 1200               Requires prior safety off + escgate, then hold motor-test M4/front-left
  trigger all 1200              Requires prior safety off + escgate, then hold all M1-M4
  triggerp fl 20                Requires prior safety off + escgate, then hold front-left at 20 percent
  triggerwin fl 1200 3          Requires prior safety off + escgate, trigger PWM for 3 seconds, then stop
  trigger off                   Zero outputs and tx off

Recommended current test:
  streams on
  audit std
  safety off
  # wait until all Hobbywing X8 LEDs are solid
  triggercheck all 3
  tx on
  limit 1500
  sniffpwm fl 1150 1450 6
  zero
  tx off

IMPORTANT:
  This script cannot prove whether Pixhawk emitted DroneCAN RawCommand by itself.
  It gives time markers and MAVLink-side evidence. Use F446/USB-CAN sniffer on
  the same CAN bus to decide whether RawCommand/com.hobbywing.esc.RawCommand exists.
===========================================================================
""")

    def handle(self, line: str):
        line = line.strip()
        if not line:
            return
        self.last_command_ok = True
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("help", "?"):
            self.print_help()
        elif cmd in ("exit", "quit"):
            self.zero()
            print("Bye.")
            raise SystemExit
        elif cmd == "tx":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: tx on/off")
                return
            if parts[1].lower() == "on":
                self.tx_enabled = True
                self.mark("TX ON: local persistent motor-test commands enabled")
            else:
                self.zero()
                self.tx_enabled = False
                self.mark("TX OFF")
        elif cmd in ("zero", "stop"):
            self.zero()
        elif cmd == "status":
            self.status()
        elif cmd == "listen":
            sec = float(parts[1]) if len(parts) > 1 else 5.0
            self.listen(sec)
        elif cmd == "mark":
            text = " ".join(parts[1:]) if len(parts) > 1 else "manual mark"
            self.mark(text)
        elif cmd == "streams":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: streams on/off")
                self.last_command_ok = False
                return
            self.request_streams(parts[1].lower() == "on")
        elif cmd in ("preflight", "flightcheck"):
            wait = float(parts[1]) if len(parts) > 1 else 2.0
            self.preflight_check(wait)
        elif cmd == "escdiag":
            wait = float(parts[1]) if len(parts) > 1 else 3.0
            self.escdiag(wait)
        elif cmd == "offsetdiag":
            wait = float(parts[1]) if len(parts) > 1 else 3.0
            self.offsetdiag(wait)
        elif cmd == "x8diag":
            wait = float(parts[1]) if len(parts) > 1 else 3.0
            self.x8diag(wait)
        elif cmd in ("nodediag", "nodedig"):
            wait = float(parts[1]) if len(parts) > 1 else 5.0
            self.nodediag(wait)
        elif cmd == "triggercheck":
            if len(parts) < 2 or len(parts) > 3:
                print("Usage: triggercheck fl 2 | triggercheck ch4 2 | triggercheck all 2")
                self.last_command_ok = False
                return
            wait = float(parts[2]) if len(parts) > 2 else 1.5
            self.trigger_check(parts[1].lower(), wait)
        elif cmd in ("prearm", "armcheck"):
            self.run_prearm_checks()
        elif cmd == "trigger":
            if len(parts) == 2 and parts[1].lower() == "off":
                self.trigger_off()
                return
            if len(parts) != 3:
                print("Usage: trigger fl 1200 | trigger ch4 1200 | trigger m4 1200 | trigger all 1200 | trigger off")
                self.last_command_ok = False
                return
            self.trigger_pwm(parts[1].lower(), int(parts[2]))
        elif cmd == "triggerp":
            if len(parts) != 3:
                print("Usage: triggerp fl 20 | triggerp ch4 20 | triggerp m4 20 | triggerp all 20")
                self.last_command_ok = False
                return
            self.trigger_percent(parts[1].lower(), float(parts[2]))
        elif cmd == "triggerwin":
            if len(parts) != 4:
                print("Usage: triggerwin fl 1200 3 | triggerwin ch4 1200 3 | triggerwin m4 1200 3 | triggerwin all 1200 3")
                self.last_command_ok = False
                return
            self.trigger_window_pwm(parts[1].lower(), int(parts[2]), float(parts[3]))
        elif cmd == "audit":
            if len(parts) != 2 or parts[1].lower() not in ("hw", "std"):
                print("Usage: audit hw|std")
                return
            self.audit_params(parts[1].lower())
        elif cmd == "setup":
            if len(parts) != 2 or parts[1].lower() not in ("hw", "std"):
                print("Usage: setup hw|std")
                return
            self.apply_setup(parts[1].lower())
        elif cmd == "modes":
            self.print_modes()
        elif cmd == "mode":
            if len(parts) != 2:
                print("Usage: mode stabilize|alt_hold|loiter|guided|guided_nogps|rtl|land|brake|acro|auto|poshold")
                self.last_command_ok = False
                return
            self.set_flight_mode(parts[1])
        elif cmd in ("stabilize", "stable"):
            self.set_flight_mode("STABILIZE")
        elif cmd == "acro":
            self.set_flight_mode("ACRO")
        elif cmd == "auto":
            self.set_flight_mode("AUTO")
        elif cmd in ("alt_hold", "althold"):
            self.set_flight_mode("ALT_HOLD")
        elif cmd == "loiter":
            self.set_flight_mode("LOITER")
        elif cmd in ("poshold", "pos_hold", "positionhold", "position_hold"):
            self.set_flight_mode("POSHOLD")
        elif cmd == "guided":
            self.set_flight_mode("GUIDED")
        elif cmd in ("guided_nogps", "guidednogps", "guided_no_gps"):
            self.set_flight_mode("GUIDED_NOGPS")
        elif cmd == "brake":
            self.set_flight_mode("BRAKE")
        elif cmd == "rtl":
            self.rtl()
        elif cmd == "land":
            self.land()
        elif cmd == "takeoff":
            altitude = float(parts[1]) if len(parts) > 1 else 2.0
            self.takeoff(altitude)
        elif cmd == "launch":
            altitude = float(parts[1]) if len(parts) > 1 else 2.0
            if not self.require_safety_off():
                return
            self.set_flight_mode("GUIDED")
            if self.last_command_ok:
                self.arm()
            if self.last_command_ok:
                self.takeoff(altitude)
        elif cmd in ("flighttest", "testflight"):
            altitude = float(parts[1]) if len(parts) > 1 else 1.0
            speed = float(parts[2]) if len(parts) > 2 else 0.2
            duration = float(parts[3]) if len(parts) > 3 else 1.0
            land_wait = float(parts[4]) if len(parts) > 4 else 8.0
            self.flight_test(altitude, speed, duration, land_wait)
        elif cmd in ("forward", "backward", "back", "left", "right", "up", "down"):
            speed = float(parts[1]) if len(parts) > 1 else 0.5
            duration = float(parts[2]) if len(parts) > 2 else 1.0
            if speed < 0 or speed > 5:
                print("speed must be 0..5 m/s")
                self.last_command_ok = False
                return
            vectors = {
                "forward": (speed, 0.0, 0.0),
                "backward": (-speed, 0.0, 0.0),
                "back": (-speed, 0.0, 0.0),
                "left": (0.0, -speed, 0.0),
                "right": (0.0, speed, 0.0),
                "up": (0.0, 0.0, -speed),
                "down": (0.0, 0.0, speed),
            }
            vx, vy, vz = vectors[cmd]
            self.guided_move(cmd, vx, vy, vz, 0.0, duration)
        elif cmd in ("yawleft", "yawright"):
            rate_deg = float(parts[1]) if len(parts) > 1 else 20.0
            duration = float(parts[2]) if len(parts) > 2 else 1.0
            if rate_deg < 0 or rate_deg > 120:
                print("yaw rate must be 0..120 deg/s")
                self.last_command_ok = False
                return
            sign = -1.0 if cmd == "yawleft" else 1.0
            self.guided_move(cmd, 0.0, 0.0, 0.0, sign * rate_deg * 0.01745329252, duration)
        elif cmd == "hover":
            self.guided_stop()
        elif cmd == "arm":
            force = len(parts) > 1 and parts[1].lower() == "force"
            self.arm(force=force)
        elif cmd == "safety":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: safety on/off")
                self.last_command_ok = False
                return
            self.set_safety_switch(dangerous=parts[1].lower() == "off")
        elif cmd == "disarm":
            self.disarm()
        elif cmd == "limit":
            if len(parts) != 2:
                print("Usage: limit 1500")
                return
            val = int(parts[1])
            if val < 1000 or val > 2200:
                print("PWM limit must be 1000..2200")
                return
            self.pwm_limit = val
            print(f"pwm_limit={self.pwm_limit}")
        elif cmd == "plimit":
            if len(parts) != 2:
                print("Usage: plimit 50")
                return
            val = float(parts[1])
            if val < 0 or val > 100:
                print("Percent limit must be 0..100")
                return
            self.percent_limit = val
            print(f"percent_limit={self.percent_limit}")
        elif cmd == "refresh":
            if len(parts) != 2:
                print("Usage: refresh 0.8")
                return
            val = float(parts[1])
            if val < 0.2 or val > 10:
                print("refresh must be 0.2..10")
                return
            self.refresh_period = val
            print(f"refresh_period={self.refresh_period}")
        elif cmd == "hold":
            if len(parts) != 2:
                print("Usage: hold 3.0")
                return
            val = float(parts[1])
            if val < 0.5 or val > 30:
                print("hold must be 0.5..30")
                return
            self.hold_duration = val
            print(f"hold_duration={self.hold_duration}")
        elif cmd == "verbose":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: verbose on/off")
                return
            self.verbose = parts[1].lower() == "on"
            print(f"verbose={self.verbose}")
        elif cmd == "outmon":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: outmon on/off")
                self.last_command_ok = False
                return
            self.outmon = parts[1].lower() == "on"
            print(f"outmon={self.outmon}")
        elif cmd == "ack":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                print("Usage: ack on/off")
                self.last_command_ok = False
                return
            self.ack_required = parts[1].lower() == "on"
            print(f"ack_required={self.ack_required}")
        elif cmd == "escgate":
            if len(parts) != 2 or parts[1].lower() not in ("all", "target", "off"):
                print("Usage: escgate all|target|off")
                self.last_command_ok = False
                return
            self.esc_gate = parts[1].lower()
            print(f"escgate={self.esc_gate}")
        elif cmd == "all":
            if len(parts) != 2:
                print("Usage: all 1200")
                return
            self.set_all_pwm(int(parts[1]))
        elif cmd == "allp":
            if len(parts) != 2:
                print("Usage: allp 5")
                return
            self.set_all_percent(float(parts[1]))
        elif len(cmd) == 2 and cmd[0] == "m" and cmd[1] in "1234":
            if len(parts) != 2:
                print("Usage: m1 1200")
                return
            self.set_active_pwm(int(cmd[1]), int(parts[1]))
        elif len(cmd) == 3 and cmd[0] == "m" and cmd[1] in "1234" and cmd[2] == "p":
            if len(parts) != 2:
                print("Usage: m1p 5")
                return
            self.set_active_percent(int(cmd[1]), float(parts[1]))
        elif self.parse_position_token(cmd) is not None:
            channel, is_percent = self.parse_position_token(cmd)
            if len(parts) != 2:
                print(f"Usage: {cmd} {'5' if is_percent else '1200'}")
                return
            if is_percent:
                self.set_output_percent(channel, float(parts[1]))
            else:
                self.set_output_pwm(channel, int(parts[1]))
        elif self.parse_output_token(cmd) is not None:
            channel, is_percent = self.parse_output_token(cmd)
            if len(parts) != 2:
                print(f"Usage: ch{channel}{'p' if is_percent else ''} {'5' if is_percent else '1200'}")
                return
            if is_percent:
                self.set_output_percent(channel, float(parts[1]))
            else:
                self.set_output_pwm(channel, int(parts[1]))
        elif cmd == "maptest":
            pwm = int(parts[1]) if len(parts) > 1 else 1200
            dwell = float(parts[2]) if len(parts) > 2 else 3.0
            self.maptest(pwm, dwell)
        elif cmd == "sniffpwm":
            if len(parts) != 5:
                print("Usage: sniffpwm m4 1150 1450 6")
                return
            motor_id = self.parse_motor_token(parts[1])
            if motor_id is None:
                return
            self.sniffseq_pwm(motor_id, int(parts[2]), int(parts[3]), float(parts[4]))
        elif cmd == "sniffch":
            if len(parts) != 5:
                print("Usage: sniffch fl 1150 1450 6 | sniffch ch4 1150 1450 6")
                return
            channel = self.parse_position_token(parts[1], allow_percent=False)
            if channel is None:
                channel = self.parse_output_token(parts[1], allow_percent=False)
            if channel is None:
                return
            motor_id = self.output_channel_to_motor_id(channel)
            if motor_id is not None:
                self.sniffseq_pwm(motor_id, int(parts[2]), int(parts[3]), float(parts[4]))
        elif cmd == "sniffpct":
            if len(parts) != 5:
                print("Usage: sniffpct m4 20 50 6")
                return
            motor_id = self.parse_motor_token(parts[1])
            if motor_id is None:
                return
            self.sniffseq_percent(motor_id, float(parts[2]), float(parts[3]), float(parts[4]))
        elif cmd == "scan":
            if len(parts) != 6:
                print("Usage: scan m4 1100 1450 50 2")
                return
            motor_id = self.parse_motor_token(parts[1])
            if motor_id is None:
                return
            self.scan_pwm(motor_id, int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]))
        elif cmd == "param":
            if len(parts) != 3:
                print("Usage: param NAME VALUE")
                return
            self.set_param(parts[1], float(parts[2]))
        elif cmd == "getparam":
            if len(parts) != 2:
                print("Usage: getparam NAME")
                return
            self.get_param(parts[1])
        elif cmd == "reboot":
            self.reboot()
        else:
            print("Unknown command. Type help.")
            self.last_command_ok = False

    def parse_motor_token(self, token: str) -> Optional[int]:
        token = token.lower()
        if len(token) == 2 and token[0] == "m" and token[1] in "1234":
            return int(token[1])
        print("motor must be m1..m4")
        return None

    def parse_motor_or_output_target(self, token: str) -> Optional[int]:
        token = token.lower()
        if len(token) == 2 and token[0] == "m" and token[1] in "1234":
            return int(token[1])
        position = self.parse_position_name(token)
        if position is not None:
            return self.position_to_motor_id(position)
        channel = self.parse_output_token(token, allow_percent=False, print_error=False)
        if channel is None:
            print("target must be m1..m4, ch1..ch4/slot1..slot4, or fl/fr/rl/rr")
            return None
        return self.output_channel_to_motor_id(channel)

    def parse_position_name(self, token: str, allow_percent: bool = False):
        token = token.lower()
        is_percent = False
        if allow_percent and token.endswith("p"):
            is_percent = True
            token = token[:-1]
        position = POSITION_ALIASES_TO_POSITION.get(token)
        if position is None:
            return None
        return (position, is_percent) if allow_percent else position

    def parse_position_token(self, token: str, allow_percent: bool = True):
        parsed = self.parse_position_name(token, allow_percent=allow_percent)
        if parsed is None:
            return None
        if allow_percent:
            position, is_percent = parsed
        else:
            position = parsed
            is_percent = False
        channel = self.position_to_output_channel(position)
        if channel is None:
            return None
        return (channel, is_percent) if allow_percent else channel

    def parse_output_token(self, token: str, allow_percent: bool = True, print_error: bool = True):
        token = token.lower()
        is_percent = False
        if allow_percent and token.endswith("p"):
            is_percent = True
            token = token[:-1]
        for prefix in ("ch", "slot"):
            if token.startswith(prefix):
                suffix = token[len(prefix):]
                if suffix in ("1", "2", "3", "4"):
                    channel = int(suffix)
                    return (channel, is_percent) if allow_percent else channel
        if print_error and not allow_percent:
            print("output must be ch1..ch4 or slot1..slot4")
        return None

    def repl(self):
        print("Type help for commands.")
        prompt = "pixhawk> "
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while True:
            self.poll_mavlink(print_messages=True)
            self.refresh_active_outputs()
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                line = sys.stdin.readline()
                try:
                    self.handle(line)
                except SystemExit:
                    return
                except Exception as e:
                    print(f"ERROR: {e}")
                sys.stdout.write(prompt)
                sys.stdout.flush()

    def run_commands(self, commands: str, settle: float = 0.2):
        command_list = []
        for chunk in commands.replace("\n", ";").split(";"):
            command = chunk.strip()
            if command:
                command_list.append(command)
        if not command_list:
            print("No commands to run.")
            return

        for command in command_list:
            print(f"\n>>> {command}")
            try:
                self.handle(command)
            except SystemExit:
                return
            if not self.last_command_ok:
                raise RuntimeError(f"Command failed: {command}")
            if settle > 0:
                end = time.time() + settle
                while time.time() < end:
                    self.poll_mavlink(print_messages=True)
                    self.refresh_active_outputs()
                    time.sleep(0.03)


class _FakeMav:
    def __init__(self, master):
        self.master = master

    def command_long_send(self, target_system, target_component, command, confirmation, p1, p2, p3, p4, p5, p6, p7):
        ack_result = self.master.next_ack_result(command)
        self.master.sent.append(
            {
                "kind": "command_long",
                "command": command,
                "params": (p1, p2, p3, p4, p5, p6, p7),
            }
        )
        self.master.messages.append(_FakeCommandAck(command, ack_result))
        if int(command) == int(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM) and int(ack_result) == int(MAV_RESULT_ACCEPTED):
            self.master.armed = bool(p1)
            self.master.messages.append(_FakeHeartbeat(self.master.current_mode, armed=self.master.armed))
        if int(command) == int(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL) and int(p2) > 0:
            esc_msg = self.master.fake_esc_telemetry_for_msg_id(int(p1))
            if esc_msg is not None:
                self.master.messages.append(esc_msg)

    def set_mode_send(self, target_system, base_mode, custom_mode):
        self.master.sent.append(
            {
                "kind": "set_mode",
                "target_system": target_system,
                "base_mode": base_mode,
                "custom_mode": custom_mode,
            }
        )
        self.master.current_mode = custom_mode
        self.master.messages.append(_FakeHeartbeat(custom_mode, armed=self.master.armed))

    def set_position_target_local_ned_send(
        self,
        time_boot_ms,
        target_system,
        target_component,
        coordinate_frame,
        type_mask,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        afx,
        afy,
        afz,
        yaw,
        yaw_rate,
    ):
        self.master.sent.append(
            {
                "kind": "set_position_target_local_ned",
                "frame": coordinate_frame,
                "type_mask": type_mask,
                "velocity": (vx, vy, vz),
                "yaw_rate": yaw_rate,
            }
        )

    def param_set_send(self, target_system, target_component, name, value, param_type):
        self.master.sent.append({"kind": "param_set", "name": name, "value": value, "param_type": param_type})

    def param_request_read_send(self, target_system, target_component, name, index):
        self.master.sent.append({"kind": "param_request_read", "name": name, "index": index})
        param_id = name.decode(errors="ignore") if isinstance(name, bytes) else str(name)
        self.master.messages.append(_FakeParamValue(param_id, self.master.params.get(param_id, 0.0)))


class _FakeMaster:
    def __init__(
        self,
        ack_result: int = MAV_RESULT_ACCEPTED,
        ack_results_by_command: Optional[Dict[int, int]] = None,
        ack_sequences_by_command: Optional[Dict[int, List[int]]] = None,
    ):
        self.target_system = 1
        self.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        self.sent = []
        self.messages = []
        self.ack_result = ack_result
        self.ack_results_by_command = {int(k): int(v) for k, v in (ack_results_by_command or {}).items()}
        self.ack_sequences_by_command = {
            int(k): [int(result) for result in v]
            for k, v in (ack_sequences_by_command or {}).items()
        }
        self.fake_esc_telemetry = _FakeEscTelemetry()
        self.current_mode = 0
        self.armed = False
        self.params = {
            "SERVO1_FUNCTION": 36.0,
            "SERVO2_FUNCTION": 35.0,
            "SERVO3_FUNCTION": 34.0,
            "SERVO4_FUNCTION": 33.0,
            "ESC_TLM_MAV_OFS": 0.0,
            "ARMING_CHECK": 1.0,
            "BRD_SAFETY_DEFLT": 0.0,
            "BRD_SAFETYOPTION": 3.0,
            "BATT_MONITOR": 9.0,
            "GPS_TYPE": 1.0,
            "EK3_ENABLE": 1.0,
            "AHRS_EKF_TYPE": 3.0,
            "FENCE_ENABLE": 0.0,
            "FS_THR_ENABLE": 0.0,
        }
        self.fake_esc_telemetry_by_msg_id = {}
        self.mav = _FakeMav(self)
        self._mode_mapping = {
            "STABILIZE": 0,
            "ACRO": 1,
            "ALT_HOLD": 2,
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "LAND": 9,
            "POSHOLD": 16,
            "BRAKE": 17,
            "GUIDED_NOGPS": 20,
        }

    def recv_match(self, blocking=False):
        if self.messages:
            return self.messages.pop(0)
        return None

    def next_ack_result(self, command: int) -> int:
        command = int(command)
        sequence = self.ack_sequences_by_command.get(command)
        if sequence:
            return sequence.pop(0)
        return self.ack_results_by_command.get(command, self.ack_result)

    def mode_mapping(self):
        return self._mode_mapping

    def fake_esc_telemetry_for_msg_id(self, msg_id: int):
        if msg_id in self.fake_esc_telemetry_by_msg_id:
            return self.fake_esc_telemetry_by_msg_id[msg_id]
        if msg_id == int(MSG_ESC_TELEMETRY_1_TO_4):
            return self.fake_esc_telemetry
        return None


class _FakeCommandAck:
    def __init__(self, command: int, result: int = mavutil.mavlink.MAV_RESULT_ACCEPTED):
        self.command = command
        self.result = result

    def get_type(self):
        return "COMMAND_ACK"


class _FakeParamValue:
    def __init__(self, param_id: str, param_value: float):
        self.param_id = param_id
        self.param_value = param_value

    def get_type(self):
        return "PARAM_VALUE"


class _FakeEscTelemetry:
    def __init__(
        self,
        present_ids: Iterable[int] = (1, 2, 3, 4),
        first_id: int = 1,
        msg_type: str = "ESC_TELEMETRY_1_TO_4",
    ):
        present = set(int(x) for x in present_ids)
        self.msg_type = msg_type
        self.temperature = [15 if i in present else 0 for i in range(first_id, first_id + 4)]
        self.voltage = [4500 if i in present else 0 for i in range(first_id, first_id + 4)]
        self.current = [0, 0, 0, 0]
        self.totalcurrent = [0, 0, 0, 0]
        self.rpm = [0, 0, 0, 0]
        self.count = [100 if i in present else 0 for i in range(first_id, first_id + 4)]

    def get_type(self):
        return self.msg_type


class _FakeEscStatus:
    def get_type(self):
        return "ESC_STATUS"


class _FakeHeartbeat:
    def __init__(self, custom_mode: int, armed: bool = False):
        self.type = mavutil.mavlink.MAV_TYPE_QUADROTOR
        self.autopilot = mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        self.base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if armed:
            self.base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        self.custom_mode = custom_mode

    def get_type(self):
        return "HEARTBEAT"


def _require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _motor_test_events(fake: _FakeMaster):
    return [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST)
    ]


def run_self_test():
    _require(decode_hobbywing_get_esc_id(2, [2, 2]) == (2, 2), "GetEscID response must decode")
    _require(decode_hobbywing_get_esc_id(2, [0]) is None, "GetEscID query must not decode as a response")
    _require(decode_hobbywing_get_esc_id(2, [3, 2]) is None, "GetEscID NodeID must match CAN source")

    class _RetryConnection:
        def __init__(self):
            self.attempts = 0
            self.close_calls = 0

        def connect(self, show_help=True):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("simulated heartbeat timeout")

        def close(self):
            self.close_calls += 1

    retry_connection = _RetryConnection()
    connect_with_retries(retry_connection, show_help=False, retries=1, delay=0)
    _require(retry_connection.attempts == 2, "heartbeat timeout should retry the connection")
    _require(retry_connection.close_calls == 1, "failed connection must release its serial handle before retry")

    cli = PixhawkX8CLI("mock", 0)
    fake = _FakeMaster()
    cli.master = fake
    cli.target_system = fake.target_system
    cli.target_component = fake.target_component

    cli.last_heartbeat = _FakeHeartbeat(fake.current_mode, armed=False)
    cli.handle("zero")
    _require(not _motor_test_events(fake), "zero must not send MAV_CMD_DO_MOTOR_TEST before safety off or active tx")

    cli.handle("tx on")
    cli.handle("tx off")
    _require(not _motor_test_events(fake), "tx off must not send MAV_CMD_DO_MOTOR_TEST before safety off")

    cli.handle("trigger fl 1200")
    _require(not cli.last_command_ok, "trigger must require explicit safety off first")
    _require(not cli.tx_enabled, "trigger must not enable tx before safety off is confirmed")
    _require(not _motor_test_events(fake), "trigger must not send MAV_CMD_DO_MOTOR_TEST before safety off")

    cli.handle("safety off")
    _require(cli.safety_off_confirmed, "safety off should unlock motor-test activation after accepted ACK")
    safety_events = [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE)
    ]
    _require(safety_events, "safety off should send MAV_CMD_DO_SET_SAFETY_SWITCH_STATE")
    _require(safety_events[-1]["params"][0] == SAFETY_SWITCH_STATE_DANGEROUS, "safety off should request DANGEROUS state")

    motor_events_before_armed_attempt = len(_motor_test_events(fake))
    cli.last_heartbeat = _FakeHeartbeat(fake.current_mode, armed=True)
    cli.handle("trigger fl 1200")
    _require(not cli.last_command_ok, "trigger must reject an ARMED Pixhawk")
    _require(not cli.tx_enabled, "ARMED rejection must not enable motor-test tx")
    _require(len(_motor_test_events(fake)) == motor_events_before_armed_attempt, "ARMED rejection must not send MAV_CMD_DO_MOTOR_TEST")
    cli.last_heartbeat = _FakeHeartbeat(fake.current_mode, armed=False)


    cli.handle("trigger fl 1200")
    _require(cli.tx_enabled, "trigger should enable tx")
    _require(4 in cli.active, "trigger fl should address front-left motor-test M4")
    _require(cli.command_acks, "trigger should receive COMMAND_ACK")
    _require(
        any(int(ack.command) == int(mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST) for ack in cli.command_acks),
        "trigger should receive MAV_CMD_DO_MOTOR_TEST ACK",
    )
    events = _motor_test_events(fake)
    _require(events, "trigger fl should send MAV_CMD_DO_MOTOR_TEST")
    _require(events[-1]["params"][0] == 4, "fl should map to motor-test order M4")
    _require(events[-1]["params"][2] == 1200, "trigger fl 1200 should send PWM 1200")

    cli.handle("trigger fr 1200")
    events = _motor_test_events(fake)
    _require(events[-1]["params"][0] == 1, "fr should map to motor-test order M1")
    cli.handle("trigger rr 1200")
    events = _motor_test_events(fake)
    _require(events[-1]["params"][0] == 2, "rr should map to motor-test order M2")
    cli.handle("trigger rl 1200")
    events = _motor_test_events(fake)
    _require(events[-1]["params"][0] == 3, "rl should map to motor-test order M3")
    cli.handle("trigger ch4 1200")
    events = _motor_test_events(fake)
    _require(events[-1]["params"][0] == 1, "with SERVO4_FUNCTION=33, ch4 should map to motor-test M1/FR")
    _require(
        cli.motor_ids_to_esc_ids((1, 2, 3, 4)) == [1, 2, 3, 4],
        "all motor-test positions must cover ThrottleID 1..4",
    )
    _require(cli.esc_id_label(1).endswith("RR/M2"), "ThrottleID 1 must be rear-right")
    _require(cli.esc_id_label(2).endswith("FL/M4"), "ThrottleID 2 must be front-left")
    _require(cli.esc_id_label(3).endswith("RL/M3"), "ThrottleID 3 must be rear-left")
    _require(cli.esc_id_label(4).endswith("FR/M1"), "ThrottleID 4 must be front-right")

    cli.handle("safety off")
    safety_events = [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE)
    ]
    _require(safety_events, "safety off should send MAV_CMD_DO_SET_SAFETY_SWITCH_STATE")
    _require(safety_events[-1]["params"][0] == SAFETY_SWITCH_STATE_DANGEROUS, "safety off should request DANGEROUS state")

    cli.handle("mode guided")
    mode_events = [event for event in fake.sent if event.get("kind") == "set_mode"]
    _require(mode_events and mode_events[-1]["custom_mode"] == 4, "mode guided should request GUIDED mode")

    cli.handle("stabilize")
    mode_events = [event for event in fake.sent if event.get("kind") == "set_mode"]
    _require(mode_events and mode_events[-1]["custom_mode"] == 0, "stabilize should request STABILIZE mode")

    cli.handle("poshold")
    mode_events = [event for event in fake.sent if event.get("kind") == "set_mode"]
    _require(mode_events and mode_events[-1]["custom_mode"] == 16, "poshold should request POSHOLD mode")

    cli.handle("guided_nogps")
    mode_events = [event for event in fake.sent if event.get("kind") == "set_mode"]
    _require(mode_events and mode_events[-1]["custom_mode"] == 20, "guided_nogps should request GUIDED_NOGPS mode")

    cli.handle("auto")
    mode_events = [event for event in fake.sent if event.get("kind") == "set_mode"]
    _require(mode_events and mode_events[-1]["custom_mode"] == 3, "auto should request AUTO mode")

    cli.handle("mode guided")

    cli.handle("arm")
    _require(cli.is_armed() is True, "arm should confirm HEARTBEAT armed state")
    cli.handle("disarm")
    _require(cli.is_armed() is False, "disarm should confirm HEARTBEAT disarmed state")

    cli.handle("preflight 0")
    preflight_param_events = [
        event for event in fake.sent
        if event.get("kind") == "param_request_read"
    ]
    _require(preflight_param_events, "preflight should request flight-critical parameters")
    _require(cli.param_cache.get("ARMING_CHECK") == 1.0, "preflight should collect parameter responses before printing")

    cli.handle("prearm")
    prearm_events = [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(MAV_CMD_RUN_PREARM_CHECKS)
    ]
    _require(prearm_events, "prearm should send MAV_CMD_RUN_PREARM_CHECKS")

    cli.handle("arm")
    _require(cli.is_armed() is True, "movement tests should start from confirmed armed state")

    cli.handle("takeoff 2")
    takeoff_events = [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
    ]
    _require(takeoff_events, "takeoff should send MAV_CMD_NAV_TAKEOFF")
    _require(abs(takeoff_events[-1]["params"][6] - 2.0) < 0.001, "takeoff 2 should request 2m altitude")

    cli.handle("forward 0.5 0.1")
    velocity_events = [event for event in fake.sent if event.get("kind") == "set_position_target_local_ned"]
    _require(velocity_events, "forward should send SET_POSITION_TARGET_LOCAL_NED")
    _require(
        any(abs(event["velocity"][0] - 0.5) < 0.001 for event in velocity_events),
        "forward should send positive body-frame x velocity",
    )
    cli.handle("hover")
    velocity_events = [event for event in fake.sent if event.get("kind") == "set_position_target_local_ned"]
    _require(
        all(abs(v) < 0.001 for v in velocity_events[-1]["velocity"]),
        "hover should send zero velocity",
    )

    cli.handle("rtl")
    _require(fake.sent[-1]["kind"] == "set_mode" and fake.sent[-1]["custom_mode"] == 6, "rtl should request RTL mode")
    cli.handle("land")
    _require(fake.sent[-1]["kind"] == "set_mode" and fake.sent[-1]["custom_mode"] == 9, "land should request LAND mode")

    cli.handle("disarm")
    _require(cli.is_armed() is False, "flighttest should start from disarmed state")
    cli.handle("flighttest 1 0.2 0.1 0")
    _require(cli.is_armed() is False, "flighttest cleanup should disarm")
    takeoff_events = [
        event for event in fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
    ]
    _require(takeoff_events, "flighttest should include takeoff")
    land_events = [
        event for event in fake.sent
        if event.get("kind") == "set_mode" and event.get("custom_mode") == 9
    ]
    _require(land_events, "flighttest should include LAND mode")

    cli.handle("triggerp fl 20")
    events = _motor_test_events(fake)
    _require(events[-1]["params"][1] == 0, "triggerp should use percent throttle type")
    _require(abs(events[-1]["params"][2] - 20.0) < 0.001, "triggerp fl 20 should send 20 percent")

    cli.handle("trigger off")
    _require(not cli.tx_enabled, "trigger off should disable tx")
    _require(not cli.active, "trigger off should clear active outputs")

    cli.run_commands("trigger fl 1200; trigger off", settle=0)
    failed = False
    try:
        cli.tx_enabled = False
        cli.run_commands("trigger ch9 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(failed, "bad trigger target should fail in --commands mode")
    _require(not cli.tx_enabled, "bad trigger target should not enable tx")

    cli.handle("safety on")
    _require(not cli.safety_off_confirmed, "safety on should lock motor-test activation")
    failed = False
    try:
        cli.run_commands("trigger fl 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(failed, "trigger should fail after safety on until safety off is entered again")
    _require(not cli.tx_enabled, "trigger after safety on should not enable tx")

    retry_cli = PixhawkX8CLI("mock", 0)
    retry_fake = _FakeMaster(
        ack_sequences_by_command={
            mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE: [
                mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED,
                MAV_RESULT_ACCEPTED,
            ],
        }
    )
    retry_cli.master = retry_fake
    retry_cli.target_system = retry_fake.target_system
    retry_cli.target_component = retry_fake.target_component
    retry_cli.safety_retry_period = 0.0
    retry_cli.handle("safety off")
    retry_safety_events = [
        event for event in retry_fake.sent
        if event.get("kind") == "command_long" and int(event.get("command")) == int(mavutil.mavlink.MAV_CMD_DO_SET_SAFETY_SWITCH_STATE)
    ]
    _require(len(retry_safety_events) == 2, "safety off should retry until ACK ACCEPTED")
    _require(retry_cli.safety_off_confirmed, "safety off retry should unlock after accepted ACK")

    missing_esc_cli = PixhawkX8CLI("mock", 0)
    missing_esc_fake = _FakeMaster()
    missing_esc_fake.fake_esc_telemetry = _FakeEscTelemetry((2, 3, 4))
    missing_esc_cli.master = missing_esc_fake
    missing_esc_cli.target_system = missing_esc_fake.target_system
    missing_esc_cli.target_component = missing_esc_fake.target_component
    missing_esc_cli.last_heartbeat = _FakeHeartbeat(missing_esc_fake.current_mode, armed=False)
    failed = False
    try:
        missing_esc_cli.run_commands("safety off; trigger fl 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(failed, "escgate=all should reject trigger when ESC1 telemetry is missing")
    _require(not missing_esc_cli.tx_enabled, "escgate failure should not enable tx")

    missing_esc_cli.handle("escgate target")
    missing_esc_cli.run_commands("trigger fl 1200; trigger off", settle=0)
    _require(not missing_esc_cli.tx_enabled, "trigger off after escgate target debug should disable tx")

    offset_cli = PixhawkX8CLI("mock", 0)
    offset_fake = _FakeMaster()
    offset_fake.fake_esc_telemetry = _FakeEscTelemetry((2, 3, 4), first_id=1)
    offset_fake.fake_esc_telemetry_by_msg_id[MSG_ESC_TELEMETRY_5_TO_8] = _FakeEscTelemetry(
        (5,),
        first_id=5,
        msg_type="ESC_TELEMETRY_5_TO_8",
    )
    offset_cli.master = offset_fake
    offset_cli.target_system = offset_fake.target_system
    offset_cli.target_component = offset_fake.target_component
    offset_cli.last_heartbeat = _FakeHeartbeat(offset_fake.current_mode, armed=False)
    failed = False
    try:
        offset_cli.run_commands("safety off; trigger all 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(not failed, "a unique MAVLink-only +1 display shift should not reject four healthy ESCs")
    _require(offset_cli.tx_enabled, "MAVLink display shift should preserve trigger readiness")
    _require(offset_cli.infer_mavlink_slot_shifts((2, 3, 4, 5)) == [1], "online slots 2..5 should detect display shift +1")
    _require(offset_cli.current_can_esc_offset() == 0, "display shift must not change CAN_D1_UC_ESC_OF")
    offset_cli.run_commands("trigger off", settle=0)
    _require(not offset_cli.tx_enabled, "trigger off after display-shift test should disable tx")

    x8diag_cli = PixhawkX8CLI("mock", 0)
    x8diag_fake = _FakeMaster()
    x8diag_cli.master = x8diag_fake
    x8diag_cli.target_system = x8diag_fake.target_system
    x8diag_cli.target_component = x8diag_fake.target_component
    before_motor_tests = len(_motor_test_events(x8diag_fake))
    x8diag_cli.handle("x8diag 0")
    after_motor_tests = len(_motor_test_events(x8diag_fake))
    _require(x8diag_cli.last_command_ok, "x8diag should be a successful read-only diagnostic")
    _require(after_motor_tests == before_motor_tests, "x8diag must not send MAV_CMD_DO_MOTOR_TEST")
    _require(not x8diag_cli.tx_enabled, "x8diag must not enable tx")

    esc_status_cli = PixhawkX8CLI("mock", 0)
    esc_status_fake = _FakeMaster()
    esc_status_cli.master = esc_status_fake
    esc_status_cli.target_system = esc_status_fake.target_system
    esc_status_cli.target_component = esc_status_fake.target_component
    esc_status_cli.last_heartbeat = _FakeHeartbeat(esc_status_fake.current_mode, armed=False)
    esc_status_cli.handle("safety off")
    esc_status_cli.poll_mavlink(print_messages=False)
    esc_status_cli.last_esc_telemetry = _FakeEscTelemetry((1, 2, 3, 4))
    esc_status_cli.last_esc_telemetry_time = time.time()
    esc_status_cli.esc_telemetry_frames = 1
    esc_status_cli.last_esc = _FakeEscStatus()
    esc_status_cli.last_esc_time = time.time()
    esc_status_cli.last_esc_status = esc_status_cli.last_esc
    esc_status_cli.last_esc_status_time = esc_status_cli.last_esc_time
    esc_status_cli.esc_status_frames = 1
    esc_status_cli.handle("trigger fl 1200")
    _require(esc_status_cli.tx_enabled, "ESC_STATUS must not hide valid ESC_TELEMETRY_1_TO_4 readiness")

    stale_esc_cli = PixhawkX8CLI("mock", 0)
    stale_esc_fake = _FakeMaster()
    stale_esc_fake.fake_esc_telemetry = None
    stale_esc_cli.master = stale_esc_fake
    stale_esc_cli.target_system = stale_esc_fake.target_system
    stale_esc_cli.target_component = stale_esc_fake.target_component
    stale_esc_cli.last_heartbeat = _FakeHeartbeat(stale_esc_fake.current_mode, armed=False)
    stale_esc_cli.handle("safety off")
    stale_esc_cli.poll_mavlink(print_messages=False)
    stale_esc_cli.last_esc_telemetry = _FakeEscTelemetry((1, 2, 3, 4))
    stale_esc_cli.last_esc_telemetry_time = time.time() - (ESC_TELEMETRY_MAX_AGE + 10.0)
    stale_esc_cli.esc_telemetry_frames = 1
    failed = False
    try:
        stale_esc_cli.run_commands("trigger fl 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(failed, "stale ESC_TELEMETRY_1_TO_4 must reject trigger even when old data showed all ESCs")
    _require(not stale_esc_cli.tx_enabled, "stale ESC telemetry rejection should not enable tx")

    denied_cli = PixhawkX8CLI("mock", 0)
    denied_fake = _FakeMaster(
        ack_results_by_command={
            mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST: mavutil.mavlink.MAV_RESULT_DENIED,
        }
    )
    denied_cli.master = denied_fake
    denied_cli.target_system = denied_fake.target_system
    denied_cli.target_component = denied_fake.target_component
    denied_cli.last_heartbeat = _FakeHeartbeat(denied_fake.current_mode, armed=False)
    failed = False
    try:
        denied_cli.run_commands("safety off; trigger fl 1200", settle=0)
    except RuntimeError:
        failed = True
    _require(failed, "DENIED motor-test ACK should fail --commands mode")

    print("DIAG SELF TEST PASS: direct trigger motor-test flow is valid.")


def main():
    parser = argparse.ArgumentParser(description="Pixhawk 6X + Hobbywing X8 G2 DroneCAN diagnostic CLI")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Pixhawk serial port, e.g. /dev/ttyACM0 or auto")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="baudrate, usually 115200 for USB")
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT, help="seconds to wait for Pixhawk heartbeat")
    parser.add_argument("--wait-port", type=float, default=0.0, help="seconds to wait for the serial port to appear")
    parser.add_argument("--connect-retries", type=int, default=0, help="retry count after serial open/connect errors")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="seconds between --connect-retries attempts")
    parser.add_argument("--list-ports", action="store_true", help="list likely Pixhawk serial ports and exit")
    parser.add_argument("--can-probe", type=float, metavar="SECONDS", help="passively decode Hobbywing DroneCAN traffic, then exit")
    parser.add_argument("--can-config-probe", type=float, metavar="SECONDS", help="read Hobbywing GetMajorConfig from ESCs, then exit")
    parser.add_argument("--can-node-info", type=float, metavar="SECONDS", help="read standard DroneCAN GetNodeInfo from ESCs, then exit")
    parser.add_argument("--set-can-throttle", type=float, metavar="SECONDS", help="set expected Hobbywing ESCs to CAN_DIGITAL throttle")
    parser.add_argument("--can-bus", type=int, default=1, choices=(1, 2), help="physical CAN bus for CAN probes (default: 1)")
    parser.add_argument("--commands", help="semicolon-separated CLI commands to run after connecting, then exit")
    parser.add_argument("--command-settle", type=float, default=0.2, help="seconds to poll MAVLink after each --commands item")
    parser.add_argument("--post-listen", type=float, default=0.0, help="seconds to listen after --commands completes")
    parser.add_argument("--repl-after-commands", action="store_true", help="enter interactive prompt after --commands")
    parser.add_argument("--trigger", nargs=2, metavar=("TARGET", "PWM"), help="disabled safety shortcut; use --commands with explicit 'safety off; trigger ...'")
    parser.add_argument("--trigger-percent", nargs=2, metavar=("TARGET", "PERCENT"), help="disabled safety shortcut; use --commands with explicit 'safety off; triggerp ...'")
    parser.add_argument("--trigger-duration", type=float, default=0.0, help="seconds to run --trigger before automatic trigger off")
    parser.add_argument("--self-test", action="store_true", help="run local direct-trigger tests without Pixhawk hardware")
    args = parser.parse_args()
    if args.list_ports:
        print_serial_ports()
        return
    if args.self_test:
        run_self_test()
        return
    if args.trigger or args.trigger_percent:
        print("ERROR: direct --trigger shortcuts are disabled.", file=sys.stderr)
        print("Use the interactive CLI, or pass an explicit command list such as:", file=sys.stderr)
        print('  --commands "safety off; triggercheck all 3; trigger fl 1200"', file=sys.stderr)
        raise SystemExit(2)

    try:
        port = wait_for_port(args.port, args.wait_port)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_serial_ports()
        raise SystemExit(2) from exc
    if port.startswith("/dev/") and not os.path.exists(port):
        print(f"ERROR: serial port does not exist: {port}", file=sys.stderr)
        print_serial_ports()
        raise SystemExit(2)
    if port.startswith("/dev/") and not os.access(port, os.R_OK | os.W_OK):
        print(f"ERROR: no read/write permission for serial port: {port}", file=sys.stderr)
        print("Add the user to the device group, commonly: sudo usermod -a -G dialout $USER", file=sys.stderr)
        print("Then log out and back in before running this script again.", file=sys.stderr)
        raise SystemExit(2)

    if args.can_probe is not None:
        if args.can_probe <= 0:
            print("ERROR: --can-probe must be greater than zero seconds.", file=sys.stderr)
            raise SystemExit(2)
        try:
            passed = run_dronecan_probe(port, args.baud, args.can_probe, args.can_bus)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        raise SystemExit(0 if passed else 4)

    if args.can_config_probe is not None:
        if args.can_config_probe <= 0:
            print("ERROR: --can-config-probe must be greater than zero seconds.", file=sys.stderr)
            raise SystemExit(2)
        try:
            passed = run_hobbywing_config_probe(port, args.baud, args.can_config_probe, args.can_bus)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        raise SystemExit(0 if passed else 4)

    if args.can_node_info is not None:
        if args.can_node_info <= 0:
            print("ERROR: --can-node-info must be greater than zero seconds.", file=sys.stderr)
            raise SystemExit(2)
        try:
            passed = run_dronecan_node_info_probe(port, args.baud, args.can_node_info, args.can_bus)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        raise SystemExit(0 if passed else 4)

    if args.set_can_throttle is not None:
        if args.set_can_throttle <= 0:
            print("ERROR: --set-can-throttle must be greater than zero seconds.", file=sys.stderr)
            raise SystemExit(2)
        try:
            passed = set_hobbywing_can_throttle(port, args.baud, args.set_can_throttle, args.can_bus)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        raise SystemExit(0 if passed else 4)

    cli = PixhawkX8CLI(port, args.baud, connect_timeout=args.connect_timeout)
    try:
        connect_with_retries(
            cli,
            show_help=not (args.commands or args.trigger or args.trigger_percent),
            retries=args.connect_retries,
            delay=args.retry_delay,
        )
        direct_commands = args.commands

        if direct_commands:
            cli.run_commands(direct_commands, settle=args.command_settle)
            if args.post_listen > 0:
                cli.listen(args.post_listen)
            if args.repl_after_commands:
                cli.repl()
            return
        cli.repl()
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except SERIAL_EXCEPTIONS as exc:
        print_serial_error(port, exc)
        raise SystemExit(2) from exc
    finally:
        cli.close()


if __name__ == "__main__":
    main()
