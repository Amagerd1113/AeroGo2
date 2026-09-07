"""Read-mostly pymavlink bridge for Pixhawk 6X / ArduCopter."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from aerogo2.common.async_utils import run_blocking
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import PixhawkConfig
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import EscTelemetry, PixhawkStatus
from aerogo2.common.numeric import finite_real
from aerogo2.common.results import OperationResult

_ESC_TELEMETRY_FIRST_SLOT = {
    "ESC_TELEMETRY_1_TO_4": 1,
    "ESC_TELEMETRY_5_TO_8": 5,
}
_ESC_RAW_UNKNOWN = 65535.0
_GROUND_ARM_AUTH_COMMAND = 31000
_GROUND_ARM_AUTH_MAGIC = 6202.0
_GROUND_ARM_AUTH_PROTOCOL = 1.0
_GROUND_ARM_AUTH_HEARTBEAT_S = 0.4
_GROUND_ARM_AUTH_ACK_TIMEOUT_S = 1.5
# COMMAND_LONG parameters are IEEE-754 float32 on the wire. Every positive
# integer through 2**24 is exactly representable, whereas a wider sequence
# domain can round distinct authorization transactions to the same value.
_GROUND_ARM_AUTH_MAX_EXACT_SEQUENCE = (1 << 24) - 1


@dataclass(frozen=True)
class _RawEscTelemetry:
    slot: int
    rpm: float
    voltage_v: Optional[float]
    current_a: Optional[float]
    temperature_c: Optional[float]
    present: bool
    healthy: bool
    timestamp: float


@dataclass(frozen=True)
class _EscTelemetryGroup:
    timestamp: float
    samples: Tuple[_RawEscTelemetry, ...]


class MavlinkPixhawkBridge:
    """Consume safety telemetry and expose only a non-arming authorization gate."""

    def __init__(
        self,
        config: PixhawkConfig,
        esc_mapping: Mapping[int, str],
        *,
        esc_mavlink_display_shift: int = 0,
        clock: Optional[Clock] = None,
        rc_timeout_s: float = 0.5,
        allow_setpoints: bool = False,
    ) -> None:
        self._config = config
        self._esc_mapping = dict(esc_mapping)
        self._esc_mavlink_display_shift = esc_mavlink_display_shift
        self._clock = clock or RealClock()
        self._rc_timeout_s = rc_timeout_s
        self._esc_timeout_s = rc_timeout_s * 2.0
        self._esc_groups: Dict[str, _EscTelemetryGroup] = {}
        self._allow_setpoints = allow_setpoints
        self._connection: Optional[Any] = None
        self._mavlink: Optional[Any] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._connected = False
        self._last_rc_timestamp = 0.0
        self._setpoint_active = False
        self._ground_arm_authorized = False
        self._ground_arm_authorization_deadline = 0.0
        # The remote live gate is closed immediately on the first observed
        # armed edge.  Retain one bounded logical receipt so SystemManager can
        # classify that same edge as authorized before explicitly consuming it.
        self._ground_arm_consumed_receipt = False
        self._ground_arm_authorization_task: Optional[asyncio.Task[None]] = None
        self._ground_arm_authorization_lock = asyncio.Lock()
        self._ground_arm_authorization_sequence = 0
        self._ground_arm_authorization_ack: Optional[Tuple[int, bool, asyncio.Future[int]]] = None
        self._ground_arm_consumption_generation = 0
        self._status = PixhawkStatus(
            esc=tuple(
                EscTelemetry(slot, position, healthy=False)
                for slot, position in sorted(self._esc_mapping.items())
            ),
            esc_mavlink_display_shift=self._esc_mavlink_display_shift,
        )

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            from pymavlink import mavutil  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BridgeError("pymavlink is required for the Pixhawk bridge") from exc
        connection: Optional[Any] = None
        try:
            connection = mavutil.mavlink_connection(
                self._config.connection,
                baud=self._config.baud,
                source_system=255,
                autoreconnect=True,
            )
            heartbeat = await asyncio.wait_for(
                run_blocking(
                    connection.wait_heartbeat,
                    timeout=self._config.heartbeat_timeout_s,
                ),
                timeout=self._config.heartbeat_timeout_s + 1.0,
            )
        except (OSError, asyncio.TimeoutError, RuntimeError) as exc:
            if connection is not None:
                try:
                    connection.close()
                except (OSError, RuntimeError):
                    pass
            raise BridgeError(
                f"Cannot receive Pixhawk heartbeat on {self._config.connection}: {exc}"
            ) from exc
        if heartbeat is None:
            connection.close()
            raise BridgeError("Pixhawk heartbeat timed out")
        self._connection = connection
        self._mavlink = mavutil
        self._connected = True
        self._handle_message(heartbeat)
        self._request_telemetry()
        self._reader_task = asyncio.create_task(self._reader_loop(), name="pixhawk-mavlink-reader")

    async def disconnect(self) -> None:
        task = self._reader_task
        if self._ground_arm_authorized:
            try:
                await self.set_ground_arm_authorization(False, 0.0)
            except (BridgeError, OSError, RuntimeError):
                self._ground_arm_authorized = False
                self._ground_arm_authorization_deadline = 0.0
        await self._cancel_ground_arm_authorization_task()
        self._reader_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        connection = self._connection
        self._connection = None
        self._connected = False
        self._setpoint_active = False
        self._ground_arm_authorized = False
        self._ground_arm_authorization_deadline = 0.0
        self._ground_arm_consumed_receipt = False
        pending = self._ground_arm_authorization_ack
        self._ground_arm_authorization_ack = None
        if pending is not None and not pending[2].done():
            pending[2].cancel()
        self._esc_groups.clear()
        if connection is not None:
            try:
                connection.close()
            except (OSError, RuntimeError):
                pass
        self._status = replace(self._status, connected=False)

    async def run(self) -> None:
        task = self._reader_task
        if task is None:
            raise BridgeError("Pixhawk is not connected")
        await task

    def get_status(self) -> PixhawkStatus:
        now = self._clock.monotonic()
        rc_stale = (
            self._last_rc_timestamp <= 0.0 or now - self._last_rc_timestamp > self._rc_timeout_s
        )
        esc, raw_present_slots, display_shift = self._build_logical_esc_status(now)
        return replace(
            self._status,
            connected=self._connected,
            # RC link validity is independent from the vehicle-wide MAV_STATE.
            # A Pixhawk may report CRITICAL for a battery, sensor, or internal
            # error while continuing to deliver fresh, valid RC_CHANNELS.
            rc_failsafe=self._status.rc_failsafe or rc_stale,
            esc=esc,
            esc_rpm={item.slot: item.rpm for item in esc},
            esc_online={item.slot: item.healthy for item in esc},
            esc_raw_present_slots=raw_present_slots,
            esc_mavlink_display_shift=display_shift,
        )

    def latest_status(self) -> PixhawkStatus:
        return self.get_status()

    async def request_mode(self, mode: str) -> bool:
        del mode
        raise BridgeError("Pixhawk mode changes remain assigned to RadioMaster/ArduPilot")

    async def set_ground_arm_authorization(
        self,
        enabled: bool,
        ttl_s: float,
    ) -> OperationResult:
        if enabled and (not math.isfinite(ttl_s) or ttl_s <= 0.0):
            return OperationResult.failure(
                "INVALID_AUTHORIZATION_TTL",
                "Ground-arm authorization TTL must be finite and positive",
            )
        async with self._ground_arm_authorization_lock:
            if enabled:
                if not self._connected:
                    return OperationResult.failure(
                        "PIXHAWK_DISCONNECTED",
                        "Cannot authorize flight while Pixhawk is disconnected",
                    )
                if self._status.armed:
                    return OperationResult.failure(
                        "PIXHAWK_ALREADY_ARMED",
                        "Ground-arm authorization cannot be opened after Pixhawk is armed",
                    )
                await self._cancel_ground_arm_authorization_task()
                self._ground_arm_consumed_receipt = False
                consumption_generation = self._ground_arm_consumption_generation
                exchange = await self._exchange_ground_arm_authorization(True, ttl_s)
                if not exchange.ok:
                    self._ground_arm_authorized = False
                    self._ground_arm_authorization_deadline = 0.0
                    return exchange
                if (
                    self._ground_arm_consumption_generation != consumption_generation
                    or self._status.armed
                ):
                    self._ground_arm_authorized = False
                    self._ground_arm_authorization_deadline = 0.0
                    try:
                        self._send_ground_arm_authorization(False, 0.0, 0)
                    except (BridgeError, OSError, RuntimeError, ValueError):
                        pass
                    return OperationResult.failure(
                        "PIXHAWK_ARM_AUTH_GATE_CONSUMED",
                        "Pixhawk armed while authorization was being acknowledged; the one-shot gate was closed",
                    )
                self._ground_arm_authorized = True
                self._ground_arm_authorization_deadline = self._clock.monotonic() + ttl_s
                self._ground_arm_authorization_task = asyncio.create_task(
                    self._ground_arm_authorization_loop(ttl_s),
                    name="pixhawk-ground-arm-authorization",
                )
                return OperationResult.success(
                    "Pixhawk ground-arm gate acknowledged authorization",
                    data={"ttl_s": ttl_s},
                )

            self._ground_arm_authorized = False
            self._ground_arm_authorization_deadline = 0.0
            self._ground_arm_consumed_receipt = False
            await self._cancel_ground_arm_authorization_task()
            if not self._connected:
                return OperationResult.success(
                    "Ground-arm authorization cleared while Pixhawk was disconnected"
                )
            return await self._exchange_ground_arm_authorization(False, 0.0)

    def ground_arm_authorization_active(self) -> bool:
        live_gate = (
            self._ground_arm_authorized
            and self._ground_arm_authorization_deadline > self._clock.monotonic()
        )
        # True may also mean the unconsumed receipt for the current armed
        # interval. The remote enable heartbeat has already stopped; this is
        # only evidence for the manager's FLIGHT_READY->FLIGHT_MANUAL commit.
        return self._connected and (
            live_gate or (self._ground_arm_consumed_receipt and self._status.armed)
        )

    async def _exchange_ground_arm_authorization(
        self,
        enabled: bool,
        ttl_s: float,
    ) -> OperationResult:
        loop = asyncio.get_running_loop()
        self._ground_arm_authorization_sequence = (
            self._ground_arm_authorization_sequence % _GROUND_ARM_AUTH_MAX_EXACT_SEQUENCE
        ) + 1
        sequence = self._ground_arm_authorization_sequence
        future: asyncio.Future[int] = loop.create_future()
        self._ground_arm_authorization_ack = (sequence, enabled, future)
        try:
            self._send_ground_arm_authorization(enabled, ttl_s, sequence)
            result = await asyncio.wait_for(future, timeout=_GROUND_ARM_AUTH_ACK_TIMEOUT_S)
        except asyncio.TimeoutError:
            return OperationResult.failure(
                "PIXHAWK_ARM_AUTH_GATE_TIMEOUT",
                "Pixhawk Lua arm gate did not acknowledge the authorization command",
            )
        except (BridgeError, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("PIXHAWK_ARM_AUTH_GATE_FAILED", str(exc))
        finally:
            pending = self._ground_arm_authorization_ack
            if pending is not None and pending[0] == sequence:
                self._ground_arm_authorization_ack = None
        accepted = int(self._require_mavlink().mavlink.MAV_RESULT_ACCEPTED)
        if result != accepted:
            return OperationResult.failure(
                "PIXHAWK_ARM_AUTH_GATE_REJECTED",
                f"Pixhawk Lua arm gate rejected authorization with MAV_RESULT={result}",
                data={"mav_result": result},
            )
        verb = "accepted" if enabled else "revoked"
        return OperationResult.success(
            f"Pixhawk Lua arm gate {verb} ground authorization",
            data={"mav_result": result},
        )

    def _send_ground_arm_authorization(
        self,
        enabled: bool,
        ttl_s: float,
        sequence: int,
    ) -> None:
        connection = self._require_connection()
        connection.mav.command_long_send(
            self._config.target_system,
            self._config.target_component,
            _GROUND_ARM_AUTH_COMMAND,
            0,
            1.0 if enabled else 0.0,
            _GROUND_ARM_AUTH_MAGIC,
            _GROUND_ARM_AUTH_PROTOCOL,
            float(sequence),
            float(ttl_s),
            0.0,
            0.0,
        )

    async def _ground_arm_authorization_loop(self, ttl_s: float) -> None:
        try:
            while self._connected and self._ground_arm_authorized:
                remaining = self._ground_arm_authorization_deadline - self._clock.monotonic()
                if remaining <= 0.0:
                    break
                await asyncio.sleep(min(_GROUND_ARM_AUTH_HEARTBEAT_S, remaining))
                if (
                    not self._connected
                    or not self._ground_arm_authorized
                    or self._clock.monotonic() >= self._ground_arm_authorization_deadline
                ):
                    break
                self._send_ground_arm_authorization(True, ttl_s, 0)
        except asyncio.CancelledError:
            raise
        except (BridgeError, OSError, RuntimeError, ValueError):
            pass
        finally:
            # Interpreter/test-loop teardown can finalize the coroutine after
            # the event loop has gone away. Cleanup must remain best-effort.
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if self._ground_arm_authorization_task is current:
                self._ground_arm_authorization_task = None
            if self._ground_arm_authorized:
                self._ground_arm_authorized = False
                self._ground_arm_authorization_deadline = 0.0
                if self._connected:
                    try:
                        self._send_ground_arm_authorization(False, 0.0, 0)
                    except (BridgeError, OSError, RuntimeError, ValueError):
                        pass

    async def _cancel_ground_arm_authorization_task(self) -> None:
        task = self._ground_arm_authorization_task
        self._ground_arm_authorization_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def send_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ) -> OperationResult:
        if not self._allow_setpoints:
            return OperationResult.failure(
                "PIXHAWK_SETPOINT_LOCKED", "Real Pixhawk setpoints are locked"
            )
        if not all(math.isfinite(item) for item in (vx, vy, vz, yaw_rate)):
            return OperationResult.failure("INVALID_SETPOINT", "Setpoint values must be finite")
        connection = self._require_connection()
        mavlink = self._require_mavlink().mavlink
        type_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 10)
        try:
            connection.mav.set_position_target_local_ned_send(
                0,
                self._config.target_system,
                self._config.target_component,
                mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0.0,
                0.0,
                0.0,
                vx,
                vy,
                vz,
                0.0,
                0.0,
                0.0,
                0.0,
                yaw_rate,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("PIXHAWK_SETPOINT_FAILED", str(exc))
        self._setpoint_active = True
        return OperationResult.success("Velocity setpoint sent")

    async def stop_external_setpoints(self) -> OperationResult:
        self._setpoint_active = False
        return OperationResult.success("External setpoint stream disabled")

    async def _reader_loop(self) -> None:
        connection = self._require_connection()
        try:
            while self._connected:
                message = await run_blocking(
                    connection.recv_match,
                    blocking=True,
                    timeout=0.2,
                )
                if message is not None:
                    self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._connected = False
            self._ground_arm_authorized = False
            self._ground_arm_authorization_deadline = 0.0
            self._ground_arm_consumed_receipt = False
            task = self._ground_arm_authorization_task
            self._ground_arm_authorization_task = None
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            pending = self._ground_arm_authorization_ack
            if pending is not None and not pending[2].done():
                pending[2].set_exception(BridgeError("Pixhawk reader stopped"))

    def _request_telemetry(self) -> None:
        connection = self._require_connection()
        mavlink = self._require_mavlink().mavlink
        names = (
            "MAVLINK_MSG_ID_HEARTBEAT",
            "MAVLINK_MSG_ID_SYS_STATUS",
            "MAVLINK_MSG_ID_ATTITUDE",
            "MAVLINK_MSG_ID_GLOBAL_POSITION_INT",
            "MAVLINK_MSG_ID_RC_CHANNELS",
            "MAVLINK_MSG_ID_EXTENDED_SYS_STATE",
            "MAVLINK_MSG_ID_ESC_TELEMETRY_1_TO_4",
            "MAVLINK_MSG_ID_ESC_TELEMETRY_5_TO_8",
        )
        for name in names:
            message_id = getattr(mavlink, name, None)
            if message_id is None:
                continue
            try:
                connection.mav.command_long_send(
                    self._config.target_system,
                    self._config.target_component,
                    mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    float(message_id),
                    100_000.0 if "RC_CHANNELS" in name else 200_000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            except (OSError, RuntimeError, ValueError):
                continue

    def _handle_message(self, message: Any) -> None:
        kind = str(message.get_type())
        now = self._clock.monotonic()
        status = self._status
        if kind == "HEARTBEAT":
            mavlink = self._require_mavlink()
            armed = bool(int(message.base_mode) & mavlink.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            armed_rising = armed and not status.armed
            if armed_rising:
                # Authorization is one-shot: stop enable heartbeats and fence
                # an enable ACK that may still be in flight at the first arm.
                self._ground_arm_consumption_generation += 1
                self._ground_arm_consumed_receipt = bool(
                    self._ground_arm_authorized and self._ground_arm_authorization_deadline > now
                )
                self._ground_arm_authorized = False
                self._ground_arm_authorization_deadline = 0.0
                authorization_task = self._ground_arm_authorization_task
                self._ground_arm_authorization_task = None
                if authorization_task is not None and not authorization_task.done():
                    authorization_task.cancel()
                pending = self._ground_arm_authorization_ack
                if pending is not None and pending[1] and not pending[2].done():
                    pending[2].set_exception(
                        BridgeError("Pixhawk armed before the authorization ACK committed")
                    )
                if self._connected:
                    try:
                        self._send_ground_arm_authorization(False, 0.0, 0)
                    except (BridgeError, OSError, RuntimeError, ValueError):
                        pass
            elif not armed and status.armed:
                # Disarm invalidates an unconsumed receipt. A later arm always
                # requires a new authorization transaction.
                self._ground_arm_consumed_receipt = False
            mode = str(mavlink.mode_string_v10(message))
            system_status = int(getattr(message, "system_status", 0))
            failsafe = system_status >= int(mavlink.mavlink.MAV_STATE_CRITICAL)
            status = replace(
                status,
                connected=True,
                armed=armed,
                flight_mode=mode,
                failsafe=failsafe,
                heartbeat_timestamp=now,
                timestamp=now,
            )
        elif kind == "COMMAND_ACK":
            if int(getattr(message, "command", -1)) == _GROUND_ARM_AUTH_COMMAND:
                pending = self._ground_arm_authorization_ack
                sequence = int(getattr(message, "result_param2", 0))
                if pending is not None and pending[0] == sequence and not pending[2].done():
                    pending[2].set_result(int(getattr(message, "result", -1)))
        elif kind == "ATTITUDE":
            roll = finite_real(getattr(message, "roll", None))
            pitch = finite_real(getattr(message, "pitch", None))
            yaw = finite_real(getattr(message, "yaw", None))
            rollspeed = finite_real(getattr(message, "rollspeed", None))
            pitchspeed = finite_real(getattr(message, "pitchspeed", None))
            yawspeed = finite_real(getattr(message, "yawspeed", None))
            if (
                roll is None
                or pitch is None
                or yaw is None
                or rollspeed is None
                or pitchspeed is None
                or yawspeed is None
            ):
                return
            status = replace(
                status,
                attitude_timestamp=now,
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw,
                angular_velocity=(rollspeed, pitchspeed, yawspeed),
            )
        elif kind == "GLOBAL_POSITION_INT":
            relative_alt = finite_real(getattr(message, "relative_alt", None))
            vx = finite_real(getattr(message, "vx", None))
            vy = finite_real(getattr(message, "vy", None))
            vz = finite_real(getattr(message, "vz", None))
            if relative_alt is None or vx is None or vy is None or vz is None:
                return
            status = replace(
                status,
                kinematics_timestamp=now,
                relative_altitude_m=relative_alt / 1000.0,
                vertical_velocity_mps=vz / 100.0,
                local_velocity=(
                    vx / 100.0,
                    vy / 100.0,
                    vz / 100.0,
                ),
            )
        elif kind == "EXTENDED_SYS_STATE":
            mavlink = self._require_mavlink().mavlink
            raw_landed_state = getattr(message, "landed_state", None)
            # pymavlink supplies a plain integer enum. Do not coerce floats,
            # strings, booleans, or arbitrary integer-like values into a
            # ground-only safety permission.
            if type(raw_landed_state) is not int:
                return
            landed_state = raw_landed_state
            if landed_state not in {0, 1, 2, 3, 4}:
                return
            landed = landed_state == int(mavlink.MAV_LANDED_STATE_ON_GROUND)
            status = replace(status, landed=landed, landed_state_timestamp=now)
        elif kind == "SYS_STATUS":
            millivolts = int(getattr(message, "voltage_battery", 65535))
            status = replace(
                status, battery_voltage=None if millivolts == 65535 else millivolts / 1000.0
            )
        elif kind == "RC_CHANNELS":
            channels = {}
            for index in range(1, 17):
                value = int(getattr(message, f"chan{index}_raw", 0))
                if 800 <= value <= 2200:
                    channels[index] = value
            self._last_rc_timestamp = now
            status = replace(status, rc_channels=channels, rc_failsafe=len(channels) < 8)
        elif kind in _ESC_TELEMETRY_FIRST_SLOT:
            self._esc_groups[kind] = _EscTelemetryGroup(
                timestamp=now,
                samples=self._parse_raw_esc_group(
                    message,
                    now,
                    first_slot=_ESC_TELEMETRY_FIRST_SLOT[kind],
                ),
            )
            esc, raw_present_slots, display_shift = self._build_logical_esc_status(now)
            status = replace(
                status,
                esc=esc,
                esc_rpm={item.slot: item.rpm for item in esc},
                esc_online={item.slot: item.healthy for item in esc},
                esc_raw_present_slots=raw_present_slots,
                esc_mavlink_display_shift=display_shift,
            )
        elif kind == "STATUSTEXT":
            text = getattr(message, "text", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            lines = (status.statustext + (str(text).rstrip("\x00"),))[-32:]
            status = replace(status, statustext=lines)
        self._status = status

    def _parse_esc(self, message: Any, timestamp: float) -> Tuple[EscTelemetry, ...]:
        """Parse the first MAVLink ESC group without display shifting."""

        raw_items = {
            item.slot: item for item in self._parse_raw_esc_group(message, timestamp, first_slot=1)
        }
        return tuple(
            self._logical_esc_item(slot, position, raw_items.get(slot))
            for slot, position in sorted(self._esc_mapping.items())
        )

    def _parse_raw_esc_group(
        self,
        message: Any,
        timestamp: float,
        *,
        first_slot: int,
    ) -> Tuple[_RawEscTelemetry, ...]:
        rpm = tuple(getattr(message, "rpm", ()))
        voltage = tuple(getattr(message, "voltage", ()))
        current = tuple(getattr(message, "current", ()))
        temperature = tuple(getattr(message, "temperature", ()))
        count = tuple(getattr(message, "count", ()))
        items = []
        for offset in range(4):
            raw_rpm = self._raw_array_value(rpm, offset)
            raw_voltage = self._raw_array_value(voltage, offset)
            raw_current = self._raw_array_value(current, offset)
            raw_temperature = self._raw_array_value(temperature, offset)
            raw_count = self._raw_array_value(count, offset)
            present = any(
                math.isfinite(value) and value != 0.0
                for value in (raw_rpm, raw_voltage, raw_current, raw_count)
            )
            rpm_valid = math.isfinite(raw_rpm) and 0.0 <= raw_rpm < _ESC_RAW_UNKNOWN
            voltage_valid = math.isfinite(raw_voltage) and 0.0 < raw_voltage < _ESC_RAW_UNKNOWN
            current_valid = math.isfinite(raw_current) and 0.0 <= raw_current < _ESC_RAW_UNKNOWN
            healthy = present and rpm_valid and voltage_valid and current_valid
            items.append(
                _RawEscTelemetry(
                    slot=first_slot + offset,
                    rpm=raw_rpm,
                    voltage_v=(
                        raw_voltage / 100.0
                        if math.isfinite(raw_voltage) and 0.0 <= raw_voltage < _ESC_RAW_UNKNOWN
                        else None
                    ),
                    current_a=(
                        raw_current / 100.0
                        if math.isfinite(raw_current) and 0.0 <= raw_current < _ESC_RAW_UNKNOWN
                        else None
                    ),
                    temperature_c=(
                        raw_temperature
                        if math.isfinite(raw_temperature) and 0.0 <= raw_temperature < 255.0
                        else None
                    ),
                    present=present,
                    healthy=healthy,
                    timestamp=timestamp,
                )
            )
        return tuple(items)

    @staticmethod
    def _raw_array_value(values: Tuple[Any, ...], offset: int) -> float:
        if offset >= len(values):
            return math.nan
        try:
            return float(values[offset])
        except (TypeError, ValueError, OverflowError):
            return math.nan

    def _build_logical_esc_status(
        self,
        now: float,
    ) -> Tuple[Tuple[EscTelemetry, ...], Tuple[int, ...], int]:
        raw_items: Dict[int, _RawEscTelemetry] = {}
        for group in self._esc_groups.values():
            age = now - group.timestamp
            if age < 0.0 or age > self._esc_timeout_s:
                continue
            for item in group.samples:
                raw_items[item.slot] = item

        raw_present_slots = tuple(sorted(slot for slot, item in raw_items.items() if item.present))
        display_shift = self._esc_mavlink_display_shift
        expected_raw_slots = {slot + display_shift for slot in self._esc_mapping}
        if any(slot not in expected_raw_slots for slot in raw_present_slots):
            return (
                tuple(
                    self._logical_esc_item(slot, position, None)
                    for slot, position in sorted(self._esc_mapping.items())
                ),
                raw_present_slots,
                display_shift,
            )

        return (
            tuple(
                self._logical_esc_item(
                    slot,
                    position,
                    raw_items.get(slot + display_shift),
                )
                for slot, position in sorted(self._esc_mapping.items())
            ),
            raw_present_slots,
            display_shift,
        )

    @staticmethod
    def _logical_esc_item(
        slot: int,
        position: str,
        raw: Optional[_RawEscTelemetry],
    ) -> EscTelemetry:
        if raw is None:
            return EscTelemetry(
                slot=slot,
                physical_position=position,
                rpm=math.nan,
                healthy=False,
            )
        return EscTelemetry(
            slot=slot,
            physical_position=position,
            rpm=raw.rpm if raw.healthy else math.nan,
            voltage_v=raw.voltage_v,
            current_a=raw.current_a,
            temperature_c=raw.temperature_c,
            healthy=raw.healthy,
            timestamp=raw.timestamp,
        )

    def _require_connection(self) -> Any:
        if not self._connected or self._connection is None:
            raise BridgeError("Pixhawk is not connected")
        return self._connection

    def _require_mavlink(self) -> Any:
        if self._mavlink is None:
            raise BridgeError("pymavlink is not initialized")
        return self._mavlink


__all__ = ["MavlinkPixhawkBridge"]
