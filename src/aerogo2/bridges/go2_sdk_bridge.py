"""Unitree SDK2 high-level Go2 bridge for aarch64 Ubuntu."""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
from dataclasses import replace
from typing import Any, Optional, Tuple

from aerogo2.common.async_utils import run_blocking
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import Go2Config
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import Go2Status


def _value(obj: Any, name: str, default: Any) -> Any:
    raw = getattr(obj, name, default)
    return raw() if callable(raw) else raw


def _vector3(raw: Any) -> Tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    if len(values) < 3 or any(not math.isfinite(item) for item in values[:3]):
        return (0.0, 0.0, 0.0)
    return (values[0], values[1], values[2])


def _vector4_int(raw: Any) -> Tuple[Tuple[int, int, int, int], bool]:
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return (0, 0, 0, 0), False
    if (
        len(values) < 4
        or any(not math.isfinite(item) or not item.is_integer() for item in values[:4])
        or any(item < -32768 or item > 32767 for item in values[:4])
    ):
        return (0, 0, 0, 0), False
    parsed = tuple(int(item) for item in values[:4])
    return (parsed[0], parsed[1], parsed[2], parsed[3]), True


class UnitreeGo2Bridge:
    """Subscribe to SportModeState and issue only conservative SportClient calls."""

    # Go2 firmware uses SportModeState.error_code as a posture/state-machine code.
    # Hardware verified: 100 is upright IDLE_STAND; 1001 is lying down.
    # Unknown codes remain fail-closed.
    _STABLE_STATE_CODES = frozenset((0, 100))

    _MODE_NAMES = {
        0: "IDLE_STAND",
        1: "BALANCE_STAND",
        2: "POSE",
        3: "LOCOMOTION",
        5: "LIE_DOWN",
        6: "JOINT_LOCK",
        7: "DAMPING",
        8: "RECOVERY_STAND",
        10: "SIT",
    }

    def __init__(
        self,
        config: Go2Config,
        *,
        clock: Optional[Clock] = None,
        allow_control: bool = False,
    ) -> None:
        self._config = config
        self._clock = clock or RealClock()
        self._allow_control = allow_control
        self._status = Go2Status()
        self._status_lock = threading.Lock()
        self._first_state = threading.Event()
        self._subscriber: Optional[Any] = None
        self._client: Optional[Any] = None
        self._connected = False
        self._joystick_disabled = False
        self._disconnect_event = asyncio.Event()

    async def connect(self) -> None:
        if self._connected:
            return
        self._disconnect_event.clear()
        self._first_state.clear()
        try:
            channel = importlib.import_module("unitree_sdk2py.core.channel")
            defaults = importlib.import_module("unitree_sdk2py.idl.default")
            dds = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")
            sport = importlib.import_module("unitree_sdk2py.go2.sport.sport_client")
        except ImportError as exc:
            raise BridgeError(
                "unitree_sdk2py is not installed; run deploy/install_aarch64.sh"
            ) from exc
        try:
            channel.ChannelFactoryInitialize(
                self._config.domain_id,
                self._config.network_interface,
            )
            message_factory = getattr(defaults, "unitree_go_msg_dds__SportModeState_", None)
            message_type = (
                type(message_factory()) if callable(message_factory) else dds.SportModeState_
            )
            self._subscriber = channel.ChannelSubscriber(
                self._config.sport_state_topic,
                message_type,
            )
            self._subscriber.Init(self._on_state, 10)
            self._client = sport.SportClient()
            self._client.SetTimeout(self._config.command_timeout_s)
            self._client.Init()
        except Exception as exc:
            await self.disconnect()
            raise BridgeError(f"Unitree SDK2 initialization failed: {exc}") from exc
        received = await run_blocking(
            self._first_state.wait,
            self._config.status_timeout_s * 5.0,
        )
        if not received:
            await self.disconnect()
            raise BridgeError(
                "No Go2 SportModeState received; check domain 0, network interface, and DDS"
            )
        self._connected = True
        with self._status_lock:
            self._status = replace(self._status, connected=True)

    async def disconnect(self) -> None:
        subscriber = self._subscriber
        client = self._client
        if self._allow_control and client is not None and self._joystick_disabled:
            enable_joystick = getattr(client, "SwitchJoystick", None)
            if callable(enable_joystick):
                try:
                    await run_blocking(enable_joystick, True)
                except Exception:
                    pass
        self._joystick_disabled = False
        self._subscriber = None
        self._client = None
        self._connected = False
        self._disconnect_event.set()
        if subscriber is not None:
            close = getattr(subscriber, "Close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        with self._status_lock:
            self._status = replace(self._status, connected=False)

    async def run(self) -> None:
        await self._disconnect_event.wait()

    def get_status(self) -> Go2Status:
        with self._status_lock:
            return replace(self._status, connected=self._connected)

    def latest_status(self) -> Go2Status:
        return self.get_status()

    async def request_stop(self) -> bool:
        if self.get_status().joints_locked:
            return True
        return await self._call("StopMove")

    async def request_stand(self) -> bool:
        if self._joystick_disabled:
            if not await self._call("SwitchJoystick", True):
                return False
            self._joystick_disabled = False
        return await self._call("BalanceStand")

    async def request_flight_pose(self) -> bool:
        already_locked = self.get_status().joints_locked
        if not already_locked:
            # SportClient has no authoritative JointLock command. StandUp enters
            # an ordinary standing posture on Go2 and must not be treated as
            # mode=6. Keep joystick/app control available so the operator can
            # select Joint Lock in the Unitree phone app.
            await self._call("StopMove")
            return False
        if not self._joystick_disabled:
            if not await self._call("SwitchJoystick", False):
                return False
            self._joystick_disabled = True
        return self.get_status().joints_locked

    async def request_landing_pose(self) -> bool:
        if not self._joystick_disabled:
            if not await self._call("SwitchJoystick", False):
                return False
            self._joystick_disabled = True
        if not await self._call("BalanceStand"):
            return False
        return await self._wait_for_locomotion_mode("BALANCE_STAND")

    async def _call(self, method: str, *args: Any) -> bool:
        if not self._allow_control:
            raise BridgeError(
                "Go2 high-level control is locked; restart with the per-process hardware unlock"
            )
        client = self._client
        if not self._connected or client is None:
            raise BridgeError("Go2 is not connected")
        operation = getattr(client, method, None)
        if not callable(operation):
            raise BridgeError(f"Unitree SportClient does not provide {method}")
        try:
            result = await run_blocking(operation, *args)
        except Exception as exc:
            raise BridgeError(f"Go2 {method} failed: {exc}") from exc
        return int(result) == 0

    async def _wait_for_joint_lock(self) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(
            self._config.command_timeout_s,
            self._config.status_timeout_s * 5.0,
        )
        while loop.time() < deadline:
            if self.get_status().joints_locked:
                return True
            await asyncio.sleep(min(0.02, self._config.status_timeout_s))
        return self.get_status().joints_locked

    async def _wait_for_locomotion_mode(self, expected: str) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(
            self._config.command_timeout_s,
            self._config.status_timeout_s * 5.0,
        )
        while loop.time() < deadline:
            status = self.get_status()
            if (
                status.locomotion_mode == expected
                and status.standing
                and status.stable
                and not status.moving
            ):
                return True
            await asyncio.sleep(min(0.02, self._config.status_timeout_s))
        status = self.get_status()
        return (
            status.locomotion_mode == expected
            and status.standing
            and status.stable
            and not status.moving
        )

    def _on_state(self, message: Any) -> None:
        now = self._clock.monotonic()
        velocity = _vector3(_value(message, "velocity", (0.0, 0.0, 0.0)))
        imu = _value(message, "imu_state", None)
        rpy = _vector3(_value(imu, "rpy", (0.0, 0.0, 0.0)))
        foot_force, foot_force_valid = _vector4_int(_value(message, "foot_force", ()))
        mode = int(_value(message, "mode", -1))
        state_code = int(_value(message, "error_code", 0))
        speed = math.hypot(velocity[0], velocity[1])
        posture_code_ok = state_code in self._STABLE_STATE_CODES
        joints_locked = mode == 6 and posture_code_ok
        standing = mode in {0, 1, 2, 6} and posture_code_ok
        moving = mode == 3 or speed >= 0.02 or abs(velocity[2]) >= 0.02
        stable = standing and abs(rpy[0]) < 0.35 and abs(rpy[1]) < 0.35 and not moving
        status = Go2Status(
            timestamp=now,
            connected=True,
            body_velocity=velocity,
            body_rpy=rpy,
            standing=standing,
            moving=moving,
            locomotion_mode=self._MODE_NAMES.get(mode, f"MODE_{mode}"),
            fault_code=state_code,
            joints_locked=joints_locked,
            foot_force=foot_force,
            foot_force_valid=foot_force_valid,
            velocity_mps=speed,
            stable=stable,
            controller_active=moving,
        )
        with self._status_lock:
            self._status = status
        self._first_state.set()


__all__ = ["UnitreeGo2Bridge"]
