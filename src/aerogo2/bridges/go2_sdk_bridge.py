"""Unitree SDK2 high-level Go2 bridge for aarch64 Ubuntu."""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
from dataclasses import replace
from typing import Any, Optional, Tuple

from aerogo2.bridges.go2_control_arbiter import ControlOwnershipError, Go2ControlArbiter
from aerogo2.common.async_utils import await_nonabandonable, run_blocking
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import Go2Config
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import Go2Status


def _value(obj: Any, name: str, default: Any) -> Any:
    raw = getattr(obj, name, default)
    return raw() if callable(raw) else raw


def _vector3(raw: Any) -> Tuple[Tuple[float, float, float], bool]:
    if isinstance(raw, (str, bytes)):
        return (0.0, 0.0, 0.0), False
    try:
        values = tuple(raw)
    except TypeError:
        return (0.0, 0.0, 0.0), False
    if len(values) != 3 or any(isinstance(item, bool) for item in values):
        return (0.0, 0.0, 0.0), False
    try:
        parsed = tuple(float(item) for item in values)
    except (TypeError, ValueError, OverflowError):
        return (0.0, 0.0, 0.0), False
    if any(not math.isfinite(item) for item in parsed):
        return (0.0, 0.0, 0.0), False
    return (parsed[0], parsed[1], parsed[2]), True


def _vector4_int(raw: Any) -> Tuple[Tuple[int, int, int, int], bool]:
    if isinstance(raw, (str, bytes)):
        return (0, 0, 0, 0), False
    try:
        source = tuple(raw)
    except TypeError:
        return (0, 0, 0, 0), False
    if len(source) != 4 or any(isinstance(item, bool) for item in source):
        return (0, 0, 0, 0), False
    try:
        values = tuple(float(item) for item in source)
    except (TypeError, ValueError, OverflowError):
        return (0, 0, 0, 0), False
    if any(not math.isfinite(item) or not item.is_integer() for item in values) or any(
        item < -32768 or item > 32767 for item in values
    ):
        return (0, 0, 0, 0), False
    parsed = tuple(int(item) for item in values)
    return (parsed[0], parsed[1], parsed[2], parsed[3]), True


class UnitreeGo2Bridge:
    """Subscribe to SportModeState and issue only conservative SportClient calls."""

    # SportModeState.error_code remains visible as raw telemetry. Firmware-specific
    # codes considered compatible with normal posture are explicitly configured;
    # they never imply JOINT_LOCK by themselves.
    # IDLE_STAND and JOINT_LOCK are the only observed modes that can be
    # considered quiescent without inferring authority from zero velocity.
    _QUIESCENT_MODE_CODES = frozenset((0, 6))

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
        control_arbiter: Optional[Go2ControlArbiter] = None,
    ) -> None:
        if allow_control and control_arbiter is None:
            raise ValueError(
                "Writable UnitreeGo2Bridge requires the shared Go2ControlArbiter; "
                "constructing a SportClient writer without it would bypass the "
                "host-wide Sport/LowCmd ownership boundary"
            )
        self._config = config
        self._clock = clock or RealClock()
        self._allow_control = allow_control
        self._control_arbiter = control_arbiter
        self._status = Go2Status()
        self._status_lock = threading.Lock()
        self._subscription_generation = 0
        self._first_state = threading.Event()
        self._subscriber: Optional[Any] = None
        self._client: Optional[Any] = None
        self._connected = False
        self._joystick_disabled = False
        self._disconnect_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lifecycle_lock:
            await self._connect_unlocked()

    async def _connect_unlocked(self) -> None:
        with self._status_lock:
            if self._connected:
                return
            self._subscription_generation += 1
            subscription_generation = self._subscription_generation
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
            if self._control_arbiter is None:
                channel.ChannelFactoryInitialize(
                    self._config.domain_id,
                    self._config.network_interface,
                )
            else:
                initialized = self._control_arbiter.initialize_channel_factory(
                    channel.ChannelFactoryInitialize,
                    self._config.domain_id,
                    self._config.network_interface,
                )
                if not initialized.ok:
                    raise BridgeError(f"{initialized.code}: {initialized.message}")
            message_factory = getattr(defaults, "unitree_go_msg_dds__SportModeState_", None)
            message_type = (
                type(message_factory()) if callable(message_factory) else dds.SportModeState_
            )
            subscriber = channel.ChannelSubscriber(
                self._config.sport_state_topic,
                message_type,
            )

            def receive_state(
                message: Any,
                generation: int = subscription_generation,
            ) -> None:
                self._on_state(message, subscription_generation=generation)

            with self._status_lock:
                self._subscriber = subscriber
            subscriber.Init(receive_state, 10)
            client = sport.SportClient()
            client.SetTimeout(self._config.command_timeout_s)
            client.Init()
            self._client = client
        except Exception as exc:
            self._abandon_connect_attempt(
                subscription_generation,
                locals().get("subscriber"),
            )
            raise BridgeError(f"Unitree SDK2 initialization failed: {exc}") from exc
        wait_task = asyncio.ensure_future(
            run_blocking(
                self._first_state.wait,
                self._config.status_timeout_s * 5.0,
            )
        )
        try:
            received = await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            self._abandon_connect_attempt(subscription_generation, subscriber)
            await await_nonabandonable(wait_task)
            raise
        except Exception as exc:
            self._abandon_connect_attempt(subscription_generation, subscriber)
            raise BridgeError(f"Waiting for Go2 SportModeState failed: {exc}") from exc
        if not received:
            self._abandon_connect_attempt(subscription_generation, subscriber)
            raise BridgeError(
                "No Go2 SportModeState received; check domain 0, network interface, and DDS"
            )
        with self._status_lock:
            subscription_replaced = self._subscription_generation != subscription_generation
            if not subscription_replaced:
                self._connected = True
                self._status = replace(self._status, connected=True)
        if subscription_replaced:
            self._abandon_connect_attempt(subscription_generation, subscriber)
            raise BridgeError("Go2 SportModeState subscription changed during connect")

    def _abandon_connect_attempt(
        self,
        subscription_generation: int,
        subscriber: Any,
    ) -> None:
        """Invalidate and close one reader without affecting a future generation."""

        with self._status_lock:
            if self._subscription_generation == subscription_generation:
                self._subscription_generation += 1
            if self._subscriber is subscriber:
                self._subscriber = None
                self._client = None
            self._connected = False
            self._status = replace(self._status, connected=False)
            self._first_state.set()
        close = getattr(subscriber, "Close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    async def disconnect(self) -> None:
        async with self._lifecycle_lock:
            await self._disconnect_unlocked()

    async def _disconnect_unlocked(self) -> None:
        with self._status_lock:
            self._subscription_generation += 1
            subscriber = self._subscriber
            client = self._client
        cancelled = False
        if self._allow_control and client is not None and self._joystick_disabled:
            enable_joystick = getattr(client, "SwitchJoystick", None)
            if callable(enable_joystick):
                try:
                    await self._run_sport_rpc(enable_joystick, True)
                except asyncio.CancelledError:
                    # _run_sport_rpc has already waited for the blocking SDK
                    # call and retained the host lease throughout. Finish the
                    # local teardown before honoring cancellation.
                    cancelled = True
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
        if cancelled:
            raise asyncio.CancelledError

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
        return bool(self.get_status().joints_locked)

    async def finalize_operator_joint_lock(self) -> bool:
        """Disable joystick input after a guarded, explicit operator confirmation.

        This does not claim that SportModeState.mode is 6. The manager records the
        operator confirmation separately from authoritative telemetry.
        """

        status = self.get_status()
        if not status.connected or status.moving:
            return False
        if not self._joystick_disabled:
            if not await self._call("SwitchJoystick", False):
                return False
            self._joystick_disabled = True
        return True

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
            result = await self._run_sport_rpc(operation, *args)
        except ControlOwnershipError as exc:
            raise BridgeError(f"Go2 {method} rejected by control arbiter: {exc}") from exc
        except Exception as exc:
            raise BridgeError(f"Go2 {method} failed: {exc}") from exc
        return int(result) == 0

    async def _run_sport_rpc(self, operation: Any, *args: Any) -> Any:
        """Keep the host-wide Sport lease until a blocking RPC really ends.

        Cancelling ``run_in_executor`` does not stop its worker.  Releasing the
        lease on coroutine cancellation would therefore admit LowCmd while the
        Sport RPC was still executing.  Cancellation is delayed until the
        worker has completed (or remains deliberately stuck with the lease).
        """

        def start_worker() -> asyncio.Task[Any]:
            return asyncio.create_task(run_blocking(operation, *args))

        async def await_nonabandonable(worker: asyncio.Task[Any]) -> Any:
            cancelled = False
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
            result = worker.result()
            if cancelled:
                raise asyncio.CancelledError
            return result

        arbiter = self._control_arbiter
        if arbiter is None:
            return await await_nonabandonable(start_worker())
        with arbiter.sport_lease():
            return await await_nonabandonable(start_worker())

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
        return bool(self.get_status().joints_locked)

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

    def _on_state(
        self,
        message: Any,
        *,
        subscription_generation: Optional[int] = None,
    ) -> None:
        with self._status_lock:
            generation = (
                self._subscription_generation
                if subscription_generation is None
                else subscription_generation
            )
            if generation != self._subscription_generation:
                return
        now = self._clock.monotonic()
        try:
            velocity, velocity_valid = _vector3(_value(message, "velocity", ()))
            imu = _value(message, "imu_state", None)
            rpy, rpy_valid = _vector3(_value(imu, "rpy", ()))
            foot_force, foot_force_valid = _vector4_int(_value(message, "foot_force", ()))
            raw_mode = _value(message, "mode", None)
            raw_state_code = _value(message, "error_code", None)
            if isinstance(raw_mode, bool) or not isinstance(raw_mode, int):
                raise ValueError("mode is not an integer")
            if isinstance(raw_state_code, bool) or not isinstance(raw_state_code, int):
                raise ValueError("error_code is not an integer")
            mode = raw_mode
            state_code = raw_state_code
        # DDS invokes this callback outside the asyncio command path.  No
        # malformed/generated accessor is allowed to escape and leave the
        # previous, potentially safe-looking status latched.
        except Exception:
            velocity = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            foot_force = (0, 0, 0, 0)
            foot_force_valid = False
            velocity_valid = False
            rpy_valid = False
            mode = -1
            state_code = -1
        speed = math.hypot(velocity[0], velocity[1])
        telemetry_valid = velocity_valid and rpy_valid
        posture_code_ok = state_code in self._config.accepted_state_codes
        # Go2 EDU firmware variants may keep mode=0 (IDLE_STAND) after the
        # phone app enters Lock On and expose the locked state through
        # error_code=1002 instead. The compatibility codes are explicit and
        # configurable; downstream FLIGHT_READY interlocks remain mandatory.
        joints_locked = posture_code_ok and (
            mode == 6 or state_code in self._config.joint_lock_state_codes
        )
        standing = mode in {0, 1, 2, 6} and posture_code_ok
        moving = not telemetry_valid or mode == 3 or speed >= 0.02 or abs(velocity[2]) >= 0.02
        stable = (
            telemetry_valid
            and standing
            and abs(rpy[0]) < 0.35
            and abs(rpy[1]) < 0.35
            and not moving
        )
        controller_active = (
            not telemetry_valid
            or not posture_code_ok
            or mode not in self._QUIESCENT_MODE_CODES
            or moving
        )
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
            controller_active=controller_active,
        )
        with self._status_lock:
            if generation != self._subscription_generation:
                return
            self._status = status
        self._first_state.set()


__all__ = ["UnitreeGo2Bridge"]
