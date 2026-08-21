"""Async serial bridge for the F446 morphology controller.

Only current-limited F446 motion is exposed through the production interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, List, Optional, Tuple

from aerogo2.bridges.f446_parser import F446TextParser
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import F446Config
from aerogo2.common.enums import Configuration, F446State
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import F446Event, F446Status
from aerogo2.common.results import OperationResult

# The deployed F446 firmware polls USART2 once per roughly 2 ms main-loop
# iteration. Sending a complete command at USB-VCP speed overruns that polling
# receiver and leaves only a prefix in the firmware command buffer. Keep a
# conservative gap between bytes; this applies to every command, including the
# guarded motion commands.
_TX_INTER_BYTE_DELAY_S = 0.010
_INITIAL_STATUS_ATTEMPTS = 3
_INITIAL_STATUS_RETRY_DELAY_S = 0.100
_MAX_MAINTENANCE_DUTY = 900


class TextF446Bridge:
    """Own the USB serial transport and verified morphology transactions."""

    def __init__(
        self,
        config: Optional[F446Config] = None,
        clock: Optional[Clock] = None,
        *,
        allow_motion: bool = False,
    ) -> None:
        self._config = config
        self._clock = clock or RealClock()
        self._parser = F446TextParser(clock=self._clock)
        self._allow_motion = allow_motion
        self._reader: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._connected = False
        self._background_error: Optional[str] = None
        self._write_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()
        self._status_changed = asyncio.Condition()
        self._event_history: List[F446Event] = []

    @property
    def parser(self) -> F446TextParser:
        return self._parser

    @property
    def event_history(self) -> Tuple[F446Event, ...]:
        return tuple(self._event_history)

    async def connect(self) -> None:
        if self._connected:
            return
        config = self._require_config()
        try:
            import serial_asyncio  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BridgeError("pyserial-asyncio is required for the F446 hardware bridge") from exc
        self._parser.reset()
        self._background_error = None
        try:
            reader, writer = await asyncio.wait_for(
                serial_asyncio.open_serial_connection(url=config.port, baudrate=config.baud),
                timeout=config.response_timeout_s,
            )
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            raise BridgeError(f"Cannot open F446 serial port {config.port}: {exc}") from exc
        self._reader = reader
        self._writer = writer
        self._connected = True
        self._reader_task = asyncio.create_task(self._reader_loop(), name="f446-reader")
        try:
            status = await self._request_initial_status()
            if status.state is F446State.UNKNOWN:
                raise BridgeError("F446 returned an unknown initial state")
        except Exception:
            await self.disconnect()
            raise
        self._poll_task = asyncio.create_task(self._poll_loop(), name="f446-status-poll")

    async def disconnect(self) -> None:
        status = self.get_status()
        if self._connected and self._allow_motion and self._is_moving(status):
            try:
                await self.stop()
            except (BridgeError, OSError, RuntimeError):
                pass
        current = asyncio.current_task()
        for task in (self._poll_task, self._reader_task):
            if task is not None and task is not current:
                task.cancel()
        for task in (self._poll_task, self._reader_task):
            if task is not None and task is not current:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        writer = self._writer
        self._reader = None
        self._writer = None
        self._reader_task = None
        self._poll_task = None
        self._connected = False
        if writer is not None:
            writer.close()
            wait_closed = getattr(writer, "wait_closed", None)
            if callable(wait_closed):
                try:
                    await wait_closed()
                except (OSError, RuntimeError):
                    pass

    async def run(self) -> None:
        task = self._reader_task
        if task is None:
            raise BridgeError("F446 is not connected")
        await task

    def get_status(self) -> F446Status:
        return replace(self._parser.latest_status, connected=self._connected)

    def latest_status(self) -> F446Status:
        return self.get_status()

    async def request_status(self) -> F446Status:
        return await self._query("status", require_current=False)

    async def request_current(self) -> F446Status:
        return await self._query("is", require_current=True)

    async def move_to_configuration(self, configuration: Configuration) -> OperationResult:
        if not self._allow_motion:
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 motion is locked; restart with the explicit per-process hardware unlock",
            )
        if configuration not in (Configuration.FLIGHT, Configuration.WALK):
            return OperationResult.failure(
                "F446_INVALID_CONFIGURATION", f"Unsupported F446 target {configuration.value}"
            )
        config = self._require_config()
        async with self._transaction_lock:
            initial = await self.request_status()
            expected = config.expected_state_for(configuration.value)
            if initial.state is expected and initial.duty == 0:
                return OperationResult.success(
                    f"F446 already confirms {configuration.value}", code="F446_ALREADY_AT_TARGET"
                )
            if initial.faulted:
                return OperationResult.failure(
                    "F446_FAULT", initial.fault_message or "F446 reports FAULT"
                )
            direction = config.direction_for(configuration.value)
            duty = config.duty_for(configuration.value)
            command = "limf" if direction == "forward" else "limr"
            await self._write_line(f"{command} {duty}")
            deadline = self._clock.monotonic() + config.transform_timeout_s
            while self._clock.monotonic() < deadline:
                status = self.get_status()
                if not status.connected:
                    return OperationResult.failure(
                        "F446_DISCONNECTED",
                        self._background_error or "F446 disconnected during transformation",
                    )
                if status.faulted:
                    return OperationResult.failure(
                        "F446_FAULT", status.fault_message or "F446 reports FAULT"
                    )
                if status.state is expected:
                    final = await self.request_status()
                    if final.state is not expected:
                        return OperationResult.failure(
                            "F446_FINAL_STATE_MISMATCH",
                            f"Expected {expected.value}, received {final.state.value}",
                        )
                    if final.duty != 0:
                        await self._force_stop()
                        return OperationResult.failure(
                            "F446_FINAL_DUTY_NONZERO",
                            f"F446 limit was reached but duty remained {final.duty}",
                        )
                    return OperationResult.success(
                        f"F446 verified {configuration.value} limit with zero duty",
                        code="F446_CONFIGURATION_VERIFIED",
                    )
                remaining = max(0.0, deadline - self._clock.monotonic())
                try:
                    async with self._status_changed:
                        await asyncio.wait_for(
                            self._status_changed.wait(),
                            timeout=min(remaining, config.response_timeout_s),
                        )
                except asyncio.TimeoutError:
                    await self._write_line("status")
            await self._force_stop()
            return OperationResult.failure(
                "F446_TRANSFORM_TIMEOUT",
                f"F446 did not reach {expected.value} within {config.transform_timeout_s:.2f}s",
            )

    async def start_maintenance_motion(
        self,
        operation: str,
        duty: int,
    ) -> OperationResult:
        """Start one explicitly typed maintenance move and verify acceptance."""

        normalized = operation.strip().lower()
        expected_by_operation = {
            "mf": F446State.MANUAL_FWD,
            "mr": F446State.MANUAL_REV,
            "limf": F446State.LIMIT_FWD,
            "limr": F446State.LIMIT_REV,
        }
        expected = expected_by_operation.get(normalized)
        if expected is None:
            return OperationResult.failure(
                "F446_INVALID_MAINTENANCE_OPERATION",
                f"Unsupported F446 maintenance operation '{operation}'",
            )
        if isinstance(duty, bool) or not 1 <= duty <= _MAX_MAINTENANCE_DUTY:
            return OperationResult.failure(
                "F446_INVALID_DUTY",
                f"Maintenance duty must be 1..{_MAX_MAINTENANCE_DUTY}",
            )
        if not self._allow_motion:
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 motion is locked; restart with the explicit per-process hardware unlock",
            )

        expected_duty = duty if normalized.endswith("f") else -duty
        async with self._transaction_lock:
            initial = await self.request_status()
            if initial.faulted:
                return OperationResult.failure(
                    "F446_FAULT",
                    initial.fault_message or "F446 reports FAULT",
                )
            if self._is_moving(initial) or initial.duty != 0:
                return OperationResult.failure(
                    "F446_ALREADY_MOVING",
                    f"Stop F446 before a new command: state={initial.state.value}, duty={initial.duty}",
                )
            if initial.manual_limit < duty:
                await self._write_line(f"mlimit {duty}")
                configured = await self.request_status()
                if configured.manual_limit < duty:
                    return OperationResult.failure(
                        "F446_MANUAL_LIMIT_REJECTED",
                        f"F446 manual_limit remained {configured.manual_limit}, requested {duty}",
                    )

            await self._write_line(f"{normalized} {duty}")
            observed = await self.request_status()
            reached_state = (
                F446State.LIMIT_REACHED_FWD if normalized == "limf" else F446State.LIMIT_REACHED_REV
            )
            if (
                normalized.startswith("lim")
                and observed.state is reached_state
                and observed.duty == 0
            ):
                return OperationResult.success(
                    f"F446 {normalized} reached its current limit and stopped",
                    code="F446_LIMIT_REACHED",
                    data={
                        "operation": normalized,
                        "duty": duty,
                        "state": observed.state.value,
                        "automatic_limit_stop": True,
                    },
                )
            if observed.state is not expected or observed.duty != expected_duty:
                stop_result = await self._force_stop()
                stop_detail = "" if stop_result.ok else f"; stop failed: {stop_result.message}"
                return OperationResult.failure(
                    "F446_MOTION_NOT_ACCEPTED",
                    (
                        f"Expected state={expected.value}, duty={expected_duty}; received "
                        f"state={observed.state.value}, duty={observed.duty}{stop_detail}"
                    ),
                )
            return OperationResult.success(
                f"F446 {normalized} started at duty {duty}",
                code="F446_MAINTENANCE_MOTION_STARTED",
                data={
                    "operation": normalized,
                    "duty": duty,
                    "signed_duty": expected_duty,
                    "state": observed.state.value,
                    "automatic_limit_stop": normalized.startswith("lim"),
                },
            )

    async def set_current_threshold_adc(self, threshold_adc: int) -> OperationResult:
        return await self._set_current_threshold(
            command="thr",
            requested=threshold_adc,
            maximum=4095,
            observed_field="threshold_adc",
            tolerance=0,
        )

    async def set_current_threshold_mv(self, threshold_mv: int) -> OperationResult:
        return await self._set_current_threshold(
            command="thrmv",
            requested=threshold_mv,
            maximum=3300,
            observed_field="threshold_mv",
            tolerance=1,
        )

    async def _set_current_threshold(
        self,
        *,
        command: str,
        requested: int,
        maximum: int,
        observed_field: str,
        tolerance: int,
    ) -> OperationResult:
        if isinstance(requested, bool) or not 0 <= requested <= maximum:
            return OperationResult.failure(
                "F446_INVALID_THRESHOLD",
                f"{command} must be 0..{maximum}",
            )
        if not self._allow_motion:
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 parameter writes are locked in this process",
            )
        async with self._transaction_lock:
            initial = await self.request_status()
            if initial.faulted:
                return OperationResult.failure(
                    "F446_FAULT",
                    initial.fault_message or "F446 reports FAULT",
                )
            if self._is_moving(initial) or initial.duty != 0:
                return OperationResult.failure(
                    "F446_PARAMETER_WRITE_WHILE_MOVING",
                    "Stop F446 before changing the current threshold",
                )
            await self._write_line(f"{command} {requested}")
            observed = await self.request_status()
            value = getattr(observed, observed_field)
            if value is None or abs(int(value) - requested) > tolerance:
                return OperationResult.failure(
                    "F446_THRESHOLD_VERIFY_FAILED",
                    f"Requested {command}={requested}, read back {value}",
                )
            return OperationResult.success(
                f"F446 current threshold verified: {value}",
                code="F446_THRESHOLD_UPDATED",
                data={
                    "threshold_adc": observed.threshold_adc,
                    "threshold_mv": observed.threshold_mv,
                },
            )

    async def stop(self) -> OperationResult:
        status = self.get_status()
        if not self._allow_motion:
            return OperationResult.success(
                "F446 hardware writes are locked and no motion was started by this process",
                code="F446_WRITE_LOCKED",
            )
        if not self._connected:
            return OperationResult.failure("F446_DISCONNECTED", "F446 is not connected")
        # Preserve LIMIT_REACHED_*: it is the current firmware's authoritative
        # indication of the physical morphology position.
        if not self._is_moving(status) and status.duty == 0:
            return OperationResult.success("F446 is already stopped", code="F446_ALREADY_STOPPED")
        return await self._force_stop()

    def feed_offline_data(self, data: bytes) -> List[F446Event]:
        return self._parser.feed(data)

    async def _query(self, command: str, *, require_current: bool) -> F446Status:
        self._raise_if_unavailable()
        config = self._require_config()
        before = self._parser.latest_status.timestamp
        await self._write_line(command)
        deadline = self._clock.monotonic() + config.response_timeout_s
        while self._clock.monotonic() < deadline:
            status = self.get_status()
            has_current = status.used_current_adc is not None and status.threshold_adc is not None
            if status.timestamp > before and (has_current or not require_current):
                return status
            remaining = max(0.0, deadline - self._clock.monotonic())
            try:
                async with self._status_changed:
                    await asyncio.wait_for(self._status_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
        raise BridgeError(
            f"F446 did not answer '{command}' within {config.response_timeout_s:.2f}s"
        )

    async def _force_stop(self) -> OperationResult:
        try:
            await self._write_line("stop")
            final = await self.request_status()
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            return OperationResult.failure("F446_STOP_FAILED", str(exc))
        if final.duty != 0 or self._is_moving(final):
            return OperationResult.failure(
                "F446_STOP_FAILED",
                f"F446 stop verification failed: state={final.state.value}, duty={final.duty}",
            )
        return OperationResult.success("F446 motion stopped and duty verified zero")

    async def _write_line(self, command: str) -> None:
        self._raise_if_unavailable()
        writer = self._writer
        if writer is None:
            raise BridgeError("F446 serial writer is unavailable")
        payload = (command.strip() + "\r").encode("ascii")
        async with self._write_lock:
            drain = getattr(writer, "drain", None)
            for offset, value in enumerate(payload):
                writer.write(bytes((value,)))
                if callable(drain):
                    await drain()
                if offset + 1 < len(payload):
                    await asyncio.sleep(_TX_INTER_BYTE_DELAY_S)

    async def _request_initial_status(self) -> F446Status:
        last_error: Optional[BridgeError] = None
        for attempt in range(_INITIAL_STATUS_ATTEMPTS):
            try:
                return await self.request_status()
            except BridgeError as exc:
                last_error = exc
                if attempt + 1 < _INITIAL_STATUS_ATTEMPTS:
                    await asyncio.sleep(_INITIAL_STATUS_RETRY_DELAY_S)
        assert last_error is not None
        raise last_error

    async def _reader_loop(self) -> None:
        reader = self._reader
        if reader is None:
            return
        try:
            while self._connected:
                data = await reader.read(512)
                if not data:
                    raise BridgeError("F446 serial stream reached EOF")
                events = self._parser.feed(bytes(data))
                if events:
                    self._event_history.extend(events)
                    del self._event_history[:-256]
                    async with self._status_changed:
                        self._status_changed.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._background_error = f"{type(exc).__name__}: {exc}"
            self._connected = False
            async with self._status_changed:
                self._status_changed.notify_all()

    async def _poll_loop(self) -> None:
        config = self._require_config()
        interval = 1.0 / config.status_poll_hz
        try:
            while self._connected:
                await asyncio.sleep(interval)
                await self._write_line("status")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._background_error = f"{type(exc).__name__}: {exc}"
            self._connected = False

    def _raise_if_unavailable(self) -> None:
        if not self._connected:
            raise BridgeError(self._background_error or "F446 is not connected")

    def _require_config(self) -> F446Config:
        if self._config is None:
            raise BridgeError("F446 configuration was not provided")
        return self._config

    @staticmethod
    def _is_moving(status: F446Status) -> bool:
        return (
            status.state
            in (
                F446State.MANUAL_FWD,
                F446State.MANUAL_REV,
                F446State.LIMIT_FWD,
                F446State.LIMIT_REV,
            )
            or status.duty != 0
        )


F446TextBridge = TextF446Bridge

__all__ = ["F446TextBridge", "TextF446Bridge"]
