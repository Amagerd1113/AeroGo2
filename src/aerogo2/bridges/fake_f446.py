"""Deterministic F446 morphology-controller simulator for Phase 1."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import List, Optional, Tuple, Union

from aerogo2.bridges.f446_interface import F446CommandRecord
from aerogo2.common.clock import Clock, ManualClock, RealClock
from aerogo2.common.config import F446Config
from aerogo2.common.enums import Configuration, F446EventType, F446State
from aerogo2.common.models import F446Event, F446Status
from aerogo2.common.results import OperationResult


def _raw_to_mv(raw: int) -> int:
    """Mirror motor_main.c raw_to_mv integer arithmetic exactly."""

    return raw * 3300 // 4095


class FakeF446:
    """A high-level fake that models guarded limit-based transformations."""

    def __init__(
        self,
        config: Optional[F446Config] = None,
        clock: Optional[Clock] = None,
        transform_delay_s: float = 0.0,
        initial_configuration: Configuration = Configuration.WALK,
    ) -> None:
        if not math.isfinite(transform_delay_s) or transform_delay_s < 0:
            raise ValueError("transform_delay_s must be finite and non-negative")
        if initial_configuration not in {
            Configuration.UNKNOWN,
            Configuration.WALK,
            Configuration.FLIGHT,
        }:
            raise ValueError("FakeF446 initial configuration must be UNKNOWN, WALK, or FLIGHT")
        self._config = config
        self._clock = clock or RealClock()
        self._transform_delay_s = transform_delay_s
        initial_state = (
            F446State.IDLE
            if initial_configuration is Configuration.UNKNOWN
            else self._expected_state(initial_configuration)
        )
        self._status = F446Status(
            state=initial_state,
            raw_state=initial_state.value,
            configuration=initial_configuration.value,
            manual_limit=350,
            sense_mode="max",
            r_is_adc=0,
            l_is_adc=0,
            used_current_adc=0,
            threshold_adc=1800,
            threshold_raw=1800,
            threshold_mv=_raw_to_mv(1800),
            blanking_ms=500,
            overcurrent_ms=180,
            timeout_ms=5000,
        )
        self._configuration = initial_configuration
        self._commands: List[F446CommandRecord] = []
        self._events: List[F446Event] = []
        self._operation_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._next_timeout = False
        self._next_fault_message: Optional[str] = None
        self._next_fault_code = "F446_FAULT"
        self._next_wrong_final_state = False
        self._next_nonzero_final_duty = False

    async def connect(self) -> None:
        self._shutdown.clear()
        self._status = replace(
            self._status,
            connected=True,
            timestamp=self._clock.monotonic(),
        )

    async def disconnect(self) -> None:
        if self._status.duty != 0:
            self._stop_without_lock("disconnect")
        self._status = replace(
            self._status,
            connected=False,
            timestamp=self._clock.monotonic(),
        )
        self._shutdown.set()

    async def run(self) -> None:
        await self._shutdown.wait()

    def get_status(self) -> F446Status:
        return self._status

    def latest_status(self) -> F446Status:
        return self.get_status()

    @property
    def configuration(self) -> Configuration:
        return self._configuration

    @property
    def command_history(self) -> Tuple[F446CommandRecord, ...]:
        return tuple(self._commands)

    @property
    def event_history(self) -> Tuple[F446Event, ...]:
        return tuple(self._events)

    async def request_status(self) -> F446Status:
        self._record_command("status")
        return self._status

    async def request_current(self) -> F446Status:
        self._record_command("is")
        return self._status

    async def move_to_configuration(
        self,
        configuration: Union[Configuration, str],
    ) -> OperationResult:
        target = self._coerce_configuration(configuration)
        if target is None:
            return OperationResult.failure(
                "INVALID_CONFIGURATION",
                "F446 target must be WALK or FLIGHT",
            )

        async with self._operation_lock:
            if not self._status.connected:
                return OperationResult.failure(
                    "F446_DISCONNECTED",
                    "Cannot transform while FakeF446 is disconnected",
                )
            self._record_command("status")
            if self._status.faulted:
                return OperationResult.failure(
                    "F446_FAULT",
                    self._status.fault_message or "F446 is in FAULT",
                )
            expected = self._expected_state(target)
            if (
                self._configuration is target
                and self._status.state is expected
                and self._status.duty == 0
            ):
                return OperationResult.success(
                    f"F446 is already in {target.value} configuration",
                    data={"configuration": target.value, "state": expected.value},
                )

            direction = self._direction(target)
            duty = self._duty(target)
            command = "{} {}".format("limf" if direction == "forward" else "limr", duty)
            self._record_command(command)
            moving_state = F446State.LIMIT_FWD if direction == "forward" else F446State.LIMIT_REV
            self._configuration = Configuration.UNKNOWN
            self._status = replace(
                self._status,
                state=moving_state,
                raw_state=moving_state.value,
                configuration="UNKNOWN",
                duty=duty,
                timestamp=self._clock.monotonic(),
            )

            deadline = self._clock.monotonic() + self._transform_timeout_s()
            try:
                completed_before_deadline = await self._wait_transform_delay(deadline)
            except asyncio.CancelledError:
                self._stop_without_lock("cancelled transform")
                raise
            if not completed_before_deadline:
                self._next_timeout = False
                self._stop_without_lock("transform timeout")
                return OperationResult.failure(
                    "F446_TRANSFORM_TIMEOUT",
                    "Transform completion exceeded the configured deadline",
                )

            if self._status.faulted:
                self._record_command("stop")
                return OperationResult.failure(
                    "F446_FAULT",
                    self._status.fault_message or "F446 faulted during transform",
                )

            if self._next_fault_message is not None:
                message = self._next_fault_message
                code = self._next_fault_code
                self._next_fault_message = None
                self._next_fault_code = "F446_FAULT"
                if code == "F446_OVERCURRENT":
                    current = max(1800, self._status.threshold_adc or 0)
                    self._status = replace(
                        self._status,
                        used_current_adc=current,
                        used_raw=current,
                        over_active=True,
                    )
                self._set_fault(message)
                return OperationResult.failure(code, message)

            if self._next_timeout:
                self._next_timeout = False
                try:
                    await self._wait_until(deadline)
                except asyncio.CancelledError:
                    self._stop_without_lock("cancelled transform")
                    raise
                self._stop_without_lock("transform timeout")
                return OperationResult.failure(
                    "F446_TRANSFORM_TIMEOUT",
                    "No matching limit event was received before the transform timeout",
                )

            observed_state = expected
            if self._next_wrong_final_state:
                self._next_wrong_final_state = False
                observed_state = (
                    F446State.LIMIT_REACHED_REV
                    if expected is F446State.LIMIT_REACHED_FWD
                    else F446State.LIMIT_REACHED_FWD
                )
            observed_duty = duty if self._next_nonzero_final_duty else 0
            self._next_nonzero_final_duty = False
            threshold_raw = self._status.threshold_adc or 1800
            limit_used_raw = min(4095, threshold_raw + 100)
            sense = self._status.sense_mode
            direction_label = "FORWARD" if direction == "forward" else "REVERSE"
            limit_line = (
                f"{direction_label} LIMIT REACHED: IS_{sense}={limit_used_raw} "
                f"({_raw_to_mv(limit_used_raw)}mV), threshold={threshold_raw} "
                f"({_raw_to_mv(threshold_raw)}mV)"
            )
            now = self._clock.monotonic()
            self._events.append(
                F446Event(
                    (
                        F446EventType.FORWARD_LIMIT_REACHED
                        if direction == "forward"
                        else F446EventType.REVERSE_LIMIT_REACHED
                    ),
                    limit_line,
                    now,
                    state=expected,
                    values={
                        "direction": direction,
                        "sense": sense,
                        "used_raw": limit_used_raw,
                        "used_mv": _raw_to_mv(limit_used_raw),
                        "threshold_raw": threshold_raw,
                        "threshold_mv": _raw_to_mv(threshold_raw),
                    },
                )
            )
            self._status = replace(
                self._status,
                state=observed_state,
                raw_state=observed_state.value,
                configuration=(
                    target.value if observed_state is expected and observed_duty == 0 else "UNKNOWN"
                ),
                duty=observed_duty,
                timestamp=now,
                raw_lines=(limit_line,),
            )
            self._record_command("status")

            if observed_state is not expected:
                observed = observed_state.value
                self._stop_without_lock("unexpected final state")
                return OperationResult.failure(
                    "F446_FINAL_STATE_MISMATCH",
                    f"Expected {expected.value}, observed {observed}",
                )
            if observed_duty != 0:
                self._stop_without_lock("nonzero final duty")
                return OperationResult.failure(
                    "F446_FINAL_DUTY_NONZERO",
                    f"Limit event was followed by nonzero duty {observed_duty}",
                )
            if self._status.faulted:
                return OperationResult.failure(
                    "F446_FAULT",
                    self._status.fault_message or "F446 faulted after limit event",
                )

            self._configuration = target
            return OperationResult.success(
                f"F446 {target.value} configuration verified",
                data={
                    "configuration": target.value,
                    "state": expected.value,
                    "duty": 0,
                    "command": command,
                },
            )

    async def start_maintenance_motion(
        self,
        operation: str,
        duty: int,
    ) -> OperationResult:
        normalized = operation.strip().lower()
        state_by_operation = {
            "mf": F446State.MANUAL_FWD,
            "mr": F446State.MANUAL_REV,
            "limf": F446State.LIMIT_FWD,
            "limr": F446State.LIMIT_REV,
        }
        moving_state = state_by_operation.get(normalized)
        if moving_state is None:
            return OperationResult.failure(
                "F446_INVALID_MAINTENANCE_OPERATION",
                f"Unsupported F446 maintenance operation '{operation}'",
            )
        if isinstance(duty, bool) or not 1 <= duty <= 900:
            return OperationResult.failure(
                "F446_INVALID_DUTY",
                "Maintenance duty must be 1..900",
            )
        signed_duty = duty if normalized.endswith("f") else -duty
        async with self._operation_lock:
            if not self._status.connected:
                return OperationResult.failure(
                    "F446_DISCONNECTED",
                    "Cannot move while FakeF446 is disconnected",
                )
            if self._status.faulted:
                return OperationResult.failure(
                    "F446_FAULT",
                    self._status.fault_message or "F446 is in FAULT",
                )
            if self._status.duty != 0:
                return OperationResult.failure(
                    "F446_ALREADY_MOVING",
                    "Stop FakeF446 before a new maintenance command",
                )
            if self._status.manual_limit < duty:
                self._record_command(f"mlimit {duty}")
                self._status = replace(self._status, manual_limit=duty)
            self._record_command(f"{normalized} {duty}")
            self._configuration = Configuration.UNKNOWN
            self._status = replace(
                self._status,
                state=moving_state,
                raw_state=moving_state.value,
                configuration=Configuration.UNKNOWN.value,
                duty=signed_duty,
                timestamp=self._clock.monotonic(),
            )
            return OperationResult.success(
                f"FakeF446 {normalized} started at duty {duty}",
                code="F446_MAINTENANCE_MOTION_STARTED",
                data={
                    "operation": normalized,
                    "duty": duty,
                    "signed_duty": signed_duty,
                    "state": moving_state.value,
                    "automatic_limit_stop": normalized.startswith("lim"),
                },
            )

    async def set_current_threshold_adc(self, threshold_adc: int) -> OperationResult:
        if isinstance(threshold_adc, bool) or not 0 <= threshold_adc <= 4095:
            return OperationResult.failure(
                "F446_INVALID_THRESHOLD",
                "thr must be 0..4095",
            )
        async with self._operation_lock:
            if self._status.duty != 0:
                return OperationResult.failure(
                    "F446_PARAMETER_WRITE_WHILE_MOVING",
                    "Stop FakeF446 before changing the current threshold",
                )
            self._record_command(f"thr {threshold_adc}")
            self._status = replace(
                self._status,
                threshold_adc=threshold_adc,
                threshold_raw=threshold_adc,
                threshold_mv=_raw_to_mv(threshold_adc),
                timestamp=self._clock.monotonic(),
            )
            return OperationResult.success(
                f"FakeF446 current threshold verified: {threshold_adc}",
                code="F446_THRESHOLD_UPDATED",
                data={
                    "threshold_adc": threshold_adc,
                    "threshold_mv": _raw_to_mv(threshold_adc),
                },
            )

    async def set_current_threshold_mv(self, threshold_mv: int) -> OperationResult:
        if isinstance(threshold_mv, bool) or not 0 <= threshold_mv <= 3300:
            return OperationResult.failure(
                "F446_INVALID_THRESHOLD",
                "thrmv must be 0..3300",
            )
        threshold_adc = threshold_mv * 4095 // 3300
        async with self._operation_lock:
            if self._status.duty != 0:
                return OperationResult.failure(
                    "F446_PARAMETER_WRITE_WHILE_MOVING",
                    "Stop FakeF446 before changing the current threshold",
                )
            self._record_command(f"thrmv {threshold_mv}")
            self._status = replace(
                self._status,
                threshold_adc=threshold_adc,
                threshold_raw=threshold_adc,
                threshold_mv=_raw_to_mv(threshold_adc),
                timestamp=self._clock.monotonic(),
            )
            return OperationResult.success(
                f"FakeF446 current threshold verified: {_raw_to_mv(threshold_adc)}mV",
                code="F446_THRESHOLD_UPDATED",
                data={
                    "threshold_adc": threshold_adc,
                    "threshold_mv": _raw_to_mv(threshold_adc),
                },
            )

    async def stop(self) -> OperationResult:
        async with self._operation_lock:
            was_moving = self._status.duty != 0
            self._stop_without_lock("operator stop")
            return OperationResult.success(
                "FakeF446 stopped",
                data={"was_moving": was_moving},
            )

    def inject_status(self, **changes: object) -> F446Status:
        changes.setdefault("timestamp", self._clock.monotonic())
        self._status = replace(self._status, **changes)  # type: ignore[arg-type]
        if self._status.state is F446State.LIMIT_REACHED_FWD:
            self._configuration = self._configuration_for_state(self._status.state)
        elif self._status.state is F446State.LIMIT_REACHED_REV:
            self._configuration = self._configuration_for_state(self._status.state)
        elif self._status.state is F446State.FAULT:
            self._configuration = Configuration.UNKNOWN
        self._status = replace(
            self._status,
            raw_state=self._status.state.value,
            configuration=self._configuration.value,
        )
        return self._status

    def inject_connection(self, connected: bool) -> F446Status:
        return self.inject_status(connected=connected)

    def inject_fault(self, message: str = "simulated F446 fault") -> F446Status:
        self._set_fault(message)
        return self._status

    def inject_next_transform_timeout(self, enabled: bool = True) -> None:
        self._next_timeout = enabled

    def inject_next_transform_fault(
        self,
        message: str = "simulated F446 fault",
        *,
        code: str = "F446_FAULT",
    ) -> None:
        if code not in {"F446_FAULT", "F446_OVERCURRENT"}:
            raise ValueError(f"unsupported simulated F446 fault code: {code}")
        self._next_fault_message = message
        self._next_fault_code = code

    def inject_next_wrong_final_state(self, enabled: bool = True) -> None:
        self._next_wrong_final_state = enabled

    def inject_next_nonzero_final_duty(self, enabled: bool = True) -> None:
        self._next_nonzero_final_duty = enabled

    def clear_fault_for_simulation(
        self,
        configuration: Configuration = Configuration.UNKNOWN,
    ) -> None:
        """Reset fake state without modelling the forbidden firmware ``clear`` command."""

        state = F446State.IDLE
        if configuration in {Configuration.WALK, Configuration.FLIGHT}:
            state = self._expected_state(configuration)
        self._configuration = configuration
        self._status = replace(
            self._status,
            state=state,
            raw_state=state.value,
            configuration=configuration.value,
            duty=0,
            fault_message=None,
            timestamp=self._clock.monotonic(),
        )

    def clear_history(self) -> None:
        self._commands.clear()
        self._events.clear()

    def _stop_without_lock(self, reason: str) -> None:
        self._record_command("stop")
        previous_state = self._status.state
        if previous_state in {
            F446State.LIMIT_FWD,
            F446State.LIMIT_REV,
            F446State.MANUAL_FWD,
            F446State.MANUAL_REV,
        }:
            next_state = F446State.IDLE
            self._configuration = Configuration.UNKNOWN
        else:
            next_state = previous_state
        self._status = replace(
            self._status,
            state=next_state,
            raw_state=next_state.value,
            configuration=self._configuration.value,
            duty=0,
            timestamp=self._clock.monotonic(),
            raw_lines=(f"Stopped: {reason}",),
        )

    def _set_fault(self, message: str) -> None:
        now = self._clock.monotonic()
        line = f"FAULT: {message}"
        self._configuration = Configuration.UNKNOWN
        self._status = replace(
            self._status,
            state=F446State.FAULT,
            raw_state=F446State.FAULT.value,
            configuration="UNKNOWN",
            duty=0,
            fault_message=message,
            timestamp=now,
            raw_lines=(line,),
        )
        self._events.append(
            F446Event(
                F446EventType.FAULT,
                line,
                now,
                state=F446State.FAULT,
                values={"message": message},
            )
        )

    def _record_command(self, command: str) -> None:
        self._commands.append(F446CommandRecord(self._clock.monotonic(), command))

    async def _wait_transform_delay(self, deadline: float) -> bool:
        remaining = max(0.0, deadline - self._clock.monotonic())
        if self._transform_delay_s >= remaining:
            await self._wait_duration(remaining)
            return False
        await self._wait_duration(self._transform_delay_s)
        return self._clock.monotonic() < deadline

    async def _wait_until(self, deadline: float) -> None:
        await self._wait_duration(max(0.0, deadline - self._clock.monotonic()))

    async def _wait_duration(self, seconds: float) -> None:
        if isinstance(self._clock, ManualClock):
            self._clock.advance(seconds)
            await asyncio.sleep(0)
            return
        await asyncio.sleep(seconds)

    def _transform_timeout_s(self) -> float:
        return 5.0 if self._config is None else self._config.transform_timeout_s

    def _direction(self, configuration: Configuration) -> str:
        if self._config is None:
            return "forward" if configuration is Configuration.FLIGHT else "reverse"
        return self._config.direction_for(configuration.value)

    def _duty(self, configuration: Configuration) -> int:
        if self._config is None:
            return 120
        return self._config.duty_for(configuration.value)

    def _expected_state(self, configuration: Configuration) -> F446State:
        if self._config is None:
            return (
                F446State.LIMIT_REACHED_FWD
                if configuration is Configuration.FLIGHT
                else F446State.LIMIT_REACHED_REV
            )
        return self._config.expected_state_for(configuration.value)

    def _configuration_for_state(self, state: F446State) -> Configuration:
        if state is self._expected_state(Configuration.FLIGHT):
            return Configuration.FLIGHT
        if state is self._expected_state(Configuration.WALK):
            return Configuration.WALK
        return Configuration.UNKNOWN

    @staticmethod
    def _coerce_configuration(
        configuration: Union[Configuration, str],
    ) -> Optional[Configuration]:
        if isinstance(configuration, Configuration):
            return (
                configuration
                if configuration in {Configuration.WALK, Configuration.FLIGHT}
                else None
            )
        try:
            parsed = Configuration(str(configuration).upper())
        except ValueError:
            return None
        return parsed if parsed in {Configuration.WALK, Configuration.FLIGHT} else None


__all__ = ["FakeF446"]
