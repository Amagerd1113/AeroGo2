"""Versioned JSONL schemas for command, transition, and telemetry records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

from aerogo2.common.clock import Clock
from aerogo2.common.models import LandingCommand, SafetyViolation, SystemSnapshot, snapshot_to_dict
from aerogo2.common.results import CommandResult

SCHEMA_VERSION = 1


class EventType(str, Enum):
    SYSTEM_STARTED = "SYSTEM_STARTED"
    DEVICE_CONNECTED = "DEVICE_CONNECTED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    TRANSFORM_FLIGHT_REQUESTED = "TRANSFORM_FLIGHT_REQUESTED"
    TRANSFORM_FLIGHT_STARTED = "TRANSFORM_FLIGHT_STARTED"
    FLIGHT_LIMIT_REACHED = "FLIGHT_LIMIT_REACHED"
    FLIGHT_CONFIGURATION_VERIFIED = "FLIGHT_CONFIGURATION_VERIFIED"
    FLIGHT_READY = "FLIGHT_READY"
    PIXHAWK_ARMED = "PIXHAWK_ARMED"
    AUTOLAND_READY = "AUTOLAND_READY"
    AUTOLAND_STARTED = "AUTOLAND_STARTED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    TOUCHDOWN_CONFIRMED = "TOUCHDOWN_CONFIRMED"
    PIXHAWK_DISARMED = "PIXHAWK_DISARMED"
    TRANSFORM_WALK_STARTED = "TRANSFORM_WALK_STARTED"
    WALK_CONFIGURATION_VERIFIED = "WALK_CONFIGURATION_VERIFIED"
    FAULT_ENTERED = "FAULT_ENTERED"
    FAULT_CLEARED = "FAULT_CLEARED"
    COMMAND_EXECUTED = "COMMAND_EXECUTED"
    TELEMETRY = "TELEMETRY"
    LOG_MARK = "LOG_MARK"
    LOGGER_ERROR = "LOGGER_ERROR"
    SYSTEM_EXITED = "SYSTEM_EXITED"


def to_jsonable(value: Any) -> Any:
    """Recursively convert project types to strict JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        enum_value = value.value
        return enum_value if isinstance(enum_value, str) else value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: to_jsonable(getattr(value, item.name)) for item in fields(cast(Any, value))
        }
    return str(value)


@dataclass(frozen=True)
class LogRecord:
    """One complete JSONL record; absent context is represented by ``null``."""

    wall_timestamp: float
    monotonic_timestamp: float
    event_type: str
    system_state: str
    previous_state: Optional[str] = None
    command_id: Optional[str] = None
    command_name: Optional[str] = None
    command_result: Optional[Mapping[str, Any]] = None
    pixhawk_status: Optional[Mapping[str, Any]] = None
    f446_status: Optional[Mapping[str, Any]] = None
    go2_status: Optional[Mapping[str, Any]] = None
    operator_request: Optional[Mapping[str, Any]] = None
    safety_violations: Tuple[Mapping[str, Any], ...] = ()
    transition_reason: Optional[str] = None
    landing_command: Optional[Mapping[str, Any]] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Mapping[str, Any]:
        # Keep every required key present so downstream tooling does not need
        # event-specific schemas.
        return {
            "schema_version": self.schema_version,
            "wall_timestamp": self.wall_timestamp,
            "monotonic_timestamp": self.monotonic_timestamp,
            "event_type": self.event_type,
            "system_state": self.system_state,
            "previous_state": self.previous_state,
            "command_id": self.command_id,
            "command_name": self.command_name,
            "command_result": to_jsonable(self.command_result),
            "pixhawk_status": to_jsonable(self.pixhawk_status),
            "f446_status": to_jsonable(self.f446_status),
            "go2_status": to_jsonable(self.go2_status),
            "operator_request": to_jsonable(self.operator_request),
            "safety_violations": to_jsonable(self.safety_violations),
            "transition_reason": self.transition_reason,
            "landing_command": to_jsonable(self.landing_command),
            "details": to_jsonable(self.details),
        }

    @classmethod
    def create(
        cls,
        *,
        clock: Clock,
        event_type: Any,
        snapshot: Optional[SystemSnapshot] = None,
        previous_state: Optional[Any] = None,
        command_id: Optional[str] = None,
        command_name: Optional[str] = None,
        command_result: Optional[CommandResult] = None,
        safety_violations: Sequence[SafetyViolation] = (),
        transition_reason: Optional[str] = None,
        landing_command: Optional[LandingCommand] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> LogRecord:
        event_name = str(event_type.value) if isinstance(event_type, Enum) else str(event_type)
        snapshot_data = snapshot_to_dict(snapshot) if snapshot is not None else {}
        state_name = snapshot.state.name if snapshot is not None else "UNKNOWN"
        if isinstance(previous_state, Enum):
            previous_name: Optional[str] = previous_state.name
        elif previous_state is None:
            previous_name = None
        else:
            previous_name = str(previous_state)
        result_data = to_jsonable(command_result) if command_result is not None else None
        return cls(
            wall_timestamp=clock.wall_time(),
            monotonic_timestamp=clock.monotonic(),
            event_type=event_name,
            system_state=state_name,
            previous_state=previous_name,
            command_id=command_id,
            command_name=command_name,
            command_result=result_data,
            pixhawk_status=snapshot_data.get("pixhawk"),
            f446_status=snapshot_data.get("f446"),
            go2_status=snapshot_data.get("go2"),
            operator_request=snapshot_data.get("operator"),
            safety_violations=tuple(to_jsonable(violation) for violation in safety_violations),
            transition_reason=transition_reason,
            landing_command=(to_jsonable(landing_command) if landing_command is not None else None),
            details={} if details is None else to_jsonable(details),
        )


EventRecord = LogRecord

__all__ = [
    "EventRecord",
    "EventType",
    "LogRecord",
    "SCHEMA_VERSION",
    "to_jsonable",
]
