"""Immutable snapshots crossing AeroGo2 subsystem boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from aerogo2.common.config import X8_ESC_SLOT_MAPPING
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446EventType,
    F446State,
    MorphologyRequest,
    SafetySeverity,
    SystemState,
)
from aerogo2.common.immutable import frozen_mapping
from aerogo2.common.numeric import finite_real


@dataclass(frozen=True)
class EscTelemetry:
    slot: int
    physical_position: str
    rpm: float = 0.0
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    temperature_c: Optional[float] = None
    healthy: bool = True
    timestamp: float = 0.0


def default_esc_telemetry() -> Tuple[EscTelemetry, ...]:
    return tuple(EscTelemetry(slot, position) for slot, position in X8_ESC_SLOT_MAPPING.items())


@dataclass(frozen=True)
class PixhawkStatus:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    landed: bool = True
    rc_failsafe: bool = False
    attitude_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery_voltage: Optional[float] = None
    rc_channels: Mapping[int, int] = field(default_factory=dict)
    esc_rpm: Mapping[int, float] = field(default_factory=dict)
    esc_online: Mapping[int, bool] = field(default_factory=dict)
    esc_raw_present_slots: Tuple[int, ...] = ()
    esc_mavlink_display_shift: Optional[int] = None

    # Compatibility/detail fields used by the Phase 1 safety implementation.
    failsafe: bool = False
    heartbeat_timestamp: float = 0.0
    vertical_velocity_mps: float = 0.0
    relative_altitude_m: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    statustext: Tuple[str, ...] = ()
    esc: Tuple[EscTelemetry, ...] = field(default_factory=default_esc_telemetry)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attitude_rpy", tuple(self.attitude_rpy))
        object.__setattr__(self, "angular_velocity", tuple(self.angular_velocity))
        object.__setattr__(self, "local_position", tuple(self.local_position))
        object.__setattr__(self, "local_velocity", tuple(self.local_velocity))
        object.__setattr__(self, "rc_channels", frozen_mapping(self.rc_channels))
        object.__setattr__(self, "esc_rpm", frozen_mapping(self.esc_rpm))
        object.__setattr__(self, "esc_online", frozen_mapping(self.esc_online))
        object.__setattr__(self, "esc_raw_present_slots", tuple(self.esc_raw_present_slots))
        object.__setattr__(self, "statustext", tuple(self.statustext))
        object.__setattr__(self, "esc", tuple(self.esc))

    @property
    def maximum_esc_rpm(self) -> float:
        raw_values = [item.rpm for item in self.esc]
        raw_values.extend(self.esc_rpm.values())
        values = []
        for raw in raw_values:
            value = finite_real(raw)
            if value is None:
                return math.inf
            values.append(abs(value))
        return max(values, default=0.0)


@dataclass(frozen=True)
class F446Status:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    raw_state: str = "UNKNOWN"
    configuration: str = "UNKNOWN"
    duty: int = 0
    manual_limit: int = 0
    sense_mode: str = "unknown"
    r_is_raw: int = 0
    r_is_mv: int = 0
    l_is_raw: int = 0
    l_is_mv: int = 0
    used_raw: int = 0
    used_mv: int = 0
    threshold_raw: int = 0
    threshold_mv: int = 0
    blanking_ms: int = 0
    overcurrent_ms: int = 0
    timeout_ms: int = 0
    over_active: bool = False
    fault_message: Optional[str] = None

    # Compatibility/detail fields used by existing guards and diagnostics.
    state: F446State = F446State.UNKNOWN
    r_is_adc: Optional[int] = None
    l_is_adc: Optional[int] = None
    used_current_adc: Optional[int] = None
    threshold_adc: Optional[int] = None
    auto_status: Optional[bool] = None
    raw_lines: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_lines", tuple(self.raw_lines))

    @property
    def faulted(self) -> bool:
        return self.state is F446State.FAULT or self.fault_message is not None


@dataclass(frozen=True)
class Go2Status:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    body_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    standing: bool = True
    moving: bool = False
    locomotion_mode: str = "UNKNOWN"
    fault_code: int = 0
    joints_locked: bool = False
    foot_force: Tuple[int, int, int, int] = (0, 0, 0, 0)
    foot_force_valid: bool = False

    # Compatibility/detail fields used by Phase 1 high-level motion guards.
    velocity_mps: float = 0.0
    stable: bool = True
    controller_active: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "body_velocity", tuple(self.body_velocity))
        object.__setattr__(self, "body_rpy", tuple(self.body_rpy))
        object.__setattr__(self, "foot_force", tuple(self.foot_force))


@dataclass(frozen=True)
class RCStatus:
    connected: bool = False
    failsafe: bool = True
    channels: Mapping[int, int] = field(default_factory=dict)
    flight_enable: bool = False
    morphology_request: MorphologyRequest = MorphologyRequest.UNKNOWN
    auto_landing_request: AutoLandingRequest = AutoLandingRequest.MANUAL
    manual_override: bool = False
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", frozen_mapping(self.channels))


@dataclass(frozen=True)
class OperatorRequest:
    timestamp: float = 0.0
    flight_enable: bool = False
    morphology_request: str = "UNKNOWN"
    auto_landing_request: str = "MANUAL"
    manual_override: bool = False


@dataclass(frozen=True)
class LandingEstimate:
    valid: bool = False
    ground_detected: bool = False
    height_m: Optional[float] = None
    vertical_velocity_mps: Optional[float] = None
    horizontal_velocity_mps: Optional[float] = None
    timestamp: float = 0.0
    reason: str = "not initialized"


@dataclass(frozen=True)
class LandingCommand:
    timestamp: float = 0.0
    valid: bool = False
    vx_des: float = 0.0
    vy_des: float = 0.0
    vz_des: float = 0.0
    yaw_rate_des: float = 0.0
    reason: str = "inactive"


@dataclass(frozen=True)
class SystemSnapshot:
    timestamp: float
    state: SystemState
    pixhawk: PixhawkStatus = field(default_factory=PixhawkStatus)
    f446: F446Status = field(default_factory=F446Status)
    go2: Go2Status = field(default_factory=Go2Status)
    operator: OperatorRequest = field(default_factory=OperatorRequest)
    rc: RCStatus = field(default_factory=RCStatus)
    configuration: Configuration = Configuration.UNKNOWN
    landing_estimate: LandingEstimate = field(default_factory=LandingEstimate)
    autoland_active: bool = False
    external_setpoint_active: bool = False
    maintenance_mode: bool = False
    ground_arm_authorized: bool = False
    ground_arm_authorization_expires_at: Optional[float] = None
    active_fault_codes: Tuple[str, ...] = ()
    configuration_source: str = "unconfirmed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_fault_codes", tuple(self.active_fault_codes))

    def with_state(self, state: SystemState, timestamp: Optional[float] = None) -> SystemSnapshot:
        return replace(
            self, state=state, timestamp=self.timestamp if timestamp is None else timestamp
        )


@dataclass(frozen=True)
class SafetyViolation:
    code: str
    severity: SafetySeverity
    message: str
    recommended_action: str
    timestamp: float


@dataclass(frozen=True)
class TransitionRecord:
    timestamp: float
    previous_state: SystemState
    new_state: SystemState
    reason: str
    permitted: bool
    guard_codes: Tuple[str, ...] = ()
    entry_action_error: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "guard_codes", tuple(self.guard_codes))


@dataclass(frozen=True)
class F446Event:
    event_type: F446EventType
    line: str
    timestamp: float
    state: Optional[F446State] = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", frozen_mapping(self.values))


@dataclass(frozen=True)
class HealthReport:
    healthy: bool
    checks: Mapping[str, bool]
    messages: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", frozen_mapping(self.checks))
        object.__setattr__(self, "messages", tuple(self.messages))


def snapshot_to_dict(snapshot: SystemSnapshot) -> Dict[str, Any]:
    """Convert a snapshot to JSON-compatible primitives without mutating it."""

    def convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value if isinstance(value.value, str) else value.name
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, set, frozenset)):
            return [convert(item) for item in value]
        if is_dataclass(value):
            return {
                item.name: convert(getattr(value, item.name)) for item in fields(cast(Any, value))
            }
        return value

    return cast(Dict[str, Any], convert(snapshot))
