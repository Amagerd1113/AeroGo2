"""Enumerations used across the system.

Enums are intentionally centralized so bridges, managers, the console, and tests
cannot silently invent incompatible string values.
"""

from enum import Enum, auto


class SystemState(Enum):
    MANUAL_POSITIONING = auto()
    BOOT_SAFE = auto()
    HOMING_TO_WALK = auto()
    WALK = auto()
    WALK_TO_FLIGHT_PRECHECK = auto()
    TRANSFORM_TO_FLIGHT = auto()
    FLIGHT_READY = auto()
    FLIGHT_MANUAL = auto()
    AUTO_LANDING_READY = auto()
    AUTO_LANDING = auto()
    TOUCHDOWN_VERIFY = auto()
    LANDING_COMPLIANT = auto()
    FLIGHT_TO_WALK_PRECHECK = auto()
    TRANSFORM_TO_WALK = auto()
    FAULT = auto()
    EMERGENCY_STOP = auto()


class RuntimeMode(Enum):
    DRY_RUN = "DRY-RUN"
    HARDWARE_READONLY = "HW-RO"
    HARDWARE = "HW"


class Configuration(Enum):
    UNKNOWN = "UNKNOWN"
    WALK = "WALK"
    FLIGHT = "FLIGHT"


class F446State(Enum):
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    MANUAL_FWD = "MANUAL_FWD"
    MANUAL_REV = "MANUAL_REV"
    LIMIT_FWD = "LIMIT_FWD"
    LIMIT_REV = "LIMIT_REV"
    LIMIT_REACHED_FWD = "LIMIT_REACHED_FWD"
    LIMIT_REACHED_REV = "LIMIT_REACHED_REV"
    FAULT = "FAULT"


class DeviceConnection(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    STALE = "STALE"
    FAULT = "FAULT"


class RCPosition(Enum):
    LOW = "LOW"
    MIDDLE = "MIDDLE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class MorphologyRequest(Enum):
    WALK = "WALK"
    HOLD = "HOLD"
    FLIGHT_REQUEST = "FLIGHT_REQUEST"
    UNKNOWN = "UNKNOWN"


class AutoLandingRequest(Enum):
    MANUAL = "MANUAL"
    AUTO_READY = "AUTO_READY"
    AUTO_EXECUTE = "AUTO_EXECUTE"
    UNKNOWN = "UNKNOWN"


class SafetySeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    FAULT = "FAULT"
    EMERGENCY = "EMERGENCY"


class ConfirmationLevel(Enum):
    NONE = auto()
    SIMPLE = auto()
    EXACT_PHRASE = auto()
    TWO_STAGE = auto()


class CommandStatus(Enum):
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class F446EventType(Enum):
    STATUS = "STATUS"
    CURRENT_STATUS = "CURRENT_STATUS"
    FORWARD_LIMIT_REACHED = "FORWARD_LIMIT_REACHED"
    REVERSE_LIMIT_REACHED = "REVERSE_LIMIT_REACHED"
    FAULT = "FAULT"
    STOPPED = "STOPPED"
    DISABLED = "DISABLED"
    FAULT_CLEARED = "FAULT_CLEARED"
    COMMAND_ECHO = "COMMAND_ECHO"
    INFO = "INFO"
    UNKNOWN_LINE = "UNKNOWN_LINE"

    # Compatibility aliases retained for Phase 1 callers created before the
    # protocol event taxonomy was made direction-specific.
    CURRENT = CURRENT_STATUS
    ECHO = COMMAND_ECHO
    UNKNOWN = UNKNOWN_LINE
