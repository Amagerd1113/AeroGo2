"""YAML configuration loading, merging, and safety validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import yaml

from aerogo2.common.enums import F446State
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.common.immutable import frozen_mapping

# Canonical AeroGo2 labels for the Hobbywing X8 NodeID/ThrottleID order.
# The validated diagnostic uses FR/FL/RL/RR while the existing AeroGo2 model
# uses RF/LF/LR/RR; only the spelling differs.
X8_ESC_SLOT_MAPPING = {1: "RR", 2: "LF", 3: "LR", 4: "RF"}

GO2_LOW_LEVEL_JOINT_COUNT = 12
GO2_LOW_LEVEL_SAFE_HOLD_POLICIES = frozenset({"capture_current", "configured_pose"})

_ROOT_KEYS = frozenset(
    {"includes", "system", "pixhawk", "f446", "go2", "rc", "safety", "landing", "esc"}
)
_SYSTEM_KEYS = frozenset({"loop_hz", "dry_run", "hardware_write_enabled", "log_directory"})
_PIXHAWK_KEYS = frozenset(
    {
        "connection",
        "baud",
        "heartbeat_timeout_s",
        "target_system",
        "target_component",
        "rc9_option",
        "rc10_option",
    }
)
_F446_KEYS = frozenset(
    {
        "port",
        "baud",
        "flight_direction",
        "walk_direction",
        "flight_duty",
        "walk_duty",
        "expected_flight_state",
        "expected_walk_state",
        "response_timeout_s",
        "transform_timeout_s",
        "status_poll_hz",
        "firmware_timeout_ms",
        "automatic_stall_threshold_adc",
        "stall_blanking_ms",
        "stall_overcurrent_ms",
        "current_safe_margin_adc",
        "current_clear_hold_s",
    }
)
_GO2_KEYS = frozenset(
    {
        "enabled",
        "status_timeout_s",
        "network_interface",
        "domain_id",
        "sport_state_topic",
        "command_timeout_s",
        "joint_lock_operator_timeout_s",
        "joint_lock_transition_grace_s",
        "joint_lock_unsafe_confirm_s",
        "joint_lock_state_codes",
        "accepted_state_codes",
        "landing_compliance_enabled",
        "foot_force_contact_thresholds",
        "landing_contact_min_feet",
        "landing_contact_confirm_s",
        "landing_compliance_settle_s",
        "low_level",
    }
)
_RC_KEYS = frozenset(
    {
        "flight_enable_channel",
        "flight_mode_channel",
        "rtl_channel",
        "land_channel",
        "morphology_channel",
        "auto_landing_channel",
        "brake_channel",
        "buzzer_channel",
        "low_max",
        "middle_min",
        "middle_max",
        "high_min",
        "debounce_s",
        "timeout_s",
        "manual_override_deadband_us",
    }
)
_SAFETY_KEYS = frozenset(
    {
        "stationary_velocity_mps",
        "stationary_confirm_s",
        "maximum_safe_esc_rpm_for_transform",
        "airborne_confirm_s",
        "touchdown_confirm_s",
        "touchdown_max_vertical_speed_mps",
        "touchdown_max_tilt_rad",
        "touchdown_max_height_delta_m",
        "touchdown_max_esc_rpm",
        "touchdown_max_source_age_s",
        "touchdown_max_source_skew_s",
        "post_touchdown_stable_confirm_s",
        "post_touchdown_stability_max_check_gap_s",
        "aborted_impact_airborne_confirm_s",
        "impact_recovery_status_max_age_s",
        "impact_recovery_completion_timeout_s",
        "impact_recovery_finalization_timeout_s",
        "pixhawk_timeout_s",
        "f446_timeout_s",
        "go2_timeout_s",
        "rc_timeout_s",
        "controller_timeout_s",
        "maximum_transform_current_adc",
    }
)
_LANDING_KEYS = frozenset(
    {
        "controller_hz",
        "maximum_descent_speed_mps",
        "maximum_horizontal_speed_mps",
        "maximum_yaw_rate_rad_s",
        "controller_timeout_s",
        "manual_override_deadband_us",
        "default_abort_mode",
    }
)
_ESC_KEYS = frozenset({"slot_1", "slot_2", "slot_3", "slot_4", "mavlink_display_shift"})
_GO2_LOW_LEVEL_OBSERVE_REQUIRED_KEYS = (
    "low_state_topic",
    "low_state_max_age_s",
    "mapping_version",
    "mapping_hash",
    "joint_names",
    "motor_ids",
    "directions",
    "zero_offsets_rad",
)
_GO2_LOW_LEVEL_ACTUATION_REQUIRED_KEYS = (
    *_GO2_LOW_LEVEL_OBSERVE_REQUIRED_KEYS,
    "low_command_topic",
    "send_period_s",
    "maximum_jitter_s",
    "target_ttl_s",
    "acquire_timeout_s",
    "release_timeout_s",
    "safe_hold_policy",
    "safe_hold_pose_rad",
    "safe_hold_position_tolerance_rad",
    "safe_hold_velocity_tolerance_rad_s",
    "tracking_position_error_limit_rad",
    "safe_hold_ack_timeout_s",
    "restore_mode_form",
    "restore_mode_name",
    "q_min_rad",
    "q_max_rad",
    "dq_max_rad_s",
    "maximum_delta_q_rad",
    "kp",
    "kd",
    "tau_ff_nm",
    "tau_limit_nm",
    "feedback_loss_degraded_kp",
    "feedback_loss_degraded_kd",
    "feedback_loss_degraded_tau_ff_nm",
    "firmware_torque_limit_nm",
    "firmware_torque_clamp_verified",
    "temperature_limit_c",
)
_GO2_LOW_LEVEL_KEYS = frozenset(
    {
        "enabled",
        "observe_only_enabled",
        *_GO2_LOW_LEVEL_ACTUATION_REQUIRED_KEYS,
    }
)


def compute_go2_joint_mapping_hash(
    mapping_version: str,
    joint_names: Tuple[str, ...],
    motor_ids: Tuple[int, ...],
    directions: Tuple[int, ...],
    zero_offsets_rad: Tuple[float, ...],
) -> str:
    """Return the canonical SHA-256 digest for a reviewed Go2 joint mapping.

    The digest deliberately covers only coordinate/motor identity data.  Servo
    gains and safety limits are validated independently and may be tuned without
    silently changing the kinematic mapping identity used by command leases.
    """

    payload = {
        "directions": [int(value) for value in directions],
        "joint_names": [str(value) for value in joint_names],
        "mapping_version": str(mapping_version),
        "motor_ids": [int(value) for value in motor_ids],
        "zero_offsets_rad": [float(value) for value in zero_offsets_rad],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SystemConfig:
    loop_hz: float
    dry_run: bool
    hardware_write_enabled: bool
    log_directory: Path


@dataclass(frozen=True)
class PixhawkConfig:
    connection: str
    baud: int
    heartbeat_timeout_s: float
    target_system: int
    target_component: int
    rc9_option: int
    rc10_option: int


@dataclass(frozen=True)
class F446Config:
    port: str
    baud: int
    flight_direction: str
    walk_direction: str
    flight_duty: int
    walk_duty: int
    expected_flight_state: F446State
    expected_walk_state: F446State
    response_timeout_s: float
    transform_timeout_s: float
    status_poll_hz: float
    firmware_timeout_ms: int
    automatic_stall_threshold_adc: int
    stall_blanking_ms: int
    stall_overcurrent_ms: int
    current_safe_margin_adc: int
    current_clear_hold_s: float

    def direction_for(self, configuration: str) -> str:
        return self.flight_direction if configuration.upper() == "FLIGHT" else self.walk_direction

    def duty_for(self, configuration: str) -> int:
        return self.flight_duty if configuration.upper() == "FLIGHT" else self.walk_duty

    def expected_state_for(self, configuration: str) -> F446State:
        return (
            self.expected_flight_state
            if configuration.upper() == "FLIGHT"
            else self.expected_walk_state
        )


@dataclass(frozen=True)
class Go2LowLevelConfig:
    """Fail-closed configuration for the sole Go2 ``rt/lowcmd`` owner.

    Every hardware-dependent value is optional while this subsystem is
    disabled.  Enabling it is accepted only after the YAML validator has seen
    and cross-checked every field.
    """

    # ``enabled`` authorizes only the fully commissioned LowCmd path.
    # ``observe_only_enabled`` may independently subscribe to LowState, but it
    # can never acquire an ownership epoch or create a LowCmd publisher.
    enabled: bool = False
    observe_only_enabled: bool = False
    low_state_topic: Optional[str] = None
    low_command_topic: Optional[str] = None
    send_period_s: Optional[float] = None
    maximum_jitter_s: Optional[float] = None
    low_state_max_age_s: Optional[float] = None
    target_ttl_s: Optional[float] = None
    acquire_timeout_s: Optional[float] = None
    release_timeout_s: Optional[float] = None
    safe_hold_policy: Optional[str] = None
    safe_hold_pose_rad: Optional[Tuple[float, ...]] = None
    safe_hold_position_tolerance_rad: Optional[Tuple[float, ...]] = None
    safe_hold_velocity_tolerance_rad_s: Optional[Tuple[float, ...]] = None
    tracking_position_error_limit_rad: Optional[Tuple[float, ...]] = None
    safe_hold_ack_timeout_s: Optional[float] = None
    restore_mode_form: Optional[str] = None
    restore_mode_name: Optional[str] = None
    mapping_version: Optional[str] = None
    mapping_hash: Optional[str] = None
    joint_names: Optional[Tuple[str, ...]] = None
    motor_ids: Optional[Tuple[int, ...]] = None
    directions: Optional[Tuple[int, ...]] = None
    zero_offsets_rad: Optional[Tuple[float, ...]] = None
    q_min_rad: Optional[Tuple[float, ...]] = None
    q_max_rad: Optional[Tuple[float, ...]] = None
    dq_max_rad_s: Optional[Tuple[float, ...]] = None
    maximum_delta_q_rad: Optional[Tuple[float, ...]] = None
    kp: Optional[Tuple[float, ...]] = None
    kd: Optional[Tuple[float, ...]] = None
    tau_ff_nm: Optional[Tuple[float, ...]] = None
    tau_limit_nm: Optional[Tuple[float, ...]] = None
    feedback_loss_degraded_kp: Optional[Tuple[float, ...]] = None
    feedback_loss_degraded_kd: Optional[Tuple[float, ...]] = None
    feedback_loss_degraded_tau_ff_nm: Optional[Tuple[float, ...]] = None
    firmware_torque_limit_nm: Optional[Tuple[float, ...]] = None
    firmware_torque_clamp_verified: Optional[bool] = None
    temperature_limit_c: Optional[Tuple[float, ...]] = None

    @property
    def observation_enabled(self) -> bool:
        """Whether the read-only LowState transport should be connected."""

        return self.observe_only_enabled or self.enabled


@dataclass(frozen=True)
class Go2Config:
    enabled: bool
    status_timeout_s: float
    network_interface: str = "eth0"
    domain_id: int = 0
    sport_state_topic: str = "rt/sportmodestate"
    command_timeout_s: float = 2.0
    joint_lock_operator_timeout_s: float = 60.0
    joint_lock_transition_grace_s: float = 2.0
    joint_lock_unsafe_confirm_s: float = 0.5
    joint_lock_state_codes: Tuple[int, ...] = (1002,)
    accepted_state_codes: Tuple[int, ...] = (0, 100, 1002)
    landing_compliance_enabled: bool = False
    foot_force_contact_thresholds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    landing_contact_min_feet: int = 3
    landing_contact_confirm_s: float = 0.5
    landing_compliance_settle_s: float = 1.5
    low_level: Go2LowLevelConfig = Go2LowLevelConfig()


@dataclass(frozen=True)
class RCConfig:
    flight_enable_channel: int
    flight_mode_channel: int
    rtl_channel: int
    land_channel: int
    morphology_channel: int
    auto_landing_channel: int
    brake_channel: int
    buzzer_channel: int
    low_max: int
    middle_min: int
    middle_max: int
    high_min: int
    debounce_s: float
    timeout_s: float
    manual_override_deadband_us: int


@dataclass(frozen=True)
class SafetyConfig:
    stationary_velocity_mps: float
    stationary_confirm_s: float
    maximum_safe_esc_rpm_for_transform: float
    airborne_confirm_s: float
    touchdown_confirm_s: float
    touchdown_max_vertical_speed_mps: float
    touchdown_max_tilt_rad: float
    touchdown_max_height_delta_m: float
    touchdown_max_esc_rpm: float
    touchdown_max_source_age_s: float
    touchdown_max_source_skew_s: float
    post_touchdown_stable_confirm_s: float
    post_touchdown_stability_max_check_gap_s: float
    aborted_impact_airborne_confirm_s: float
    impact_recovery_status_max_age_s: float
    impact_recovery_completion_timeout_s: float
    impact_recovery_finalization_timeout_s: float
    pixhawk_timeout_s: float
    f446_timeout_s: float
    go2_timeout_s: float
    rc_timeout_s: float
    controller_timeout_s: float
    maximum_transform_current_adc: int


@dataclass(frozen=True)
class LandingConfig:
    controller_hz: float
    maximum_descent_speed_mps: float
    maximum_horizontal_speed_mps: float
    maximum_yaw_rate_rad_s: float
    controller_timeout_s: float
    manual_override_deadband_us: int
    default_abort_mode: str


@dataclass(frozen=True)
class EscConfig:
    slots: Mapping[int, str]
    mavlink_display_shift: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", frozen_mapping(self.slots))


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    system: SystemConfig
    pixhawk: PixhawkConfig
    f446: F446Config
    go2: Go2Config
    rc: RCConfig
    safety: SafetyConfig
    landing: LandingConfig
    esc: EscConfig
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", frozen_mapping(self.raw))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.raw
        for key in dotted_key.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current


def _deep_merge(
    target: MutableMapping[str, Any], incoming: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in incoming.items():
        if key == "includes":
            continue
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            existing = dict(target[key])
            target[key] = _deep_merge(existing, value)
        else:
            target[key] = value
    return target


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigurationError(f"Top level of {path} must be a mapping")
    return loaded


def _unknown_keys_error(
    section: Mapping[str, Any],
    allowed: AbstractSet[str],
    label: str,
) -> Optional[str]:
    unknown = sorted(
        (key if isinstance(key, str) else repr(key)) for key in section if key not in allowed
    )
    if not unknown:
        return None
    return f"{label} contains unknown keys: " + ", ".join(unknown)


def _load_merged(path: Path, seen: Optional[Tuple[Path, ...]] = None) -> Dict[str, Any]:
    resolved = path.resolve()
    chain = () if seen is None else seen
    if resolved in chain:
        raise ConfigurationError(f"Configuration include cycle at {resolved}")
    document = _read_yaml(resolved)
    result: Dict[str, Any] = {}
    includes = document.get("includes", [])
    if not isinstance(includes, list):
        raise ConfigurationError("'includes' must be a list")
    for item in includes:
        if not isinstance(item, str):
            raise ConfigurationError("Every include path must be a string")
        included = _load_merged(resolved.parent / item, chain + (resolved,))
        _deep_merge(result, included)
    _deep_merge(result, document)
    return result


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing or invalid '{name}' section")
    return value


def _required(section: Mapping[str, Any], key: str) -> Any:
    if key not in section:
        raise ConfigurationError(f"Missing configuration key '{key}'")
    return section[key]


def _validate_raw(raw: Mapping[str, Any]) -> List[str]:
    """Return deterministic validation errors without permissive coercion."""

    errors: List[str] = []
    root_error = _unknown_keys_error(raw, _ROOT_KEYS, "configuration root")
    if root_error is not None:
        errors.append(root_error)
    system = _section(raw, "system")
    pixhawk = _section(raw, "pixhawk")
    f446 = _section(raw, "f446")
    go2 = _section(raw, "go2")
    rc = _section(raw, "rc")
    safety = _section(raw, "safety")
    landing = _section(raw, "landing")
    esc = _section(raw, "esc")

    section_schemas = (
        (system, _SYSTEM_KEYS, "system"),
        (pixhawk, _PIXHAWK_KEYS, "pixhawk"),
        (f446, _F446_KEYS, "f446"),
        (go2, _GO2_KEYS, "go2"),
        (rc, _RC_KEYS, "rc"),
        (safety, _SAFETY_KEYS, "safety"),
        (landing, _LANDING_KEYS, "landing"),
        (esc, _ESC_KEYS, "esc"),
    )
    for section, allowed, label in section_schemas:
        unknown_error = _unknown_keys_error(section, allowed, label)
        if unknown_error is not None:
            errors.append(unknown_error)

    def value(section: Mapping[str, Any], key: str, label: str) -> Any:
        if key not in section:
            errors.append(f"Missing configuration key '{label}'")
            return None
        return section[key]

    def boolean(section: Mapping[str, Any], key: str, label: str) -> Optional[bool]:
        item = value(section, key, label)
        if item is None:
            return None
        if type(item) is not bool:
            errors.append(f"{label} must be a YAML boolean")
            return None
        return item

    def integer(
        section: Mapping[str, Any],
        key: str,
        label: str,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> Optional[int]:
        item = value(section, key, label)
        if item is None:
            return None
        if type(item) is not int:
            errors.append(f"{label} must be an integer")
            return None
        if minimum is not None and item < minimum:
            errors.append(f"{label} must be at least {minimum}")
        if maximum is not None and item > maximum:
            errors.append(f"{label} must be at most {maximum}")
        return item

    def integer_quad(
        section: Mapping[str, Any],
        key: str,
        label: str,
        *,
        minimum: int,
        maximum: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        item = value(section, key, label)
        if item is None:
            return None
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            errors.append(f"{label} must contain exactly four integers")
            return None
        parsed: List[int] = []
        valid = True
        for index, raw_item in enumerate(item):
            if type(raw_item) is not int:
                errors.append(f"{label}[{index}] must be an integer")
                valid = False
                continue
            if raw_item < minimum or raw_item > maximum:
                errors.append(f"{label}[{index}] must be within {minimum}..{maximum}")
                valid = False
            parsed.append(raw_item)
        if not valid or len(parsed) != 4:
            return None
        return (parsed[0], parsed[1], parsed[2], parsed[3])

    def integer_list(
        section: Mapping[str, Any],
        key: str,
        label: str,
        *,
        minimum: int,
        maximum: int,
    ) -> Optional[Tuple[int, ...]]:
        item = value(section, key, label)
        if item is None:
            return None
        if not isinstance(item, (list, tuple)) or not item:
            errors.append(f"{label} must contain at least one integer")
            return None
        parsed: List[int] = []
        valid = True
        for index, raw_item in enumerate(item):
            if type(raw_item) is not int:
                errors.append(f"{label}[{index}] must be an integer")
                valid = False
                continue
            if raw_item < minimum or raw_item > maximum:
                errors.append(f"{label}[{index}] must be within {minimum}..{maximum}")
                valid = False
            parsed.append(raw_item)
        if len(set(parsed)) != len(parsed):
            errors.append(f"{label} must not contain duplicate values")
            valid = False
        return tuple(parsed) if valid else None

    def finite_number(
        section: Mapping[str, Any],
        key: str,
        label: str,
        *,
        positive: bool = False,
    ) -> Optional[float]:
        item = value(section, key, label)
        if item is None:
            return None
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{label} must be a finite number")
            return None
        numeric = float(item)
        if not math.isfinite(numeric):
            errors.append(f"{label} must be finite")
            return None
        if positive and numeric <= 0.0:
            errors.append(f"{label} must be positive")
        return numeric

    def nonempty_text(
        section: Mapping[str, Any],
        key: str,
        label: str,
    ) -> Optional[str]:
        item = value(section, key, label)
        if item is None:
            return None
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label} must be a non-empty string")
            return None
        return item

    def number_vector(
        section: Mapping[str, Any],
        key: str,
        label: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> Optional[Tuple[float, ...]]:
        item = section.get(key)
        if item is None:
            return None
        if not isinstance(item, (list, tuple)) or len(item) != GO2_LOW_LEVEL_JOINT_COUNT:
            errors.append(
                f"{label} must contain exactly {GO2_LOW_LEVEL_JOINT_COUNT} finite numbers"
            )
            return None
        parsed: List[float] = []
        valid = True
        for index, raw_item in enumerate(item):
            if isinstance(raw_item, bool) or not isinstance(raw_item, (int, float)):
                errors.append(f"{label}[{index}] must be a finite number")
                valid = False
                continue
            numeric = float(raw_item)
            if not math.isfinite(numeric):
                errors.append(f"{label}[{index}] must be finite")
                valid = False
                continue
            if positive and numeric <= 0.0:
                errors.append(f"{label}[{index}] must be positive")
                valid = False
            if nonnegative and numeric < 0.0:
                errors.append(f"{label}[{index}] must be nonnegative")
                valid = False
            parsed.append(numeric)
        return tuple(parsed) if valid and len(parsed) == GO2_LOW_LEVEL_JOINT_COUNT else None

    def integer_vector(
        section: Mapping[str, Any],
        key: str,
        label: str,
    ) -> Optional[Tuple[int, ...]]:
        item = section.get(key)
        if item is None:
            return None
        if not isinstance(item, (list, tuple)) or len(item) != GO2_LOW_LEVEL_JOINT_COUNT:
            errors.append(f"{label} must contain exactly {GO2_LOW_LEVEL_JOINT_COUNT} integers")
            return None
        parsed: List[int] = []
        valid = True
        for index, raw_item in enumerate(item):
            if type(raw_item) is not int:
                errors.append(f"{label}[{index}] must be an integer")
                valid = False
                continue
            parsed.append(raw_item)
        return tuple(parsed) if valid and len(parsed) == GO2_LOW_LEVEL_JOINT_COUNT else None

    def text_vector(
        section: Mapping[str, Any],
        key: str,
        label: str,
    ) -> Optional[Tuple[str, ...]]:
        item = section.get(key)
        if item is None:
            return None
        if not isinstance(item, (list, tuple)) or len(item) != GO2_LOW_LEVEL_JOINT_COUNT:
            errors.append(f"{label} must contain exactly {GO2_LOW_LEVEL_JOINT_COUNT} strings")
            return None
        parsed: List[str] = []
        valid = True
        for index, raw_item in enumerate(item):
            if not isinstance(raw_item, str) or not raw_item.strip():
                errors.append(f"{label}[{index}] must be a non-empty string")
                valid = False
                continue
            parsed.append(raw_item)
        return tuple(parsed) if valid and len(parsed) == GO2_LOW_LEVEL_JOINT_COUNT else None

    loop_hz = finite_number(system, "loop_hz", "system.loop_hz", positive=True)
    del loop_hz
    dry_run = boolean(system, "dry_run", "system.dry_run")
    hardware_write = boolean(
        system,
        "hardware_write_enabled",
        "system.hardware_write_enabled",
    )
    if dry_run is True and hardware_write is True:
        errors.append("Dry-run cannot enable physical hardware writes")
    nonempty_text(system, "log_directory", "system.log_directory")

    nonempty_text(pixhawk, "connection", "pixhawk.connection")
    integer(pixhawk, "baud", "pixhawk.baud", minimum=1, maximum=4_000_000)
    finite_number(
        pixhawk,
        "heartbeat_timeout_s",
        "pixhawk.heartbeat_timeout_s",
        positive=True,
    )
    integer(pixhawk, "target_system", "pixhawk.target_system", minimum=1, maximum=255)
    integer(
        pixhawk,
        "target_component",
        "pixhawk.target_component",
        minimum=1,
        maximum=255,
    )
    rc9_option = integer(pixhawk, "rc9_option", "pixhawk.rc9_option", minimum=0)
    rc10_option = integer(pixhawk, "rc10_option", "pixhawk.rc10_option", minimum=0)
    if rc9_option is not None and rc9_option != 0:
        errors.append("Pixhawk RC9_OPTION and RC10_OPTION must both remain 0")
    if rc10_option is not None and rc10_option != 0:
        errors.append("Pixhawk RC9_OPTION and RC10_OPTION must both remain 0")

    nonempty_text(f446, "port", "f446.port")
    integer(f446, "baud", "f446.baud", minimum=1, maximum=4_000_000)
    flight_direction = nonempty_text(f446, "flight_direction", "f446.flight_direction")
    walk_direction = nonempty_text(f446, "walk_direction", "f446.walk_direction")
    normalized_flight = "" if flight_direction is None else flight_direction.lower()
    normalized_walk = "" if walk_direction is None else walk_direction.lower()
    if (
        flight_direction is not None
        and walk_direction is not None
        and {normalized_flight, normalized_walk} != {"forward", "reverse"}
    ):
        errors.append("F446 flight/walk directions must be distinct forward/reverse values")
    integer(f446, "flight_duty", "f446.flight_duty", minimum=1, maximum=350)
    integer(f446, "walk_duty", "f446.walk_duty", minimum=1, maximum=350)
    expected_flight = nonempty_text(
        f446,
        "expected_flight_state",
        "f446.expected_flight_state",
    )
    expected_walk = nonempty_text(
        f446,
        "expected_walk_state",
        "f446.expected_walk_state",
    )
    expected_by_direction = {
        "forward": "LIMIT_REACHED_FWD",
        "reverse": "LIMIT_REACHED_REV",
    }
    if (
        expected_flight is not None
        and normalized_flight in expected_by_direction
        and expected_flight != expected_by_direction[normalized_flight]
    ):
        errors.append("expected_flight_state conflicts with flight_direction")
    if (
        expected_walk is not None
        and normalized_walk in expected_by_direction
        and expected_walk != expected_by_direction[normalized_walk]
    ):
        errors.append("expected_walk_state conflicts with walk_direction")
    finite_number(f446, "response_timeout_s", "f446.response_timeout_s", positive=True)
    transform_timeout_s = finite_number(
        f446,
        "transform_timeout_s",
        "f446.transform_timeout_s",
        positive=True,
    )
    finite_number(f446, "status_poll_hz", "f446.status_poll_hz", positive=True)
    current_safe_margin_adc = integer(
        f446,
        "current_safe_margin_adc",
        "f446.current_safe_margin_adc",
        minimum=0,
        maximum=4095,
    )
    default_firmware_timeout_ms = (
        15000 if transform_timeout_s is None else max(100, int(round(transform_timeout_s * 1000.0)))
    )
    firmware_timeout_ms = (
        integer(
            f446,
            "firmware_timeout_ms",
            "f446.firmware_timeout_ms",
            minimum=100,
            maximum=60000,
        )
        if "firmware_timeout_ms" in f446
        else default_firmware_timeout_ms
    )
    automatic_stall_threshold_adc = (
        integer(
            f446,
            "automatic_stall_threshold_adc",
            "f446.automatic_stall_threshold_adc",
            minimum=0,
            maximum=4095,
        )
        if "automatic_stall_threshold_adc" in f446
        else 0
    )
    stall_blanking_ms = (
        integer(f446, "stall_blanking_ms", "f446.stall_blanking_ms", minimum=0, maximum=5000)
        if "stall_blanking_ms" in f446
        else 500
    )
    stall_overcurrent_ms = (
        integer(f446, "stall_overcurrent_ms", "f446.stall_overcurrent_ms", minimum=10, maximum=3000)
        if "stall_overcurrent_ms" in f446
        else 180
    )
    finite_number(
        f446,
        "current_clear_hold_s",
        "f446.current_clear_hold_s",
        positive=True,
    )

    boolean(go2, "enabled", "go2.enabled")
    finite_number(go2, "status_timeout_s", "go2.status_timeout_s", positive=True)
    if "network_interface" in go2:
        nonempty_text(go2, "network_interface", "go2.network_interface")
    if "domain_id" in go2:
        integer(go2, "domain_id", "go2.domain_id", minimum=0, maximum=232)
    if "sport_state_topic" in go2:
        nonempty_text(go2, "sport_state_topic", "go2.sport_state_topic")
    if "command_timeout_s" in go2:
        finite_number(go2, "command_timeout_s", "go2.command_timeout_s", positive=True)
    if "joint_lock_operator_timeout_s" in go2:
        finite_number(
            go2,
            "joint_lock_operator_timeout_s",
            "go2.joint_lock_operator_timeout_s",
            positive=True,
        )
    if "joint_lock_transition_grace_s" in go2:
        finite_number(
            go2,
            "joint_lock_transition_grace_s",
            "go2.joint_lock_transition_grace_s",
            positive=True,
        )
    if "joint_lock_unsafe_confirm_s" in go2:
        finite_number(
            go2,
            "joint_lock_unsafe_confirm_s",
            "go2.joint_lock_unsafe_confirm_s",
            positive=True,
        )
    if "joint_lock_state_codes" in go2:
        integer_list(
            go2,
            "joint_lock_state_codes",
            "go2.joint_lock_state_codes",
            minimum=0,
            maximum=4_294_967_295,
        )
    if "accepted_state_codes" in go2:
        integer_list(
            go2,
            "accepted_state_codes",
            "go2.accepted_state_codes",
            minimum=0,
            maximum=4_294_967_295,
        )

    compliance_enabled: Optional[bool] = False
    if "landing_compliance_enabled" in go2:
        compliance_enabled = boolean(
            go2,
            "landing_compliance_enabled",
            "go2.landing_compliance_enabled",
        )
    foot_force_thresholds: Optional[Tuple[int, int, int, int]] = None
    if "foot_force_contact_thresholds" in go2:
        foot_force_thresholds = integer_quad(
            go2,
            "foot_force_contact_thresholds",
            "go2.foot_force_contact_thresholds",
            minimum=0,
            maximum=32767,
        )
    if "landing_contact_min_feet" in go2:
        integer(
            go2,
            "landing_contact_min_feet",
            "go2.landing_contact_min_feet",
            minimum=1,
            maximum=4,
        )
    for key in ("landing_contact_confirm_s", "landing_compliance_settle_s"):
        if key in go2:
            finite_number(go2, key, f"go2.{key}", positive=True)
    if compliance_enabled is True:
        required_compliance_keys = (
            "foot_force_contact_thresholds",
            "landing_contact_min_feet",
            "landing_contact_confirm_s",
            "landing_compliance_settle_s",
        )
        for key in required_compliance_keys:
            if key not in go2:
                errors.append(
                    f"Missing configuration key 'go2.{key}' while landing compliance is enabled"
                )
        if foot_force_thresholds is not None and any(item <= 0 for item in foot_force_thresholds):
            errors.append(
                "go2.foot_force_contact_thresholds must all be positive when "
                "landing compliance is enabled"
            )

    raw_low_level = go2.get("low_level")
    if raw_low_level is None:
        low_level: Mapping[str, Any] = {}
    elif not isinstance(raw_low_level, Mapping):
        errors.append("go2.low_level must be a mapping")
        low_level = {}
    else:
        low_level = raw_low_level

    low_level_enabled: Optional[bool]
    if not low_level:
        low_level_enabled = False
    elif "enabled" not in low_level:
        errors.append("Missing configuration key 'go2.low_level.enabled'")
        low_level_enabled = None
    elif low_level["enabled"] is None:
        errors.append("go2.low_level.enabled must be a YAML boolean")
        low_level_enabled = None
    else:
        low_level_enabled = boolean(low_level, "enabled", "go2.low_level.enabled")

    observe_only_enabled: Optional[bool]
    if low_level.get("observe_only_enabled") is None:
        observe_only_enabled = False
    else:
        observe_only_enabled = boolean(
            low_level,
            "observe_only_enabled",
            "go2.low_level.observe_only_enabled",
        )

    unknown_low_level_error = _unknown_keys_error(
        low_level,
        _GO2_LOW_LEVEL_KEYS,
        "go2.low_level",
    )
    if unknown_low_level_error is not None:
        errors.append(unknown_low_level_error)
    if low_level_enabled is True or observe_only_enabled is True:
        for key in _GO2_LOW_LEVEL_OBSERVE_REQUIRED_KEYS:
            if key not in low_level or low_level[key] is None:
                errors.append(
                    f"Missing explicit configuration value 'go2.low_level.{key}' "
                    "while LowState observation is enabled"
                )
    if low_level_enabled is True:
        for key in _GO2_LOW_LEVEL_ACTUATION_REQUIRED_KEYS:
            if key in _GO2_LOW_LEVEL_OBSERVE_REQUIRED_KEYS:
                continue
            if key not in low_level or low_level[key] is None:
                errors.append(
                    f"Missing explicit configuration value 'go2.low_level.{key}' "
                    "while LowCmd actuation is enabled"
                )

    low_state_topic = (
        nonempty_text(low_level, "low_state_topic", "go2.low_level.low_state_topic")
        if low_level.get("low_state_topic") is not None
        else None
    )
    low_command_topic = (
        nonempty_text(low_level, "low_command_topic", "go2.low_level.low_command_topic")
        if low_level.get("low_command_topic") is not None
        else None
    )
    if (
        low_state_topic is not None
        and low_command_topic is not None
        and low_state_topic == low_command_topic
    ):
        errors.append("go2.low_level state and command topics must be different")

    timing_values: Dict[str, Optional[float]] = {}
    safety_numbers: Dict[str, Optional[float]] = {}
    for key in (
        "send_period_s",
        "low_state_max_age_s",
        "target_ttl_s",
        "acquire_timeout_s",
        "release_timeout_s",
        "safe_hold_ack_timeout_s",
    ):
        timing_values[key] = (
            finite_number(low_level, key, f"go2.low_level.{key}", positive=True)
            if low_level.get(key) is not None
            else None
        )
    maximum_jitter_s = (
        finite_number(low_level, "maximum_jitter_s", "go2.low_level.maximum_jitter_s")
        if low_level.get("maximum_jitter_s") is not None
        else None
    )
    if maximum_jitter_s is not None and maximum_jitter_s < 0.0:
        errors.append("go2.low_level.maximum_jitter_s must be nonnegative")
    send_period_s = timing_values["send_period_s"]
    if (
        send_period_s is not None
        and maximum_jitter_s is not None
        and maximum_jitter_s >= send_period_s
    ):
        errors.append("go2.low_level.maximum_jitter_s must be less than send_period_s")
    for key in ("low_state_max_age_s", "target_ttl_s", "safe_hold_ack_timeout_s"):
        item = timing_values[key]
        if send_period_s is not None and item is not None and item < send_period_s:
            errors.append(f"go2.low_level.{key} must be at least send_period_s")
    if (
        timing_values["safe_hold_ack_timeout_s"] is not None
        and timing_values["release_timeout_s"] is not None
        and timing_values["safe_hold_ack_timeout_s"] > timing_values["release_timeout_s"]
    ):
        errors.append("go2.low_level.safe_hold_ack_timeout_s must not exceed release_timeout_s")

    safe_hold_policy = (
        nonempty_text(low_level, "safe_hold_policy", "go2.low_level.safe_hold_policy")
        if low_level.get("safe_hold_policy") is not None
        else None
    )
    if safe_hold_policy is not None and safe_hold_policy not in GO2_LOW_LEVEL_SAFE_HOLD_POLICIES:
        choices = ", ".join(sorted(GO2_LOW_LEVEL_SAFE_HOLD_POLICIES))
        errors.append(f"go2.low_level.safe_hold_policy must be one of: {choices}")

    for key in ("restore_mode_form", "restore_mode_name"):
        if low_level.get(key) is not None:
            nonempty_text(low_level, key, f"go2.low_level.{key}")

    mapping_version = (
        nonempty_text(low_level, "mapping_version", "go2.low_level.mapping_version")
        if low_level.get("mapping_version") is not None
        else None
    )
    mapping_hash = (
        nonempty_text(low_level, "mapping_hash", "go2.low_level.mapping_hash")
        if low_level.get("mapping_hash") is not None
        else None
    )
    mapping_digest = (
        None
        if mapping_hash is None or not mapping_hash.startswith("sha256:")
        else mapping_hash[len("sha256:") :]
    )
    if mapping_hash is not None and (
        not mapping_hash.startswith("sha256:")
        or mapping_digest is None
        or len(mapping_digest) != 64
        or any(char not in "0123456789abcdef" for char in mapping_digest)
    ):
        errors.append(
            "go2.low_level.mapping_hash must use the form sha256:<64 lowercase hex digits>"
        )

    joint_names = text_vector(low_level, "joint_names", "go2.low_level.joint_names")
    motor_ids = integer_vector(low_level, "motor_ids", "go2.low_level.motor_ids")
    directions = integer_vector(low_level, "directions", "go2.low_level.directions")
    zero_offsets_rad = number_vector(
        low_level, "zero_offsets_rad", "go2.low_level.zero_offsets_rad"
    )
    q_min_rad = number_vector(low_level, "q_min_rad", "go2.low_level.q_min_rad")
    q_max_rad = number_vector(low_level, "q_max_rad", "go2.low_level.q_max_rad")
    dq_max_rad_s = number_vector(
        low_level, "dq_max_rad_s", "go2.low_level.dq_max_rad_s", positive=True
    )
    number_vector(
        low_level,
        "maximum_delta_q_rad",
        "go2.low_level.maximum_delta_q_rad",
        positive=True,
    )
    kp = number_vector(low_level, "kp", "go2.low_level.kp", nonnegative=True)
    kd = number_vector(low_level, "kd", "go2.low_level.kd", nonnegative=True)
    tau_ff_nm = number_vector(low_level, "tau_ff_nm", "go2.low_level.tau_ff_nm")
    tau_limit_nm = number_vector(
        low_level, "tau_limit_nm", "go2.low_level.tau_limit_nm", positive=True
    )
    feedback_loss_degraded_kp = number_vector(
        low_level,
        "feedback_loss_degraded_kp",
        "go2.low_level.feedback_loss_degraded_kp",
        nonnegative=True,
    )
    feedback_loss_degraded_kd = number_vector(
        low_level,
        "feedback_loss_degraded_kd",
        "go2.low_level.feedback_loss_degraded_kd",
        nonnegative=True,
    )
    feedback_loss_degraded_tau_ff_nm = number_vector(
        low_level,
        "feedback_loss_degraded_tau_ff_nm",
        "go2.low_level.feedback_loss_degraded_tau_ff_nm",
    )
    firmware_torque_limit_nm = number_vector(
        low_level,
        "firmware_torque_limit_nm",
        "go2.low_level.firmware_torque_limit_nm",
        positive=True,
    )
    firmware_torque_clamp_verified = (
        boolean(
            low_level,
            "firmware_torque_clamp_verified",
            "go2.low_level.firmware_torque_clamp_verified",
        )
        if low_level.get("firmware_torque_clamp_verified") is not None
        else None
    )
    temperature_limit_c = number_vector(
        low_level,
        "temperature_limit_c",
        "go2.low_level.temperature_limit_c",
        positive=True,
    )
    safe_hold_pose_rad = number_vector(
        low_level,
        "safe_hold_pose_rad",
        "go2.low_level.safe_hold_pose_rad",
    )
    safe_hold_position_tolerance_rad = number_vector(
        low_level,
        "safe_hold_position_tolerance_rad",
        "go2.low_level.safe_hold_position_tolerance_rad",
        positive=True,
    )
    safe_hold_velocity_tolerance_rad_s = number_vector(
        low_level,
        "safe_hold_velocity_tolerance_rad_s",
        "go2.low_level.safe_hold_velocity_tolerance_rad_s",
        positive=True,
    )
    tracking_position_error_limit_rad = number_vector(
        low_level,
        "tracking_position_error_limit_rad",
        "go2.low_level.tracking_position_error_limit_rad",
        positive=True,
    )

    if joint_names is not None and len(set(joint_names)) != GO2_LOW_LEVEL_JOINT_COUNT:
        errors.append("go2.low_level.joint_names must be unique")
    if motor_ids is not None:
        if set(motor_ids) != set(range(GO2_LOW_LEVEL_JOINT_COUNT)):
            errors.append(
                "go2.low_level.motor_ids must be a permutation of the 12 Go2 leg slots 0..11"
            )
    if directions is not None and any(item not in (-1, 1) for item in directions):
        errors.append("go2.low_level.directions must contain only -1 or 1")

    if q_min_rad is not None and q_max_rad is not None:
        for index, (lower, upper) in enumerate(zip(q_min_rad, q_max_rad)):
            if lower >= upper:
                errors.append(
                    f"go2.low_level q_min_rad[{index}] must be less than q_max_rad[{index}]"
                )
    if safe_hold_pose_rad is not None and q_min_rad is not None and q_max_rad is not None:
        for index, (pose, lower, upper) in enumerate(zip(safe_hold_pose_rad, q_min_rad, q_max_rad)):
            if pose < lower or pose > upper:
                errors.append(f"go2.low_level.safe_hold_pose_rad[{index}] must be within q limits")
    if tau_ff_nm is not None and tau_limit_nm is not None:
        for index, (feedforward, limit) in enumerate(zip(tau_ff_nm, tau_limit_nm)):
            if abs(feedforward) > limit:
                errors.append(
                    f"go2.low_level.tau_ff_nm[{index}] must not exceed tau_limit_nm[{index}]"
                )
    if (
        feedback_loss_degraded_kp is not None
        and feedback_loss_degraded_kd is not None
        and feedback_loss_degraded_tau_ff_nm is not None
        and firmware_torque_limit_nm is not None
        and tau_limit_nm is not None
    ):
        for index in range(GO2_LOW_LEVEL_JOINT_COUNT):
            if kp is not None and feedback_loss_degraded_kp[index] > kp[index]:
                errors.append(
                    f"go2.low_level.feedback_loss_degraded_kp[{index}] must not exceed kp[{index}]"
                )
            if kd is not None and feedback_loss_degraded_kd[index] > kd[index]:
                errors.append(
                    f"go2.low_level.feedback_loss_degraded_kd[{index}] must not exceed kd[{index}]"
                )
            if firmware_torque_limit_nm[index] > tau_limit_nm[index]:
                errors.append(
                    f"go2.low_level.firmware_torque_limit_nm[{index}] must not exceed tau_limit_nm[{index}]"
                )
            if abs(feedback_loss_degraded_tau_ff_nm[index]) > firmware_torque_limit_nm[index]:
                errors.append(
                    "go2.low_level.feedback_loss_degraded_tau_ff_nm"
                    f"[{index}] must not exceed firmware_torque_limit_nm[{index}]"
                )
    if low_level_enabled is True and firmware_torque_clamp_verified is not True:
        errors.append(
            "go2.low_level.firmware_torque_clamp_verified must be true after "
            "a robot-specific torque-clamp test before low-level control is enabled"
        )
    if temperature_limit_c is not None and any(item > 150.0 for item in temperature_limit_c):
        errors.append("go2.low_level.temperature_limit_c must not exceed 150 C")
    if safe_hold_velocity_tolerance_rad_s is not None and dq_max_rad_s is not None:
        if any(
            tolerance > limit
            for tolerance, limit in zip(safe_hold_velocity_tolerance_rad_s, dq_max_rad_s)
        ):
            errors.append(
                "go2.low_level.safe_hold_velocity_tolerance_rad_s must not exceed dq_max_rad_s"
            )
    if (
        safe_hold_position_tolerance_rad is not None
        and q_min_rad is not None
        and q_max_rad is not None
        and any(
            tolerance >= upper - lower
            for tolerance, lower, upper in zip(
                safe_hold_position_tolerance_rad, q_min_rad, q_max_rad
            )
        )
    ):
        errors.append(
            "go2.low_level.safe_hold_position_tolerance_rad must be smaller than each q range"
        )
    if (
        tracking_position_error_limit_rad is not None
        and q_min_rad is not None
        and q_max_rad is not None
        and any(
            tolerance >= upper - lower
            for tolerance, lower, upper in zip(
                tracking_position_error_limit_rad, q_min_rad, q_max_rad
            )
        )
    ):
        errors.append(
            "go2.low_level.tracking_position_error_limit_rad must be smaller than each q range"
        )

    mapping_parts = (
        mapping_version,
        joint_names,
        motor_ids,
        directions,
        zero_offsets_rad,
    )
    if mapping_hash is not None and all(item is not None for item in mapping_parts):
        assert mapping_version is not None
        assert joint_names is not None
        assert motor_ids is not None
        assert directions is not None
        assert zero_offsets_rad is not None
        expected_mapping_hash = compute_go2_joint_mapping_hash(
            mapping_version,
            joint_names,
            motor_ids,
            directions,
            zero_offsets_rad,
        )
        if mapping_hash != expected_mapping_hash:
            errors.append(
                "go2.low_level.mapping_hash does not match the canonical reviewed mapping"
            )

    channel_keys = (
        "flight_enable_channel",
        "flight_mode_channel",
        "rtl_channel",
        "land_channel",
        "morphology_channel",
        "auto_landing_channel",
        "brake_channel",
        "buzzer_channel",
    )
    channels = [integer(rc, key, f"rc.{key}", minimum=1, maximum=16) for key in channel_keys]
    valid_channels = [item for item in channels if item is not None]
    if len(valid_channels) == len(channels) and len(set(valid_channels)) != len(valid_channels):
        errors.append("RC channel assignments must be positive, unique, and within 1..16")

    threshold_keys = ("low_max", "middle_min", "middle_max", "high_min")
    thresholds = [
        integer(rc, key, f"rc.{key}", minimum=800, maximum=2200) for key in threshold_keys
    ]
    if all(item is not None for item in thresholds):
        low_max, middle_min, middle_max, high_min = thresholds
        assert low_max is not None
        assert middle_min is not None
        assert middle_max is not None
        assert high_min is not None
        if not low_max < middle_min <= middle_max < high_min:
            errors.append("RC LOW/MIDDLE/HIGH thresholds overlap or are out of order")
    finite_number(rc, "debounce_s", "rc.debounce_s", positive=True)
    finite_number(rc, "timeout_s", "rc.timeout_s", positive=True)
    integer(
        rc,
        "manual_override_deadband_us",
        "rc.manual_override_deadband_us",
        minimum=1,
        maximum=700,
    )

    if "airborne_confirm_s" in safety:
        finite_number(
            safety,
            "airborne_confirm_s",
            "safety.airborne_confirm_s",
            positive=True,
        )

    for key in (
        "stationary_velocity_mps",
        "stationary_confirm_s",
        "maximum_safe_esc_rpm_for_transform",
        "touchdown_confirm_s",
        "touchdown_max_vertical_speed_mps",
        "touchdown_max_tilt_rad",
        "touchdown_max_height_delta_m",
        "touchdown_max_esc_rpm",
        "touchdown_max_source_age_s",
        "touchdown_max_source_skew_s",
        "post_touchdown_stable_confirm_s",
        "post_touchdown_stability_max_check_gap_s",
        "aborted_impact_airborne_confirm_s",
        "impact_recovery_status_max_age_s",
        "impact_recovery_completion_timeout_s",
        "impact_recovery_finalization_timeout_s",
        "pixhawk_timeout_s",
        "f446_timeout_s",
        "go2_timeout_s",
        "rc_timeout_s",
        "controller_timeout_s",
    ):
        safety_numbers[key] = finite_number(
            safety,
            key,
            f"safety.{key}",
            positive=True,
        )
    stable_confirm = safety_numbers.get("post_touchdown_stable_confirm_s")
    touchdown_confirm = safety_numbers.get("touchdown_confirm_s")
    stationary_confirm = safety_numbers.get("stationary_confirm_s")
    stability_gap = safety_numbers.get("post_touchdown_stability_max_check_gap_s")
    airborne_confirm = safety_numbers.get("aborted_impact_airborne_confirm_s")
    touchdown_source_age = safety_numbers.get("touchdown_max_source_age_s")
    touchdown_source_skew = safety_numbers.get("touchdown_max_source_skew_s")
    if stable_confirm is not None and stability_gap is not None and stability_gap >= stable_confirm:
        errors.append(
            "safety.post_touchdown_stability_max_check_gap_s must be less than "
            "post_touchdown_stable_confirm_s"
        )
    if (
        stable_confirm is not None
        and touchdown_source_age is not None
        and touchdown_source_age >= stable_confirm
    ):
        errors.append(
            "safety.touchdown_max_source_age_s must be less than post_touchdown_stable_confirm_s"
        )
    if (
        touchdown_confirm is not None
        and touchdown_source_age is not None
        and touchdown_source_age >= touchdown_confirm
    ):
        errors.append("safety.touchdown_max_source_age_s must be less than touchdown_confirm_s")
    if (
        stationary_confirm is not None
        and touchdown_source_age is not None
        and touchdown_source_age >= stationary_confirm
    ):
        errors.append("safety.touchdown_max_source_age_s must be less than stationary_confirm_s")
    if (
        touchdown_source_age is not None
        and touchdown_source_skew is not None
        and touchdown_source_skew > touchdown_source_age
    ):
        errors.append(
            "safety.touchdown_max_source_skew_s must be no greater than touchdown_max_source_age_s"
        )
    if (
        airborne_confirm is not None
        and touchdown_source_age is not None
        and touchdown_source_age >= airborne_confirm
    ):
        errors.append(
            "safety.touchdown_max_source_age_s must be less than aborted_impact_airborne_confirm_s"
        )
    if (
        airborne_confirm is not None
        and stability_gap is not None
        and stability_gap >= airborne_confirm
    ):
        errors.append(
            "safety.post_touchdown_stability_max_check_gap_s must be less than "
            "aborted_impact_airborne_confirm_s"
        )
    recovery_timeout = safety_numbers.get("impact_recovery_completion_timeout_s")
    finalization_timeout = safety_numbers.get("impact_recovery_finalization_timeout_s")
    if (
        recovery_timeout is not None
        and finalization_timeout is not None
        and stable_confirm is not None
        and recovery_timeout <= finalization_timeout + stable_confirm
    ):
        errors.append(
            "safety.impact_recovery_completion_timeout_s must exceed the sum of "
            "impact_recovery_finalization_timeout_s and post_touchdown_stable_confirm_s"
        )
    maximum_transform_current_adc = integer(
        safety,
        "maximum_transform_current_adc",
        "safety.maximum_transform_current_adc",
        minimum=1,
        maximum=4095,
    )
    if (
        transform_timeout_s is not None
        and firmware_timeout_ms is not None
        and firmware_timeout_ms > int(round(transform_timeout_s * 1000.0))
    ):
        errors.append("f446.firmware_timeout_ms must not exceed transform_timeout_s")
    if (
        firmware_timeout_ms is not None
        and stall_blanking_ms is not None
        and stall_overcurrent_ms is not None
        and stall_blanking_ms + stall_overcurrent_ms >= firmware_timeout_ms
    ):
        errors.append(
            "f446.stall_blanking_ms plus stall_overcurrent_ms must be less than firmware_timeout_ms"
        )
    if (
        automatic_stall_threshold_adc is not None
        and automatic_stall_threshold_adc != 0
        and current_safe_margin_adc is not None
        and maximum_transform_current_adc is not None
    ):
        safe_ceiling = maximum_transform_current_adc - current_safe_margin_adc
        if not current_safe_margin_adc < automatic_stall_threshold_adc <= safe_ceiling:
            errors.append(
                "f446.automatic_stall_threshold_adc must be 0 or within the host safety envelope "
                f"{current_safe_margin_adc + 1}..{safe_ceiling}"
            )

    for key in (
        "controller_hz",
        "maximum_descent_speed_mps",
        "maximum_horizontal_speed_mps",
        "maximum_yaw_rate_rad_s",
        "controller_timeout_s",
    ):
        finite_number(landing, key, f"landing.{key}", positive=True)
    integer(
        landing,
        "manual_override_deadband_us",
        "landing.manual_override_deadband_us",
        minimum=1,
        maximum=700,
    )
    nonempty_text(landing, "default_abort_mode", "landing.default_abort_mode")

    slots = [nonempty_text(esc, f"slot_{index}", f"esc.slot_{index}") for index in range(1, 5)]
    if all(item is not None for item in slots):
        actual_mapping = {index: str(slots[index - 1]) for index in range(1, 5)}
        if actual_mapping != X8_ESC_SLOT_MAPPING:
            errors.append("X8 ESC slots must match NodeID/ThrottleID order 1=RR, 2=LF, 3=LR, 4=RF")
    if "mavlink_display_shift" in esc:
        integer(
            esc,
            "mavlink_display_shift",
            "esc.mavlink_display_shift",
            minimum=0,
            maximum=1,
        )
    return errors


def validate_config(path: Path) -> Tuple[str, ...]:
    raw = _load_merged(path)
    return tuple(_validate_raw(raw))


def _build_config(source: Path, raw: Mapping[str, Any]) -> AppConfig:
    system = _section(raw, "system")
    pixhawk = _section(raw, "pixhawk")
    f446 = _section(raw, "f446")
    go2 = _section(raw, "go2")
    rc = _section(raw, "rc")
    safety = _section(raw, "safety")
    landing = _section(raw, "landing")
    esc = _section(raw, "esc")

    raw_low_level = go2.get("low_level", {})
    low_level: Mapping[str, Any] = raw_low_level if isinstance(raw_low_level, Mapping) else {}

    def optional_float_tuple(key: str) -> Optional[Tuple[float, ...]]:
        item = low_level.get(key)
        if item is None:
            return None
        return tuple(float(value) for value in item)

    def optional_int_tuple(key: str) -> Optional[Tuple[int, ...]]:
        item = low_level.get(key)
        if item is None:
            return None
        return tuple(int(value) for value in item)

    def optional_text_tuple(key: str) -> Optional[Tuple[str, ...]]:
        item = low_level.get(key)
        if item is None:
            return None
        return tuple(str(value) for value in item)

    def optional_float(key: str) -> Optional[float]:
        item = low_level.get(key)
        return None if item is None else float(item)

    def optional_text(key: str) -> Optional[str]:
        item = low_level.get(key)
        return None if item is None else str(item)

    def optional_bool(key: str) -> Optional[bool]:
        item = low_level.get(key)
        return None if item is None else bool(item)

    raw_foot_force_thresholds = go2.get(
        "foot_force_contact_thresholds",
        (0, 0, 0, 0),
    )
    foot_force_thresholds = (
        int(raw_foot_force_thresholds[0]),
        int(raw_foot_force_thresholds[1]),
        int(raw_foot_force_thresholds[2]),
        int(raw_foot_force_thresholds[3]),
    )

    log_directory = Path(_required(system, "log_directory"))
    if not log_directory.is_absolute():
        packaged_config_directory = Path(__file__).resolve().parents[1] / "default_configs"
        if source.resolve().parent == packaged_config_directory.resolve():
            # Installed package data may be read-only. Runtime artifacts belong
            # to the operator's working directory, not site-packages.
            log_directory = Path.cwd() / log_directory
        else:
            log_directory = source.parent.parent / log_directory

    return AppConfig(
        source_path=source,
        system=SystemConfig(
            loop_hz=float(_required(system, "loop_hz")),
            dry_run=_required(system, "dry_run"),
            hardware_write_enabled=_required(system, "hardware_write_enabled"),
            log_directory=log_directory.resolve(),
        ),
        pixhawk=PixhawkConfig(
            connection=_required(pixhawk, "connection"),
            baud=_required(pixhawk, "baud"),
            heartbeat_timeout_s=float(_required(pixhawk, "heartbeat_timeout_s")),
            target_system=_required(pixhawk, "target_system"),
            target_component=_required(pixhawk, "target_component"),
            rc9_option=_required(pixhawk, "rc9_option"),
            rc10_option=_required(pixhawk, "rc10_option"),
        ),
        f446=F446Config(
            port=_required(f446, "port"),
            baud=_required(f446, "baud"),
            flight_direction=_required(f446, "flight_direction").lower(),
            walk_direction=_required(f446, "walk_direction").lower(),
            flight_duty=_required(f446, "flight_duty"),
            walk_duty=_required(f446, "walk_duty"),
            expected_flight_state=F446State[_required(f446, "expected_flight_state")],
            expected_walk_state=F446State[_required(f446, "expected_walk_state")],
            response_timeout_s=float(_required(f446, "response_timeout_s")),
            transform_timeout_s=float(_required(f446, "transform_timeout_s")),
            status_poll_hz=float(_required(f446, "status_poll_hz")),
            firmware_timeout_ms=int(
                f446.get(
                    "firmware_timeout_ms",
                    round(float(_required(f446, "transform_timeout_s")) * 1000.0),
                )
            ),
            automatic_stall_threshold_adc=int(f446.get("automatic_stall_threshold_adc", 0)),
            stall_blanking_ms=int(f446.get("stall_blanking_ms", 500)),
            stall_overcurrent_ms=int(f446.get("stall_overcurrent_ms", 180)),
            current_safe_margin_adc=_required(f446, "current_safe_margin_adc"),
            current_clear_hold_s=float(_required(f446, "current_clear_hold_s")),
        ),
        go2=Go2Config(
            enabled=_required(go2, "enabled"),
            status_timeout_s=float(_required(go2, "status_timeout_s")),
            network_interface=str(go2.get("network_interface", "eth0")),
            domain_id=int(go2.get("domain_id", 0)),
            sport_state_topic=str(go2.get("sport_state_topic", "rt/sportmodestate")),
            command_timeout_s=float(go2.get("command_timeout_s", 2.0)),
            joint_lock_operator_timeout_s=float(go2.get("joint_lock_operator_timeout_s", 60.0)),
            joint_lock_transition_grace_s=float(go2.get("joint_lock_transition_grace_s", 2.0)),
            joint_lock_unsafe_confirm_s=float(go2.get("joint_lock_unsafe_confirm_s", 0.5)),
            joint_lock_state_codes=tuple(
                int(item) for item in go2.get("joint_lock_state_codes", (1002,))
            ),
            accepted_state_codes=tuple(
                int(item) for item in go2.get("accepted_state_codes", (0, 100, 1002))
            ),
            landing_compliance_enabled=bool(go2.get("landing_compliance_enabled", False)),
            foot_force_contact_thresholds=foot_force_thresholds,
            landing_contact_min_feet=int(go2.get("landing_contact_min_feet", 3)),
            landing_contact_confirm_s=float(go2.get("landing_contact_confirm_s", 0.5)),
            landing_compliance_settle_s=float(go2.get("landing_compliance_settle_s", 1.5)),
            low_level=Go2LowLevelConfig(
                enabled=bool(low_level.get("enabled", False)),
                observe_only_enabled=bool(low_level.get("observe_only_enabled", False)),
                low_state_topic=optional_text("low_state_topic"),
                low_command_topic=optional_text("low_command_topic"),
                send_period_s=optional_float("send_period_s"),
                maximum_jitter_s=optional_float("maximum_jitter_s"),
                low_state_max_age_s=optional_float("low_state_max_age_s"),
                target_ttl_s=optional_float("target_ttl_s"),
                acquire_timeout_s=optional_float("acquire_timeout_s"),
                release_timeout_s=optional_float("release_timeout_s"),
                safe_hold_policy=optional_text("safe_hold_policy"),
                safe_hold_pose_rad=optional_float_tuple("safe_hold_pose_rad"),
                safe_hold_position_tolerance_rad=optional_float_tuple(
                    "safe_hold_position_tolerance_rad"
                ),
                safe_hold_velocity_tolerance_rad_s=optional_float_tuple(
                    "safe_hold_velocity_tolerance_rad_s"
                ),
                tracking_position_error_limit_rad=optional_float_tuple(
                    "tracking_position_error_limit_rad"
                ),
                safe_hold_ack_timeout_s=optional_float("safe_hold_ack_timeout_s"),
                restore_mode_form=optional_text("restore_mode_form"),
                restore_mode_name=optional_text("restore_mode_name"),
                mapping_version=optional_text("mapping_version"),
                mapping_hash=optional_text("mapping_hash"),
                joint_names=optional_text_tuple("joint_names"),
                motor_ids=optional_int_tuple("motor_ids"),
                directions=optional_int_tuple("directions"),
                zero_offsets_rad=optional_float_tuple("zero_offsets_rad"),
                q_min_rad=optional_float_tuple("q_min_rad"),
                q_max_rad=optional_float_tuple("q_max_rad"),
                dq_max_rad_s=optional_float_tuple("dq_max_rad_s"),
                maximum_delta_q_rad=optional_float_tuple("maximum_delta_q_rad"),
                kp=optional_float_tuple("kp"),
                kd=optional_float_tuple("kd"),
                tau_ff_nm=optional_float_tuple("tau_ff_nm"),
                tau_limit_nm=optional_float_tuple("tau_limit_nm"),
                feedback_loss_degraded_kp=optional_float_tuple("feedback_loss_degraded_kp"),
                feedback_loss_degraded_kd=optional_float_tuple("feedback_loss_degraded_kd"),
                feedback_loss_degraded_tau_ff_nm=optional_float_tuple(
                    "feedback_loss_degraded_tau_ff_nm"
                ),
                firmware_torque_limit_nm=optional_float_tuple("firmware_torque_limit_nm"),
                firmware_torque_clamp_verified=optional_bool("firmware_torque_clamp_verified"),
                temperature_limit_c=optional_float_tuple("temperature_limit_c"),
            ),
        ),
        rc=RCConfig(
            flight_enable_channel=_required(rc, "flight_enable_channel"),
            flight_mode_channel=_required(rc, "flight_mode_channel"),
            rtl_channel=_required(rc, "rtl_channel"),
            land_channel=_required(rc, "land_channel"),
            morphology_channel=_required(rc, "morphology_channel"),
            auto_landing_channel=_required(rc, "auto_landing_channel"),
            brake_channel=_required(rc, "brake_channel"),
            buzzer_channel=_required(rc, "buzzer_channel"),
            low_max=_required(rc, "low_max"),
            middle_min=_required(rc, "middle_min"),
            middle_max=_required(rc, "middle_max"),
            high_min=_required(rc, "high_min"),
            debounce_s=float(_required(rc, "debounce_s")),
            timeout_s=float(_required(rc, "timeout_s")),
            manual_override_deadband_us=_required(rc, "manual_override_deadband_us"),
        ),
        safety=SafetyConfig(
            stationary_velocity_mps=float(_required(safety, "stationary_velocity_mps")),
            stationary_confirm_s=float(_required(safety, "stationary_confirm_s")),
            maximum_safe_esc_rpm_for_transform=float(
                _required(safety, "maximum_safe_esc_rpm_for_transform")
            ),
            airborne_confirm_s=float(safety.get("airborne_confirm_s", 1.0)),
            touchdown_confirm_s=float(_required(safety, "touchdown_confirm_s")),
            touchdown_max_vertical_speed_mps=float(
                _required(safety, "touchdown_max_vertical_speed_mps")
            ),
            touchdown_max_tilt_rad=float(_required(safety, "touchdown_max_tilt_rad")),
            touchdown_max_height_delta_m=float(_required(safety, "touchdown_max_height_delta_m")),
            touchdown_max_esc_rpm=float(_required(safety, "touchdown_max_esc_rpm")),
            touchdown_max_source_age_s=float(_required(safety, "touchdown_max_source_age_s")),
            touchdown_max_source_skew_s=float(_required(safety, "touchdown_max_source_skew_s")),
            post_touchdown_stable_confirm_s=float(
                _required(safety, "post_touchdown_stable_confirm_s")
            ),
            post_touchdown_stability_max_check_gap_s=float(
                _required(safety, "post_touchdown_stability_max_check_gap_s")
            ),
            aborted_impact_airborne_confirm_s=float(
                _required(safety, "aborted_impact_airborne_confirm_s")
            ),
            impact_recovery_status_max_age_s=float(
                _required(safety, "impact_recovery_status_max_age_s")
            ),
            impact_recovery_completion_timeout_s=float(
                _required(safety, "impact_recovery_completion_timeout_s")
            ),
            impact_recovery_finalization_timeout_s=float(
                _required(safety, "impact_recovery_finalization_timeout_s")
            ),
            pixhawk_timeout_s=float(_required(safety, "pixhawk_timeout_s")),
            f446_timeout_s=float(_required(safety, "f446_timeout_s")),
            go2_timeout_s=float(_required(safety, "go2_timeout_s")),
            rc_timeout_s=float(_required(safety, "rc_timeout_s")),
            controller_timeout_s=float(_required(safety, "controller_timeout_s")),
            maximum_transform_current_adc=_required(
                safety,
                "maximum_transform_current_adc",
            ),
        ),
        landing=LandingConfig(
            controller_hz=float(_required(landing, "controller_hz")),
            maximum_descent_speed_mps=float(_required(landing, "maximum_descent_speed_mps")),
            maximum_horizontal_speed_mps=float(_required(landing, "maximum_horizontal_speed_mps")),
            maximum_yaw_rate_rad_s=float(_required(landing, "maximum_yaw_rate_rad_s")),
            controller_timeout_s=float(_required(landing, "controller_timeout_s")),
            manual_override_deadband_us=_required(landing, "manual_override_deadband_us"),
            default_abort_mode=_required(landing, "default_abort_mode"),
        ),
        esc=EscConfig(
            slots={index: _required(esc, f"slot_{index}") for index in range(1, 5)},
            mavlink_display_shift=int(esc.get("mavlink_display_shift", 0)),
        ),
        raw=raw,
    )


def load_config(path: Path) -> AppConfig:
    source = Path(path).resolve()
    raw = _load_merged(source)
    errors = _validate_raw(raw)
    if errors:
        raise ConfigurationError("; ".join(errors))
    try:
        return _build_config(source, raw)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration value: {exc}") from exc
