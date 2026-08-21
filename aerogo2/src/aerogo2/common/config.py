"""YAML configuration loading, merging, and safety validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import yaml

from aerogo2.common.enums import F446State
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.common.immutable import frozen_mapping

# Canonical AeroGo2 labels for the Hobbywing X8 NodeID/ThrottleID order.
# The validated diagnostic uses FR/FL/RL/RR while the existing AeroGo2 model
# uses RF/LF/LR/RR; only the spelling differs.
X8_ESC_SLOT_MAPPING = {1: "RR", 2: "LF", 3: "LR", 4: "RF"}


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
class Go2Config:
    enabled: bool
    status_timeout_s: float
    network_interface: str = "eth0"
    domain_id: int = 0
    sport_state_topic: str = "rt/sportmodestate"
    command_timeout_s: float = 2.0
    landing_compliance_enabled: bool = False
    foot_force_contact_thresholds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    landing_contact_min_feet: int = 3
    landing_contact_confirm_s: float = 0.5
    landing_compliance_settle_s: float = 1.5


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
    touchdown_confirm_s: float
    touchdown_max_vertical_speed_mps: float
    touchdown_max_tilt_rad: float
    touchdown_max_height_delta_m: float
    touchdown_max_esc_rpm: float
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


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigurationError(f"Top level of {path} must be a mapping")
    return loaded


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
    system = _section(raw, "system")
    pixhawk = _section(raw, "pixhawk")
    f446 = _section(raw, "f446")
    go2 = _section(raw, "go2")
    rc = _section(raw, "rc")
    safety = _section(raw, "safety")
    landing = _section(raw, "landing")
    esc = _section(raw, "esc")

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
    for key in ("response_timeout_s", "transform_timeout_s", "status_poll_hz"):
        finite_number(f446, key, f"f446.{key}", positive=True)
    integer(
        f446,
        "current_safe_margin_adc",
        "f446.current_safe_margin_adc",
        minimum=0,
        maximum=4095,
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

    for key in (
        "stationary_velocity_mps",
        "stationary_confirm_s",
        "maximum_safe_esc_rpm_for_transform",
        "touchdown_confirm_s",
        "touchdown_max_vertical_speed_mps",
        "touchdown_max_tilt_rad",
        "touchdown_max_height_delta_m",
        "touchdown_max_esc_rpm",
        "pixhawk_timeout_s",
        "f446_timeout_s",
        "go2_timeout_s",
        "rc_timeout_s",
        "controller_timeout_s",
    ):
        finite_number(safety, key, f"safety.{key}", positive=True)
    integer(
        safety,
        "maximum_transform_current_adc",
        "safety.maximum_transform_current_adc",
        minimum=1,
        maximum=4095,
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
            landing_compliance_enabled=bool(go2.get("landing_compliance_enabled", False)),
            foot_force_contact_thresholds=foot_force_thresholds,
            landing_contact_min_feet=int(go2.get("landing_contact_min_feet", 3)),
            landing_contact_confirm_s=float(go2.get("landing_contact_confirm_s", 0.5)),
            landing_compliance_settle_s=float(go2.get("landing_compliance_settle_s", 1.5)),
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
            touchdown_confirm_s=float(_required(safety, "touchdown_confirm_s")),
            touchdown_max_vertical_speed_mps=float(
                _required(safety, "touchdown_max_vertical_speed_mps")
            ),
            touchdown_max_tilt_rad=float(_required(safety, "touchdown_max_tilt_rad")),
            touchdown_max_height_delta_m=float(_required(safety, "touchdown_max_height_delta_m")),
            touchdown_max_esc_rpm=float(_required(safety, "touchdown_max_esc_rpm")),
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
        esc=EscConfig(slots={index: _required(esc, f"slot_{index}") for index in range(1, 5)}),
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
