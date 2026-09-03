from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pytest
import yaml

from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.common.immutable import deep_thaw


def test_loads_included_configuration(app_config: AppConfig) -> None:
    assert app_config.system.dry_run is True
    assert app_config.system.hardware_write_enabled is False
    assert app_config.f446.flight_direction == "forward"
    assert app_config.rc.morphology_channel == 9
    assert app_config.f446.walk_duty == 300
    assert app_config.f446.transform_timeout_s == 15.0
    assert app_config.f446.firmware_timeout_ms == 15000
    assert app_config.f446.automatic_stall_threshold_adc == 0
    assert app_config.f446.stall_blanking_ms == 500
    assert app_config.f446.stall_overcurrent_ms == 180
    assert app_config.safety.airborne_confirm_s == 1.0
    assert app_config.go2.joint_lock_transition_grace_s == 2.0
    assert app_config.go2.joint_lock_unsafe_confirm_s == 0.5
    assert app_config.go2.joint_lock_state_codes == (1002,)
    assert app_config.go2.accepted_state_codes == (0, 100, 1002)


def test_non_dry_configuration_can_describe_hardware_write_capability(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    raw["system"]["dry_run"] = False
    raw["system"]["hardware_write_enabled"] = True
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.system.dry_run is False
    assert loaded.system.hardware_write_enabled is True


def test_dot_get_returns_nested_value(app_config: AppConfig) -> None:
    assert app_config.get("safety.f446_timeout_s") == 0.5
    assert app_config.get("does.not.exist", "fallback") == "fallback"


def test_f446_mapping_is_configuration_driven(app_config: AppConfig) -> None:
    assert app_config.f446.direction_for("FLIGHT") == "forward"
    assert app_config.f446.direction_for("WALK") == "reverse"
    assert app_config.f446.duty_for("FLIGHT") == 120


def test_esc_mapping_is_fixed_and_unique(app_config: AppConfig) -> None:
    assert app_config.esc.slots == {1: "RR", 2: "LF", 3: "LR", 4: "RF"}
    assert app_config.esc.mavlink_display_shift == 0


def test_missing_airborne_confirmation_defaults_to_one_second(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    del raw["safety"]["airborne_confirm_s"]
    path = tmp_path / "legacy-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.safety.airborne_confirm_s == 1.0


def test_missing_go2_accepted_state_codes_uses_firmware_compatibility_default(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    del raw["go2"]["accepted_state_codes"]
    path = tmp_path / "legacy-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.go2.accepted_state_codes == (0, 100, 1002)


def test_missing_joint_lock_filter_settings_use_safe_compatibility_defaults(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    del raw["go2"]["joint_lock_transition_grace_s"]
    del raw["go2"]["joint_lock_unsafe_confirm_s"]
    del raw["go2"]["joint_lock_state_codes"]
    path = tmp_path / "legacy-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.go2.joint_lock_transition_grace_s == 2.0
    assert loaded.go2.joint_lock_unsafe_confirm_s == 0.5
    assert loaded.go2.joint_lock_state_codes == (1002,)


def test_duplicate_go2_accepted_state_code_is_rejected(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    raw["go2"]["accepted_state_codes"] = [0, 1002, 1002]
    path = tmp_path / "unsafe-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate"):
        load_config(path)


def test_duplicate_go2_joint_lock_state_code_is_rejected(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    raw["go2"]["joint_lock_state_codes"] = [1002, 1002]
    path = tmp_path / "unsafe-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate"):
        load_config(path)


def test_missing_esc_display_shift_defaults_to_zero(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    del raw["esc"]["mavlink_display_shift"]
    path = tmp_path / "legacy-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.esc.mavlink_display_shift == 0


Mutator = Callable[[Dict[str, Any]], None]


def _same_direction(raw: Dict[str, Any]) -> None:
    raw["f446"]["walk_direction"] = "forward"


def _bad_expected_flight(raw: Dict[str, Any]) -> None:
    raw["f446"]["expected_flight_state"] = "LIMIT_REACHED_REV"


def _zero_duty(raw: Dict[str, Any]) -> None:
    raw["f446"]["flight_duty"] = 0


def _excess_duty(raw: Dict[str, Any]) -> None:
    raw["f446"]["walk_duty"] = 351


def _zero_timeout(raw: Dict[str, Any]) -> None:
    raw["f446"]["transform_timeout_s"] = 0


def _firmware_timeout_exceeds_host(raw: Dict[str, Any]) -> None:
    raw["f446"]["firmware_timeout_ms"] = 16000


def _unsafe_persistent_threshold(raw: Dict[str, Any]) -> None:
    raw["f446"]["automatic_stall_threshold_adc"] = 1800


def _invalid_timing_combination(raw: Dict[str, Any]) -> None:
    raw["f446"]["stall_blanking_ms"] = 14900


def _duplicate_rc(raw: Dict[str, Any]) -> None:
    raw["rc"]["morphology_channel"] = raw["rc"]["auto_landing_channel"]


def _overlap_rc(raw: Dict[str, Any]) -> None:
    raw["rc"]["middle_min"] = 1100


def _rc9_option(raw: Dict[str, Any]) -> None:
    raw["pixhawk"]["rc9_option"] = 42


def _duplicate_esc(raw: Dict[str, Any]) -> None:
    raw["esc"]["slot_4"] = "LR"


def _invalid_esc_display_shift(raw: Dict[str, Any]) -> None:
    raw["esc"]["mavlink_display_shift"] = 2


def _zero_safety_timeout(raw: Dict[str, Any]) -> None:
    raw["safety"]["pixhawk_timeout_s"] = 0


def _zero_airborne_confirmation(raw: Dict[str, Any]) -> None:
    raw["safety"]["airborne_confirm_s"] = 0


def _negative_landing_speed(raw: Dict[str, Any]) -> None:
    raw["landing"]["maximum_descent_speed_mps"] = -0.1


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (_same_direction, "directions"),
        (_bad_expected_flight, "expected_flight_state"),
        (_zero_duty, "flight_duty"),
        (_excess_duty, "walk_duty"),
        (_zero_timeout, "transform_timeout_s"),
        (_duplicate_rc, "channel assignments"),
        (_firmware_timeout_exceeds_host, "must not exceed"),
        (_unsafe_persistent_threshold, "safety envelope"),
        (_invalid_timing_combination, "stall_blanking_ms"),
        (_overlap_rc, "thresholds"),
        (_rc9_option, "RC9_OPTION"),
        (_duplicate_esc, "ESC slots"),
        (_invalid_esc_display_shift, "mavlink_display_shift"),
        (_zero_safety_timeout, "pixhawk_timeout_s"),
        (_zero_airborne_confirmation, "airborne_confirm_s"),
        (_negative_landing_speed, "maximum_descent_speed_mps"),
    ],
)
def test_unsafe_configuration_is_rejected(
    tmp_path: Path,
    app_config: AppConfig,
    mutator: Mutator,
    expected: str,
) -> None:
    raw = deep_thaw(app_config.raw)
    mutator(raw)
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=expected):
        load_config(path)


def test_legacy_f446_timing_keys_are_derived_safely(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    raw["f446"]["transform_timeout_s"] = 6.0
    for key in (
        "firmware_timeout_ms",
        "automatic_stall_threshold_adc",
        "stall_blanking_ms",
        "stall_overcurrent_ms",
    ):
        del raw["f446"][key]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.f446.firmware_timeout_ms == 6000
    assert loaded.f446.automatic_stall_threshold_adc == 0


def test_include_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("includes: [second.yaml]\n", encoding="utf-8")
    second.write_text("includes: [first.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="include cycle"):
        load_config(first)


def test_missing_file_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
