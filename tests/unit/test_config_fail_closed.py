from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from aerogo2.common.config import AppConfig, load_config, validate_config
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.common.immutable import deep_thaw


def _write_config(
    tmp_path: Path,
    app_config: AppConfig,
    dotted_key: str,
    value: Any,
) -> Path:
    raw = deep_thaw(app_config.raw)
    parts = dotted_key.split(".")
    current: dict[str, Any] = raw
    for part in parts[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = child
    current[parts[-1]] = value
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("system.loop_hz", float("nan"), "system.loop_hz must be finite"),
        (
            "f446.transform_timeout_s",
            float("inf"),
            "f446.transform_timeout_s must be finite",
        ),
        (
            "landing.maximum_descent_speed_mps",
            float("-inf"),
            "landing.maximum_descent_speed_mps must be finite",
        ),
        ("safety.rc_timeout_s", "fast", "must be a finite number"),
    ],
)
def test_nonfinite_and_nonnumeric_values_fail_closed(
    tmp_path: Path,
    app_config: AppConfig,
    key: str,
    value: Any,
    message: str,
) -> None:
    path = _write_config(tmp_path, app_config, key, value)

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)

    assert any(message in error for error in validate_config(path))


@pytest.mark.parametrize(
    "key",
    [
        "system.dry_run",
        "system.hardware_write_enabled",
        "go2.enabled",
    ],
)
def test_boolean_strings_are_never_coerced(
    tmp_path: Path,
    app_config: AppConfig,
    key: str,
) -> None:
    path = _write_config(tmp_path, app_config, key, "false")

    with pytest.raises(ConfigurationError, match="must be a YAML boolean"):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("f446.flight_duty", True, "must be an integer"),
        ("f446.walk_duty", 120.5, "must be an integer"),
        ("f446.flight_duty", 0, "at least 1"),
        ("f446.walk_duty", 351, "at most 350"),
        ("rc.flight_enable_channel", 0, "at least 1"),
        ("rc.flight_enable_channel", 17, "at most 16"),
        ("rc.flight_enable_channel", 5.5, "must be an integer"),
        ("rc.low_max", 799, "at least 800"),
        ("rc.high_min", 2201, "at most 2200"),
        ("pixhawk.baud", True, "must be an integer"),
        ("esc.mavlink_display_shift", True, "must be an integer"),
        ("esc.mavlink_display_shift", -1, "at least 0"),
        ("esc.mavlink_display_shift", 2, "at most 1"),
    ],
)
def test_integer_and_protocol_bounds_are_strict(
    tmp_path: Path,
    app_config: AppConfig,
    key: str,
    value: Any,
    message: str,
) -> None:
    path = _write_config(tmp_path, app_config, key, value)

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_missing_section_and_required_key_are_configuration_errors(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw = deep_thaw(app_config.raw)
    del raw["go2"]
    missing_section = tmp_path / "missing-section.yaml"
    missing_section.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Missing or invalid 'go2' section"):
        load_config(missing_section)

    raw = deep_thaw(app_config.raw)
    system = raw["system"]
    assert isinstance(system, dict)
    del system["loop_hz"]
    missing_key = tmp_path / "missing-key.yaml"
    missing_key.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="system.loop_hz"):
        load_config(missing_key)


@pytest.mark.parametrize(
    "payload",
    [
        "system: [unterminated\n",
        "- system\n- pixhawk\n",
    ],
)
def test_malformed_or_non_mapping_yaml_is_configuration_error(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_loaded_boolean_and_integer_types_remain_exact(app_config: AppConfig) -> None:
    values: Mapping[str, Any] = {
        "dry_run": app_config.system.dry_run,
        "hardware_write_enabled": app_config.system.hardware_write_enabled,
        "go2_enabled": app_config.go2.enabled,
        "flight_duty": app_config.f446.flight_duty,
        "flight_enable_channel": app_config.rc.flight_enable_channel,
    }

    assert type(values["dry_run"]) is bool
    assert type(values["hardware_write_enabled"]) is bool
    assert type(values["go2_enabled"]) is bool
    assert type(values["flight_duty"]) is int
    assert type(values["flight_enable_channel"]) is int


def test_landing_compliance_rejects_uncalibrated_zero_thresholds(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = _write_config(
        tmp_path,
        app_config,
        "go2.landing_compliance_enabled",
        True,
    )

    with pytest.raises(ConfigurationError, match="must all be positive"):
        load_config(path)


def test_foot_force_thresholds_require_exactly_four_integers(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = _write_config(
        tmp_path,
        app_config,
        "go2.foot_force_contact_thresholds",
        [10, 20, 30],
    )

    with pytest.raises(ConfigurationError, match="exactly four integers"):
        load_config(path)
