from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.immutable import deep_thaw
from aerogo2.tools.configure_landing_compliance import configure_landing_compliance


def _standalone_config(tmp_path: Path, app_config: AppConfig) -> Path:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        yaml.safe_dump(deep_thaw(app_config.raw), sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_configuration_tool_validates_backs_up_and_atomically_enables(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = _standalone_config(tmp_path, app_config)

    backup = configure_landing_compliance(
        path,
        enabled=True,
        thresholds=(101, 202, 303, 404),
        minimum_contact_feet=3,
        contact_confirm_s=0.6,
        settle_s=1.7,
    )

    loaded = load_config(path)
    original = load_config(backup)
    assert loaded.go2.landing_compliance_enabled
    assert loaded.go2.foot_force_contact_thresholds == (101, 202, 303, 404)
    assert loaded.go2.landing_contact_min_feet == 3
    assert loaded.go2.landing_contact_confirm_s == 0.6
    assert loaded.go2.landing_compliance_settle_s == 1.7
    assert not original.go2.landing_compliance_enabled


def test_invalid_thresholds_leave_configuration_unchanged(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = _standalone_config(tmp_path, app_config)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="1..32767"):
        configure_landing_compliance(
            path,
            enabled=True,
            thresholds=(10, 20, 0, 40),
        )

    assert path.read_bytes() == before
    assert not (tmp_path / "hardware.yaml.bak").exists()
