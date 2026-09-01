from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.immutable import deep_thaw
from aerogo2.tools.configure_f446 import configure_f446, main


def _write_config(path: Path, app_config: AppConfig) -> None:
    path.write_text(yaml.safe_dump(deep_thaw(app_config.raw), sort_keys=False), encoding="utf-8")


def test_configure_f446_updates_and_backs_up_atomically(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = tmp_path / "hardware.yaml"
    _write_config(path, app_config)

    backup, configured = configure_f446(
        path,
        walk_duty=300,
        transform_timeout_s=15.0,
        threshold_adc=900,
        blanking_ms=450,
        overcurrent_ms=200,
    )

    loaded = load_config(path)
    original = load_config(backup)
    assert configured.f446.walk_duty == loaded.f446.walk_duty == 300
    assert loaded.f446.transform_timeout_s == 15.0
    assert loaded.f446.firmware_timeout_ms == 15000
    assert loaded.f446.automatic_stall_threshold_adc == 900
    assert loaded.f446.stall_blanking_ms == 450
    assert loaded.f446.stall_overcurrent_ms == 200
    assert original.f446.automatic_stall_threshold_adc == 0


def test_configure_f446_rejects_unsafe_threshold_without_touching_file(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = tmp_path / "hardware.yaml"
    _write_config(path, app_config)
    before = path.read_bytes()

    with pytest.raises(Exception, match="safety envelope"):
        configure_f446(path, threshold_adc=1800)

    assert path.read_bytes() == before
    assert not (tmp_path / "hardware.yaml.bak").exists()


def test_configure_f446_requires_at_least_one_change(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = tmp_path / "hardware.yaml"
    _write_config(path, app_config)

    with pytest.raises(ValueError, match="At least one"):
        configure_f446(path)


def test_configure_f446_cli_reports_effective_values(
    tmp_path: Path,
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "hardware.yaml"
    _write_config(path, app_config)

    result = main(
        [
            "--config",
            str(path),
            "--walk-duty",
            "300",
            "--transform-timeout-s",
            "15",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "walk_duty=300" in output
    assert "firmware_timeout_ms=15000" in output
