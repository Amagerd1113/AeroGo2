from __future__ import annotations

from pathlib import Path

import pytest

from aerogo2.common.config import load_config
from aerogo2.main import _default_config_path, build_parser


def test_default_config_falls_back_to_packaged_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    path = _default_config_path()
    config = load_config(path)

    assert path.name == "system.yaml"
    assert path.parent.name == "default_configs"
    assert config.system.dry_run is True
    assert config.system.hardware_write_enabled is False
    assert config.pixhawk.connection == "/dev/ttyACM0"
    assert config.pixhawk.baud == 115200
    assert config.pixhawk.heartbeat_timeout_s == 30.0
    assert config.esc.slots == {1: "RR", 2: "LF", 3: "LR", 4: "RF"}
    assert config.esc.mavlink_display_shift == 0
    assert config.system.log_directory == (tmp_path / "logs").resolve()


def test_demo_parser_uses_packaged_config_outside_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["demo"])

    assert args.config == _default_config_path()
