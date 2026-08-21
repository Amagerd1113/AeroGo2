from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import aerogo2.main as main_module
from aerogo2.common.config import AppConfig, PixhawkConfig
from aerogo2.main import async_main, build_parser
from aerogo2.x8_bench import (
    X8BenchError,
    build_diag_command,
    build_x8_spin_args,
    check_x8_alignment,
)


def test_x8_alignment_matches_canonical_diag(app_config: AppConfig) -> None:
    report = check_x8_alignment(app_config)

    assert report.ok
    assert report.errors == ()
    assert report.script_path.name == "pixhawk_x8_cli_diag.py"
    assert "x8 slots={1: 'RR', 2: 'LF', 3: 'LR', 4: 'RF'}" in report.details
    assert "standard parameter audit=17 critical values" in report.details


def test_explicit_missing_diag_fails_closed(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing_diag.py"

    with pytest.raises(X8BenchError, match="does not exist"):
        check_x8_alignment(app_config, missing)


def test_x8_alignment_rejects_transport_drift(app_config: AppConfig) -> None:
    drifted_pixhawk = replace(
        app_config.pixhawk,
        baud=57600,
        heartbeat_timeout_s=2.0,
    )
    drifted = replace(app_config, pixhawk=drifted_pixhawk)

    report = check_x8_alignment(drifted)

    assert not report.ok
    assert any("pixhawk.baud=57600" in error for error in report.errors)
    assert any("heartbeat_timeout_s=2.0" in error for error in report.errors)


def test_build_diag_command_injects_aligned_transport(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    script = tmp_path / "diag.py"
    command = build_diag_command(
        app_config,
        script,
        ("--", "--commands", "streams on; audit std; x8diag 3"),
    )

    assert command[1] == str(script)
    assert command[2:8] == (
        "--port",
        "/dev/ttyACM0",
        "--baud",
        "115200",
        "--connect-timeout",
        "30.0",
    )
    assert command[8:] == (
        "--commands",
        "streams on; audit std; x8diag 3",
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--port", "/dev/ttyUSB0"),
        ("--baud=57600",),
        ("--connect-timeout", "2"),
    ],
)
def test_build_diag_command_rejects_transport_override(
    app_config: AppConfig,
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(X8BenchError, match="managed by AeroGo2 config"):
        build_diag_command(app_config, tmp_path / "diag.py", arguments)


def test_x8_bench_parser_keeps_diag_arguments() -> None:
    args = build_parser().parse_args(
        [
            "x8-bench",
            "--check",
            "--",
            "--self-test",
        ]
    )

    assert args.command == "x8-bench"
    assert args.check is True
    assert args.diag_args == ["--", "--self-test"]


@pytest.mark.asyncio
async def test_x8_bench_check_does_not_open_hardware(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(
        [
            "x8-bench",
            "--config",
            str(app_config.source_path),
            "--check",
        ]
    )

    result = await async_main(args)

    assert result == 0
    assert "X8 alignment: PASS" in capsys.readouterr().out


def test_pixhawk_dataclass_remains_replaceable(app_config: AppConfig) -> None:
    updated: PixhawkConfig = replace(app_config.pixhawk, connection="/dev/ttyUSB0")

    assert updated.connection == "/dev/ttyUSB0"


def test_build_x8_spin_args_is_bounded_and_stops() -> None:
    arguments = build_x8_spin_args("lf", 10.0, 2.0)

    assert arguments[0] == "--commands"
    commands = arguments[1]
    assert "audit std" in commands
    assert "x8diag 2" in commands
    assert "triggercheck fl 2" in commands
    assert "triggerwin fl 1100 2.00" in commands
    assert commands.endswith("trigger off; safety on")
    assert arguments[2:] == ("--command-settle", "0.2", "--post-listen", "1")


@pytest.mark.parametrize(
    ("target", "percent", "duration"),
    [
        ("invalid", 10.0, 2.0),
        ("lf", 4.9, 2.0),
        ("lf", 20.1, 2.0),
        ("lf", float("nan"), 2.0),
        ("lf", 10.0, 0.4),
        ("lf", 10.0, 5.1),
    ],
)
def test_build_x8_spin_args_rejects_unsafe_values(
    target: str,
    percent: float,
    duration: float,
) -> None:
    with pytest.raises(X8BenchError):
        build_x8_spin_args(target, percent, duration)


@pytest.mark.asyncio
async def test_x8_spin_requires_all_physical_confirmations(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(
        [
            "x8-spin",
            "--config",
            str(app_config.source_path),
            "--target",
            "lf",
            "--percent",
            "10",
            "--duration",
            "2",
        ]
    )

    result = await async_main(args)

    assert result == 2
    assert "X8 spin requires" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_x8_spin_launches_only_the_bounded_sequence(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run_x8_bench(config, report, diag_args):  # type: ignore[no-untyped-def]
        captured["args"] = tuple(diag_args)
        return 0

    monkeypatch.setattr(main_module, "run_x8_bench", fake_run_x8_bench)
    args = build_parser().parse_args(
        [
            "x8-spin",
            "--config",
            str(app_config.source_path),
            "--target",
            "all",
            "--percent",
            "12",
            "--duration",
            "1.5",
            "--props-removed",
            "--airframe-secured",
            "--confirm-x8",
            "X8_PROPS_REMOVED_AND_AIRFRAME_SECURED",
        ]
    )

    assert await async_main(args) == 0
    assert "triggerwin all 1120 1.50" in captured["args"][1]
