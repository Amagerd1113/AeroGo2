from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import aerogo2.main as main_module
from aerogo2.common.config import AppConfig, PixhawkConfig
from aerogo2.main import async_main, build_parser
from aerogo2.x8_bench import (
    X8_SPIN_CONFIRMATION,
    X8AlignmentReport,
    X8BenchError,
    build_diag_command,
    build_x8_spin_args,
    check_x8_alignment,
    run_x8_bench,
    run_x8_spin,
)


def test_x8_alignment_matches_canonical_diag(app_config: AppConfig) -> None:
    report = check_x8_alignment(app_config)

    assert report.ok
    assert report.errors == ()
    assert report.script_path.name == "pixhawk_x8_cli_diag.py"
    assert report.script_sha256 == (
        "sha256:7987dbf41d17e9c6d9dbd811b9be1fda0eea37c25028def17c4fca2986123dbb"
    )
    assert "x8 slots={1: 'RR', 2: 'LF', 3: 'LR', 4: 'RF'}" in report.details
    assert "standard parameter audit=17 critical values" in report.details


def test_x8_alignment_resolves_project_script_outside_project_cwd(
    app_config: AppConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    report = check_x8_alignment(app_config)

    assert report.ok
    assert report.script_path == app_config.source_path.resolve().parents[1] / (
        "pixhawk_x8_cli_diag.py"
    )


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
        ("--", "--commands", "audit std; x8diag 3"),
    )

    assert command[1] == "-"
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
        "audit std; x8diag 3",
    )


def test_x8_alignment_rejects_unreviewed_script_before_import(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    script = tmp_path / "diag.py"
    script.write_bytes(b"raise RuntimeError('must never execute')\n")

    with pytest.raises(X8BenchError, match="SHA-256"):
        check_x8_alignment(app_config, script)


def test_run_x8_bench_revalidates_report_and_executes_verified_snapshot(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = check_x8_alignment(app_config)
    captured: dict[str, object] = {}

    def fake_run(command, *, check, input):  # type: ignore[no-untyped-def,redefined-builtin]
        captured["command"] = command
        captured["check"] = check
        captured["input"] = input

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("aerogo2.x8_bench.subprocess.run", fake_run)
    assert run_x8_bench(app_config, report, ("--self-test",)) == 0
    assert captured["command"][1] == "-"  # type: ignore[index]
    assert captured["input"] == report.script_path.read_bytes()

    forged = X8AlignmentReport(
        script_path=report.script_path,
        script_sha256="sha256:" + "0" * 64,
        details=report.details,
        errors=(),
    )
    with pytest.raises(X8BenchError, match="changed after alignment"):
        run_x8_bench(app_config, forged, ("--self-test",))


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


@pytest.mark.parametrize(
    "arguments",
    [
        ("--list-ports",),
        ("--self-test",),
        ("--can-probe", "5", "--can-bus", "1"),
        ("--can-config-probe=5", "--can-bus=2"),
        ("--can-node-info", "5"),
        (
            "--commands",
            "help; status; modes; listen 1; preflight 1; escdiag 1; "
            "offsetdiag 1; x8diag 1; nodediag 1; audit std; getparam FRAME_CLASS",
            "--command-settle",
            "0.2",
            "--post-listen",
            "1",
        ),
    ],
)
def test_build_diag_command_accepts_only_documented_read_only_operations(
    app_config: AppConfig,
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    command = build_diag_command(app_config, tmp_path / "diag.py", arguments)

    assert command[8:] == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--commands", "streams on"),
        ("--commands", "status; safety off"),
        ("--commands", "arm"),
        ("--commands", "arm force"),
        ("--commands", "launch 1"),
        ("--commands", "flighttest 1 0.2 1"),
        ("--commands", "setup std"),
        ("--commands", "param FRAME_CLASS 1"),
        ("--commands", "reboot"),
        ("--commands", "triggerwin fl 1100 1"),
        ("--commands", "tx on"),
        ("--commands", "m1 1100"),
        ("--commands", "rfp 10"),
        ("--commands", "mode guided"),
        ("--commands", "takeoff 1"),
        ("--commands", "disarm"),
        ("--set-can-throttle", "5"),
        ("--repl-after-commands",),
        ("--trigger", "fl", "1100"),
        ("--commands", "status", "--repl-after-commands"),
        ("--commands", "status", "--post-listen", "31"),
        ("--can-probe", "61"),
        ("--can-probe", "5", "--commands", "status"),
    ],
)
def test_build_diag_command_rejects_interactive_writing_or_unbounded_operations(
    app_config: AppConfig,
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(X8BenchError):
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


@pytest.mark.asyncio
async def test_x8_bench_check_rejects_unused_diag_arguments(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(
        [
            "x8-bench",
            "--config",
            str(app_config.source_path),
            "--check",
            "--",
            "--self-test",
        ]
    )

    result = await async_main(args)

    assert result == 2
    assert "--check cannot be combined" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_x8_bench_rejects_write_before_subprocess_launch(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unsafe diagnostic reached subprocess.run")

    monkeypatch.setattr("aerogo2.x8_bench.subprocess.run", unexpected_run)
    args = build_parser().parse_args(
        [
            "x8-bench",
            "--config",
            str(app_config.source_path),
            "--",
            "--commands",
            "status; safety off; arm force",
        ]
    )

    result = await async_main(args)

    assert result == 2
    assert "not on the x8-bench read-only allowlist" in capsys.readouterr().out


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
        ("all", 10.0, 2.0),
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


def test_run_x8_spin_rechecks_physical_confirmation(app_config: AppConfig) -> None:
    report = check_x8_alignment(app_config)

    with pytest.raises(X8BenchError, match="removed propellers"):
        run_x8_spin(
            app_config,
            report,
            "lf",
            10.0,
            2.0,
            props_removed=False,
            airframe_secured=True,
            confirmation=X8_SPIN_CONFIRMATION,
        )


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
    captured: dict[str, object] = {}

    def fake_run_x8_spin(  # type: ignore[no-untyped-def]
        config,
        report,
        target,
        percent,
        duration,
        **confirmations,
    ):
        captured["target"] = target
        captured["percent"] = percent
        captured["duration"] = duration
        captured["confirmations"] = confirmations
        return 0

    monkeypatch.setattr(main_module, "run_x8_spin", fake_run_x8_spin)
    args = build_parser().parse_args(
        [
            "x8-spin",
            "--config",
            str(app_config.source_path),
            "--target",
            "rr",
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
    assert captured["target"] == "rr"
    assert captured["percent"] == 12.0
    assert captured["duration"] == 1.5
    assert captured["confirmations"] == {
        "props_removed": True,
        "airframe_secured": True,
        "confirmation": X8_SPIN_CONFIRMATION,
    }
