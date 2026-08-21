"""Guarded adapter for the validated Pixhawk/Hobbywing X8 bench CLI."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from collections.abc import Mapping as RuntimeMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence, Tuple

from aerogo2.common.config import X8_ESC_SLOT_MAPPING, AppConfig


class X8BenchError(RuntimeError):
    """The canonical X8 diagnostic cannot be located or is not aligned."""


@dataclass(frozen=True)
class X8AlignmentReport:
    """Result of comparing AeroGo2 configuration with the canonical diagnostic."""

    script_path: Path
    details: Tuple[str, ...]
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


_EXPECTED_DIAG_POSITION_TO_CHANNEL = {"rr": 1, "fl": 2, "rl": 3, "fr": 4}
_EXPECTED_STANDARD_PARAMS = {
    "FRAME_CLASS": 1.0,
    "FRAME_TYPE": 1.0,
    "SERVO1_FUNCTION": 36.0,
    "SERVO2_FUNCTION": 35.0,
    "SERVO3_FUNCTION": 34.0,
    "SERVO4_FUNCTION": 33.0,
    "CAN_P1_DRIVER": 1.0,
    "CAN_P1_BITRATE": 1_000_000.0,
    "CAN_D1_PROTOCOL": 1.0,
    "CAN_D1_UC_ESC_BM": 15.0,
    "CAN_D1_UC_ESC_OF": 0.0,
    "CAN_D1_UC_SRV_BM": 0.0,
    "CAN_D1_UC_ESC_RV": 0.0,
    "CAN_D1_UC_OPTION": 0.0,
    "ESC_TLM_MAV_OFS": 0.0,
    "MOT_PWM_MIN": 1000.0,
    "MOT_PWM_MAX": 2000.0,
}
_EXPECTED_RUNTIME_DEFAULTS = {
    "DEFAULT_PORT": "/dev/ttyACM0",
    "DEFAULT_BAUD": 115200,
    "DEFAULT_CONNECT_TIMEOUT": 30.0,
    "DEFAULT_PWM_MIN": 1000,
    "DEFAULT_PWM_LIMIT": 1450,
    "DEFAULT_PERCENT_LIMIT": 50.0,
    "DEFAULT_REFRESH_PERIOD": 0.8,
    "DEFAULT_HOLD_DURATION": 3.0,
}
_PROJECT_POSITION_LABELS = {"rr": "RR", "fl": "LF", "rl": "LR", "fr": "RF"}
_RESERVED_CONNECTION_OPTIONS = ("--port", "--baud", "--connect-timeout")
_X8_SPIN_TARGETS = {"rr": "rr", "lf": "fl", "lr": "rl", "rf": "fr", "all": "all"}


def resolve_diag_script(config: AppConfig, explicit: Optional[Path] = None) -> Path:
    """Locate the repository-level canonical diagnostic without duplicating it."""

    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file():
            raise X8BenchError(f"Explicit X8 diagnostic does not exist: {resolved}")
        return resolved

    configured = os.environ.get("AEROGO2_X8_DIAG")
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if not resolved.is_file():
            raise X8BenchError(f"AEROGO2_X8_DIAG does not exist: {resolved}")
        return resolved

    candidates = [
        config.source_path.resolve().parents[2] / "pixhawk_x8_cli_diag.py",
        Path(__file__).resolve().parents[3] / "pixhawk_x8_cli_diag.py",
        Path.cwd() / "pixhawk_x8_cli_diag.py",
        Path.cwd().parent / "pixhawk_x8_cli_diag.py",
    ]

    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved
    locations = ", ".join(str(path) for path in checked)
    raise X8BenchError(
        f"Cannot locate pixhawk_x8_cli_diag.py. Checked: {locations}. "
        "Use --diag-script PATH or AEROGO2_X8_DIAG."
    )


def _load_diag_module(script_path: Path) -> ModuleType:
    module_name = "_aerogo2_pixhawk_x8_cli_diag"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise X8BenchError(f"Cannot load X8 diagnostic module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as exc:
        sys.modules.pop(module_name, None)
        raise X8BenchError(f"Cannot import X8 diagnostic: {exc}") from exc
    return module


def _standard_param_values(module: ModuleType) -> Dict[str, Optional[float]]:
    raw = getattr(module, "PARAM_EXPECT_STANDARD", None)
    if not isinstance(raw, (list, tuple)):
        raise X8BenchError("Diagnostic PARAM_EXPECT_STANDARD is missing or invalid")
    values: Dict[str, Optional[float]] = {}
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise X8BenchError("Diagnostic PARAM_EXPECT_STANDARD contains an invalid entry")
        name, expected = entry[0], entry[1]
        if not isinstance(name, str):
            raise X8BenchError("Diagnostic parameter name is not text")
        if expected is None:
            values[name] = None
        elif isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise X8BenchError(f"Diagnostic parameter {name} has an invalid expected value")
        else:
            values[name] = float(expected)
    return values


def check_x8_alignment(
    config: AppConfig,
    explicit_script: Optional[Path] = None,
) -> X8AlignmentReport:
    """Fail closed when the project and canonical bench diagnostic drift."""

    script_path = resolve_diag_script(config, explicit_script)
    module = _load_diag_module(script_path)
    details = []
    errors = []

    for name, expected in _EXPECTED_RUNTIME_DEFAULTS.items():
        actual = getattr(module, name, None)
        if actual != expected:
            errors.append(f"diag {name}={actual!r}; expected {expected!r}")
    details.append(
        f"transport port={config.pixhawk.connection} baud={config.pixhawk.baud} heartbeat_timeout_s={config.pixhawk.heartbeat_timeout_s:.1f}"
    )

    diag_baud = getattr(module, "DEFAULT_BAUD", None)
    if config.pixhawk.baud != diag_baud:
        errors.append(
            f"pixhawk.baud={config.pixhawk.baud} does not match diag DEFAULT_BAUD={diag_baud}"
        )
    diag_timeout = getattr(module, "DEFAULT_CONNECT_TIMEOUT", None)
    if config.pixhawk.heartbeat_timeout_s != diag_timeout:
        errors.append(
            f"pixhawk.heartbeat_timeout_s={config.pixhawk.heartbeat_timeout_s} does not match diag "
            f"DEFAULT_CONNECT_TIMEOUT={diag_timeout}"
        )
    if "PIXHAWK_DEVICE" in config.pixhawk.connection:
        errors.append("pixhawk.connection still contains the placeholder PIXHAWK_DEVICE")

    raw_position_map = getattr(module, "POSITION_TO_OUTPUT_CHANNEL", None)
    if not isinstance(raw_position_map, RuntimeMapping):
        errors.append("diag POSITION_TO_OUTPUT_CHANNEL is missing or invalid")
        position_map: Dict[str, int] = {}
    else:
        position_map = {
            str(position): int(channel)
            for position, channel in raw_position_map.items()
            if isinstance(channel, int)
        }
        if position_map != _EXPECTED_DIAG_POSITION_TO_CHANNEL:
            errors.append(
                f"diag motor mapping {position_map!r}; expected {_EXPECTED_DIAG_POSITION_TO_CHANNEL!r}"
            )

    expected_project_map = {
        channel: _PROJECT_POSITION_LABELS[position]
        for position, channel in _EXPECTED_DIAG_POSITION_TO_CHANNEL.items()
    }
    actual_project_map = dict(config.esc.slots)
    details.append(f"x8 slots={actual_project_map}")
    if expected_project_map != X8_ESC_SLOT_MAPPING:
        errors.append("internal AeroGo2 X8 mapping constant is inconsistent")
    if actual_project_map != expected_project_map:
        errors.append(
            f"AeroGo2 ESC mapping {actual_project_map!r}; expected {expected_project_map!r}"
        )

    actual_params = _standard_param_values(module)
    for name, expected in _EXPECTED_STANDARD_PARAMS.items():
        actual = actual_params.get(name)
        if actual != expected:
            errors.append(f"diag standard parameter {name}={actual!r}; expected {expected!r}")
    details.append(f"standard parameter audit={len(_EXPECTED_STANDARD_PARAMS)} critical values")
    details.append(f"canonical diagnostic={script_path}")
    return X8AlignmentReport(
        script_path=script_path,
        details=tuple(details),
        errors=tuple(errors),
    )


def _normalized_diag_args(diag_args: Sequence[str]) -> Tuple[str, ...]:
    args = tuple(diag_args)
    if args[:1] == ("--",):
        args = args[1:]
    for token in args:
        for option in _RESERVED_CONNECTION_OPTIONS:
            if token == option or token.startswith(option + "="):
                raise X8BenchError(
                    f"{option} is managed by AeroGo2 config and cannot be overridden"
                )
    return args


def build_x8_spin_args(target: str, percent: float, duration_s: float) -> Tuple[str, ...]:
    """Build a bounded, fail-closed X8 motor-test command list."""

    normalized_target = target.strip().lower()
    diag_target = _X8_SPIN_TARGETS.get(normalized_target)
    if diag_target is None:
        raise X8BenchError("X8 target must be one of: rr, lf, lr, rf, all")
    if not math.isfinite(percent) or not 5.0 <= percent <= 20.0:
        raise X8BenchError("X8 percent must be finite and within 5..20")
    if not math.isfinite(duration_s) or not 0.5 <= duration_s <= 5.0:
        raise X8BenchError("X8 duration must be finite and within 0.5..5.0 seconds")

    pwm = round(1000.0 + percent * 10.0)
    commands = "; ".join(
        (
            "streams on",
            "audit std",
            "x8diag 2",
            "safety off",
            f"triggercheck {diag_target} 2",
            f"triggerwin {diag_target} {pwm} {duration_s:.2f}",
            "trigger off",
            "safety on",
        )
    )
    return ("--commands", commands, "--command-settle", "0.2", "--post-listen", "1")


def build_diag_command(
    config: AppConfig,
    script_path: Path,
    diag_args: Sequence[str],
) -> Tuple[str, ...]:
    """Build an argv-only subprocess command; no shell is involved."""

    passthrough = _normalized_diag_args(diag_args)
    return (
        sys.executable,
        str(script_path),
        "--port",
        config.pixhawk.connection,
        "--baud",
        str(config.pixhawk.baud),
        "--connect-timeout",
        str(config.pixhawk.heartbeat_timeout_s),
        *passthrough,
    )


def run_x8_bench(
    config: AppConfig,
    report: X8AlignmentReport,
    diag_args: Sequence[str],
) -> int:
    """Launch the canonical diagnostic with inherited interactive I/O."""

    if not report.ok:
        raise X8BenchError("Refusing to launch an unaligned X8 diagnostic")
    command = build_diag_command(config, report.script_path, diag_args)
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise X8BenchError(f"Cannot launch X8 diagnostic: {exc}") from exc
    return int(completed.returncode)


def print_alignment_report(report: X8AlignmentReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"X8 alignment: {status}")
    for detail in report.details:
        print(f"  {detail}")
    for error in report.errors:
        print(f"  ERROR: {error}")


__all__ = [
    "X8AlignmentReport",
    "X8BenchError",
    "build_diag_command",
    "build_x8_spin_args",
    "check_x8_alignment",
    "print_alignment_report",
    "resolve_diag_script",
    "run_x8_bench",
]
