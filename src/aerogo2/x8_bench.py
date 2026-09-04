"""Guarded adapter for the validated Pixhawk/Hobbywing X8 bench CLI."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
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
    script_sha256: str
    details: Tuple[str, ...]
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


_EXPECTED_DIAG_POSITION_TO_CHANNEL = {"rr": 1, "fl": 2, "rl": 3, "fr": 4}
_EXPECTED_DIAG_SHA256 = "sha256:7987dbf41d17e9c6d9dbd811b9be1fda0eea37c25028def17c4fca2986123dbb"
_MAX_DIAG_BYTES = 512 * 1024
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
# Deliberately omit ``all``.  The project has no signed commissioning record
# proving that all four physical outputs have passed individual mapping and
# direction checks, so a simultaneous spin cannot be authorized here.
_X8_SPIN_TARGETS = {"rr": "rr", "lf": "fl", "lr": "rl", "rf": "fr"}
X8_SPIN_CONFIRMATION = "X8_PROPS_REMOVED_AND_AIRFRAME_SECURED"

# ``x8-bench`` is deliberately not a generic forwarding wrapper.  The
# repository-level diagnostic also contains configuration, arming and motor
# commands, so every accepted top-level option and every item in ``--commands``
# is enumerated here.  These operations may send telemetry/parameter *read*
# requests, but cannot arm, change a flight mode, write a parameter, disable the
# safety switch or command an ESC.
_READ_ONLY_PRIMARY_OPTIONS = {
    "--list-ports",
    "--self-test",
    "--can-probe",
    "--can-config-probe",
    "--can-node-info",
    "--commands",
}
_READ_ONLY_VALUE_OPTIONS = {
    "--can-probe",
    "--can-config-probe",
    "--can-node-info",
    "--can-bus",
    "--commands",
    "--command-settle",
    "--post-listen",
}
_READ_ONLY_FLAG_OPTIONS = {"--list-ports", "--self-test"}
_CAN_READ_OPTIONS = {"--can-probe", "--can-config-probe", "--can-node-info"}
_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
_READ_ONLY_NO_ARGUMENT_COMMANDS = {"help", "?", "status", "modes"}
_READ_ONLY_TIMED_COMMANDS = {
    "listen",
    "preflight",
    "flightcheck",
    "escdiag",
    "offsetdiag",
    "x8diag",
    "nodediag",
    "nodedig",
}


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
        config.source_path.resolve().parents[1] / "pixhawk_x8_cli_diag.py",
        Path(__file__).resolve().parents[2] / "pixhawk_x8_cli_diag.py",
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


def _read_pinned_diag_payload(script_path: Path) -> tuple[bytes, str]:
    """Read one bounded, reviewed diagnostic snapshot and verify its identity."""

    try:
        with script_path.open("rb") as stream:
            payload = stream.read(_MAX_DIAG_BYTES + 1)
    except OSError as exc:
        raise X8BenchError(f"Cannot read X8 diagnostic: {exc}") from exc
    if len(payload) > _MAX_DIAG_BYTES:
        raise X8BenchError("X8 diagnostic exceeds the reviewed size bound")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not hmac.compare_digest(digest, _EXPECTED_DIAG_SHA256):
        raise X8BenchError("X8 diagnostic SHA-256 does not match the reviewed repository version")
    return payload, digest


def _load_diag_module(script_path: Path, payload: bytes) -> ModuleType:
    """Load the exact byte snapshot already checked by `_read_pinned_diag_payload`."""

    module_name = "_aerogo2_pixhawk_x8_cli_diag"
    module = ModuleType(module_name)
    module.__file__ = str(script_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(payload, str(script_path), "exec")
        exec(code, module.__dict__)
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
    payload, script_sha256 = _read_pinned_diag_payload(script_path)
    module = _load_diag_module(script_path, payload)
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
    details.append(f"canonical diagnostic digest={script_sha256}")
    return X8AlignmentReport(
        script_path=script_path,
        script_sha256=script_sha256,
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


def _finite_float_in_range(value: str, *, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise X8BenchError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise X8BenchError(f"{name} must be finite and within {minimum:g}..{maximum:g}")
    return parsed


def _validate_read_only_commands(commands: str) -> None:
    command_list = [part.strip() for part in commands.replace("\n", ";").split(";")]
    command_list = [command for command in command_list if command]
    if not command_list:
        raise X8BenchError("--commands must contain at least one read-only diagnostic command")

    for command in command_list:
        parts = command.split()
        name = parts[0].lower()
        if name in _READ_ONLY_NO_ARGUMENT_COMMANDS:
            if len(parts) != 1:
                raise X8BenchError(f"read-only command {name!r} accepts no arguments")
            continue
        if name in _READ_ONLY_TIMED_COMMANDS:
            if len(parts) > 2:
                raise X8BenchError(f"read-only command {name!r} accepts at most one duration")
            if len(parts) == 2:
                _finite_float_in_range(
                    parts[1],
                    name=f"{name} duration",
                    minimum=0.0,
                    maximum=30.0,
                )
            continue
        if name == "audit":
            if len(parts) != 2 or parts[1].lower() not in {"hw", "std"}:
                raise X8BenchError("read-only audit must be exactly 'audit hw' or 'audit std'")
            continue
        if name == "getparam":
            if len(parts) != 2 or _PARAMETER_NAME.fullmatch(parts[1]) is None:
                raise X8BenchError("read-only getparam requires one 1..16 character parameter name")
            continue
        raise X8BenchError(
            f"diagnostic command {name!r} is not on the x8-bench read-only allowlist"
        )


def _parse_read_only_options(diag_args: Sequence[str]) -> Dict[str, Optional[str]]:
    args = _normalized_diag_args(diag_args)
    if not args:
        raise X8BenchError(
            "x8-bench requires an explicit read-only operation; the interactive diagnostic is disabled"
        )

    parsed: Dict[str, Optional[str]] = {}
    index = 0
    while index < len(args):
        token = args[index]
        option, separator, inline_value = token.partition("=")
        if option in _READ_ONLY_FLAG_OPTIONS:
            if separator:
                raise X8BenchError(f"{option} does not accept a value")
            value: Optional[str] = None
        elif option in _READ_ONLY_VALUE_OPTIONS:
            if separator:
                if not inline_value:
                    raise X8BenchError(f"{option} requires a value")
                value = inline_value
            else:
                index += 1
                if index >= len(args):
                    raise X8BenchError(f"{option} requires a value")
                value = args[index]
        else:
            raise X8BenchError(
                f"diagnostic option {option!r} is not on the x8-bench read-only allowlist"
            )
        if option in parsed:
            raise X8BenchError(f"duplicate diagnostic option is not allowed: {option}")
        parsed[option] = value
        index += 1

    primary = set(parsed).intersection(_READ_ONLY_PRIMARY_OPTIONS)
    if len(primary) != 1:
        raise X8BenchError("x8-bench requires exactly one read-only primary operation")
    selected = next(iter(primary))

    if selected in _READ_ONLY_FLAG_OPTIONS and set(parsed) != {selected}:
        raise X8BenchError(f"{selected} cannot be combined with other diagnostic options")
    if selected in _CAN_READ_OPTIONS:
        if not set(parsed).issubset({selected, "--can-bus"}):
            raise X8BenchError(f"{selected} may only be combined with --can-bus")
        duration = parsed[selected]
        assert duration is not None
        _finite_float_in_range(
            duration,
            name=f"{selected} duration",
            minimum=0.1,
            maximum=60.0,
        )
    elif selected == "--commands":
        if not set(parsed).issubset({selected, "--command-settle", "--post-listen"}):
            raise X8BenchError(
                "--commands may only be combined with --command-settle and --post-listen"
            )
        commands = parsed[selected]
        assert commands is not None
        _validate_read_only_commands(commands)
        if "--command-settle" in parsed:
            settle = parsed["--command-settle"]
            assert settle is not None
            _finite_float_in_range(
                settle,
                name="--command-settle",
                minimum=0.0,
                maximum=2.0,
            )
        if "--post-listen" in parsed:
            post_listen = parsed["--post-listen"]
            assert post_listen is not None
            _finite_float_in_range(
                post_listen,
                name="--post-listen",
                minimum=0.0,
                maximum=30.0,
            )

    if "--can-bus" in parsed:
        can_bus = parsed["--can-bus"]
        if selected not in _CAN_READ_OPTIONS or can_bus not in {"1", "2"}:
            raise X8BenchError("--can-bus must be 1 or 2 and accompany a read-only CAN probe")
    return parsed


def build_x8_spin_args(target: str, percent: float, duration_s: float) -> Tuple[str, ...]:
    """Build a bounded, fail-closed X8 motor-test command list."""

    normalized_target = target.strip().lower()
    diag_target = _X8_SPIN_TARGETS.get(normalized_target)
    if diag_target is None:
        raise X8BenchError("X8 target must be one individual arm: rr, lf, lr, rf")
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
    """Build a read-only argv-only subprocess command; no shell is involved."""

    passthrough = _normalized_diag_args(diag_args)
    _parse_read_only_options(passthrough)
    return _build_unchecked_diag_command(config, script_path, passthrough)


def _build_unchecked_diag_command(
    config: AppConfig,
    script_path: Path,
    diag_args: Sequence[str],
) -> Tuple[str, ...]:
    """Inject the managed transport after a caller-specific policy check."""

    return (
        sys.executable,
        "-",
        "--port",
        config.pixhawk.connection,
        "--baud",
        str(config.pixhawk.baud),
        "--connect-timeout",
        str(config.pixhawk.heartbeat_timeout_s),
        *diag_args,
    )


def _run_diag_command(command: Sequence[str], payload: bytes) -> int:
    try:
        completed = subprocess.run(tuple(command), check=False, input=payload)
    except OSError as exc:
        raise X8BenchError(f"Cannot launch X8 diagnostic: {exc}") from exc
    return int(completed.returncode)


def _runtime_verified_snapshot(
    config: AppConfig,
    report: X8AlignmentReport,
) -> tuple[X8AlignmentReport, bytes]:
    """Revalidate immediately before launch and return the exact bytes to execute."""

    if not isinstance(report, X8AlignmentReport):
        raise X8BenchError("Refusing an invalid X8 alignment report")
    fresh = check_x8_alignment(config, report.script_path)
    if not fresh.ok:
        raise X8BenchError("Refusing to launch an unaligned X8 diagnostic")
    if fresh.script_path != report.script_path or not hmac.compare_digest(
        fresh.script_sha256, report.script_sha256
    ):
        raise X8BenchError("X8 diagnostic changed after alignment validation")
    payload, digest = _read_pinned_diag_payload(fresh.script_path)
    if not hmac.compare_digest(digest, fresh.script_sha256):
        raise X8BenchError("X8 diagnostic changed while preparing the launch snapshot")
    return fresh, payload


def run_x8_bench(
    config: AppConfig,
    report: X8AlignmentReport,
    diag_args: Sequence[str],
) -> int:
    """Launch one validated read-only diagnostic with inherited I/O."""

    fresh, payload = _runtime_verified_snapshot(config, report)
    command = build_diag_command(config, fresh.script_path, diag_args)
    return _run_diag_command(command, payload)


def run_x8_spin(
    config: AppConfig,
    report: X8AlignmentReport,
    target: str,
    percent: float,
    duration_s: float,
    *,
    props_removed: bool,
    airframe_secured: bool,
    confirmation: str,
) -> int:
    """Launch only the generated bounded motor-test sequence after physical confirmation."""

    if not props_removed or not airframe_secured or confirmation != X8_SPIN_CONFIRMATION:
        raise X8BenchError(
            "X8 spin requires removed propellers, a secured airframe and the exact confirmation"
        )
    diag_args = build_x8_spin_args(target, percent, duration_s)
    fresh, payload = _runtime_verified_snapshot(config, report)
    command = _build_unchecked_diag_command(config, fresh.script_path, diag_args)
    return _run_diag_command(command, payload)


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
    "X8_SPIN_CONFIRMATION",
    "build_diag_command",
    "build_x8_spin_args",
    "check_x8_alignment",
    "print_alignment_report",
    "resolve_diag_script",
    "run_x8_bench",
    "run_x8_spin",
]
