"""Console entry point."""

from __future__ import annotations

import argparse
import asyncio
import signal
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from aerogo2.app import AeroGo2Application
from aerogo2.cli.signal_shutdown import SupervisedSignalRouter
from aerogo2.common.config import load_config
from aerogo2.common.enums import RuntimeMode
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.hardware.runtime import HardwareWorld
from aerogo2.simulation.world import SimulationWorld
from aerogo2.x8_bench import (
    X8_SPIN_CONFIRMATION,
    X8BenchError,
    check_x8_alignment,
    print_alignment_report,
    run_x8_bench,
    run_x8_spin,
)

_DEFAULT_CONFIG = Path("configs/system.yaml")


def _default_config_path() -> Path:
    if _DEFAULT_CONFIG.is_file():
        return _DEFAULT_CONFIG
    packaged = Path(__file__).resolve().parent / "default_configs" / "system.yaml"
    return packaged if packaged.is_file() else _DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerogo2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shell = subparsers.add_parser("shell", help="Start the resident console")
    mode = shell.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--hardware-readonly", action="store_true")
    mode.add_argument("--hardware", action="store_true")
    shell.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
    )
    shell.add_argument("--confirm-hardware", default="")
    shell.add_argument("--enable-hardware-write", action="store_true")

    monitor = subparsers.add_parser("monitor", help="Run the fail-closed hardware monitor")
    monitor.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    monitor.add_argument("--confirm-hardware", default="")

    bench = subparsers.add_parser(
        "x8-bench",
        help="Run an allowlisted read-only Pixhawk/Hobbywing X8 diagnostic",
    )
    bench.add_argument("--config", type=Path, default=_default_config_path())
    bench.add_argument(
        "--diag-script",
        type=Path,
        help="Override the canonical pixhawk_x8_cli_diag.py location",
    )
    bench.add_argument(
        "--check",
        action="store_true",
        help="Validate alignment without opening a serial port",
    )
    bench.add_argument(
        "diag_args",
        nargs=argparse.REMAINDER,
        help="Allowlisted read-only cli_diag arguments; place them after --",
    )

    spin = subparsers.add_parser(
        "x8-spin",
        help="Run a bounded DISARMED X8 motor test with propellers removed",
    )
    spin.add_argument("--config", type=Path, default=_default_config_path())
    spin.add_argument(
        "--diag-script",
        type=Path,
        help="Override the canonical pixhawk_x8_cli_diag.py location",
    )
    spin.add_argument(
        "--target",
        required=True,
        choices=("rr", "lf", "lr", "rf"),
        help="One AeroGo2 arm target (rr/lf/lr/rf); simultaneous all-arm spin is blocked",
    )
    spin.add_argument("--percent", required=True, type=float, help="5..20 percent")
    spin.add_argument(
        "--duration",
        required=True,
        type=float,
        help="0.5..5.0 seconds",
    )
    spin.add_argument("--props-removed", action="store_true")
    spin.add_argument("--airframe-secured", action="store_true")
    spin.add_argument("--confirm-x8", default="")

    demo = subparsers.add_parser("demo", help="Run a non-interactive dry-run scenario")
    demo.add_argument("--scenario", default="nominal")
    demo.add_argument("--config", type=Path, default=_default_config_path())
    return parser


async def async_main(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2

    if args.command == "x8-spin":
        if (
            not args.props_removed
            or not args.airframe_secured
            or args.confirm_x8 != X8_SPIN_CONFIRMATION
        ):
            print(
                "X8 spin requires --props-removed --airframe-secured "
                f"--confirm-x8 {X8_SPIN_CONFIRMATION}"
            )
            return 2
        try:
            report = check_x8_alignment(config, args.diag_script)
            print_alignment_report(report)
            if not report.ok:
                return 2
            print(
                f"X8 bounded motor test: target={args.target.upper()} "
                f"power={args.percent:.1f}% duration={args.duration:.2f}s; Pixhawk must be DISARMED"
            )
            return run_x8_spin(
                config,
                report,
                args.target,
                args.percent,
                args.duration,
                props_removed=args.props_removed,
                airframe_secured=args.airframe_secured,
                confirmation=args.confirm_x8,
            )
        except X8BenchError as exc:
            print(f"X8 spin error: {exc}")
            return 2
    if args.command == "x8-bench":
        try:
            if args.check and args.diag_args:
                raise X8BenchError("--check cannot be combined with diagnostic arguments")
            report = check_x8_alignment(config, args.diag_script)
            print_alignment_report(report)
            if not report.ok:
                return 2
            if args.check:
                return 0
            return run_x8_bench(config, report, args.diag_args)
        except X8BenchError as exc:
            print(f"X8 bench error: {exc}")
            return 2

    if args.command == "shell":
        runtime_mode = RuntimeMode.DRY_RUN
        write_enabled = False
        if args.hardware_readonly or args.hardware:
            if args.confirm_hardware != "I_UNDERSTAND_HARDWARE_RISK":
                print("Hardware mode requires --confirm-hardware I_UNDERSTAND_HARDWARE_RISK")
                return 2
            runtime_mode = RuntimeMode.HARDWARE if args.hardware else RuntimeMode.HARDWARE_READONLY
        if args.hardware:
            if not args.enable_hardware_write:
                print("Hardware control requires --enable-hardware-write for this process")
                return 2
            write_enabled = True
        elif args.enable_hardware_write:
            print("--enable-hardware-write is valid only together with --hardware")
            return 2
        config = replace(
            config,
            system=replace(
                config.system,
                dry_run=runtime_mode is RuntimeMode.DRY_RUN,
                hardware_write_enabled=write_enabled,
            ),
        )
        app = AeroGo2Application(config, runtime_mode=runtime_mode)
        stopped = None
        signal_router = None
        if isinstance(app.world, HardwareWorld):
            signal_router = SupervisedSignalRouter(app.shell.request_supervised_shutdown)
            signal_router.install()
        try:
            try:
                shell_result = int(await app.shell.run())
            finally:
                # InteractiveShell owns the supervised manager handoff. HardwareWorld
                # additionally owns the LowState/LowCmd bridge, so its idempotent
                # cleanup must not be abandoned if the prompt raises or is cancelled.
                if isinstance(app.world, HardwareWorld):
                    stopped = await app.world.shutdown_until_safe()
                else:
                    await app.world.shutdown()
        finally:
            if signal_router is not None:
                signal_router.close()
        if stopped is not None and not stopped.ok:
            print(
                "Hardware shutdown failed after ownership cleared: "
                f"{stopped.code}: {stopped.message}"
            )
            return 1
        return shell_result

    if args.command == "monitor":
        if args.confirm_hardware != "I_UNDERSTAND_HARDWARE_RISK":
            print("Monitor mode requires --confirm-hardware I_UNDERSTAND_HARDWARE_RISK")
            return 2
        config = replace(
            config,
            system=replace(config.system, dry_run=False, hardware_write_enabled=False),
        )
        app = AeroGo2Application(config, runtime_mode=RuntimeMode.HARDWARE_READONLY)
        stop_event = asyncio.Event()
        if not isinstance(app.world, HardwareWorld):
            raise RuntimeError("Monitor composition did not create HardwareWorld")
        loop = asyncio.get_running_loop()
        for event_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(event_signal, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        monitor_result = await app.world.monitor_until_stopped(stop_event)
        if not monitor_result.ok:
            print(f"Hardware monitor failed: {monitor_result.code}: {monitor_result.message}")
            return 1
        return 0

    if args.command == "demo":
        app = AeroGo2Application(config, runtime_mode=RuntimeMode.DRY_RUN)
        if not isinstance(app.world, SimulationWorld):
            raise RuntimeError("Demo composition did not create SimulationWorld")
        scenario_result = await app.world.run_scenario(args.scenario)
        print(
            "{}: {} (final_state={})".format(
                scenario_result.name,
                "PASS" if scenario_result.ok else "FAIL",
                scenario_result.final_state.name,
            )
        )
        for state in scenario_result.states:
            print(f"  -> {state.name}")
        for message in scenario_result.messages:
            print(f"  {message}")
        await app.world.shutdown()
        return 0 if scenario_result.ok else 1
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
