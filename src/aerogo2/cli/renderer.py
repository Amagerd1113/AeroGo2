"""Rich-based console rendering with a plain, testable prompt contract."""

from __future__ import annotations

import json
from collections import defaultdict
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from aerogo2.cli.command_models import CommandSpec
from aerogo2.common.enums import CommandStatus, RuntimeMode, SystemState
from aerogo2.common.models import (
    F446Status,
    HealthReport,
    SafetyViolation,
    SystemSnapshot,
    snapshot_to_dict,
)
from aerogo2.common.results import CommandResult, GuardResult
from aerogo2.logging.schemas import to_jsonable

STATE_COLORS: Final[Mapping[SystemState, str]] = MappingProxyType(
    {
        SystemState.BOOT_SAFE: "yellow",
        SystemState.MANUAL_POSITIONING: "bold yellow",
        SystemState.WALK: "green",
        SystemState.WALK_TO_FLIGHT_PRECHECK: "yellow",
        SystemState.TRANSFORM_TO_FLIGHT: "yellow",
        SystemState.GO2_JOINT_LOCK_WAIT: "bold yellow",
        SystemState.FLIGHT_READY: "cyan",
        SystemState.FLIGHT_MANUAL: "cyan",
        SystemState.AUTO_LANDING_READY: "blue",
        SystemState.AUTO_LANDING: "blue",
        SystemState.TOUCHDOWN_VERIFY: "yellow",
        SystemState.LANDING_COMPLIANT: "yellow",
        SystemState.FLIGHT_TO_WALK_PRECHECK: "yellow",
        SystemState.TRANSFORM_TO_WALK: "yellow",
        SystemState.FAULT: "bold red",
        SystemState.EMERGENCY_STOP: "bold white on red",
    }
)


class ConsoleRenderer:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def prompt_text(self, runtime_mode: RuntimeMode, snapshot: SystemSnapshot) -> str:
        if snapshot.maintenance_mode:
            label = "F446-MAINTENANCE"
        elif snapshot.state is SystemState.FAULT:
            fault = snapshot.active_fault_codes[0] if snapshot.active_fault_codes else "UNSPECIFIED"
            label = f"FAULT|{fault}"
        else:
            label = f"{runtime_mode.value}|{snapshot.state.name}"
        return f"aerogo2[{label}]> "

    def prompt(self, runtime_mode: RuntimeMode, snapshot: SystemSnapshot) -> str:
        return self.prompt_text(runtime_mode, snapshot)

    def render_banner(
        self,
        runtime_mode: RuntimeMode,
        snapshot: SystemSnapshot,
        logging_enabled: bool = True,
    ) -> None:
        self.console.print("[bold cyan]AeroGo2 Integrated Control Console[/bold cyan]")
        self.console.print("=" * 34)
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Runtime", runtime_mode.value)
        table.add_row(
            "System state",
            Text(snapshot.state.name, style=STATE_COLORS.get(snapshot.state, "white")),
        )
        table.add_row("Pixhawk", "CONNECTED" if snapshot.pixhawk.connected else "DISCONNECTED")
        table.add_row("F446", "CONNECTED" if snapshot.f446.connected else "DISCONNECTED")
        table.add_row("Go2", "CONNECTED" if snapshot.go2.connected else "DISCONNECTED")
        table.add_row("Logging", "ON" if logging_enabled else "OFF")
        self.console.print(table)
        self.console.print('\nType "help" to list commands.\n')

    banner = render_banner

    def render_result(self, result: CommandResult) -> None:
        styles = {
            CommandStatus.SUCCESS: "green",
            CommandStatus.REJECTED: "yellow",
            CommandStatus.FAILED: "bold red",
            CommandStatus.UNAVAILABLE: "magenta",
        }
        self.console.print(result.message, style=styles.get(result.status, "white"))
        if result.data:
            self.render_mapping(result.data)

    def render_snapshot(
        self, snapshot: SystemSnapshot, *, full: bool = False, as_json: bool = False
    ) -> None:
        if as_json:
            self.console.print_json(self.snapshot_json(snapshot))
            return

        table = Table(title="AeroGo2 status")
        table.add_column("Subsystem", style="bold")
        table.add_column("State")
        table.add_column("Details")
        table.add_row(
            "System",
            Text(snapshot.state.name, style=STATE_COLORS.get(snapshot.state, "white")),
            "configuration={} source={} faults={}".format(
                snapshot.configuration.value,
                ",".join(snapshot.active_fault_codes) or "none",
                snapshot.configuration_source,
            ),
        )
        table.add_row(
            "Pixhawk",
            "CONNECTED" if snapshot.pixhawk.connected else "DISCONNECTED",
            f"armed={snapshot.pixhawk.armed} mode={snapshot.pixhawk.flight_mode} landed={snapshot.pixhawk.landed} max_rpm={snapshot.pixhawk.maximum_esc_rpm:.1f}",
        )
        table.add_row(
            "F446",
            snapshot.f446.state.value,
            f"connected={snapshot.f446.connected} duty={snapshot.f446.duty} current={snapshot.f446.used_current_adc}",
        )
        table.add_row(
            "Go2",
            "CONNECTED" if snapshot.go2.connected else "DISCONNECTED",
            f"velocity={snapshot.go2.velocity_mps:.3f}m/s stable={snapshot.go2.stable}",
        )
        table.add_row(
            "RC",
            "FAILSAFE" if snapshot.rc.failsafe else "OK",
            f"flight_enable={snapshot.rc.flight_enable} morphology={snapshot.rc.morphology_request.value} autoland={snapshot.rc.auto_landing_request.value}",
        )
        self.console.print(table)
        if full:
            self.render_mapping(snapshot_to_dict(snapshot))

    render_status = render_snapshot

    @staticmethod
    def snapshot_json(snapshot: SystemSnapshot) -> str:
        return json.dumps(
            snapshot_to_dict(snapshot),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def render_f446_current(self, status: F446Status) -> None:
        """Render one compact HW-039 sample without blocking the command prompt."""

        self.console.print(
            f"HW039 state={status.state.value} duty={status.duty} "
            f"R_IS={status.r_is_raw}/{status.r_is_mv}mV L_IS={status.l_is_raw}/{status.l_is_mv}mV "
            f"used={status.used_raw}/{status.used_mv}mV "
            f"threshold={status.threshold_raw}/{status.threshold_mv}mV over_active={int(status.over_active)}",
            style="cyan",
        )

    def render_help(self, specs: Iterable[CommandSpec], title: str = "Commands") -> None:
        grouped = defaultdict(list)
        for spec in specs:
            if not spec.hidden:
                grouped[spec.category].append(spec)
        self.console.print(f"[bold]{title}[/bold]")
        for category in sorted(grouped):
            table = Table(title=category, show_header=True)
            table.add_column("Command", style="cyan", no_wrap=True)
            table.add_column("Description")
            table.add_column("Permission")
            table.add_column("Confirmation")
            for spec in sorted(grouped[category], key=lambda item: item.path):
                table.add_row(
                    spec.effective_usage,
                    spec.description,
                    spec.permission.capability.value,
                    spec.confirmation.level.name,
                )
            self.console.print(table)

    def render_command_help(self, spec: CommandSpec) -> None:
        table = Table(show_header=False, box=None)
        table.add_row("Usage", spec.effective_usage)
        table.add_row("Description", spec.description)
        table.add_row("Category", spec.category)
        table.add_row("Permission", spec.permission.capability.value)
        table.add_row("Confirmation", spec.confirmation.level.name)
        if spec.aliases:
            table.add_row("Aliases", ", ".join(" ".join(alias) for alias in spec.aliases))
        if spec.options:
            table.add_row("Options", ", ".join(spec.options))
        self.console.print(table)

    def render_guard(self, guard: GuardResult, title: str = "Checks") -> None:
        table = Table(title=title)
        table.add_column("Result")
        table.add_column("Code")
        table.add_column("Message")
        if guard.permitted:
            table.add_row("[green]PASS[/green]", "OK", "All checks passed")
        else:
            for code, message in zip(guard.codes, guard.messages):
                table.add_row("[red]FAIL[/red]", code, message)
        self.console.print(table)

    def render_health(self, report: HealthReport) -> None:
        table = Table(title="Health")
        table.add_column("Check")
        table.add_column("Result")
        for name, passed in sorted(report.checks.items()):
            table.add_row(name, "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
        for message in report.messages:
            table.add_row("detail", message)
        self.console.print(table)

    def render_violations(self, violations: Sequence[SafetyViolation]) -> None:
        table = Table(title="Safety violations")
        table.add_column("Severity")
        table.add_column("Code")
        table.add_column("Message")
        table.add_column("Recommended action")
        if not violations:
            table.add_row("INFO", "NONE", "No active violations", "")
        for violation in violations:
            style = "red" if violation.severity.value in {"FAULT", "EMERGENCY"} else "yellow"
            table.add_row(
                Text(violation.severity.value, style=style),
                violation.code,
                violation.message,
                violation.recommended_action,
            )
        self.console.print(table)

    def render_mapping(self, mapping: Mapping[str, Any]) -> None:
        table = Table(show_header=False, box=None)
        table.add_column(style="bold")
        table.add_column()
        for key, value in mapping.items():
            if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                rendered = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
            else:
                rendered = str(value)
            table.add_row(str(key), rendered)
        self.console.print(table)

    def render_suggestions(self, suggestions: Sequence[str]) -> None:
        if suggestions:
            self.console.print("Did you mean: {}?".format(", ".join(suggestions)), style="yellow")

    def info(self, message: str) -> None:
        self.console.print(message)

    def warning(self, message: str) -> None:
        self.console.print(message, style="bold yellow")

    def error(self, message: str) -> None:
        self.console.print(message, style="bold red")


Renderer = ConsoleRenderer

__all__ = ["ConsoleRenderer", "Renderer", "STATE_COLORS"]
