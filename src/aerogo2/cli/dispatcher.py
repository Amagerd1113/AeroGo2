"""Resolve, authorize, confirm, and dispatch console commands."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from aerogo2 import __version__
from aerogo2.cli.command_models import (
    CommandContext,
    CommandInvocation,
    ConfirmationPolicy,
)
from aerogo2.cli.confirmation import ConfirmationService
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.registry import (
    CommandNotFoundError,
    CommandRegistry,
)
from aerogo2.cli.renderer import ConsoleRenderer
from aerogo2.common.config import validate_config
from aerogo2.common.enums import CommandStatus
from aerogo2.common.exceptions import CommandParseError, ConfigurationError
from aerogo2.common.results import CommandResult
from aerogo2.manager.command_service import CommandService
from aerogo2.simulation.fault_injection import SimulatedFault


@dataclass(frozen=True)
class DispatchOutcome:
    result: CommandResult
    should_exit: bool = False
    watch_target: Optional[str] = None
    watch_interval_s: float = 0.5


class CommandDispatcher:
    """The shell-facing command boundary; it imports no device bridge."""

    def __init__(
        self,
        registry: CommandRegistry,
        world: Any,
        renderer: Optional[ConsoleRenderer] = None,
        confirmation: Optional[ConfirmationService] = None,
        history: Optional[CommandHistory] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.world = world
        self.manager = world.manager
        self.renderer = ConsoleRenderer() if renderer is None else renderer
        self.confirmation = ConfirmationService() if confirmation is None else confirmation
        self.history = CommandHistory() if history is None else history
        self._event_sink = event_sink
        self._service = CommandService(self.manager)

    async def dispatch(
        self,
        line: str,
        render: bool = True,
        *,
        record_history: bool = True,
    ) -> DispatchOutcome:
        try:
            parsed = self.registry._parser.parse(line)
        except CommandParseError as exc:
            return self._finish(CommandResult(CommandStatus.REJECTED, str(exc)), render)
        if parsed.empty:
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, ""))
        if record_history:
            self.history.record(line)
        try:
            match = self.registry.resolve(parsed)
        except CommandNotFoundError as exc:
            if render:
                self.renderer.error(str(exc))
                self.renderer.render_suggestions(exc.suggestions)
            return DispatchOutcome(CommandResult(CommandStatus.REJECTED, str(exc)))

        invocation = CommandInvocation.from_match(match, parsed.raw)
        validation_error = self._validate_arguments(invocation)
        if validation_error is not None:
            return self._complete(invocation, DispatchOutcome(validation_error), render=render)
        context = CommandContext(
            runtime_mode=self.manager.runtime_mode,
            state=self.manager.state,
            phase=1,
            maintenance_mode=self.manager.snapshot.maintenance_mode,
            hardware_write_enabled=self.manager.config.system.hardware_write_enabled,
            pixhawk_armed=self.manager.snapshot.pixhawk.armed,
        )
        permission = invocation.spec.permission.evaluate(context)
        if not permission.allowed:
            return self._complete(
                invocation,
                DispatchOutcome(
                    CommandResult(
                        CommandStatus.UNAVAILABLE,
                        permission.reason,
                        {"code": permission.code},
                        invocation.command_id,
                    )
                ),
                render=render,
            )
        if invocation.spec.action in (
            "manual_enter",
            "home_walk",
            "transform_flight",
            "transform_walk",
        ):
            profile = {
                "manual_enter": "manual-position",
                "home_walk": "home-walk",
                "transform_flight": "transform-flight",
                "transform_walk": "transform-walk",
            }[invocation.spec.action]

            try:
                preflight = await self.manager.preflight(profile)
            except Exception as exc:
                return self._complete(
                    invocation,
                    DispatchOutcome(
                        CommandResult(
                            CommandStatus.FAILED,
                            f"Preflight failed safely: {type(exc).__name__}: {exc}",
                            {"code": "PREFLIGHT_EXCEPTION"},
                            invocation.command_id,
                        )
                    ),
                    render=render,
                )
            preflight_result = self._from_operation(preflight)
            if render:
                self.renderer.render_result(preflight_result)
            if not preflight.ok:
                return self._complete(
                    invocation,
                    DispatchOutcome(preflight_result),
                    render=False,
                )

        try:
            confirmed = await self.confirmation.confirm(invocation.spec.confirmation)
        except Exception as exc:
            return self._complete(
                invocation,
                DispatchOutcome(
                    CommandResult(
                        CommandStatus.FAILED,
                        f"Confirmation failed safely: {type(exc).__name__}: {exc}",
                        {"code": "CONFIRMATION_EXCEPTION"},
                        invocation.command_id,
                    )
                ),
                render=render,
            )
        if not confirmed:
            return self._complete(
                invocation,
                DispatchOutcome(
                    CommandResult(
                        CommandStatus.REJECTED,
                        "Operator confirmation was not accepted",
                        command_id=invocation.command_id,
                    )
                ),
                render=render,
            )

        try:
            outcome = await self._execute(invocation, render=render)
        except Exception as exc:
            outcome = DispatchOutcome(
                CommandResult(
                    CommandStatus.FAILED,
                    f"Command failed safely: {type(exc).__name__}: {exc}",
                    {"code": "COMMAND_EXCEPTION"},
                    invocation.command_id,
                )
            )
        return self._complete(invocation, outcome, render=render)

    async def _execute(
        self,
        invocation: CommandInvocation,
        *,
        render: bool,
    ) -> DispatchOutcome:
        action = invocation.spec.action
        args = invocation.arguments
        if action == "help":
            if args:
                help_target = tuple(args[0].split()) if len(args) == 1 else args
                try:
                    self.renderer.render_command_help(self.registry.get(help_target))
                except (CommandNotFoundError, ValueError) as exc:
                    return DispatchOutcome(CommandResult(CommandStatus.REJECTED, str(exc)))
            else:
                self.renderer.render_help(self.registry.specs())
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, ""))
        if action == "version":
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, f"AeroGo2 {__version__}"))
        if action == "clear_screen":
            self.renderer.console.clear()
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, ""))
        if action == "history":
            return DispatchOutcome(
                CommandResult(
                    CommandStatus.SUCCESS,
                    "Command history",
                    {"entries": list(self.history.entries())},
                )
            )
        if action == "exit":
            if self.manager.snapshot.pixhawk.armed:
                accepted = await self.confirmation.confirm(
                    ConfirmationPolicy.exact(
                        "EXIT_WHILE_ARMED",
                        warning=(
                            "Pixhawk is armed. Exiting only stops external setpoints; "
                            "it never disarms or stops rotors."
                        ),
                    )
                )
                if not accepted:
                    return DispatchOutcome(CommandResult(CommandStatus.REJECTED, "Exit cancelled"))
            stop_result = await self.manager.stop_supervised()
            if not stop_result.ok:
                return DispatchOutcome(
                    CommandResult(
                        CommandStatus.REJECTED,
                        "Console exit inhibited because supervised stop was incomplete: "
                        + stop_result.message,
                    )
                )
            return DispatchOutcome(
                CommandResult(CommandStatus.SUCCESS, "Console exit requested"),
                should_exit=True,
            )

        if action in ("query_status", "health") and "--watch" in args:
            interval = self._parse_watch_interval(args)
            target = "status" if action == "query_status" else "health"
            return DispatchOutcome(
                CommandResult(CommandStatus.SUCCESS, f"Watch {target} started"),
                watch_target=target,
                watch_interval_s=interval,
            )

        if action == "query_status" and invocation.spec.path == ("status",):
            data = self.manager.query("status")
            if render:
                self.renderer.render_snapshot(
                    self.manager.snapshot,
                    full="--full" in args,
                    as_json="--json" in args,
                )
                return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, ""))
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, action, data))

        if action == "query_faults":
            return DispatchOutcome(self._fault_query(invocation))

        if invocation.spec.path[0] == "preflight":
            profile = invocation.spec.path[-1] if len(invocation.spec.path) > 1 else "all"
            return DispatchOutcome(self._from_operation(await self.manager.preflight(profile)))

        if action == "config_validate":
            return DispatchOutcome(self._validate_configuration())
        if action.startswith("query_") or action == "health":
            data = self.manager.query(invocation.command_name)
            if "error" in data:
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, str(data["error"]), data)
                )
            return DispatchOutcome(
                CommandResult(CommandStatus.SUCCESS, invocation.command_name, data)
            )
        if action == "config_get":
            if len(args) != 1:
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, "Usage: config get KEY")
                )
            key = args[0]
            value = self.manager.config.get(key, None)
            if value is None:
                return DispatchOutcome(CommandResult(CommandStatus.REJECTED, f"Unknown key {key}"))
            return DispatchOutcome(CommandResult(CommandStatus.SUCCESS, key, {"value": value}))

        if action.startswith("watch_"):
            interval = self._parse_watch_interval(args)
            return DispatchOutcome(
                CommandResult(CommandStatus.SUCCESS, "Watch started"),
                watch_target=action[6:],
                watch_interval_s=interval,
            )

        if action == "sim_status":
            return DispatchOutcome(
                CommandResult(CommandStatus.SUCCESS, "Simulation status", self.world.status())
            )
        if action == "sim_scenario":
            selected = invocation.spec.path[-1]
            result = await self.world.select_scenario(selected)
            return DispatchOutcome(self._from_operation(result))
        if action == "sim_run":
            scenario = await self.world.run_selected()
            self.manager = self.world.manager
            self._service = CommandService(self.manager)
            return DispatchOutcome(
                CommandResult(
                    CommandStatus.SUCCESS if scenario.ok else CommandStatus.FAILED,
                    "Scenario {} {}".format(scenario.name, "passed" if scenario.ok else "failed"),
                    {
                        "final_state": scenario.final_state.name,
                        "states": [state.name for state in scenario.states],
                        "messages": list(scenario.messages),
                        "details": dict(scenario.details),
                    },
                )
            )
        if action == "sim_reset":
            result = await self.world.reset(start=False)
            self.manager = self.world.manager
            self._service = CommandService(self.manager)
            return DispatchOutcome(self._from_operation(result))
        if action == "sim_pause":
            return DispatchOutcome(self._from_operation(await self.world.pause()))
        if action == "sim_step":
            if len(args) > 1:
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, "Usage: sim step [SECONDS]")
                )
            try:
                seconds = float(args[0]) if args else 0.05
            except ValueError:
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, "Usage: sim step [SECONDS]")
                )
            if not math.isfinite(seconds):
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, "Step duration must be finite")
                )
            return DispatchOutcome(self._from_operation(await self.world.step(seconds)))
        if action == "sim_clear":
            result = await self.world.reset(start=True)
            self.manager = self.world.manager
            self._service = CommandService(self.manager)
            return DispatchOutcome(self._from_operation(result))
        if action == "sim_inject":
            if len(args) != 1:
                return DispatchOutcome(
                    CommandResult(CommandStatus.REJECTED, "Usage: sim inject FAULT")
                )
            try:
                fault = SimulatedFault.parse(args[0])
            except ValueError as exc:
                return DispatchOutcome(CommandResult(CommandStatus.REJECTED, str(exc)))
            return DispatchOutcome(self._from_operation(await self.world.inject(fault)))

        if action.startswith("log_"):
            return DispatchOutcome(self._logging_action(action, args))
        if action == "phase_unavailable":
            return DispatchOutcome(
                CommandResult(
                    CommandStatus.UNAVAILABLE,
                    "This command is registered for a later hardware phase",
                )
            )

        service_result = await self._service.run(action, args)
        return DispatchOutcome(service_result)

    def _fault_query(self, invocation: CommandInvocation) -> CommandResult:
        faults = self.manager.query("faults")
        active = list(faults.get("active", ()))
        history = list(faults.get("history", ()))
        path = invocation.spec.path
        if path == ("faults", "history"):
            return CommandResult(
                CommandStatus.SUCCESS,
                "Fault history",
                {"history": history},
            )
        if path == ("faults", "explain"):
            if len(invocation.arguments) != 1:
                return CommandResult(
                    CommandStatus.REJECTED,
                    "Usage: faults explain CODE",
                )
            code = invocation.arguments[0].strip().upper()
            observed = active + history
            detail = next(
                (
                    item
                    for item in reversed(observed)
                    if isinstance(item, Mapping) and str(item.get("code", "")).upper() == code
                ),
                None,
            )
            if detail is None:
                return CommandResult(
                    CommandStatus.REJECTED,
                    f"Fault code {code} has not been observed in this session",
                    {"code": code},
                )
            return CommandResult(
                CommandStatus.SUCCESS,
                f"Fault {code}",
                {
                    "fault": dict(detail),
                    "active": any(
                        isinstance(item, Mapping) and str(item.get("code", "")).upper() == code
                        for item in active
                    ),
                },
            )
        return CommandResult(
            CommandStatus.SUCCESS,
            "Active faults",
            {"active": active},
        )

    def _validate_configuration(self) -> CommandResult:
        path = self.manager.config.source_path
        try:
            errors = validate_config(path)
        except (ConfigurationError, OSError, ValueError) as exc:
            return CommandResult(
                CommandStatus.REJECTED,
                "Configuration validation failed",
                {"valid": False, "path": str(path), "errors": [str(exc)]},
            )
        if errors:
            return CommandResult(
                CommandStatus.REJECTED,
                "Configuration is invalid",
                {"valid": False, "path": str(path), "errors": list(errors)},
            )
        return CommandResult(
            CommandStatus.SUCCESS,
            "Configuration is valid",
            {"valid": True, "path": str(path), "errors": []},
        )

    def _logging_action(self, action: str, args: Tuple[str, ...]) -> CommandResult:
        sink = self._event_sink
        if sink is None:
            return CommandResult(CommandStatus.UNAVAILABLE, "Logging is not configured")
        if action == "log_status":
            return CommandResult(
                CommandStatus.SUCCESS,
                "Logger status",
                {
                    "enabled": sink.enabled,
                    "running": sink.running,
                    "path": str(sink.path),
                    "records_written": sink.records_written,
                },
            )
        if action == "log_mark" and not args:
            return CommandResult(CommandStatus.REJECTED, "Usage: log mark TEXT")
        if action == "log_mark" and not sink.enabled:
            return CommandResult(
                CommandStatus.REJECTED, "Logging is stopped; run `log start` before `log mark`"
            )
        if action == "log_export" and len(args) != 1:
            return CommandResult(CommandStatus.REJECTED, "Usage: log export PATH")
        try:
            if action == "log_start":
                sink.start()
                return CommandResult(CommandStatus.SUCCESS, "Logging started")
            if action == "log_stop":
                return CommandResult(CommandStatus.SUCCESS, "Logging stopped")
            if action == "log_mark":
                snapshot = self.manager.snapshot
                marker = " ".join(args).replace("\r", " ").replace("\n", " ").strip()
                sink.emit(
                    event_type="LOG_MARK",
                    system_state=self.manager.state.name,
                    pixhawk_status=snapshot.pixhawk,
                    f446_status=snapshot.f446,
                    go2_status=snapshot.go2,
                    operator_request=snapshot.operator,
                    safety_violations=self.manager.violations,
                    landing_command=self.manager.last_landing_command,
                    marker_text=marker,
                )
                return CommandResult(CommandStatus.SUCCESS, "Log marker added")
            if action == "log_tail":
                return CommandResult(
                    CommandStatus.SUCCESS,
                    "Recent records",
                    {"records": sink.tail()},
                )
            if action == "log_export":
                path = sink.export(Path(args[0]))
                return CommandResult(CommandStatus.SUCCESS, "Log exported", {"path": str(path)})
        except (OSError, RuntimeError, ValueError) as exc:
            return CommandResult(
                CommandStatus.REJECTED,
                f"Logging action failed: {exc}",
            )
        return CommandResult(CommandStatus.UNAVAILABLE, "Unknown logging action")

    def _validate_arguments(
        self,
        invocation: CommandInvocation,
    ) -> Optional[CommandResult]:
        """Reject malformed or surplus arguments before confirmation/execution."""

        action = invocation.spec.action
        path = invocation.spec.path
        args = invocation.arguments

        if action == "help":
            return None

        if action == "query_status" and path == ("status",):
            if not args or (len(args) == 1 and args[0] in {"--full", "--json"}):
                return None
            if len(args) == 2 and args[0] == "--watch" and self._valid_watch_interval(args[1]):
                return None
            return self._usage_error(invocation)

        if action == "health":
            if not args:
                return None
            if len(args) == 2 and args[0] == "--watch" and self._valid_watch_interval(args[1]):
                return None
            return self._usage_error(invocation)

        if action.startswith("watch_"):
            if not args or (len(args) == 1 and self._valid_watch_interval(args[0])):
                return None
            return self._usage_error(invocation)

        if path == ("faults", "explain"):
            return None if len(args) == 1 and args[0].strip() else self._usage_error(invocation)

        if action in {"config_get", "log_export", "sim_inject"}:
            return None if len(args) == 1 and args[0].strip() else self._usage_error(invocation)

        if action == "log_mark":
            marker = " ".join(args).replace("\r", " ").replace("\n", " ").strip()
            return None if marker else self._usage_error(invocation)

        if action == "sim_step":
            if len(args) > 1:
                return self._usage_error(invocation)
            if not args:
                return None
            try:
                seconds = float(args[0])
            except ValueError:
                return self._usage_error(invocation)
            if not math.isfinite(seconds) or seconds <= 0.0:
                return self._usage_error(invocation)
            return None

        unsigned_motor_values = {
            ("motor", "mf"),
            ("motor", "mr"),
            ("motor", "limf"),
            ("motor", "limr"),
            ("motor", "threshold"),
            ("motor", "threshold-mv"),
            ("motor", "blank"),
            ("motor", "overms"),
            ("motor", "timeout"),
        }
        if path in unsigned_motor_values or path == ("motor", "raw"):
            if len(args) != 1:
                return self._usage_error(invocation)
            try:
                value = int(args[0], 10)
            except ValueError:
                return self._usage_error(invocation)
            if path == ("motor", "raw"):
                valid = -1000 <= value <= 1000
            elif path in {
                ("motor", "mf"),
                ("motor", "mr"),
                ("motor", "limf"),
                ("motor", "limr"),
            }:
                valid = 1 <= value <= 900
            else:
                valid = value >= 0
            return None if valid else self._usage_error(invocation)

        if args:
            return self._usage_error(invocation)
        return None

    @staticmethod
    def _valid_watch_interval(value: str) -> bool:
        try:
            seconds = float(value)
        except ValueError:
            return False
        return math.isfinite(seconds) and 0.1 <= seconds <= 60.0

    @staticmethod
    def _usage_error(invocation: CommandInvocation) -> CommandResult:
        usage = invocation.spec.effective_usage
        return CommandResult(
            CommandStatus.REJECTED,
            f"Usage: {usage}",
            {"code": "INVALID_ARGUMENTS", "usage": usage},
            invocation.command_id,
        )

    def _complete(
        self,
        invocation: CommandInvocation,
        outcome: DispatchOutcome,
        *,
        render: bool,
    ) -> DispatchOutcome:
        result = outcome.result
        if not result.command_id:
            result = replace(result, command_id=invocation.command_id)
            outcome = replace(outcome, result=result)
        try:
            self._log_command(invocation, result)
            if invocation.spec.action == "log_stop" and result.ok:
                if self._event_sink is not None:
                    self._event_sink.stop()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            result = CommandResult(
                CommandStatus.FAILED,
                f"{result.message}; audit logging failed: {type(exc).__name__}: {exc}",
                {
                    "code": "AUDIT_LOG_FAILED",
                    "original_status": result.status.value,
                },
                invocation.command_id,
            )
            outcome = replace(outcome, result=result)
        if render and result.message:
            self.renderer.render_result(result)
        return outcome

    def _log_command(self, invocation: CommandInvocation, result: CommandResult) -> None:
        if self._event_sink is not None:
            snapshot = self.manager.snapshot
            self._event_sink.emit(
                event_type="COMMAND_EXECUTED",
                system_state=self.manager.state.name,
                command_id=invocation.command_id,
                command_name=invocation.command_name,
                command_result={
                    "status": result.status.value,
                    "message": result.message,
                    "data": dict(result.data),
                },
                pixhawk_status=snapshot.pixhawk,
                f446_status=snapshot.f446,
                go2_status=snapshot.go2,
                operator_request=snapshot.operator,
                safety_violations=self.manager.violations,
                landing_command=self.manager.last_landing_command,
                raw_command=invocation.raw,
            )

    def _finish(self, result: CommandResult, render: bool) -> DispatchOutcome:
        if render and result.message:
            self.renderer.render_result(result)
        return DispatchOutcome(result)

    @staticmethod
    def _from_operation(result: Any) -> CommandResult:
        return CommandResult(
            CommandStatus.SUCCESS if result.ok else CommandStatus.REJECTED,
            result.message,
            result.data,
        )

    @staticmethod
    def _parse_watch_interval(args: Tuple[str, ...]) -> float:
        if not args:
            return 0.5
        try:
            value = float(args[-1])
        except ValueError:
            return 0.5
        if not math.isfinite(value):
            return 0.5
        return min(60.0, max(0.1, value))


__all__ = ["CommandDispatcher", "DispatchOutcome"]
