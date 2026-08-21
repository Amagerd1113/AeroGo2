"""Integration tests for the parser/registry/permission/confirmation boundary."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

import aerogo2.cli
from aerogo2.cli.command_models import CommandContext
from aerogo2.cli.commands import build_registry
from aerogo2.cli.confirmation import ConfirmationService, ScriptedConfirmationReader
from aerogo2.cli.dispatcher import CommandDispatcher
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.shell import InteractiveShell
from aerogo2.common.enums import CommandStatus, RuntimeMode, SystemState
from aerogo2.common.models import PixhawkStatus, SystemSnapshot, snapshot_to_dict


async def _authorize_and_confirm(
    command_name: str,
    context: CommandContext,
    confirmation: ConfirmationService,
) -> bool:
    """Model the dispatcher ordering: permission before confirmation."""

    spec = build_registry().get(command_name)
    decision = spec.permission.evaluate(context)
    if not decision.allowed:
        return False
    return await confirmation.confirm(spec.confirmation)


def test_read_only_command_does_not_prompt_for_confirmation() -> None:
    reader = ScriptedConfirmationReader(())
    confirmation = ConfirmationService(reader)

    accepted = asyncio.run(
        _authorize_and_confirm(
            "status",
            CommandContext(
                runtime_mode=RuntimeMode.DRY_RUN,
                state=SystemState.BOOT_SAFE,
            ),
            confirmation,
        ),
    )

    assert accepted
    assert reader.prompts == []


def test_transform_requires_the_exact_case_sensitive_phrase() -> None:
    wrong_reader = ScriptedConfirmationReader(("transform_to_flight",))
    right_reader = ScriptedConfirmationReader(("TRANSFORM_TO_FLIGHT",))
    context = CommandContext(
        runtime_mode=RuntimeMode.DRY_RUN,
        state=SystemState.WALK,
    )

    wrong = asyncio.run(
        _authorize_and_confirm("transform flight", context, ConfirmationService(wrong_reader))
    )
    right = asyncio.run(
        _authorize_and_confirm("transform flight", context, ConfirmationService(right_reader))
    )

    assert not wrong
    assert right


def test_two_stage_manual_motor_confirmation_requires_both_stages() -> None:
    policy = build_registry().get("motor mr").confirmation
    declined = ScriptedConfirmationReader(("n",))
    wrong_phrase = ScriptedConfirmationReader(("yes", "RUN_MANUAL_motor"))
    accepted = ScriptedConfirmationReader(("yes", "RUN_MANUAL_MOTOR"))

    assert not asyncio.run(ConfirmationService(declined).confirm(policy))
    assert not asyncio.run(ConfirmationService(wrong_phrase).confirm(policy))
    assert asyncio.run(ConfirmationService(accepted).confirm(policy))
    assert len(accepted.prompts) == 2


def test_confirmation_text_never_enters_persistent_history(tmp_path: Path) -> None:
    history_path = tmp_path / "console-history.jsonl"
    history = CommandHistory(history_path)
    history.record("transform flight")
    reader = ScriptedConfirmationReader(("TRANSFORM_TO_FLIGHT",))

    confirmed = asyncio.run(
        ConfirmationService(reader).confirm(build_registry().get("transform flight").confirmation)
    )

    assert confirmed
    assert history.entries() == ("transform flight",)
    assert "TRANSFORM_TO_FLIGHT" not in history_path.read_text(encoding="utf-8")


def test_phase_one_manual_motor_rejection_occurs_before_prompt() -> None:
    reader = ScriptedConfirmationReader(("yes", "RUN_MANUAL_MOTOR"))
    confirmation = ConfirmationService(reader)

    accepted = asyncio.run(
        _authorize_and_confirm(
            "motor mr",
            CommandContext(
                runtime_mode=RuntimeMode.DRY_RUN,
                state=SystemState.BOOT_SAFE,
                phase=1,
                maintenance_mode=False,
            ),
            confirmation,
        ),
    )

    assert not accepted
    assert reader.prompts == []


class _InterruptingReader:
    async def __call__(self, prompt: str) -> NoReturn:
        del prompt
        raise KeyboardInterrupt


def test_ctrl_c_during_confirmation_cancels_without_approving() -> None:
    service = ConfirmationService(_InterruptingReader())

    confirmed = asyncio.run(service.confirm(build_registry().get("transform flight").confirmation))

    assert not confirmed


@pytest.mark.parametrize(
    "forbidden",
    [
        "arm",
        "disarm",
        "takeoff",
        "raw-throttle",
        "pixhawk arm",
        "pixhawk disarm",
        "pixhawk motor-test",
    ],
)
def test_phase_one_does_not_register_forbidden_flight_write_commands(
    forbidden: str,
) -> None:
    registry = build_registry()

    with pytest.raises(LookupError):
        registry.resolve(forbidden)


def test_cli_source_tree_never_imports_device_bridges() -> None:
    """Enforce InteractiveShell -> manager rather than Shell -> Bridge."""

    cli_root = Path(aerogo2.cli.__file__).resolve().parent
    violations = []
    for source_path in cli_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(name.startswith("aerogo2.bridges") for name in imported):
                violations.append(f"{source_path.name}:{node.lineno}")

    assert violations == []


class _ManagerStub:
    def __init__(
        self,
        state: SystemState = SystemState.BOOT_SAFE,
        *,
        armed: bool = False,
    ) -> None:
        self.runtime_mode = RuntimeMode.DRY_RUN
        self.snapshot = SystemSnapshot(
            timestamp=1.0,
            state=state,
            pixhawk=PixhawkStatus(armed=armed),
        )
        self.config = SimpleNamespace(
            system=SimpleNamespace(hardware_write_enabled=False, loop_hz=50.0),
            esc=SimpleNamespace(slots={1: "RR", 2: "LF", 3: "LR", 4: "RF"}),
        )
        self._safety_monitor = SimpleNamespace(evaluate=lambda snapshot: ())
        self.stop_calls = 0
        self.interrupt_calls = 0
        self.shutdown_calls = 0

    @property
    def state(self) -> SystemState:
        return self.snapshot.state

    async def start(self) -> Any:
        return SimpleNamespace(ok=True, message="started", data={})

    async def stop_supervised(self) -> Any:
        self.stop_calls += 1
        return SimpleNamespace(ok=True, message="stopped", data={})

    async def interrupt_transform(self) -> Any:
        self.interrupt_calls += 1
        self.stop_calls += 1
        self.snapshot = replace(
            self.snapshot,
            state=SystemState.FAULT,
            active_fault_codes=("TRANSFORM_INTERRUPTED",),
        )
        return SimpleNamespace(ok=True, message="transform interrupted", data={})

    async def shutdown(self) -> Any:
        self.shutdown_calls += 1
        return SimpleNamespace(ok=True, message="shutdown", data={})

    async def tick(self) -> tuple[Any, ...]:
        return ()

    def query(self, name: str) -> dict[str, Any]:
        del name
        return snapshot_to_dict(self.snapshot)


class _RendererSpy:
    def __init__(self) -> None:
        self.help_specs: tuple[Any, ...] = ()
        self.command_help: Any = None
        self.errors: list[str] = []
        self.suggestions: tuple[str, ...] = ()
        self.console = SimpleNamespace(clear=lambda: None)

    def render_help(self, specs: Any) -> None:
        self.help_specs = tuple(specs)

    def render_command_help(self, spec: Any) -> None:
        self.command_help = spec

    def render_result(self, result: Any) -> None:
        del result

    def render_suggestions(self, suggestions: Any) -> None:
        self.suggestions = tuple(suggestions)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        del message


def _dispatcher(
    manager: _ManagerStub,
    *,
    reader: ScriptedConfirmationReader | None = None,
    history: CommandHistory | None = None,
    renderer: _RendererSpy | None = None,
) -> CommandDispatcher:
    world = SimpleNamespace(manager=manager)
    return CommandDispatcher(
        build_registry(),
        world,
        renderer=renderer,
        confirmation=ConfirmationService(reader or ScriptedConfirmationReader(())),
        history=history or CommandHistory(),
    )


def test_dispatcher_help_lists_tree_and_resolves_command_help() -> None:
    renderer = _RendererSpy()
    dispatcher = _dispatcher(_ManagerStub(), renderer=renderer)

    list_result = asyncio.run(dispatcher.dispatch("help", render=False))
    command_result = asyncio.run(dispatcher.dispatch("help transform flight", render=False))

    assert list_result.result.status is CommandStatus.SUCCESS
    assert {spec.name for spec in renderer.help_specs} == set(build_registry().command_names())
    assert command_result.result.status is CommandStatus.SUCCESS
    assert renderer.command_help.name == "transform flight"


def test_dispatcher_rejects_phase_one_motor_raw_before_prompt() -> None:
    reader = ScriptedConfirmationReader(("yes", "RUN_MANUAL_MOTOR"))
    dispatcher = _dispatcher(_ManagerStub(), reader=reader)

    outcome = asyncio.run(dispatcher.dispatch("motor mr 120", render=False))

    assert outcome.result.status is CommandStatus.UNAVAILABLE
    assert outcome.result.data["code"] == "STATE_DENIED"
    assert reader.prompts == []


def test_armed_exit_requires_exact_phrase_and_never_records_it(tmp_path: Path) -> None:
    history = CommandHistory(tmp_path / "armed-exit-history.jsonl")
    wrong_reader = ScriptedConfirmationReader(("exit_while_armed",))
    wrong_manager = _ManagerStub(armed=True)
    wrong_dispatcher = _dispatcher(wrong_manager, reader=wrong_reader, history=history)

    rejected = asyncio.run(wrong_dispatcher.dispatch("exit", render=False))

    assert rejected.result.status is CommandStatus.REJECTED
    assert not rejected.should_exit
    assert wrong_manager.stop_calls == 0

    accepted_reader = ScriptedConfirmationReader(("EXIT_WHILE_ARMED",))
    accepted_manager = _ManagerStub(armed=True)
    accepted_dispatcher = _dispatcher(accepted_manager, reader=accepted_reader, history=history)
    accepted = asyncio.run(accepted_dispatcher.dispatch("exit", render=False))

    assert accepted.should_exit
    assert accepted_manager.stop_calls == 1
    assert "EXIT_WHILE_ARMED" not in history.path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("line", "target", "interval"),
    [
        ("watch rc 0.25", "rc", 0.25),
        ("status --watch 0.5", "status", 0.5),
        ("health --watch 1", "health", 1.0),
    ],
)
def test_watch_forms_return_target_and_bounded_interval(
    line: str, target: str, interval: float
) -> None:
    dispatcher = _dispatcher(_ManagerStub())

    outcome = asyncio.run(dispatcher.dispatch(line, render=False))

    assert outcome.result.status is CommandStatus.SUCCESS
    assert outcome.watch_target == target
    assert outcome.watch_interval_s == pytest.approx(interval)


def test_ctrl_c_exits_watch_without_supervised_stop() -> None:
    manager = _ManagerStub()
    shell = InteractiveShell(_dispatcher(manager))
    shell._watching = True

    asyncio.run(shell.handle_ctrl_c())

    assert not shell._watching
    assert manager.stop_calls == 0


def test_ctrl_c_during_transform_requests_supervised_stop() -> None:
    manager = _ManagerStub(SystemState.TRANSFORM_TO_FLIGHT)
    shell = InteractiveShell(_dispatcher(manager))

    asyncio.run(shell.handle_ctrl_c())

    assert manager.interrupt_calls == 1
    assert manager.stop_calls == 1
    assert manager.state is SystemState.FAULT
    assert manager.snapshot.active_fault_codes == ("TRANSFORM_INTERRUPTED",)


def test_shell_close_cancels_and_joins_all_background_tasks() -> None:
    manager = _ManagerStub()
    shell = InteractiveShell(_dispatcher(manager))

    async def exercise() -> None:
        blocker = asyncio.Event()
        shell._spawn(blocker.wait(), "test-background-task")
        assert shell.background_task_count == 1
        await shell.close()

    asyncio.run(exercise())

    assert shell.background_task_count == 0
    assert manager.shutdown_calls == 1
