"""Regression coverage for the resident Phase 1 CLI/runtime boundary."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from io import StringIO
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from aerogo2.cli.commands import build_registry
from aerogo2.cli.confirmation import ConfirmationService, ScriptedConfirmationReader
from aerogo2.cli.dispatcher import CommandDispatcher
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.renderer import ConsoleRenderer
from aerogo2.cli.shell import InteractiveShell
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import CommandStatus, RuntimeMode, SystemState
from aerogo2.common.models import LandingCommand, SystemSnapshot, snapshot_to_dict
from aerogo2.common.results import OperationResult
from aerogo2.logging.ordered_event_sink import OrderedEventSink
from aerogo2.manager.command_service import CommandService
from aerogo2.simulation.world import SimulationWorld


class _RendererSpy:
    def __init__(self) -> None:
        self.snapshots: list[tuple[bool, bool]] = []
        self.results: list[Any] = []
        self.command_help: Any = None
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.console = SimpleNamespace(clear=lambda: None)

    def render_snapshot(
        self,
        snapshot: SystemSnapshot,
        *,
        full: bool = False,
        as_json: bool = False,
    ) -> None:
        del snapshot
        self.snapshots.append((full, as_json))

    def render_result(self, result: Any) -> None:
        self.results.append(result)

    def render_help(self, specs: Any) -> None:
        del specs

    def render_command_help(self, spec: Any) -> None:
        self.command_help = spec

    def render_suggestions(self, suggestions: Any) -> None:
        del suggestions

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _ManagerStub:
    def __init__(
        self,
        config: AppConfig,
        *,
        state: SystemState = SystemState.BOOT_SAFE,
    ) -> None:
        self.config = config
        self.runtime_mode = RuntimeMode.DRY_RUN
        self.snapshot = SystemSnapshot(timestamp=1.0, state=state)
        self.calls: list[tuple[str, str | None]] = []
        self.monitor_failures: list[str] = []
        self.tick_calls = 0

    @property
    def state(self) -> SystemState:
        return self.snapshot.state

    @property
    def violations(self) -> tuple[Any, ...]:
        return ()

    @property
    def last_landing_command(self) -> LandingCommand:
        return LandingCommand(timestamp=self.snapshot.timestamp)

    def query(self, name: str) -> dict[str, Any]:
        if name == "faults":
            return {
                "active": [
                    {
                        "code": "F446_FAULT",
                        "severity": "FAULT",
                        "message": "motor fault",
                        "recommended_action": "stop",
                    }
                ],
                "history": [
                    {
                        "code": "RC_TIMEOUT",
                        "severity": "FAULT",
                        "message": "receiver stale",
                        "recommended_action": "inspect RC",
                    }
                ],
            }
        if name != "status":
            return {"view": name}
        return snapshot_to_dict(self.snapshot)

    async def preflight(self, profile: str) -> OperationResult:
        self.calls.append(("preflight", profile))
        return OperationResult.success(
            f"Preflight {profile} passed",
            {"profile": profile, "permitted": True},
        )

    async def stop_supervised(self) -> OperationResult:
        self.calls.append(("stop_supervised", None))
        return OperationResult.success("supervised stop")

    async def stop_transform_motion(self) -> OperationResult:
        self.calls.append(("stop_transform_motion", None))
        return OperationResult.success("transform stop")

    async def request_transform_flight(self, operator_confirmed: bool) -> OperationResult:
        self.calls.append(("request_transform_flight", str(operator_confirmed)))
        return OperationResult.success("transformed to flight")

    async def request_transform_walk(self, operator_confirmed: bool) -> OperationResult:
        self.calls.append(("request_transform_walk", str(operator_confirmed)))
        return OperationResult.success("transformed to walk")

    async def connect_device(self, name: str) -> OperationResult:
        self.calls.append(("connect_device", name))
        return OperationResult.success(f"{name} connected")

    async def disconnect_device(self, name: str) -> OperationResult:
        self.calls.append(("disconnect_device", name))
        return OperationResult.success(f"{name} disconnected")

    async def walk_stop(self) -> OperationResult:
        self.calls.append(("walk_stop", None))
        return OperationResult.success("walk stopped")

    async def walk_stand(self) -> OperationResult:
        self.calls.append(("walk_stand", None))
        return OperationResult.success("standing")

    async def reset_controller(self) -> OperationResult:
        self.calls.append(("reset_controller", None))
        return OperationResult.success("controller reset")

    async def config_diff(self) -> OperationResult:
        self.calls.append(("config_diff", None))
        return OperationResult.success("config diff", {"changes": []})

    async def reload_config(self) -> OperationResult:
        self.calls.append(("reload_config", None))
        return OperationResult.success("config reloaded")

    async def abort_autoland(self, reason: str = "operator request") -> OperationResult:
        self.calls.append(("abort_autoland", reason))
        return OperationResult.success("autoland aborted")

    async def report_monitor_failure(self, message: str) -> OperationResult:
        self.monitor_failures.append(message)
        return OperationResult.success("monitor failure reported")

    async def tick(self) -> tuple[Any, ...]:
        self.tick_calls += 1
        return ()

    async def shutdown(self) -> OperationResult:
        return OperationResult.success("shutdown")


class _WorldStub:
    def __init__(self, manager: _ManagerStub) -> None:
        self.manager = manager
        self.step_calls: list[float] = []
        self.step_event = asyncio.Event()
        self.step_error: Exception | None = None
        self.control_calls: list[tuple[str, Any]] = []
        self.paused = False

    async def step(self, seconds: float) -> OperationResult:
        self.step_calls.append(seconds)
        self.step_event.set()
        if self.step_error is not None:
            raise self.step_error
        return OperationResult.success("advanced")

    async def select_scenario(self, name: str) -> OperationResult:
        self.control_calls.append(("select_scenario", name))
        return OperationResult.success(f"selected {name}")

    async def pause(self) -> OperationResult:
        self.paused = True
        self.control_calls.append(("pause", None))
        return OperationResult.success("paused")

    async def inject(self, fault: Any) -> OperationResult:
        self.control_calls.append(("inject", fault.value))
        return OperationResult.success(f"injected {fault.value}")


def _dispatcher(
    manager: _ManagerStub,
    *,
    renderer: Any = None,
    responses: tuple[str, ...] = (),
    event_sink: Any = None,
) -> CommandDispatcher:
    world = _WorldStub(manager)
    return CommandDispatcher(
        build_registry(),
        world,  # type: ignore[arg-type]
        renderer=renderer,  # type: ignore[arg-type]
        confirmation=ConfirmationService(ScriptedConfirmationReader(responses)),
        history=CommandHistory(),
        event_sink=event_sink,
    )


@pytest.mark.asyncio
async def test_help_accepts_a_quoted_multiword_command(app_config: AppConfig) -> None:
    renderer = _RendererSpy()
    dispatcher = _dispatcher(_ManagerStub(app_config), renderer=renderer)

    outcome = await dispatcher.dispatch('help "transform flight"', render=False)

    assert outcome.result.status is CommandStatus.SUCCESS
    assert renderer.command_help.name == "transform flight"


@pytest.mark.asyncio
async def test_dispatch_can_defer_history_to_prompt_toolkit(app_config: AppConfig) -> None:
    dispatcher = _dispatcher(_ManagerStub(app_config))

    await dispatcher.dispatch("status", render=False, record_history=False)

    assert dispatcher.history.entries() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["not-a-number", "nan", "inf"])
async def test_invalid_sim_step_is_rejected_without_raising(
    app_config: AppConfig,
    argument: str,
) -> None:
    outcome = await _dispatcher(_ManagerStub(app_config)).dispatch(
        f"sim step {argument}",
        render=False,
    )

    assert outcome.result.status is CommandStatus.REJECTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "line",
    [
        "stop unexpected",
        "transform flight unexpected",
        "status --full unexpected",
        "health --watch nan",
        "watch rc 0.01",
        "motor mr not-an-integer",
        "motor mr 901",
        "sim step -1",
    ],
)
async def test_malformed_or_surplus_arguments_fail_before_execution(
    app_config: AppConfig,
    line: str,
) -> None:
    manager = _ManagerStub(app_config)

    outcome = await _dispatcher(manager).dispatch(line, render=False)

    assert outcome.result.status is CommandStatus.REJECTED
    assert outcome.result.data["code"] == "INVALID_ARGUMENTS"
    assert manager.calls == []


@pytest.mark.asyncio
async def test_simulation_cli_awaits_serialized_world_controls(app_config: AppConfig) -> None:
    dispatcher = _dispatcher(_ManagerStub(app_config))

    selected = await dispatcher.dispatch("sim scenario nominal", render=False)
    paused = await dispatcher.dispatch("sim pause", render=False)
    injected = await dispatcher.dispatch("sim inject rc-loss", render=False)

    assert [selected.result.status, paused.result.status, injected.result.status] == [
        CommandStatus.SUCCESS,
        CommandStatus.SUCCESS,
        CommandStatus.SUCCESS,
    ]
    assert dispatcher.world.control_calls == [
        ("select_scenario", "nominal"),
        ("pause", None),
        ("inject", "RC_LOSS"),
    ]


@pytest.mark.asyncio
async def test_transform_preflight_is_rendered_before_exact_confirmation(
    app_config: AppConfig,
) -> None:
    manager = _ManagerStub(app_config, state=SystemState.WALK)
    renderer = _RendererSpy()
    warnings: list[str] = []

    async def reader(prompt: str) -> str:
        assert renderer.results
        assert renderer.results[0].data["profile"] == "transform-flight"
        assert "TRANSFORM_TO_FLIGHT" in prompt
        return "TRANSFORM_TO_FLIGHT"

    dispatcher = _dispatcher(manager, renderer=renderer)
    dispatcher.confirmation = ConfirmationService(reader, warnings.append)

    outcome = await dispatcher.dispatch("transform flight", render=True)

    assert outcome.result.status is CommandStatus.SUCCESS
    assert manager.calls == [
        ("preflight", "transform-flight"),
        ("request_transform_flight", "True"),
    ]
    assert len(renderer.results) == 2
    assert "Go2 original remote" in warnings[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "line",
    ["state", "rc raw", "pixhawk params", "go2 controller", "esc mapping", "controller timing"],
)
async def test_semantic_queries_use_the_public_manager_query_api(
    app_config: AppConfig,
    line: str,
) -> None:
    outcome = await _dispatcher(_ManagerStub(app_config)).dispatch(line, render=False)

    assert outcome.result.status is CommandStatus.SUCCESS
    assert outcome.result.data == {"view": line}


@pytest.mark.asyncio
async def test_config_show_renders_nested_immutable_mappings_as_json(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _ManagerStub(app_config)
    nested = MappingProxyType({"system": MappingProxyType({"dry_run": True, "phase": 1})})

    def config_query(name: str) -> Any:
        assert name == "config show"
        return nested

    monkeypatch.setattr(manager, "query", config_query)
    output = StringIO()
    renderer = ConsoleRenderer(Console(file=output, force_terminal=False, width=120))
    outcome = await _dispatcher(manager, renderer=renderer).dispatch(
        "config show",
        render=True,
    )

    assert outcome.result.status is CommandStatus.SUCCESS
    assert '"dry_run": true' in output.getvalue()
    assert outcome.result.data == nested


@pytest.mark.asyncio
async def test_permission_confirmation_and_action_failures_are_audited(
    tmp_path: Path,
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denied_sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(30.0),
        filename="permission.jsonl",
    )
    denied_manager = _ManagerStub(app_config)
    denied_manager.runtime_mode = RuntimeMode.HARDWARE_READONLY
    denied = await _dispatcher(denied_manager, event_sink=denied_sink).dispatch(
        "stop", render=False
    )
    assert denied.result.status is CommandStatus.UNAVAILABLE
    assert denied_sink.tail(1)[0]["command_result"]["data"]["code"] == "HARDWARE_WRITE_DISABLED"
    denied_sink.stop()

    confirmation_sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(31.0),
        filename="confirmation.jsonl",
    )
    confirmation_manager = _ManagerStub(app_config, state=SystemState.WALK)
    confirmation_dispatcher = _dispatcher(confirmation_manager, event_sink=confirmation_sink)

    async def reject_exact_phrase(prompt: str) -> str:
        del prompt
        return "NEVER_PERSIST_THIS_CONFIRMATION"

    confirmation_dispatcher.confirmation = ConfirmationService(
        reject_exact_phrase,
        lambda message: None,
    )
    rejected = await confirmation_dispatcher.dispatch("transform flight", render=False)
    assert rejected.result.status is CommandStatus.REJECTED
    assert confirmation_dispatcher.history.entries() == ("transform flight",)
    assert "NEVER_PERSIST_THIS_CONFIRMATION" not in confirmation_sink.path.read_text(
        encoding="utf-8"
    )
    assert confirmation_sink.tail(1)[0]["command_result"]["status"] == "REJECTED"
    confirmation_sink.stop()

    failure_sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(32.0),
        filename="failure.jsonl",
    )
    failure_manager = _ManagerStub(app_config)

    def explode(name: str) -> dict[str, Any]:
        raise RuntimeError(f"query exploded: {name}")

    monkeypatch.setattr(failure_manager, "query", explode)
    failed = await _dispatcher(failure_manager, event_sink=failure_sink).dispatch(
        "state", render=False
    )
    assert failed.result.status is CommandStatus.FAILED
    assert failed.result.data["code"] == "COMMAND_EXCEPTION"
    assert failure_sink.tail(1)[0]["command_result"]["status"] == "FAILED"
    failure_sink.stop()


@pytest.mark.asyncio
async def test_custom_shell_history_is_shared_without_duplicate_records(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    dispatcher = _dispatcher(_ManagerStub(app_config))
    shell = InteractiveShell(dispatcher, history_path=tmp_path / "history.jsonl")

    shell.history.record("status")


@pytest.mark.asyncio
async def test_command_logging_has_context_and_stop_remains_stopped(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(10.0),
        filename="commands.jsonl",
    )
    dispatcher = _dispatcher(_ManagerStub(app_config), event_sink=sink)

    await dispatcher.dispatch("status", render=False)
    command_record = sink.tail(1)[0]
    assert command_record["event_type"] == "COMMAND_EXECUTED"
    assert command_record["pixhawk_status"] is not None
    assert command_record["f446_status"] is not None
    assert command_record["go2_status"] is not None
    assert command_record["operator_request"] is not None
    assert command_record["safety_violations"] == []
    assert command_record["landing_command"] is not None
    assert command_record["details"]["raw_command"] == "status"
    assert command_record["command_result"]["status"] == "SUCCESS"

    before_stop = sink.records_written
    stopped = await dispatcher.dispatch("log stop", render=False)
    await dispatcher.dispatch("status", render=False)
    assert stopped.result.status is CommandStatus.SUCCESS
    assert not sink.enabled
    assert not sink.running
    assert sink.records_written == before_stop + 1

    started = await dispatcher.dispatch("log start", render=False)
    assert started.result.status is CommandStatus.SUCCESS
    assert sink.enabled
    assert sink.running
    assert sink.records_written == before_stop + 2
    sink.stop()


@pytest.mark.asyncio
async def test_logging_argument_and_filesystem_errors_are_results(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(20.0),
        filename="source.jsonl",
    )
    dispatcher = _dispatcher(_ManagerStub(app_config), event_sink=sink)
    empty_mark = await dispatcher.dispatch("log mark", render=False)
    destination = tmp_path / "existing.jsonl"
    destination.write_text("do not overwrite", encoding="utf-8")
    existing_export = await dispatcher.dispatch(
        f'log export "{destination}"',
        render=False,
    )

    assert empty_mark.result.status is CommandStatus.REJECTED
    assert existing_export.result.status is CommandStatus.REJECTED
    assert destination.read_text(encoding="utf-8") == "do not overwrite"
    sink.stop()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_cli_log_marker_has_full_context_and_cannot_revive_stopped_logging(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(25.0),
        filename="markers.jsonl",
    )
    dispatcher = _dispatcher(_ManagerStub(app_config), event_sink=sink)

    marked = await dispatcher.dispatch("log mark operator-note", render=False)
    marker = next(record for record in sink.tail(2) if record["event_type"] == "LOG_MARK")

    assert marked.result.status is CommandStatus.SUCCESS
    assert marker["pixhawk_status"] is not None
    assert marker["f446_status"] is not None
    assert marker["go2_status"] is not None
    assert marker["operator_request"] is not None
    assert marker["safety_violations"] == []
    assert marker["landing_command"] is not None
    assert marker["details"]["marker_text"] == "operator-note"

    stopped = await dispatcher.dispatch("log stop", render=False)
    after_stop = sink.records_written
    rejected = await dispatcher.dispatch(
        "log mark must-not-restart",
        render=False,
    )

    assert stopped.result.status is CommandStatus.SUCCESS
    assert rejected.result.status is CommandStatus.REJECTED
    assert not sink.enabled
    assert not sink.running
    assert sink.records_written == after_stop


async def test_status_full_and_json_use_dedicated_snapshot_renderer(
    app_config: AppConfig,
) -> None:
    renderer = _RendererSpy()
    dispatcher = _dispatcher(_ManagerStub(app_config), renderer=renderer)

    full = await dispatcher.dispatch("status --full", render=True)
    as_json = await dispatcher.dispatch("status --json", render=True)

    assert full.result.status is CommandStatus.SUCCESS
    assert as_json.result.status is CommandStatus.SUCCESS
    assert renderer.snapshots == [(True, False), (False, True)]


@pytest.mark.asyncio
async def test_fault_commands_return_distinct_views(app_config: AppConfig) -> None:
    dispatcher = _dispatcher(_ManagerStub(app_config))

    active = await dispatcher.dispatch("faults active", render=False)
    history = await dispatcher.dispatch("faults history", render=False)
    explained = await dispatcher.dispatch("faults explain rc_timeout", render=False)
    unknown = await dispatcher.dispatch("faults explain never_seen", render=False)

    assert set(active.result.data) == {"active"}
    assert set(history.result.data) == {"history"}
    assert explained.result.data["fault"]["code"] == "RC_TIMEOUT"
    assert explained.result.data["active"] is False
    assert unknown.result.status is CommandStatus.REJECTED


@pytest.mark.asyncio
async def test_config_validate_uses_the_config_source(app_config: AppConfig) -> None:
    outcome = await _dispatcher(_ManagerStub(app_config)).dispatch(
        "config validate",
        render=False,
    )

    assert outcome.result.status is CommandStatus.SUCCESS
    assert outcome.result.data["valid"] is True
    assert outcome.result.data["path"] == str(app_config.source_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("connect_pixhawk", ("connect_device", "pixhawk")),
        ("disconnect_go2", ("disconnect_device", "go2")),
        ("walk_stop", ("walk_stop", None)),
        ("walk_stand", ("walk_stand", None)),
        ("controller_reset", ("reset_controller", None)),
        ("config_diff", ("config_diff", None)),
        ("config_reload", ("reload_config", None)),
    ],
)
async def test_command_service_routes_phase_one_actions(
    app_config: AppConfig,
    action: str,
    expected: tuple[str, str | None],
) -> None:
    manager = _ManagerStub(app_config)

    result = await CommandService(manager).run(action, ())

    assert result.status is CommandStatus.SUCCESS
    assert manager.calls == [expected]


@pytest.mark.asyncio
async def test_top_level_and_transform_stop_have_distinct_semantics(
    app_config: AppConfig,
) -> None:
    manager = _ManagerStub(app_config)
    dispatcher = _dispatcher(manager)

    await dispatcher.dispatch("stop", render=False)
    await dispatcher.dispatch("s", render=False)
    await dispatcher.dispatch("transform stop", render=False)
    await dispatcher.dispatch("motor stop", render=False)
    await dispatcher.dispatch("ms", render=False)

    assert manager.calls == [
        ("stop_supervised", None),
        ("stop_supervised", None),
        ("stop_transform_motion", None),
        ("stop_transform_motion", None),
        ("stop_transform_motion", None),
    ]


@pytest.mark.asyncio
async def test_ctrl_c_can_confirm_direct_autoland_abort(app_config: AppConfig) -> None:
    manager = _ManagerStub(app_config, state=SystemState.AUTO_LANDING)
    renderer = _RendererSpy()
    shell = InteractiveShell(
        _dispatcher(manager, renderer=renderer, responses=("yes",)),
        renderer=renderer,  # type: ignore[arg-type]
    )

    await shell.handle_ctrl_c()

    assert manager.calls == [("abort_autoland", "operator Ctrl+C")]
    assert renderer.warnings[-1] == "autoland aborted"


@pytest.mark.asyncio
async def test_dry_run_monitor_advances_world_instead_of_direct_tick(
    app_config: AppConfig,
) -> None:
    manager = _ManagerStub(app_config, state=SystemState.WALK)
    dispatcher = _dispatcher(manager)
    world = dispatcher.world
    shell = InteractiveShell(dispatcher)

    task = asyncio.create_task(shell._monitor_loop())
    await asyncio.wait_for(world.step_event.wait(), timeout=1.0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert world.step_calls
    assert manager.tick_calls == 0


@pytest.mark.asyncio
async def test_monitor_does_not_latch_expected_boot_safe_disconnects(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    await world.manager.start()
    dispatcher = CommandDispatcher(build_registry(), world, history=CommandHistory())
    shell = InteractiveShell(dispatcher)
    monitor = asyncio.create_task(shell._monitor_loop())
    try:
        await asyncio.sleep(0.03)
        assert world.manager.state is SystemState.BOOT_SAFE
        assert world.manager.snapshot.active_fault_codes == ()

        for line in ("connect pixhawk", "connect f446", "connect go2"):
            outcome = await dispatcher.dispatch(line, render=False)
            assert outcome.result.status is CommandStatus.SUCCESS

        await asyncio.sleep(0.03)
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.active_fault_codes == ()
    finally:
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        await world.manager.shutdown()


@pytest.mark.asyncio
async def test_monitor_exception_is_reported_and_requests_shell_close(
    app_config: AppConfig,
) -> None:
    manager = _ManagerStub(app_config, state=SystemState.WALK)
    renderer = _RendererSpy()
    dispatcher = _dispatcher(manager, renderer=renderer)
    world = dispatcher.world
    world.step_error = RuntimeError("simulated monitor crash")
    shell = InteractiveShell(dispatcher, renderer=renderer)  # type: ignore[arg-type]

    await shell._monitor_loop()

    assert shell._closing
    assert manager.monitor_failures == ["RuntimeError: simulated monitor crash"]
    assert renderer.errors == ["Safety monitor failed: RuntimeError: simulated monitor crash"]
