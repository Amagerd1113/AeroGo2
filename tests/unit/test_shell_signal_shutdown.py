from __future__ import annotations

import asyncio
import signal
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from aerogo2.cli.history import CommandHistory
from aerogo2.cli.shell import InteractiveShell
from aerogo2.cli.signal_shutdown import SupervisedSignalRouter
from aerogo2.common.enums import RuntimeMode, SystemState
from aerogo2.common.models import Go2LowLevelStatus, LowCmdOwnershipState, SystemSnapshot
from aerogo2.common.results import OperationResult


class _FakeSignalLoop:
    def __init__(self) -> None:
        self.handlers: dict[signal.Signals, tuple[Any, tuple[Any, ...]]] = {}
        self.removed: list[signal.Signals] = []

    def add_signal_handler(self, sig: signal.Signals, callback: Any, *args: Any) -> None:
        self.handlers[sig] = (callback, args)

    def remove_signal_handler(self, sig: signal.Signals) -> bool:
        self.removed.append(sig)
        return True

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        callback(*args)


def test_signal_router_routes_both_signals_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _FakeSignalLoop()
    requests: list[str] = []
    restored: list[tuple[signal.Signals, Any]] = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: f"previous-{sig.name}")
    monkeypatch.setattr(signal, "signal", lambda sig, handler: restored.append((sig, handler)))
    router = SupervisedSignalRouter(lambda reason: not requests.append(reason))

    router.install(loop)  # type: ignore[arg-type]
    for event_signal in (signal.SIGINT, signal.SIGTERM):
        callback, args = loop.handlers[event_signal]
        callback(*args)
    router.close()
    router.close()

    assert requests == ["process signal SIGINT", "process signal SIGTERM"]
    assert set(loop.removed) == {signal.SIGINT, signal.SIGTERM}
    assert set(restored) == {
        (signal.SIGINT, "previous-SIGINT"),
        (signal.SIGTERM, "previous-SIGTERM"),
    }


class _BlockingSession:
    async def prompt_async(self, _prompt: str) -> str:
        await asyncio.Future()
        raise AssertionError("unreachable")


class _PendingOwnerManager:
    def __init__(self) -> None:
        self.runtime_mode = RuntimeMode.HARDWARE
        self.config = SimpleNamespace(system=SimpleNamespace(loop_hz=200.0))
        base = SystemSnapshot(timestamp=0.0, state=SystemState.BOOT_SAFE)
        self.snapshot = SystemSnapshot(
            timestamp=0.0,
            state=SystemState.BOOT_SAFE,
            go2=replace(
                base.go2,
                low_level_status=Go2LowLevelStatus(
                    ownership_state=LowCmdOwnershipState.SAFE_HOLD,
                    owner_epoch=7,
                    writer_alive=True,
                    safe_hold_active=True,
                ),
            ),
        )
        self.shutdown_calls = 0
        self.shutdown_called = asyncio.Event()

    @property
    def state(self) -> Any:
        return self.snapshot.state

    async def start(self) -> OperationResult:
        return OperationResult.success("started")

    async def tick(self) -> tuple[Any, ...]:
        return ()

    async def shutdown(self) -> OperationResult:
        self.shutdown_calls += 1
        self.shutdown_called.set()
        if self.snapshot.go2.low_level_status.ownership_pending:
            return OperationResult.failure("OWNER_PENDING", "explicit handoff required")
        return OperationResult.success("stopped")


class _Renderer:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def render_banner(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def prompt_text(self, *_args: Any, **_kwargs: Any) -> str:
        return "> "

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, _message: str) -> None:
        pass


@pytest.mark.asyncio
async def test_signal_shutdown_stays_resident_until_lowcmd_owner_is_clear() -> None:
    manager = _PendingOwnerManager()
    renderer = _Renderer()
    dispatcher = SimpleNamespace(
        manager=manager,
        renderer=renderer,
        history=CommandHistory(),
        registry=SimpleNamespace(),
        world=SimpleNamespace(),
        _event_sink=None,
    )
    shell = InteractiveShell(dispatcher, renderer=renderer)  # type: ignore[arg-type]
    shell._session = _BlockingSession()  # type: ignore[assignment]
    run_task = asyncio.create_task(shell.run())
    await asyncio.sleep(0)

    assert shell.request_supervised_shutdown("SIGTERM")
    assert not shell.request_supervised_shutdown("duplicate SIGTERM")
    await asyncio.wait_for(manager.shutdown_called.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not run_task.done()
    assert manager.shutdown_calls == 1

    manager.shutdown_called.clear()
    manager.snapshot = replace(
        manager.snapshot,
        go2=replace(
            manager.snapshot.go2,
            low_level_status=Go2LowLevelStatus(ownership_state=LowCmdOwnershipState.OBSERVE_ONLY),
        ),
    )
    assert shell.request_supervised_shutdown("SIGINT after handoff")
    assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    assert manager.shutdown_calls == 2
    assert any("ownership was not safely handed back" in item for item in renderer.warnings)
