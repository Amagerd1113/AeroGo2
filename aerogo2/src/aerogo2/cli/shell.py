"""Resident prompt-toolkit shell with clean task shutdown."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any, Coroutine, Optional, Set

from prompt_toolkit import PromptSession

from aerogo2.cli.command_models import ConfirmationPolicy
from aerogo2.cli.completer import AeroGo2Completer
from aerogo2.cli.dispatcher import CommandDispatcher, DispatchOutcome
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.renderer import ConsoleRenderer
from aerogo2.common.enums import RuntimeMode, SystemState


class InteractiveShell:
    def __init__(
        self,
        dispatcher: CommandDispatcher,
        *,
        renderer: Optional[ConsoleRenderer] = None,
        history_path: Optional[Path] = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.renderer = dispatcher.renderer if renderer is None else renderer
        if history_path is None:
            self.history = dispatcher.history
        else:
            self.history = CommandHistory(history_path)
            # PromptSession and the `history` command must share one store.
            self.dispatcher.history = self.history
        self._session: Optional[PromptSession[str]] = None
        self._tasks: Set[asyncio.Task[None]] = set()
        self._closing = False
        self._closed = False
        self._watching = False
        self._run_task: Optional[asyncio.Task[Any]] = None

    def _ensure_session(self) -> PromptSession[str]:
        if self._session is None:
            self._session = PromptSession(
                history=self.history,
                completer=AeroGo2Completer(self.dispatcher.registry),
                complete_while_typing=True,
            )
        return self._session

    @property
    def background_task_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    async def run(self) -> int:
        session = self._ensure_session()
        self._run_task = asyncio.current_task()
        await self.dispatcher.manager.start()
        self.renderer.render_banner(
            self.dispatcher.manager.runtime_mode,
            self.dispatcher.manager.snapshot,
            logging_enabled=self.dispatcher._event_sink is not None,
        )
        self._spawn(self._monitor_loop(), "aerogo2-safety-monitor")
        try:
            while not self._closing:
                try:
                    line = await session.prompt_async(
                        self.renderer.prompt_text(
                            self.dispatcher.manager.runtime_mode,
                            self.dispatcher.manager.snapshot,
                        )
                    )
                except KeyboardInterrupt:
                    await self.handle_ctrl_c()
                    continue
                except asyncio.CancelledError:
                    self._consume_current_task_cancellation()
                    if self._closing:
                        break
                    await self.handle_ctrl_c()
                    continue
                except EOFError:
                    outcome = await self.dispatcher.dispatch(
                        "exit",
                        record_history=False,
                    )
                    if outcome.should_exit:
                        break
                    continue
                try:
                    outcome = await self.dispatcher.dispatch(
                        line,
                        record_history=False,
                    )
                except KeyboardInterrupt:
                    await self.handle_ctrl_c()
                    continue
                except asyncio.CancelledError:
                    self._consume_current_task_cancellation()
                    if self._closing:
                        break
                    await self.handle_ctrl_c()
                    continue
                if outcome.watch_target is not None:
                    await self._watch(outcome)
                if outcome.should_exit:
                    break
        finally:
            try:
                await self.close()
            finally:
                self._run_task = None
        return 0

    async def run_line(self, line: str, render: bool = False) -> DispatchOutcome:
        """Test/script entry point using the same parser and dispatcher."""

        return await self.dispatcher.dispatch(line, render=render)

    async def handle_ctrl_c(self) -> None:
        if self._watching:
            self._watching = False
            return
        state = self.dispatcher.manager.state
        if state is SystemState.AUTO_LANDING:
            confirmed = await self.dispatcher.confirmation.confirm(
                ConfirmationPolicy.simple(
                    "Abort automatic landing and return control to RadioMaster?",
                    warning=("Automatic landing is active. Ctrl+C never disarms or stops rotors."),
                )
            )
            if confirmed:
                result = await self.dispatcher.manager.abort_autoland("operator Ctrl+C")
                self.renderer.warning(result.message)
            else:
                self.renderer.warning("Automatic landing continues; abort was declined.")
        elif state in (
            SystemState.WALK_TO_FLIGHT_PRECHECK,
            SystemState.MANUAL_POSITIONING,
            SystemState.TRANSFORM_TO_FLIGHT,
            SystemState.FLIGHT_TO_WALK_PRECHECK,
            SystemState.TRANSFORM_TO_WALK,
        ):
            result = await self.dispatcher.manager.interrupt_transform()
            self.renderer.warning(result.message)

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        current = asyncio.current_task()
        for task in tuple(self._tasks):
            if task is not current:
                task.cancel()
        for task in tuple(self._tasks):
            if task is not current:
                with suppress(asyncio.CancelledError):
                    await task
        self._tasks.clear()
        await self.dispatcher.manager.shutdown()
        self._closed = True

    async def _watch(self, outcome: DispatchOutcome) -> None:
        self._watching = True
        try:
            while self._watching:
                self.renderer.render_snapshot(
                    self.dispatcher.manager.snapshot,
                    full=outcome.watch_target not in ("status", None),
                )
                await asyncio.sleep(outcome.watch_interval_s)
        except KeyboardInterrupt:
            self._watching = False
        except asyncio.CancelledError:
            self._consume_current_task_cancellation()
            self._watching = False

    @staticmethod
    def _consume_current_task_cancellation() -> None:
        """Consume SIGINT-style task cancellation on Python 3.11+."""

        task = asyncio.current_task()
        if task is None:
            return
        uncancel = getattr(task, "uncancel", None)
        if callable(uncancel):
            uncancel()

    async def _monitor_loop(self) -> None:
        interval = 1.0 / self.dispatcher.manager.config.system.loop_hz
        next_current_report_at = 0.0
        try:
            while True:
                manager = self.dispatcher.manager
                if manager.runtime_mode is RuntimeMode.DRY_RUN:
                    snapshot = manager.snapshot
                    all_connected = (
                        snapshot.pixhawk.connected
                        and snapshot.f446.connected
                        and (snapshot.go2.connected or not manager.config.go2.enabled)
                    )
                    if manager.state is SystemState.BOOT_SAFE and not all_connected:
                        await asyncio.sleep(interval)
                        continue
                    await self.dispatcher.world.step(interval)
                else:
                    await manager.tick()
                snapshot = manager.snapshot
                if snapshot.state is SystemState.MANUAL_POSITIONING and snapshot.f446.duty != 0:
                    now = asyncio.get_running_loop().time()
                    if now >= next_current_report_at:
                        self.renderer.render_f446_current(snapshot.f446)
                        next_current_report_at = now + 0.5
                else:
                    next_current_report_at = 0.0
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.renderer.error(f"Safety monitor failed: {message}")
            try:
                await self.dispatcher.manager.report_monitor_failure(message)
            except Exception as report_exc:
                self.renderer.error(
                    "Safety monitor failure reporting also failed: "
                    f"{type(report_exc).__name__}: {report_exc}"
                )
            self._closing = True
            run_task = self._run_task
            if run_task is not None and run_task is not asyncio.current_task():
                run_task.cancel()

    def _spawn(self, coroutine: Coroutine[Any, Any, None], name: str) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


__all__ = ["InteractiveShell"]
