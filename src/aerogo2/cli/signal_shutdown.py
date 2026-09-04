"""Route process termination signals into the supervised shell shutdown path."""

from __future__ import annotations

import asyncio
import signal
from types import FrameType
from typing import Any, Callable, Dict, Optional


class SupervisedSignalRouter:
    """Install removable SIGINT/SIGTERM handlers without doing I/O in a signal callback."""

    def __init__(self, request_shutdown: Callable[[str], bool]) -> None:
        self._request_shutdown = request_shutdown
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._previous: Dict[signal.Signals, Any] = {}
        self._loop_handlers: set[signal.Signals] = set()
        self._fallback_handlers: set[signal.Signals] = set()
        self._closed = False

    def install(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if self._loop is not None or self._closed:
            return
        active_loop = asyncio.get_running_loop() if loop is None else loop
        self._loop = active_loop
        for event_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(event_signal)
                active_loop.add_signal_handler(event_signal, self._route, event_signal)
            except (NotImplementedError, RuntimeError):
                try:
                    previous = signal.getsignal(event_signal)
                    signal.signal(
                        event_signal,
                        self._fallback_handler(active_loop, event_signal),
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                self._previous[event_signal] = previous
                self._fallback_handlers.add(event_signal)
            else:
                self._previous[event_signal] = previous
                self._loop_handlers.add(event_signal)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            for event_signal in self._loop_handlers:
                loop.remove_signal_handler(event_signal)
        for event_signal in self._loop_handlers | self._fallback_handlers:
            previous = self._previous.get(event_signal)
            if previous is None:
                continue
            try:
                signal.signal(event_signal, previous)
            except (OSError, RuntimeError, ValueError):
                pass
        self._loop_handlers.clear()
        self._fallback_handlers.clear()

    def _route(self, event_signal: signal.Signals) -> None:
        if not self._closed:
            self._request_shutdown(f"process signal {event_signal.name}")

    def _fallback_handler(
        self,
        loop: asyncio.AbstractEventLoop,
        event_signal: signal.Signals,
    ) -> Callable[[int, Optional[FrameType]], None]:
        def handler(_received: int, _frame: Optional[FrameType]) -> None:
            loop.call_soon_threadsafe(self._route, event_signal)

        return handler


__all__ = ["SupervisedSignalRouter"]
