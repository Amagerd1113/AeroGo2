"""Cancellation-safe periodic snapshot logging."""

from __future__ import annotations

import asyncio
import logging as standard_logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

from aerogo2.common.models import SystemSnapshot
from aerogo2.logging.event_logger import EventLogger
from aerogo2.logging.schemas import EventType

SnapshotProvider = Callable[[], SystemSnapshot]


@dataclass(frozen=True)
class TelemetryLoggerStatus:
    running: bool
    sample_hz: float
    samples_written: int
    last_error: Optional[str]


class TelemetryLogger:
    """Sample immutable snapshots without blocking console command handling."""

    def __init__(
        self,
        event_logger: EventLogger,
        snapshot_provider: SnapshotProvider,
        *,
        sample_hz: float = 10.0,
    ) -> None:
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("sample_hz must be finite and positive")
        self._event_logger = event_logger
        self._snapshot_provider = snapshot_provider
        self._sample_hz = sample_hz
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_requested = asyncio.Event()
        self._samples_written = 0
        self._last_error: Optional[str] = None
        self._logger = standard_logging.getLogger(__name__)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def sample_hz(self) -> float:
        return self._sample_hz

    def status(self) -> TelemetryLoggerStatus:
        return TelemetryLoggerStatus(
            running=self.running,
            sample_hz=self._sample_hz,
            samples_written=self._samples_written,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        if self.running:
            return
        await self._event_logger.start()
        self._stop_requested.clear()
        self._task = asyncio.create_task(self.run(), name="aerogo2-telemetry-logger")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._stop_requested.set()
        try:
            await task
        finally:
            self._task = None

    pause = stop

    async def resume(self) -> None:
        await self.start()

    def set_sample_hz(self, sample_hz: float) -> None:
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("sample_hz must be finite and positive")
        self._sample_hz = sample_hz

    async def sample_once(self) -> None:
        snapshot = self._snapshot_provider()
        if not isinstance(snapshot, SystemSnapshot):
            raise TypeError("snapshot_provider must return SystemSnapshot")
        await self._event_logger.log(EventType.TELEMETRY, snapshot=snapshot)
        self._samples_written += 1
        self._last_error = None

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_requested.is_set():
            started = loop.time()
            try:
                await self.sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._logger.exception("Telemetry sample failed")
                try:
                    await self._event_logger.log(
                        EventType.LOGGER_ERROR,
                        details={"component": "telemetry", "error": self._last_error},
                    )
                except Exception:
                    self._logger.exception("Could not record telemetry logger failure")

            remaining = max(0.0, (1.0 / self._sample_hz) - (loop.time() - started))
            try:
                await asyncio.wait_for(self._stop_requested.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                continue

    async def __aenter__(self) -> TelemetryLogger:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.stop()


__all__ = ["SnapshotProvider", "TelemetryLogger", "TelemetryLoggerStatus"]
