"""Append-only JSONL event logging for safety and audit events."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TextIO, Tuple

from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.models import (
    LandingCommand,
    SafetyViolation,
    SystemSnapshot,
    TransitionRecord,
)
from aerogo2.common.results import CommandResult
from aerogo2.logging.schemas import EventType, LogRecord


@dataclass(frozen=True)
class LoggerStatus:
    running: bool
    path: Path
    records_written: int


class EventLogger:
    """Serialize all writes through one lock and flush every complete record."""

    def __init__(
        self,
        destination: Path,
        *,
        clock: Optional[Clock] = None,
        filename: Optional[str] = None,
    ) -> None:
        self._clock = clock or RealClock()
        supplied = Path(destination)
        self._auto_filename = filename is None and supplied.suffix.lower() != ".jsonl"
        if filename is not None:
            self._path = supplied / filename
        elif supplied.suffix.lower() == ".jsonl":
            self._path = supplied
        else:
            timestamp = datetime.fromtimestamp(self._clock.wall_time(), tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            self._path = supplied / f"events-{timestamp}.jsonl"
        self._base_path = self._path
        self._handle: Optional[TextIO] = None
        self._enabled = True
        self._lock = asyncio.Lock()
        self._records_written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._handle.closed

    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def enabled(self) -> bool:
        return self._enabled

    def status(self) -> LoggerStatus:
        return LoggerStatus(self.running, self._path, self._records_written)

    async def start(self) -> Path:
        async with self._lock:
            self._enabled = True
            if self.running:
                return self._path
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._auto_filename:
                base = self._base_path
                suffix = 0
                while True:
                    candidate = (
                        base
                        if suffix == 0
                        else base.with_name(f"{base.stem}-{suffix}{base.suffix}")
                    )
                    try:
                        handle = candidate.open("x", encoding="utf-8", newline="\n")
                    except FileExistsError:
                        suffix += 1
                        continue
                    self._path = candidate
                    self._handle = handle
                    break
            else:
                self._handle = self._path.open("a", encoding="utf-8", newline="\n")
        return self._path

    async def stop(self) -> None:
        async with self._lock:
            self._enabled = False
            handle = self._handle
            self._handle = None
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    async def emit(self, record: LogRecord) -> None:
        if not self._enabled:
            raise RuntimeError("Event logger is stopped; call start() before writing")
        if not self.running:
            await self.start()
        payload = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            if self._handle is None or self._handle.closed:
                raise RuntimeError("Event logger stopped while a record was pending")
            self._handle.write(payload)
            self._handle.write("\n")
            self._handle.flush()
            self._records_written += 1

    async def log(
        self,
        event_type: Any,
        *,
        snapshot: Optional[SystemSnapshot] = None,
        previous_state: Optional[Any] = None,
        command_id: Optional[str] = None,
        command_name: Optional[str] = None,
        command_result: Optional[CommandResult] = None,
        safety_violations: Sequence[SafetyViolation] = (),
        transition_reason: Optional[str] = None,
        landing_command: Optional[LandingCommand] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> LogRecord:
        record = LogRecord.create(
            clock=self._clock,
            event_type=event_type,
            snapshot=snapshot,
            previous_state=previous_state,
            command_id=command_id,
            command_name=command_name,
            command_result=command_result,
            safety_violations=safety_violations,
            transition_reason=transition_reason,
            landing_command=landing_command,
            details=details,
        )
        await self.emit(record)
        return record

    log_event = log

    async def log_command(
        self,
        command_name: str,
        result: CommandResult,
        *,
        snapshot: Optional[SystemSnapshot] = None,
        command_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> LogRecord:
        return await self.log(
            EventType.COMMAND_EXECUTED,
            snapshot=snapshot,
            command_id=command_id or result.command_id or None,
            command_name=command_name,
            command_result=result,
            details=details,
        )

    async def log_transition(
        self,
        transition: TransitionRecord,
        *,
        snapshot: Optional[SystemSnapshot] = None,
        event_type: Any = "STATE_TRANSITION",
        safety_violations: Sequence[SafetyViolation] = (),
    ) -> LogRecord:
        return await self.log(
            event_type,
            snapshot=snapshot,
            previous_state=transition.previous_state,
            safety_violations=safety_violations,
            transition_reason=transition.reason,
            details={
                "new_state": transition.new_state.name,
                "permitted": transition.permitted,
                "guard_codes": transition.guard_codes,
                "entry_action_error": transition.entry_action_error,
            },
        )

    async def mark(self, text: str, *, snapshot: Optional[SystemSnapshot] = None) -> LogRecord:
        normalized = text.replace("\r", " ").replace("\n", " ").strip()
        if not normalized:
            raise ValueError("Log marker text cannot be empty")
        return await self.log(
            EventType.LOG_MARK,
            snapshot=snapshot,
            details={"text": normalized},
        )

    async def tail(self, limit: int = 50) -> Tuple[Mapping[str, Any], ...]:
        if limit < 0:
            raise ValueError("limit cannot be negative")
        if limit == 0 or not self._path.exists():
            return ()
        async with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.flush()
            lines = self._path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"_invalid_json": line}
            if isinstance(value, Mapping):
                records.append(value)
        return tuple(records)

    async def export(self, destination: Path, *, overwrite: bool = False) -> Path:
        target = Path(destination)
        if target.resolve() == self._path.resolve():
            return self._path
        async with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.flush()
            if not self._path.exists():
                raise FileNotFoundError(str(self._path))
            if target.exists() and not overwrite:
                raise FileExistsError(str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(self._path), str(target))
        return target

    async def __aenter__(self) -> EventLogger:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        await self.stop()


__all__ = ["EventLogger", "LoggerStatus"]
