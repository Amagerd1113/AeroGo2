"""Synchronous, ordered transition sink used by the state-machine callback.

The general EventLogger remains async for telemetry tasks. State transitions
need immediate ordering and flush semantics, so SystemManager receives this
small synchronous sink instead of scheduling unowned coroutines.
"""

from __future__ import annotations

import json
import math
import shutil
import threading
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO, Tuple, cast

from aerogo2.common.clock import Clock, RealClock

LOG_FIELDS = (
    "wall_timestamp",
    "monotonic_timestamp",
    "event_type",
    "system_state",
    "previous_state",
    "command_id",
    "command_name",
    "command_result",
    "pixhawk_status",
    "f446_status",
    "go2_status",
    "operator_request",
    "safety_violations",
    "transition_reason",
    "landing_command",
)


class OrderedEventSink:
    """Append and flush a complete JSON object before returning."""

    def __init__(
        self,
        directory: Path,
        clock: Optional[Clock] = None,
        filename: Optional[str] = None,
    ) -> None:
        self._clock = RealClock() if clock is None else clock
        timestamp = datetime.fromtimestamp(self._clock.wall_time(), tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        self.path = Path(directory) / (filename or f"events-{timestamp}.jsonl")
        self._base_path = self.path
        self._auto_filename = filename is None
        self._handle: Optional[TextIO] = None
        self._enabled = True
        self._lock = threading.Lock()
        self._records_written = 0

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._handle.closed

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def records_written(self) -> int:
        return self._records_written

    def start(self) -> Path:
        with self._lock:
            self._enabled = True
            if self.running:
                return self.path
            self.path.parent.mkdir(parents=True, exist_ok=True)
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
                    self.path = candidate
                    self._handle = handle
                    break
            else:
                self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self.path

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            handle = self._handle
            self._handle = None
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    def emit(
        self,
        *,
        event_type: str,
        system_state: str,
        monotonic_timestamp: Optional[float] = None,
        **fields: Any,
    ) -> Mapping[str, Any]:
        if self._enabled and not self.running:
            self.start()
        record: Dict[str, Any] = {field: None for field in LOG_FIELDS}
        record.update(
            {
                "wall_timestamp": datetime.fromtimestamp(
                    self._clock.wall_time(), tz=timezone.utc
                ).isoformat(),
                "monotonic_timestamp": (
                    self._clock.monotonic() if monotonic_timestamp is None else monotonic_timestamp
                ),
                "event_type": event_type,
                "system_state": system_state,
            }
        )
        details: Dict[str, Any] = {}
        for key, value in fields.items():
            if key in record:
                record[key] = self._primitive(value)
            else:
                details[key] = self._primitive(value)
        if details:
            record["details"] = details
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock:
            if not self._enabled:
                return record
            if self._handle is None or self._handle.closed:
                raise RuntimeError("Event sink is not running")
            self._handle.write(payload + "\n")
            self._handle.flush()
            self._records_written += 1
        return record

    def mark(self, text: str, system_state: str) -> Mapping[str, Any]:
        if not self._enabled:
            raise RuntimeError("Logging is stopped; start logging before adding a marker")
        normalized = text.replace("\r", " ").replace("\n", " ").strip()
        if not normalized:
            raise ValueError("Log marker cannot be empty")
        return self.emit(
            event_type="LOG_MARK",
            system_state=system_state,
            marker_text=normalized,
        )

    def tail(self, limit: int = 50) -> Tuple[Mapping[str, Any], ...]:
        if limit < 0:
            raise ValueError("limit cannot be negative")
        if limit == 0 or not self.path.exists():
            return ()
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        records = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"_invalid_json": line}
            if isinstance(value, Mapping):
                records.append(value)
        return tuple(records)

    def export(self, destination: Path, overwrite: bool = False) -> Path:
        target = Path(destination)
        if target.resolve() == self.path.resolve():
            return self.path
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
            if target.exists() and not overwrite:
                raise FileExistsError(str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(self.path), str(target))
        return target

    @classmethod
    def _primitive(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if is_dataclass(value):
            return {
                item.name: cls._primitive(getattr(value, item.name))
                for item in fields(cast(Any, value))
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): cls._primitive(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, set, frozenset)):
            return [cls._primitive(item) for item in value]
        return value


__all__ = ["LOG_FIELDS", "OrderedEventSink"]
