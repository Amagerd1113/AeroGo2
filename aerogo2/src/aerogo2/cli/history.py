"""Persistent command history that is never shared with confirmations."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, List, Optional, Tuple

from prompt_toolkit.history import DummyHistory, History


class CommandHistory(History):
    """A small JSON-lines prompt-toolkit history.

    Only normal command lines should be sent to this class.  ConfirmationService
    uses ``DummyHistory`` in a separate prompt session, so exact safety phrases
    cannot leak into this file.
    """

    def __init__(self, path: Optional[Path] = None, maximum_entries: int = 1_000) -> None:
        if maximum_entries <= 0:
            raise ValueError("maximum_entries must be positive")
        super().__init__()
        self._path = path
        self._maximum_entries = maximum_entries
        self._memory_entries: List[str] = []
        self._recording_enabled = True
        self._lock = Lock()

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def load_history_strings(self) -> Iterable[str]:
        # prompt-toolkit expects newest entries first.
        return reversed(self._read_entries())

    def store_string(self, string: str) -> None:
        if not self._recording_enabled:
            return
        normalized = string.replace("\r", " ").replace("\n", " ").strip()
        if not normalized:
            return
        with self._lock:
            self._memory_entries.append(normalized)
            self._memory_entries = self._memory_entries[-self._maximum_entries :]
            if self._path is None:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=False))
                handle.write("\n")
            self._compact_if_needed()

    def record(self, command_line: str, *, confirmation: bool = False) -> None:
        """Append a command unless the caller identifies it as confirmation input."""

        if confirmation:
            return
        self.append_string(command_line)

    def entries(self) -> Tuple[str, ...]:
        """Return history in chronological order."""

        return tuple(self._read_entries())

    def clear(self) -> None:
        with self._lock:
            self._memory_entries.clear()
            if self._path is not None and self._path.exists():
                self._path.write_text("", encoding="utf-8")
        # History keeps an in-memory cache independently of store_string.
        self._strings: List[str] = []
        self._loaded = False

    @contextmanager
    def suspended(self) -> Iterator[None]:
        previous = self._recording_enabled
        self._recording_enabled = False
        try:
            yield
        finally:
            self._recording_enabled = previous

    @staticmethod
    def confirmation_history() -> DummyHistory:
        """Return a sink used exclusively by non-persistent confirmation prompts."""

        return DummyHistory()

    def _read_entries(self) -> List[str]:
        if self._path is None or not self._path.exists():
            return list(self._memory_entries[-self._maximum_entries :])
        entries: List[str] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return list(self._memory_entries[-self._maximum_entries :])
        for line in lines:
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Accept an older plain-text history rather than making the
                # console unusable after an upgrade.
                value = line
            if isinstance(value, str) and value.strip():
                entries.append(value)
        return entries[-self._maximum_entries :]

    def _compact_if_needed(self) -> None:
        if self._path is None:
            return
        try:
            if self._path.stat().st_size < 1_000_000:
                return
            entries = self._read_entries()
            payload = "".join(f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in entries)
            self._path.write_text(payload, encoding="utf-8")
        except OSError:
            # History persistence is useful but never safety-critical.
            return


SafeCommandHistory = CommandHistory

__all__ = ["CommandHistory", "SafeCommandHistory"]
