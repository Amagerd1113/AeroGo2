"""Small deterministic watchdog helpers.

The timestamp helpers intentionally treat future, non-finite, and missing
timestamps as stale.  They do not read a process clock, so evaluating a
``SystemSnapshot`` is reproducible in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def timestamp_age(now: float, timestamp: float) -> float:
    """Return a fail-closed age for a timestamp.

    ``inf`` means the timestamp cannot safely be used.  A timestamp equal to
    ``now`` has age zero; equality with the configured timeout is still fresh.
    """

    if (
        not math.isfinite(now)
        or now <= 0.0
        or not math.isfinite(timestamp)
        or timestamp <= 0.0
        or timestamp > now
    ):
        return math.inf
    return now - timestamp


def timestamp_is_fresh(now: float, timestamp: float, timeout_s: float) -> bool:
    """Return whether ``timestamp`` is no older than ``timeout_s``."""

    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        return False
    age = timestamp_age(now, timestamp)
    return age <= timeout_s and timestamp >= now - timeout_s


@dataclass
class Watchdog:
    """Explicitly kicked watchdog for manager-owned asynchronous activities."""

    timeout_s: float
    _last_kick: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("watchdog timeout must be finite and positive")

    def reset(self) -> None:
        self._last_kick = None

    def kick(self, timestamp: float) -> None:
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("watchdog timestamp must be finite and positive")
        if self._last_kick is not None and timestamp < self._last_kick:
            raise ValueError("watchdog timestamp cannot move backwards")
        self._last_kick = timestamp

    def expired(self, now: float) -> bool:
        if self._last_kick is None:
            return True
        return not timestamp_is_fresh(now, self._last_kick, self.timeout_s)

    def age(self, now: float) -> float:
        if self._last_kick is None:
            return math.inf
        return timestamp_age(now, self._last_kick)
