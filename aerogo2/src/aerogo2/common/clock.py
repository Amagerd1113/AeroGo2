"""Injectable monotonic clocks make timeout logic deterministic in tests."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Optional


class Clock(ABC):
    @abstractmethod
    def monotonic(self) -> float:
        """Return monotonic seconds."""

    def wall_time(self) -> float:
        return time.time()


class RealClock(Clock):
    def monotonic(self) -> float:
        return time.monotonic()


class ManualClock(Clock):
    def __init__(
        self,
        initial: float = 1.0,
        wall_initial: Optional[float] = None,
    ) -> None:
        if not math.isfinite(initial) or initial <= 0.0:
            raise ValueError("ManualClock initial time must be finite and positive")
        if wall_initial is not None and (not math.isfinite(wall_initial) or wall_initial <= 0.0):
            raise ValueError("ManualClock initial wall time must be finite and positive")
        self._value = float(initial)
        self._wall_value = 1_700_000_000.0 if wall_initial is None else float(wall_initial)

    def monotonic(self) -> float:
        return self._value

    def wall_time(self) -> float:
        return self._wall_value

    def advance(self, seconds: float) -> float:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("ManualClock advance must be finite and non-negative")
        next_value = self._value + seconds
        next_wall_value = self._wall_value + seconds
        if not math.isfinite(next_value) or not math.isfinite(next_wall_value):
            raise ValueError("ManualClock advance must keep both clocks finite")
        self._value = next_value
        self._wall_value = next_wall_value
        return self._value
