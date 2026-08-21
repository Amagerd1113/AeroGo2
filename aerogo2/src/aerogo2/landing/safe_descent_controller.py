"""Conservative, simulator-only Phase 1 descent controller."""

from __future__ import annotations

import math
from typing import Optional

from aerogo2.common.config import AppConfig
from aerogo2.common.models import LandingCommand, SystemSnapshot
from aerogo2.landing.controller_base import LandingControllerBase
from aerogo2.landing.safety_filter import LandingSafetyFilter
from aerogo2.landing.trajectory import SafeDescentTrajectory


class SafeDescentController(LandingControllerBase):
    """Generate vertical-only local-NED commands behind strict safety gates.

    The controller has no bridge reference.  In Phase 1 the SystemManager may
    forward a valid result only to ``FakePixhawk``; real setpoint output remains
    feature-disabled.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._filter = LandingSafetyFilter(config)
        self._trajectory = SafeDescentTrajectory(
            descent_speed_mps=config.landing.maximum_descent_speed_mps
        )
        self._last_update_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._last_update_timestamp = None

    def update(self, snapshot: SystemSnapshot, dt: float) -> LandingCommand:
        timing_error = self._timing_error(snapshot, dt)
        self._last_update_timestamp = (
            snapshot.timestamp if math.isfinite(snapshot.timestamp) else None
        )
        if timing_error is not None:
            return self._filter.invalid(snapshot.timestamp, timing_error)
        candidate = self._trajectory.command(snapshot.timestamp)
        return self._filter.apply(candidate, snapshot, dt)

    def _timing_error(self, snapshot: SystemSnapshot, dt: float) -> Optional[str]:
        timeout = self._config.landing.controller_timeout_s
        if not math.isfinite(dt) or dt <= 0.0 or dt > timeout:
            return "autoland controller timeout or invalid dt"
        if not math.isfinite(snapshot.timestamp):
            return "invalid snapshot timestamp"
        if self._last_update_timestamp is None:
            return None
        elapsed = snapshot.timestamp - self._last_update_timestamp
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            return "controller timestamp did not advance"
        if elapsed > timeout:
            return "autoland controller timeout"
        return None
