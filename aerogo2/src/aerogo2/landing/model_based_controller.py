"""Deliberately unavailable model-based controller for Phase 1."""

from __future__ import annotations

from aerogo2.common.config import AppConfig
from aerogo2.common.models import LandingCommand, SystemSnapshot
from aerogo2.landing.controller_base import LandingControllerBase
from aerogo2.landing.safety_filter import LandingSafetyFilter


class ModelBasedController(LandingControllerBase):
    """Fail closed until a later, independently reviewed implementation phase."""

    def __init__(self, config: AppConfig) -> None:
        self._filter = LandingSafetyFilter(config)

    def reset(self) -> None:
        return None

    def update(self, snapshot: SystemSnapshot, dt: float) -> LandingCommand:
        del dt
        return self._filter.invalid(
            snapshot.timestamp,
            "model-based landing is unavailable in Phase 1",
        )
