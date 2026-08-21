"""Controller contract shared by simulated landing implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aerogo2.common.models import LandingCommand, SystemSnapshot


class LandingControllerBase(ABC):
    """Produce high-level commands; never send them to a vehicle."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller timing and internal state."""

    @abstractmethod
    def update(self, snapshot: SystemSnapshot, dt: float) -> LandingCommand:
        """Return a command which may be invalid and must be manager-gated."""
