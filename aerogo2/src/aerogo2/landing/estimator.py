"""Landing-estimator contracts for simulation snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aerogo2.common.models import LandingEstimate, SystemSnapshot


class LandingEstimatorBase(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset estimator state."""

    @abstractmethod
    def update(self, snapshot: SystemSnapshot) -> LandingEstimate:
        """Return the estimate associated with a snapshot."""


class SnapshotLandingEstimator(LandingEstimatorBase):
    """Use the estimate injected by the Phase 1 simulation world."""

    def reset(self) -> None:
        return None

    def update(self, snapshot: SystemSnapshot) -> LandingEstimate:
        return snapshot.landing_estimate
