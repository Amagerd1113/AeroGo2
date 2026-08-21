"""Conservative Phase 1 descent trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass

from aerogo2.common.models import LandingCommand


@dataclass(frozen=True)
class SafeDescentTrajectory:
    """Generate a vertical-only command in MAVLink local-NED coordinates.

    Positive ``vz_des`` means down.  The safety filter still clamps every
    component before the command can be considered valid.
    """

    descent_speed_mps: float

    def command(self, timestamp: float) -> LandingCommand:
        return LandingCommand(
            vx_des=0.0,
            vy_des=0.0,
            vz_des=self.descent_speed_mps,
            yaw_rate_des=0.0,
            valid=True,
            reason="safe vertical descent candidate (local NED)",
            timestamp=timestamp,
        )
