"""Final fail-closed gate and limiter for landing commands."""

from __future__ import annotations

import math
from typing import Optional

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import AutoLandingRequest, Configuration, SystemState
from aerogo2.common.models import LandingCommand, SystemSnapshot
from aerogo2.safety.watchdog import timestamp_is_fresh


class LandingSafetyFilter:
    """Validate inputs and clamp a candidate without sending it anywhere."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def apply(
        self,
        candidate: LandingCommand,
        snapshot: SystemSnapshot,
        dt: float,
    ) -> LandingCommand:
        rejection = self._rejection_reason(candidate, snapshot, dt)
        if rejection is not None:
            return self.invalid(snapshot.timestamp, rejection)

        horizontal_speed = math.hypot(candidate.vx_des, candidate.vy_des)
        horizontal_limit = self._config.landing.maximum_horizontal_speed_mps
        vx_des = candidate.vx_des
        vy_des = candidate.vy_des
        if horizontal_speed > horizontal_limit:
            scale = horizontal_limit / horizontal_speed
            vx_des *= scale
            vy_des *= scale

        # LOCAL_POSITION_NED is the Phase 1 frame: +Z is downward.  A negative
        # candidate would climb and is reduced to zero by this landing-only
        # safety skeleton.
        vz_des = min(
            max(candidate.vz_des, 0.0),
            self._config.landing.maximum_descent_speed_mps,
        )
        yaw_limit = self._config.landing.maximum_yaw_rate_rad_s
        yaw_rate_des = min(max(candidate.yaw_rate_des, -yaw_limit), yaw_limit)

        return LandingCommand(
            vx_des=vx_des,
            vy_des=vy_des,
            vz_des=vz_des,
            yaw_rate_des=yaw_rate_des,
            valid=True,
            reason=candidate.reason,
            timestamp=snapshot.timestamp,
        )

    @staticmethod
    def invalid(timestamp: float, reason: str) -> LandingCommand:
        """Return an invalid command whose numerical outputs are all safe zeros."""

        return LandingCommand(
            vx_des=0.0,
            vy_des=0.0,
            vz_des=0.0,
            yaw_rate_des=0.0,
            valid=False,
            reason=reason,
            timestamp=timestamp,
        )

    def _rejection_reason(
        self,
        candidate: LandingCommand,
        snapshot: SystemSnapshot,
        dt: float,
    ) -> Optional[str]:
        if not math.isfinite(snapshot.timestamp):
            return "invalid snapshot timestamp"
        if not math.isfinite(dt) or dt <= 0.0 or dt > self._config.landing.controller_timeout_s:
            return "autoland controller timeout or invalid dt"
        if snapshot.state is not SystemState.AUTO_LANDING or not snapshot.autoland_active:
            return "automatic landing is not active"
        if snapshot.configuration is not Configuration.FLIGHT:
            return "FLIGHT configuration is not verified"
        if not snapshot.pixhawk.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.pixhawk.heartbeat_timestamp,
            self._config.safety.pixhawk_timeout_s,
        ):
            return "Pixhawk status is unavailable or stale"
        if not snapshot.pixhawk.armed:
            return "Pixhawk is not armed"
        if snapshot.pixhawk.failsafe:
            return "Pixhawk failsafe is active"
        if not snapshot.rc.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.rc.timestamp,
            self._config.safety.rc_timeout_s,
        ):
            return "RC status is unavailable or stale"
        if snapshot.rc.failsafe:
            return "RC failsafe is active"
        if snapshot.rc.manual_override:
            return "manual override requested"
        if snapshot.rc.auto_landing_request is not AutoLandingRequest.AUTO_EXECUTE:
            return "CH10 is not AUTO_EXECUTE"
        estimate = snapshot.landing_estimate
        if not estimate.valid or not estimate.ground_detected:
            return "landing estimate or ground observation is invalid"
        if not timestamp_is_fresh(
            snapshot.timestamp,
            estimate.timestamp,
            self._config.landing.controller_timeout_s,
        ):
            return "landing estimate is stale"
        estimate_values = (
            estimate.height_m,
            estimate.vertical_velocity_mps,
            estimate.horizontal_velocity_mps,
        )
        if any(value is None or not math.isfinite(value) for value in estimate_values):
            return "landing estimate contains a non-finite value"
        if snapshot.active_fault_codes:
            return "active safety faults inhibit automatic landing"
        if not candidate.valid:
            return candidate.reason or "landing candidate is invalid"
        if not timestamp_is_fresh(
            snapshot.timestamp,
            candidate.timestamp,
            self._config.landing.controller_timeout_s,
        ):
            return "landing command is stale"
        command_values = (
            candidate.vx_des,
            candidate.vy_des,
            candidate.vz_des,
            candidate.yaw_rate_des,
        )
        if any(not math.isfinite(value) for value in command_values):
            return "landing command contains a non-finite value"
        return None
