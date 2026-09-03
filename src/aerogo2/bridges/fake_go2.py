"""Safe high-level Go2 simulator for Phase 1."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Optional, Tuple, cast

from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.models import Go2Status


class FakeGo2:
    """In-memory Go2 model; it contains no joint-level actuator operations."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or RealClock()
        self._status = Go2Status(locomotion_mode="STAND")
        self._stop_failure = False
        self._flight_lock_failure = False
        self._shutdown = asyncio.Event()

    async def connect(self) -> None:
        self._shutdown.clear()
        self.inject_status(connected=True)

    async def disconnect(self) -> None:
        self.inject_status(
            connected=False,
            controller_active=False,
            locomotion_mode="STOPPED",
        )
        self._shutdown.set()

    async def run(self) -> None:
        await self._shutdown.wait()

    def get_status(self) -> Go2Status:
        return self._status

    def latest_status(self) -> Go2Status:
        return self.get_status()

    async def request_stop(self) -> bool:
        if not self._status.connected or self._stop_failure:
            return False
        if self._status.joints_locked:
            return True
        self.inject_status(
            velocity_mps=0.0,
            stable=True,
            controller_active=False,
            locomotion_mode="STOPPED",
            joints_locked=False,
        )
        return True

    async def request_stand(self) -> bool:
        if not self._status.connected:
            return False
        self.inject_status(
            velocity_mps=0.0,
            stable=True,
            standing=True,
            controller_active=True,
            locomotion_mode="STAND",
            joints_locked=False,
        )
        return True

    async def request_flight_pose(self) -> bool:
        if not self._status.connected or self._status.moving or self._flight_lock_failure:
            return False
        self.inject_status(
            velocity_mps=0.0,
            stable=True,
            standing=True,
            controller_active=False,
            locomotion_mode="JOINT_LOCK",
            joints_locked=True,
        )
        return True

    async def finalize_operator_joint_lock(self) -> bool:
        if not self._status.connected or self._status.moving or self._flight_lock_failure:
            return False
        return True

    async def request_landing_pose(self) -> bool:
        if not self._status.connected or self._status.moving:
            return False
        self.inject_status(
            velocity_mps=0.0,
            stable=True,
            standing=True,
            controller_active=False,
            locomotion_mode="BALANCE_STAND",
            joints_locked=False,
        )
        return True

    def inject_status(self, **changes: object) -> Go2Status:
        if "body_velocity" in changes and "velocity_mps" not in changes:
            body_velocity = cast(Tuple[float, float, float], changes["body_velocity"])
            changes["velocity_mps"] = math.sqrt(
                sum(component * component for component in body_velocity)
            )
        changes.setdefault("timestamp", self._clock.monotonic())
        status = replace(self._status, **changes)  # type: ignore[arg-type]
        if "body_velocity" in changes:
            body_velocity = status.body_velocity
            moving = any(abs(component) > 0.0 for component in body_velocity)
        elif "velocity_mps" in changes:
            body_velocity = (status.velocity_mps, 0.0, 0.0)
            moving = abs(status.velocity_mps) > 0.0
        else:
            body_velocity = status.body_velocity
            moving = status.moving
        self._status = replace(
            status,
            message_age_s=0.0,
            body_velocity=body_velocity,
            moving=moving,
        )
        return self._status

    def inject_motion(
        self,
        velocity_mps: float,
        stable: Optional[bool] = None,
    ) -> Go2Status:
        if not math.isfinite(velocity_mps):
            raise ValueError("Go2 velocity must be finite")
        is_stable = abs(velocity_mps) == 0.0 if stable is None else stable
        moving = abs(velocity_mps) > 0.0
        return self.inject_status(
            velocity_mps=float(velocity_mps),
            stable=is_stable,
            controller_active=moving,
            locomotion_mode="WALK" if moving else "STAND",
        )

    def inject_stop_failure(self, enabled: bool = True) -> None:
        self._stop_failure = enabled

    def inject_flight_lock_failure(self, enabled: bool = True) -> None:
        self._flight_lock_failure = enabled


__all__ = ["FakeGo2"]
