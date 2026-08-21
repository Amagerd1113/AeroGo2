"""Typed boundary for Pixhawk telemetry and simulated setpoint output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aerogo2.common.models import PixhawkStatus
from aerogo2.common.results import OperationResult


@dataclass(frozen=True)
class VelocitySetpoint:
    """A high-level NED velocity command recorded by the Phase 1 simulator."""

    timestamp: float
    vx: float
    vy: float
    vz: float
    yaw_rate: float


class PixhawkInterface(Protocol):
    """The only Pixhawk operations visible to ``SystemManager``.

    Deliberately absent are direct arm, disarm, motor-test, raw motor, and
    servo-PWM operations.  The narrowly scoped ground-arm authorization does
    not arm the vehicle; the Pixhawk-side gate still requires a fresh RC
    switch transition before it invokes ArduPilot's checked arming path.
    """

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def run(self) -> None: ...

    def get_status(self) -> PixhawkStatus: ...

    def latest_status(self) -> PixhawkStatus: ...

    async def request_mode(self, mode: str) -> bool: ...

    async def set_ground_arm_authorization(
        self,
        enabled: bool,
        ttl_s: float,
    ) -> OperationResult: ...

    def ground_arm_authorization_active(self) -> bool: ...

    async def send_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ) -> OperationResult: ...

    async def stop_external_setpoints(self) -> OperationResult: ...


__all__ = ["PixhawkInterface", "VelocitySetpoint"]
