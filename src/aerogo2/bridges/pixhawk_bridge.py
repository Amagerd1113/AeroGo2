"""Phase 1 Pixhawk hardware stub.

This module intentionally does not import ``pymavlink`` and cannot open a
device.  Real read-only MAVLink integration belongs to a later phase.
"""

from __future__ import annotations

from typing import Optional

from aerogo2.common.config import PixhawkConfig
from aerogo2.common.exceptions import UnsupportedPhaseOperation
from aerogo2.common.models import PixhawkStatus
from aerogo2.common.results import OperationResult

_PHASE_MESSAGE = (
    "Real Pixhawk access is disabled in AeroGo2 Phase 1; use FakePixhawk in dry-run mode"
)


class ReadOnlyPixhawkBridge:
    """Fail-closed placeholder for future read-only MAVLink telemetry."""

    def __init__(self, config: Optional[PixhawkConfig] = None) -> None:
        self._config = config
        self._status = PixhawkStatus()

    async def connect(self) -> None:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    async def disconnect(self) -> None:
        self._status = PixhawkStatus()

    async def run(self) -> None:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    def get_status(self) -> PixhawkStatus:
        return self._status

    def latest_status(self) -> PixhawkStatus:
        return self.get_status()

    async def request_mode(self, mode: str) -> bool:
        del mode
        raise UnsupportedPhaseOperation("Pixhawk mode changes are disabled in AeroGo2 Phase 1")

    async def send_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ) -> OperationResult:
        del vx, vy, vz, yaw_rate
        raise UnsupportedPhaseOperation("Real Pixhawk setpoints are disabled in AeroGo2 Phase 1")

    async def stop_external_setpoints(self) -> OperationResult:
        # No real connection or sender can exist in this stub, so stopping is a
        # safe, idempotent no-op and performs no hardware I/O.
        return OperationResult.success(
            "No real Pixhawk setpoint stream exists in Phase 1",
            code="NO_EXTERNAL_SETPOINTS",
        )


PixhawkBridgeStub = ReadOnlyPixhawkBridge

__all__ = ["PixhawkBridgeStub", "ReadOnlyPixhawkBridge"]
