"""Phase 1 Unitree Go2 hardware stub.

The Unitree SDK is intentionally not imported.  Real high-level status and
stop integration is deferred to Phase 6.
"""

from __future__ import annotations

from typing import Optional

from aerogo2.common.config import Go2Config
from aerogo2.common.exceptions import UnsupportedPhaseOperation
from aerogo2.common.models import Go2Status

_PHASE_MESSAGE = "Real Go2 access is disabled in AeroGo2 Phase 1; use FakeGo2 in dry-run mode"


class Go2BridgeStub:
    """Fail-closed placeholder with no network or actuator side effects."""

    def __init__(self, config: Optional[Go2Config] = None) -> None:
        self._config = config
        self._status = Go2Status()

    async def connect(self) -> None:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    async def disconnect(self) -> None:
        self._status = Go2Status()

    async def run(self) -> None:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    def get_status(self) -> Go2Status:
        return self._status

    def latest_status(self) -> Go2Status:
        return self.get_status()

    async def request_stop(self) -> bool:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    async def request_stand(self) -> bool:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    async def request_flight_pose(self) -> bool:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)

    async def request_landing_pose(self) -> bool:
        raise UnsupportedPhaseOperation(_PHASE_MESSAGE)


Go2Bridge = Go2BridgeStub

__all__ = ["Go2Bridge", "Go2BridgeStub"]
