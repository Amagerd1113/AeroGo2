"""Typed boundary for the F446 morphology controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aerogo2.common.enums import Configuration
from aerogo2.common.models import F446Status
from aerogo2.common.results import OperationResult


@dataclass(frozen=True)
class F446CommandRecord:
    """An auditable command emitted by the fake F446 implementation."""

    timestamp: float
    command: str


class F446Interface(Protocol):
    """High-level F446 operations available to ``SystemManager``.

    Raw serial writes remain private. Explicitly typed maintenance operations
    let the manager preserve safety checks and audit boundaries.
    """

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def run(self) -> None: ...

    def get_status(self) -> F446Status: ...

    def latest_status(self) -> F446Status: ...

    async def request_status(self) -> F446Status: ...

    async def request_current(self) -> F446Status: ...

    async def move_to_configuration(
        self,
        configuration: Configuration,
    ) -> OperationResult: ...

    async def start_maintenance_motion(
        self,
        operation: str,
        duty: int,
    ) -> OperationResult: ...

    async def set_current_threshold_adc(self, threshold_adc: int) -> OperationResult: ...

    async def set_current_threshold_mv(self, threshold_mv: int) -> OperationResult: ...

    async def stop(self) -> OperationResult: ...


__all__ = ["F446CommandRecord", "F446Interface"]
