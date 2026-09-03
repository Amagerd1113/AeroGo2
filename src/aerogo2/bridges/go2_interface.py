"""Typed high-level boundary for Unitree Go2 integration."""

from __future__ import annotations

from typing import Protocol

from aerogo2.common.models import Go2Status


class Go2Interface(Protocol):
    """Safe, high-level Go2 operations.

    No joint current, torque, or other low-level actuator API is exposed.
    """

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def run(self) -> None: ...

    def get_status(self) -> Go2Status: ...

    def latest_status(self) -> Go2Status: ...

    async def request_stop(self) -> bool: ...

    async def request_stand(self) -> bool: ...

    async def request_flight_pose(self) -> bool: ...

    async def finalize_operator_joint_lock(self) -> bool: ...

    async def request_landing_pose(self) -> bool: ...


__all__ = ["Go2Interface"]
