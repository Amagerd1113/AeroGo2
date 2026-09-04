"""Typed boundary for the process that exclusively owns Unitree ``rt/lowcmd``.

The low-rate landing controller submits expiring targets through this boundary.
It never publishes DDS messages itself.  Ownership transfer is deliberately
guarded by a short-lived, ground-only permit so a crashed/restarted application
cannot silently acquire or release low-level control while airborne.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Protocol

from aerogo2.common.models import Go2LowLevelStatus
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.integration import Go2JointPositionCommand


@dataclass(frozen=True)
class Go2OwnershipPermit:
    """Short-lived evidence that a control-mode transfer is safe.

    This is an application-layer gate, not a sensor attestation.  The system
    manager must construct it only from fresh telemetry and an explicit
    operator action.  Both acquisition and final release require the robot to
    be mechanically supported, the flight controller disarmed, and rotors
    stopped.  Revoking MPC targets does *not* require this permit because it
    leaves the sole publisher running in safe-hold.
    """

    timestamp_s: float
    valid_until_s: float
    operator_authorized: bool
    robot_supported: bool
    pixhawk_disarmed: bool
    rotors_stopped: bool
    mapping_version: str
    mapping_hash: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("timestamp_s", "valid_until_s"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.valid_until_s <= self.timestamp_s:
            raise ValueError("valid_until_s must be later than timestamp_s")
        for name in (
            "operator_authorized",
            "robot_supported",
            "pixhawk_disarmed",
            "rotors_stopped",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in ("mapping_version", "mapping_hash", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")

    def authorizes(
        self,
        now_s: float,
        *,
        mapping_version: str,
        mapping_hash: str,
    ) -> bool:
        """Return whether the permit authorizes a transfer at ``now_s``."""

        if isinstance(now_s, bool) or not isinstance(now_s, Real):
            return False
        now = float(now_s)
        return (
            math.isfinite(now)
            and self.timestamp_s <= now < self.valid_until_s
            and self.operator_authorized
            and self.robot_supported
            and self.pixhawk_disarmed
            and self.rotors_stopped
            and self.mapping_version == mapping_version
            and self.mapping_hash == mapping_hash
        )


class Go2LowLevelInterface(Protocol):
    """Exclusive low-level owner used by runtime, safety, and landing code."""

    async def connect(self) -> OperationResult:
        """Observe LowState without publishing.

        This operation may be available with only the read-side topic, age,
        and coordinate mapping commissioned.  It must not imply that
        :meth:`acquire` or :meth:`submit` is ready.
        """

        ...

    async def acquire(self, permit: Go2OwnershipPermit) -> OperationResult:
        """Release high-level motion service and begin verified safe-hold.

        Implementations must revalidate the complete write-side configuration;
        a successful observe-only ``connect`` is never sufficient authority.
        """

        ...

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        """Replace the low-rate target mailbox for the fixed-rate owner.

        Success acknowledges staging only.  Implementations must report writer
        enqueue and actuator application as separate evidence; absence of such
        evidence must never be inferred from a successful return value.
        """

        ...

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        """Revoke the MPC lease and enter safe-hold without stopping DDS."""

        ...

    async def release(
        self,
        permit: Go2OwnershipPermit,
        reason: str,
        *,
        ownership_epoch: int,
    ) -> OperationResult:
        """Stop publishing only for the exact epoch and after ground checks."""

        ...

    def status(self) -> Go2LowLevelStatus:
        """Return an immutable health/ownership snapshot in bounded time.

        When ``writer_enqueue_ack_available`` is true, a matching target
        sequence must carry a strictly increasing writer generation and the
        exact 12 joint positions remaining after the owner's final software
        limits.  These fields prove only that local DDS ``Write`` accepted the
        frame; they are never motor-side application evidence.
        """

        ...


__all__ = ["Go2LowLevelInterface", "Go2OwnershipPermit"]
