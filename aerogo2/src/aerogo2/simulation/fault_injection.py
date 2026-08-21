"""Named simulation-only fault injection."""

from __future__ import annotations

from enum import Enum


class SimulatedFault(Enum):
    RC_LOSS = "RC_LOSS"
    PIXHAWK_TIMEOUT = "PIXHAWK_TIMEOUT"
    F446_TIMEOUT = "F446_TIMEOUT"
    F446_OVERCURRENT = "F446_OVERCURRENT"
    F446_WRONG_FINAL_STATE = "F446_WRONG_FINAL_STATE"
    F446_NONZERO_FINAL_DUTY = "F446_NONZERO_FINAL_DUTY"
    GO2_MOVING = "GO2_MOVING"
    ESC_RPM_NONZERO = "ESC_RPM_NONZERO"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"

    @classmethod
    def parse(cls, value: str) -> SimulatedFault:
        normalized = value.strip().upper().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown simulated fault '{value}'") from exc


__all__ = ["SimulatedFault"]
