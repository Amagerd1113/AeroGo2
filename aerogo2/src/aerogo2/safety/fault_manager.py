"""In-memory fault latching for the Phase 1 manager."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from aerogo2.common.enums import SafetySeverity
from aerogo2.common.models import SafetyViolation


class FaultManager:
    """Latch reported violations until an explicit manager-owned clear."""

    def __init__(self) -> None:
        self._active: Dict[str, SafetyViolation] = {}
        self._history: List[SafetyViolation] = []

    def record(self, violation: SafetyViolation) -> None:
        self._active[violation.code] = violation
        self._history.append(violation)

    def record_many(self, violations: Iterable[SafetyViolation]) -> None:
        for violation in violations:
            self.record(violation)

    def clear(self, code: Optional[str] = None) -> Tuple[str, ...]:
        """Clear software latches only.

        This method deliberately does not call F446 ``clear`` or any bridge.
        Hardware fault recovery remains an explicit, separately authorized
        SystemManager operation.
        """

        if code is not None:
            if code in self._active:
                del self._active[code]
                return (code,)
            return ()
        cleared = tuple(sorted(self._active))
        self._active.clear()
        return cleared

    @property
    def active(self) -> Tuple[SafetyViolation, ...]:
        return tuple(self._active[code] for code in sorted(self._active))

    @property
    def history(self) -> Tuple[SafetyViolation, ...]:
        return tuple(self._history)

    @property
    def active_codes(self) -> Tuple[str, ...]:
        return tuple(sorted(self._active))

    @property
    def has_blocking_fault(self) -> bool:
        return any(
            item.severity in (SafetySeverity.FAULT, SafetySeverity.EMERGENCY)
            for item in self._active.values()
        )
