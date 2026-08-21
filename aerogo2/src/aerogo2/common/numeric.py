"""Strict numeric helpers for untrusted telemetry boundaries."""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional


def finite_real(value: object) -> Optional[float]:
    """Return a finite float only for genuine real-number inputs.

    Booleans and string coercions are deliberately rejected: both can hide
    malformed telemetry while still being accepted by ``float(value)``.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


__all__ = ["finite_real"]
