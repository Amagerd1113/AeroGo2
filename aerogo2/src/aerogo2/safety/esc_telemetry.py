"""Canonical, fail-closed ESC telemetry assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from aerogo2.common.models import SystemSnapshot
from aerogo2.common.numeric import finite_real


@dataclass(frozen=True)
class EscTelemetryAssessment:
    """Integrity and safety facets for the three canonical ESC views."""

    complete: bool
    consistent: bool
    healthy: bool
    rpm_safe: bool

    @property
    def safe(self) -> bool:
        return self.complete and self.consistent and self.healthy and self.rpm_safe


def assess_esc_telemetry(
    snapshot: SystemSnapshot,
    expected_mapping: Mapping[int, str],
    *,
    exact_zero: bool = False,
    maximum_abs_rpm: Optional[float] = None,
) -> EscTelemetryAssessment:
    """Assess tuple, RPM-map, and online-map telemetry as one atomic view.

    Missing slots, duplicates, malformed numbers, non-boolean health flags,
    physical mapping conflicts, or disagreement between views all fail closed.
    """

    expected = dict(expected_mapping)
    expected_slots = set(expected)
    tuple_items = snapshot.pixhawk.esc
    tuple_slots = [item.slot for item in tuple_items]
    rpm_by_slot = dict(snapshot.pixhawk.esc_rpm)
    online_by_slot = dict(snapshot.pixhawk.esc_online)
    try:
        observed_tuple_slots = set(tuple_slots)
        observed_rpm_slots = set(rpm_by_slot)
        observed_online_slots = set(online_by_slot)
    except (TypeError, ValueError):
        return EscTelemetryAssessment(False, False, False, False)
    complete = (
        bool(expected_slots)
        and len(tuple_slots) == len(expected_slots)
        and observed_tuple_slots == expected_slots
        and observed_rpm_slots == expected_slots
        and observed_online_slots == expected_slots
    )
    if not complete:
        return EscTelemetryAssessment(False, False, False, False)

    maximum = None if maximum_abs_rpm is None else finite_real(maximum_abs_rpm)
    if maximum_abs_rpm is not None and (maximum is None or maximum <= 0.0):
        return EscTelemetryAssessment(True, False, False, False)

    consistent = True
    healthy = True
    rpm_safe = True
    for item in tuple_items:
        tuple_rpm = finite_real(item.rpm)
        mapped_rpm = finite_real(rpm_by_slot[item.slot])
        tuple_health_is_bool = type(item.healthy) is bool
        mapped_online_is_bool = type(online_by_slot[item.slot]) is bool
        item_consistent = (
            item.physical_position == expected[item.slot]
            and tuple_rpm is not None
            and mapped_rpm is not None
            and tuple_rpm == mapped_rpm
            and tuple_health_is_bool
            and mapped_online_is_bool
            and item.healthy is online_by_slot[item.slot]
        )
        consistent = consistent and item_consistent
        healthy = (
            healthy
            and tuple_health_is_bool
            and mapped_online_is_bool
            and item.healthy is True
            and online_by_slot[item.slot] is True
        )
        if tuple_rpm is None or mapped_rpm is None:
            rpm_safe = False
        elif exact_zero:
            rpm_safe = rpm_safe and tuple_rpm == 0.0
        elif maximum is not None:
            rpm_safe = rpm_safe and abs(tuple_rpm) < maximum

    return EscTelemetryAssessment(complete, consistent, healthy, rpm_safe)


__all__ = ["EscTelemetryAssessment", "assess_esc_telemetry"]
