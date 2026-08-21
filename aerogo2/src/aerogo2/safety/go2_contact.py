"""Fail-closed assessment of Unitree Go2 raw foot-force contact telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from aerogo2.common.config import Go2Config
from aerogo2.common.models import Go2Status


@dataclass(frozen=True)
class FootContactAssessment:
    enabled: bool
    valid: bool
    contacts: Tuple[bool, bool, bool, bool]
    contact_count: int
    required_count: int
    safe: bool


def assess_foot_contact(
    status: Go2Status,
    config: Go2Config,
) -> FootContactAssessment:
    """Compare all four authoritative raw channels against calibrated thresholds."""

    forces = tuple(status.foot_force)
    thresholds = tuple(config.foot_force_contact_thresholds)
    shape_valid = len(forces) == 4 and len(thresholds) == 4
    values_valid = shape_valid and all(
        type(force) is int and type(threshold) is int and threshold > 0
        for force, threshold in zip(forces, thresholds)
    )
    valid = bool(status.foot_force_valid and values_valid)
    contacts = (False, False, False, False)
    if valid:
        contacts = (
            forces[0] >= thresholds[0],
            forces[1] >= thresholds[1],
            forces[2] >= thresholds[2],
            forces[3] >= thresholds[3],
        )
    count = sum(1 for contact in contacts if contact)
    enabled = config.landing_compliance_enabled
    required = config.landing_contact_min_feet
    return FootContactAssessment(
        enabled=enabled,
        valid=valid,
        contacts=contacts,
        contact_count=count,
        required_count=required,
        safe=enabled and valid and count >= required,
    )


__all__ = ["FootContactAssessment", "assess_foot_contact"]
