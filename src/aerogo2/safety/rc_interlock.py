"""Fail-closed consistency checks for safety-critical RC switches."""

from __future__ import annotations

from dataclasses import dataclass

from aerogo2.bridges.rc_monitor import classify_rc_position
from aerogo2.common.config import RCConfig
from aerogo2.common.enums import RCPosition
from aerogo2.common.models import RCStatus


@dataclass(frozen=True)
class FlightEnableAssessment:
    """Raw CH5 classification and its agreement with parsed RC state."""

    position: RCPosition
    parsed_is_bool: bool
    parsed_consistent: bool

    @property
    def valid(self) -> bool:
        return (
            self.position in (RCPosition.LOW, RCPosition.HIGH)
            and self.parsed_is_bool
            and self.parsed_consistent
        )

    @property
    def high(self) -> bool:
        return self.valid and self.position is RCPosition.HIGH

    @property
    def low(self) -> bool:
        return self.valid and self.position is RCPosition.LOW


def assess_flight_enable(rc: RCStatus, config: RCConfig) -> FlightEnableAssessment:
    """Validate raw CH5 presence/classification and parsed-boolean agreement."""

    raw = rc.channels.get(config.flight_enable_channel)
    position = classify_rc_position(config, raw)
    parsed_is_bool = type(rc.flight_enable) is bool
    parsed_consistent = parsed_is_bool and rc.flight_enable is (position is RCPosition.HIGH)
    return FlightEnableAssessment(position, parsed_is_bool, parsed_consistent)


__all__ = ["FlightEnableAssessment", "assess_flight_enable"]
