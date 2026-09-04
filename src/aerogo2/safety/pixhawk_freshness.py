"""Fail-closed freshness checks for independently published Pixhawk signals.

MAVLink HEARTBEAT, ATTITUDE, GLOBAL_POSITION_INT, and
EXTENDED_SYS_STATE are independent streams.  A new heartbeat therefore must
not make an old attitude, kinematics, or landed-state sample suitable as a
safety permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aerogo2.common.models import PixhawkStatus
from aerogo2.common.numeric import finite_real
from aerogo2.safety.watchdog import timestamp_is_fresh


@dataclass(frozen=True)
class PixhawkSourceFreshness:
    """Freshness of each safety-relevant Pixhawk source at one instant."""

    heartbeat: bool
    attitude: bool
    kinematics: bool
    landed_state: bool

    @property
    def ground_state(self) -> bool:
        """Whether ``armed`` and ``landed`` may be used as ground evidence."""

        return self.heartbeat and self.landed_state

    @property
    def touchdown(self) -> bool:
        """Whether every Pixhawk input used by touchdown logic is fresh."""

        return self.ground_state and self.attitude and self.kinematics


def pixhawk_ground_payload_is_valid(status: PixhawkStatus) -> bool:
    """Validate booleans used to grant ground-only authority."""

    return type(status.armed) is bool and type(status.landed) is bool


def pixhawk_touchdown_payload_is_valid(status: PixhawkStatus) -> bool:
    """Validate the exact Pixhawk values consumed by touchdown logic."""

    return bool(
        pixhawk_ground_payload_is_valid(status)
        and finite_real(status.roll_rad) is not None
        and finite_real(status.pitch_rad) is not None
        and finite_real(status.vertical_velocity_mps) is not None
        and finite_real(status.relative_altitude_m) is not None
    )


def assess_pixhawk_source_freshness(
    status: PixhawkStatus,
    now: float,
    heartbeat_maximum_age_s: float,
    source_maximum_age_s: float | None = None,
) -> PixhawkSourceFreshness:
    """Assess source timestamps without allowing heartbeat freshness to leak."""

    # This helper is a trust boundary for immutable snapshots as well as the
    # production bridge.  Do not let truthy integers/strings or malformed
    # timestamp objects turn into safety permissions (or escape as TypeError).
    connected = type(status.connected) is bool and status.connected is True
    now_value = finite_real(now)
    heartbeat_age = finite_real(heartbeat_maximum_age_s)
    source_age = (
        heartbeat_age if source_maximum_age_s is None else finite_real(source_maximum_age_s)
    )

    def current(timestamp: object, maximum_age_s: float | None) -> bool:
        timestamp_value = finite_real(timestamp)
        return bool(
            connected
            and now_value is not None
            and timestamp_value is not None
            and maximum_age_s is not None
            and timestamp_is_fresh(now_value, timestamp_value, maximum_age_s)
        )

    return PixhawkSourceFreshness(
        heartbeat=current(status.heartbeat_timestamp, heartbeat_age),
        attitude=current(status.attitude_timestamp, source_age),
        kinematics=current(status.kinematics_timestamp, source_age),
        landed_state=current(status.landed_state_timestamp, source_age),
    )


def timestamps_are_coherent(
    timestamps: Iterable[float],
    maximum_skew_s: float,
) -> bool:
    """Return whether positive finite source times fit one bounded window."""

    values = tuple(finite_real(value) for value in timestamps)
    maximum_skew = finite_real(maximum_skew_s)
    if (
        len(values) < 2
        or maximum_skew is None
        or maximum_skew <= 0.0
        or any(value is None or value <= 0.0 for value in values)
    ):
        return False
    finite_values = tuple(value for value in values if value is not None)
    return max(finite_values) - min(finite_values) <= maximum_skew


def pixhawk_ground_state_is_current(
    status: PixhawkStatus,
    now: float,
    heartbeat_maximum_age_s: float,
    source_maximum_age_s: float,
) -> bool:
    """Validate independent HEARTBEAT and landed-state ground proof ages."""

    freshness = assess_pixhawk_source_freshness(
        status,
        now,
        heartbeat_maximum_age_s,
        source_maximum_age_s,
    )
    # HEARTBEAT is commonly slower than EXTENDED_SYS_STATE.  Both ages must be
    # valid, but their receive timestamps are not required to share the much
    # tighter touchdown sensor-fusion window.
    return freshness.ground_state and pixhawk_ground_payload_is_valid(status)


def pixhawk_touchdown_sources_are_current(
    status: PixhawkStatus,
    now: float,
    heartbeat_maximum_age_s: float,
    source_maximum_age_s: float,
    maximum_skew_s: float,
) -> bool:
    """Validate freshness and mutual skew of every Pixhawk touchdown source."""

    freshness = assess_pixhawk_source_freshness(
        status,
        now,
        heartbeat_maximum_age_s,
        source_maximum_age_s,
    )
    return (
        freshness.touchdown
        and pixhawk_touchdown_payload_is_valid(status)
        and timestamps_are_coherent(
            (
                status.attitude_timestamp,
                status.kinematics_timestamp,
                status.landed_state_timestamp,
            ),
            maximum_skew_s,
        )
    )


__all__ = [
    "PixhawkSourceFreshness",
    "assess_pixhawk_source_freshness",
    "pixhawk_ground_payload_is_valid",
    "pixhawk_ground_state_is_current",
    "pixhawk_touchdown_payload_is_valid",
    "pixhawk_touchdown_sources_are_current",
    "timestamps_are_coherent",
]
