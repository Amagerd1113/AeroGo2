from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.models import PixhawkStatus
from aerogo2.safety.pixhawk_freshness import (
    pixhawk_ground_state_is_current,
    pixhawk_touchdown_sources_are_current,
    timestamps_are_coherent,
)


def _status() -> PixhawkStatus:
    return PixhawkStatus(
        connected=True,
        heartbeat_timestamp=9.0,
        attitude_timestamp=9.80,
        kinematics_timestamp=9.90,
        landed_state_timestamp=10.0,
    )


def test_touchdown_sources_accept_exact_age_and_skew_boundaries() -> None:
    status = _status()

    assert pixhawk_ground_state_is_current(status, 10.0, 1.0, 0.30)
    assert pixhawk_touchdown_sources_are_current(status, 10.0, 1.0, 0.20, 0.20)


def test_touchdown_skew_excludes_slower_heartbeat_but_rejects_sensor_skew() -> None:
    status = _status()

    # HEARTBEAT is independently fresh but intentionally outside the 0.20 s
    # source-fusion window used by the three touchdown streams.
    assert pixhawk_touchdown_sources_are_current(status, 10.0, 1.0, 0.30, 0.20)
    assert not pixhawk_touchdown_sources_are_current(
        replace(status, attitude_timestamp=9.799999),
        10.0,
        1.0,
        0.30,
        0.20,
    )


def test_ground_and_touchdown_proofs_fail_closed_on_missing_or_stale_source() -> None:
    status = _status()

    assert not pixhawk_ground_state_is_current(
        replace(status, landed_state_timestamp=0.0),
        10.0,
        1.0,
        0.30,
    )
    assert not pixhawk_touchdown_sources_are_current(
        replace(status, kinematics_timestamp=9.69),
        10.0,
        1.0,
        0.30,
        0.25,
    )


def test_touchdown_payload_rejects_nonboolean_landed_and_nonfinite_motion() -> None:
    status = _status()

    assert not pixhawk_ground_state_is_current(
        replace(status, landed=1),  # type: ignore[arg-type]
        10.0,
        1.0,
        0.30,
    )
    assert not pixhawk_touchdown_sources_are_current(
        replace(status, roll_rad=float("nan")),
        10.0,
        1.0,
        0.30,
        0.25,
    )


@pytest.mark.parametrize("connected", (1, "yes", object()))
def test_truthy_nonboolean_connected_is_rejected(connected: object) -> None:
    status = replace(_status(), connected=connected)  # type: ignore[arg-type]

    assert not pixhawk_ground_state_is_current(status, 10.0, 1.0, 0.30)
    assert not pixhawk_touchdown_sources_are_current(
        status,
        10.0,
        1.0,
        0.30,
        0.20,
    )


@pytest.mark.parametrize("bad_timestamp", ("10", object(), True))
def test_malformed_source_timestamp_fails_closed(bad_timestamp: object) -> None:
    status = replace(  # type: ignore[arg-type]
        _status(),
        attitude_timestamp=bad_timestamp,
    )

    assert not pixhawk_touchdown_sources_are_current(
        status,
        10.0,
        1.0,
        0.30,
        0.20,
    )


@pytest.mark.parametrize("bad_timestamp", ("10", object(), True))
def test_malformed_timestamp_coherence_fails_closed(bad_timestamp: object) -> None:
    assert not timestamps_are_coherent((9.9, bad_timestamp), 0.20)  # type: ignore[arg-type]
