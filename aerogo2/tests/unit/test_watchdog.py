from __future__ import annotations

import math

import pytest

from aerogo2.common.clock import ManualClock
from aerogo2.safety.watchdog import timestamp_age, timestamp_is_fresh


@pytest.mark.parametrize(
    ("now", "timestamp"),
    [
        (10.0, 0.0),
        (10.0, -1.0),
        (0.0, 0.0),
        (-1.0, -2.0),
        (10.0, 10.001),
        (float("nan"), 1.0),
        (10.0, float("nan")),
        (float("inf"), 1.0),
        (10.0, float("inf")),
    ],
)
def test_invalid_or_sentinel_timestamps_fail_closed(
    now: float,
    timestamp: float,
) -> None:
    assert math.isinf(timestamp_age(now, timestamp))
    assert not timestamp_is_fresh(now, timestamp, 1.0)


def test_timestamp_timeout_boundary_is_inclusive() -> None:
    assert timestamp_age(10.0, 9.0) == 1.0
    assert timestamp_is_fresh(10.0, 9.0, 1.0)
    assert not timestamp_is_fresh(
        10.0,
        math.nextafter(9.0, -math.inf),
        1.0,
    )
    assert not timestamp_is_fresh(10.0, 8.999999, 1.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("-inf")])
def test_manual_clock_requires_positive_finite_initial_time(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ManualClock(value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("-inf")])
def test_manual_clock_requires_positive_finite_wall_time(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ManualClock(10.0, wall_initial=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_watchdog_kick_rejects_invalid_timestamps(value: float) -> None:
    from aerogo2.safety.watchdog import Watchdog

    with pytest.raises(ValueError, match="finite and positive"):
        Watchdog(1.0).kick(value)
