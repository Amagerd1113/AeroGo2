from __future__ import annotations

import numpy as np
import pytest

from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetectorConfig,
    FootContactDetector,
)


def _detector() -> FootContactDetector:
    return FootContactDetector(
        ContactDetectorConfig(
            contact_on_threshold_n=[20.0] * 4,
            contact_off_threshold_n=[10.0] * 4,
            filter_time_constant_s=0.01,
            contact_confirm_s=0.05,
            release_confirm_s=0.05,
        )
    )


def test_thresholds_require_real_hysteresis() -> None:
    with pytest.raises(ValueError, match="below"):
        ContactDetectorConfig([10.0] * 4, [10.0] * 4, 0.01, 0.05, 0.05)


def test_config_rejects_boolean_and_numeric_string_coercion() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        ContactDetectorConfig(["20"] * 4, [10.0] * 4, 0.01, 0.05, 0.05)
    with pytest.raises(TypeError, match="real number"):
        ContactDetectorConfig([20.0] * 4, [10.0] * 4, 0.01, True, 0.05)


def test_validated_threshold_arrays_are_immutable() -> None:
    config = ContactDetectorConfig([20.0] * 4, [10.0] * 4, 0.01, 0.05, 0.05)
    assert not np.asarray(config.contact_on_threshold_n).flags.writeable
    assert not np.asarray(config.contact_off_threshold_n).flags.writeable
    with pytest.raises(ValueError):
        config.contact_on_threshold_n[0] = 1.0  # type: ignore[index]


def test_single_sample_does_not_confirm_touchdown() -> None:
    detector = _detector()
    first = detector.update([30.0, 0.0, 0.0, 0.0], 1.0)
    second = detector.update([30.0, 0.0, 0.0, 0.0], 1.04)
    assert first.contacts == (False, False, False, False)
    assert second.contacts == (False, False, False, False)


def test_touchdown_and_release_are_confirmed_and_timestamped() -> None:
    detector = _detector()
    detector.update([30.0, 0.0, 0.0, 0.0], 1.0)
    touchdown = detector.update([30.0, 0.0, 0.0, 0.0], 1.06)
    assert touchdown.contacts[0]
    assert touchdown.touchdown_events[0]
    assert touchdown.touchdown_times_s[0] == pytest.approx(1.06)

    detector.update([0.0, 0.0, 0.0, 0.0], 1.20)
    released = detector.update([0.0, 0.0, 0.0, 0.0], 1.26)
    assert not released.contacts[0]
    assert released.release_events[0]
    assert released.touchdown_times_s[0] is None


def test_intermediate_force_does_not_chatter_across_hysteresis_band() -> None:
    detector = _detector()
    detector.update([30.0] * 4, 1.0)
    detector.update([30.0] * 4, 1.06)
    middle = detector.update([15.0] * 4, 1.20)
    assert middle.contacts == (True, True, True, True)
    assert not any(middle.release_events)


def test_timestamp_must_increase() -> None:
    detector = _detector()
    detector.update([0.0] * 4, 1.0)
    with pytest.raises(ValueError, match="increase"):
        detector.update([0.0] * 4, 1.0)

    with pytest.raises(TypeError, match="real number"):
        _detector().update([0.0] * 4, "1.0")  # type: ignore[arg-type]
