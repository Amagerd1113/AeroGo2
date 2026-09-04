from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.models import Go2FootForceFeedback
from aerogo2.landing.impact_aware.go2_foot_force import (
    Go2FootForceAdapterError,
    Go2FootForceCalibration,
    Go2FootForceSource,
    calibrate_go2_normal_forces,
    compute_go2_foot_force_calibration_hash,
    compute_go2_foot_force_mapping_hash,
)


def _calibration(
    *,
    source: Go2FootForceSource = Go2FootForceSource.RAW_INT16,
    indices: tuple[int, int, int, int] = (2, 0, 3, 1),
    offsets: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 40.0),
    scales: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    signs: tuple[int, int, int, int] = (1, 1, 1, 1),
    maximum: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 100.0),
) -> Go2FootForceCalibration:
    leg_order = ("FR", "FL", "RR", "RL")
    mapping_hash = compute_go2_foot_force_mapping_hash(
        "robot-sn-map-v1",
        leg_order,
        indices,
    )
    calibration_hash = compute_go2_foot_force_calibration_hash(
        mapping_hash=mapping_hash,
        calibration_version="robot-sn-cal-v1",
        source=source,
        offsets_sdk_by_algorithm_leg=offsets,
        scales_n_per_sdk_unit_by_algorithm_leg=scales,
        signs_by_algorithm_leg=signs,
        maximum_valid_normal_force_n_by_algorithm_leg=maximum,
    )
    return Go2FootForceCalibration(
        mapping_version="robot-sn-map-v1",
        mapping_hash=mapping_hash,
        calibration_version="robot-sn-cal-v1",
        calibration_hash=calibration_hash,
        algorithm_leg_order=leg_order,
        sdk_indices_by_leg=indices,
        source=source,
        offsets_sdk_by_algorithm_leg=offsets,
        scales_n_per_sdk_unit_by_algorithm_leg=scales,
        signs_by_algorithm_leg=signs,
        maximum_valid_normal_force_n_by_algorithm_leg=maximum,
    )


def _feedback(**changes: object) -> Go2FootForceFeedback:
    values: dict[str, object] = {
        "receipt_timestamp_s": 12.5,
        "receipt_sequence": 9,
        "subscription_generation": 2,
        "source_tick": 1234,
        "source_tick_valid": True,
        "source_tick_monotonic": True,
        "raw_sdk_int16": (120, 140, 210, 330),
        "estimated_sdk_int16": (12, 14, 21, 33),
        "raw_valid": True,
        "estimated_valid": True,
    }
    values.update(changes)
    return Go2FootForceFeedback(**values)  # type: ignore[arg-type]


def test_explicit_mapping_and_affine_calibration_preserve_source_identity() -> None:
    calibration = _calibration()

    sample = calibrate_go2_normal_forces(_feedback(), calibration)

    # Algorithm order uses SDK indices [2, 0, 3, 1].
    assert sample.algorithm_leg_order == ("FR", "FL", "RR", "RL")
    assert sample.normal_forces_n == pytest.approx((20.0, 20.0, 90.0, 40.0))
    assert sample.source is Go2FootForceSource.RAW_INT16
    assert sample.source_tick == 1234
    assert sample.receipt_timestamp_s == pytest.approx(12.5)
    assert sample.receipt_sequence == 9
    assert sample.subscription_generation == 2
    assert sample.mapping_hash == calibration.mapping_hash
    assert sample.calibration_hash == calibration.calibration_hash


def test_raw_and_estimated_arrays_are_never_implicitly_interchanged() -> None:
    estimated = _calibration(
        source=Go2FootForceSource.ESTIMATED_INT16,
        offsets=(0.0, 0.0, 0.0, 0.0),
        scales=(1.0, 1.0, 1.0, 1.0),
        maximum=(100.0, 100.0, 100.0, 100.0),
    )
    sample = calibrate_go2_normal_forces(_feedback(), estimated)
    assert sample.normal_forces_n == pytest.approx((21.0, 12.0, 33.0, 14.0))

    with pytest.raises(Go2FootForceAdapterError, match="foot_force_est"):
        calibrate_go2_normal_forces(
            _feedback(estimated_valid=False),
            estimated,
        )


@pytest.mark.parametrize(
    "feedback",
    [
        _feedback(source_tick_monotonic=False),
        _feedback(source_tick=None, source_tick_valid=False, source_tick_monotonic=False),
        _feedback(receipt_sequence=0),
        _feedback(subscription_generation=0),
    ],
)
def test_invalid_source_identity_fails_closed(feedback: Go2FootForceFeedback) -> None:
    with pytest.raises(Go2FootForceAdapterError, match="identity is invalid"):
        calibrate_go2_normal_forces(feedback, _calibration())


def test_saturation_negative_and_out_of_envelope_values_fail_closed() -> None:
    calibration = _calibration()
    with pytest.raises(Go2FootForceAdapterError, match="saturated"):
        calibrate_go2_normal_forces(
            _feedback(raw_sdk_int16=(120, 140, 32767, 330)),
            calibration,
        )
    with pytest.raises(Go2FootForceAdapterError, match="outside"):
        calibrate_go2_normal_forces(
            _feedback(raw_sdk_int16=(120, 140, 0, 330)),
            calibration,
        )
    with pytest.raises(Go2FootForceAdapterError, match="outside"):
        calibrate_go2_normal_forces(
            _feedback(raw_sdk_int16=(120, 140, 2000, 330)),
            calibration,
        )


def test_mapping_and_calibration_hashes_cannot_be_forged_or_reused() -> None:
    calibration = _calibration()
    with pytest.raises(ValueError, match="mapping_hash"):
        replace(calibration, sdk_indices_by_leg=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="calibration_hash"):
        replace(
            calibration,
            scales_n_per_sdk_unit_by_algorithm_leg=(1.0, 1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="permutation"):
        _calibration(indices=(0, 0, 2, 3))


def test_feedback_dto_rejects_non_int16_and_inconsistent_tick_validity() -> None:
    with pytest.raises(ValueError, match="signed int16"):
        _feedback(raw_sdk_int16=(0, 0, 0, 32768))
    with pytest.raises(ValueError, match="agree"):
        _feedback(source_tick=None, source_tick_valid=True)
    with pytest.raises(ValueError, match="monotonic"):
        _feedback(source_tick=None, source_tick_valid=False, source_tick_monotonic=True)
