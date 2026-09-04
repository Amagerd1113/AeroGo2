from __future__ import annotations

import numpy as np
import pytest

from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)


def _config(*, gain: float, rise: float = 100.0) -> RotorCorrectionSafetyConfig:
    return RotorCorrectionSafetyConfig(
        target_gain=gain,
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 10.0),
        maximum_correction_n=np.full(4, 10.0),
        maximum_gain_rise_per_s=rise,
    )


@pytest.mark.parametrize("gain", [-0.01, 1.01, float("nan")])
def test_gain_range_is_hard_limited(gain: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _config(gain=gain)


@pytest.mark.parametrize("gain", [True, "0.1"])
def test_gain_rejects_boolean_and_string_coercion(gain: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        _config(gain=gain)  # type: ignore[arg-type]


def test_vector_and_ramp_limits_reject_string_or_boolean_coercion() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        RotorCorrectionSafetyConfig(
            target_gain=0.1,
            thrust_min_n=["0"] * 4,
            thrust_max_n=[10.0] * 4,
            maximum_correction_n=[1.0] * 4,
            maximum_gain_rise_per_s=1.0,
        )
    with pytest.raises(TypeError, match="real number"):
        RotorCorrectionSafetyConfig(
            target_gain=0.1,
            thrust_min_n=[0.0] * 4,
            thrust_max_n=[10.0] * 4,
            maximum_correction_n=[1.0] * 4,
            maximum_gain_rise_per_s=True,
        )


def test_validated_safety_limit_arrays_are_immutable() -> None:
    config = _config(gain=0.1)
    for values in (
        config.thrust_min_n,
        config.thrust_max_n,
        config.maximum_correction_n,
    ):
        assert not np.asarray(values).flags.writeable
        with pytest.raises(ValueError):
            values[0] = 999.0


def test_zero_gain_preserves_flight_controller_baseline() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.0))
    result = blender.blend([4.0] * 4, [7.0, 3.0, 6.0, 2.0], 0.01)
    assert result.applied_total_thrusts_n == (4.0, 4.0, 4.0, 4.0)
    assert result.applied_gain == 0.0
    assert result.transport_target_thrusts_n is None
    assert result.transport_raw_correction_n is None


def test_unit_gain_reproduces_paper_total_command() -> None:
    blender = RotorCorrectionBlender(_config(gain=1.0))
    result = blender.blend([4.0] * 4, [7.0, 3.0, 6.0, 2.0], 0.01)
    assert result.applied_total_thrusts_n == pytest.approx((7.0, 3.0, 6.0, 2.0))
    assert result.applied_residual_thrusts_n == pytest.approx((3.0, -1.0, 2.0, -2.0))


def test_common_headroom_gain_preserves_correction_direction() -> None:
    blender = RotorCorrectionBlender(_config(gain=1.0))
    result = blender.blend([5.0] * 4, [15.0, 4.0, 5.0, 5.0], 0.01)
    assert result.applied_gain == pytest.approx(0.5)
    assert result.applied_total_thrusts_n == pytest.approx((10.0, 4.5, 5.0, 5.0))
    assert result.headroom_limited


def test_gain_rises_from_zero_at_configured_rate() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.5, rise=0.2))
    result = blender.blend([4.0] * 4, [5.0] * 4, 0.5)
    assert result.applied_gain == pytest.approx(0.1)
    assert result.applied_total_thrusts_n == pytest.approx((4.1, 4.1, 4.1, 4.1))


def test_gain_preview_is_non_mutating_and_matches_execution() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.5, rise=0.2))
    assert blender.preview_gains(0.5, 3) == pytest.approx((0.1, 0.2, 0.3))
    assert blender.applied_gain == 0.0

    result = blender.blend_modeled_applied(
        [4.0] * 4,
        [4.2] * 4,
        0.5,
        expected_gain=0.1,
    )

    assert result.applied_gain == pytest.approx(0.1)
    assert result.transport_raw_correction_n == pytest.approx((2.0,) * 4)
    assert result.applied_total_thrusts_n == pytest.approx((4.2,) * 4)


def test_zero_gain_modeled_execution_cannot_hide_nonbaseline_command() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.0))

    with pytest.raises(ValueError, match="zero correction gain"):
        blender.blend_modeled_applied(
            [4.0] * 4,
            [4.1] * 4,
            0.01,
            expected_gain=0.0,
        )


def test_small_positive_gain_remains_defined_instead_of_becoming_zero() -> None:
    gain = 1.0e-8
    blender = RotorCorrectionBlender(_config(gain=gain))
    modeled = np.full(4, 4.0 + 2.0 * gain)
    result = blender.blend_modeled_applied(
        [4.0] * 4,
        modeled,
        0.01,
        expected_gain=gain,
    )

    assert result.transport_raw_correction_n is not None
    assert result.transport_raw_correction_n == pytest.approx((2.0,) * 4, abs=1.0e-7)
    assert result.transport_target_semantics == "gain_limited_algebraic_reconstruction"


def test_near_one_gain_is_still_partial_and_matches_nlp_mode() -> None:
    gain = 0.99999999995
    blender = RotorCorrectionBlender(_config(gain=gain))
    result = blender.blend_modeled_applied(
        [4.0] * 4,
        [5.0] * 4,
        0.01,
        expected_gain=gain,
    )
    assert result.transport_target_semantics == "gain_limited_algebraic_reconstruction"


def test_partial_gain_transport_target_may_exceed_total_thrust_bound() -> None:
    config = RotorCorrectionSafetyConfig(
        target_gain=0.2,
        thrust_min_n=np.zeros(4),
        thrust_max_n=np.full(4, 20.0),
        maximum_correction_n=np.full(4, 5.0),
        maximum_gain_rise_per_s=100.0,
    )
    result = RotorCorrectionBlender(config).blend_modeled_applied(
        [19.0] * 4,
        [20.0] * 4,
        0.01,
        expected_gain=0.2,
    )
    assert result.transport_target_thrusts_n == pytest.approx((24.0,) * 4)
    assert result.applied_total_thrusts_n == pytest.approx((20.0,) * 4)
    assert result.transport_target_semantics == "gain_limited_algebraic_reconstruction"


def test_modeled_execution_rejects_gain_or_headroom_mismatch() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.5, rise=0.2))
    with pytest.raises(ValueError, match="expected_gain"):
        blender.blend_modeled_applied(
            [4.0] * 4,
            [4.1] * 4,
            0.5,
            expected_gain=0.2,
        )

    with pytest.raises(ValueError, match="raw correction"):
        blender.blend_modeled_applied(
            [4.0] * 4,
            [6.0] * 4,
            0.5,
            expected_gain=0.1,
        )


def test_unhealthy_cycle_removes_only_mpc_correction() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.5))
    blender.blend([4.0] * 4, [5.0] * 4, 0.01)
    result = blender.blend([4.2] * 4, [8.0] * 4, 0.01, healthy=False)
    assert not result.valid
    assert result.applied_gain == 0.0
    assert result.applied_total_thrusts_n == pytest.approx((4.2, 4.2, 4.2, 4.2))

    blocked = blender.blend([4.2] * 4, [5.2] * 4, 0.01, healthy=True)
    assert blender.fault_latched
    assert not blocked.valid
    assert blocked.applied_gain == 0.0
    assert blocked.applied_total_thrusts_n == pytest.approx((4.2,) * 4)

    blender.reset()
    resumed = blender.blend([4.2] * 4, [5.2] * 4, 0.01, healthy=True)
    assert not blender.fault_latched
    assert resumed.valid
    assert resumed.applied_gain > 0.0


def test_baseline_outside_identified_limits_fails_closed() -> None:
    blender = RotorCorrectionBlender(_config(gain=0.1))
    with pytest.raises(ValueError, match="baseline"):
        blender.blend([11.0, 4.0, 4.0, 4.0], [5.0] * 4, 0.01)
