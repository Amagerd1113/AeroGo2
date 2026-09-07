from __future__ import annotations

import numpy as np
import pytest

from aerogo2.landing.impact_aware.normal_admittance import (
    HARDWARE_OUTPUT_PERMITTED,
    ContactLossPolicy,
    ForceObservation,
    ForceObservationMode,
    NormalAdmittanceConfig,
    NormalAdmittanceController,
    NormalAdmittanceError,
    NormalAdmittanceState,
    resolve_normal_force,
)


def _config(
    *,
    deadband_n: float = 0.0,
    position_limit_m: float = 0.1,
    velocity_limit_m_per_s: float = 0.5,
    contact_loss_policy: ContactLossPolicy = ContactLossPolicy.RESET,
) -> NormalAdmittanceConfig:
    return NormalAdmittanceConfig(
        virtual_mass_kg=1.0,
        damping_n_s_per_m=2.0,
        stance_stiffness_n_per_m=20.0,
        force_error_deadband_n=deadband_n,
        correction_position_limit_m=position_limit_m,
        correction_velocity_limit_m_per_s=velocity_limit_m_per_s,
        contact_loss_policy=contact_loss_policy,
    )


def _calibrated(force_n: float, *, contact: bool = True) -> ForceObservation:
    return ForceObservation(
        mode=ForceObservationMode.CALIBRATED_NORMAL_ONLY_N,
        contact_detected=contact,
        calibrated_normal_force_n=force_n,
    )


def _counts(count: int, *, contact: bool) -> ForceObservation:
    return ForceObservation(
        mode=ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS,
        contact_detected=contact,
        contact_count=count,
    )


def test_force_observation_modes_are_mutually_exclusive() -> None:
    with pytest.raises(NormalAdmittanceError, match="only one scalar"):
        ForceObservation(
            mode=ForceObservationMode.CALIBRATED_NORMAL_ONLY_N,
            contact_detected=True,
            calibrated_normal_force_n=10.0,
            independent_force_world_n=np.array([0.0, 0.0, 9.0]),
        )

    with pytest.raises(NormalAdmittanceError, match="only one world-frame"):
        ForceObservation(
            mode=ForceObservationMode.INDEPENDENT_3D_WORLD_N,
            contact_detected=True,
            calibrated_normal_force_n=10.0,
            independent_force_world_n=np.array([0.0, 0.0, 9.0]),
        )

    with pytest.raises(NormalAdmittanceError, match="cannot also contain calibrated"):
        ForceObservation(
            mode=ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS,
            contact_detected=True,
            contact_count=123,
            calibrated_normal_force_n=1.0,
        )


def test_contact_only_counts_cannot_drive_newton_admittance() -> None:
    controller = NormalAdmittanceController(_config())
    observation = _counts(1234, contact=True)

    with pytest.raises(NormalAdmittanceError, match="no newton semantics"):
        resolve_normal_force(observation, [0.0, 0.0, 1.0])
    with pytest.raises(NormalAdmittanceError, match="cannot drive admittance"):
        controller.preview(
            observation=observation,
            desired_normal_force_n=20.0,
            ground_normal_world=[0.0, 0.0, 1.0],
            dt_s=0.01,
        )


def test_calibrated_scalar_uniquely_derives_world_force_and_arrays_are_readonly() -> None:
    observation = _calibrated(42.0)
    resolved = resolve_normal_force(observation, [0.0, 1.0, 0.0])

    assert resolved.normal_force_n == pytest.approx(42.0)
    np.testing.assert_allclose(resolved.force_world_n, [0.0, 42.0, 0.0])
    assert not resolved.force_world_n.flags.writeable
    with pytest.raises(ValueError):
        resolved.force_world_n[0] = 1.0


def test_independent_3d_force_is_projected_and_output_has_no_tangential_component() -> None:
    observation = ForceObservation(
        mode=ForceObservationMode.INDEPENDENT_3D_WORLD_N,
        contact_detected=True,
        independent_force_world_n=np.array([30.0, -40.0, 25.0]),
    )
    controller = NormalAdmittanceController(_config())
    transition = controller.preview(
        observation=observation,
        desired_normal_force_n=20.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )

    assert transition.output.estimated_normal_force_n == pytest.approx(25.0)
    assert transition.output.correction_position_world_m[0] == pytest.approx(0.0)
    assert transition.output.correction_position_world_m[1] == pytest.approx(0.0)
    assert transition.output.correction_velocity_world_m_per_s[0] == pytest.approx(0.0)
    assert transition.output.correction_velocity_world_m_per_s[1] == pytest.approx(0.0)


def test_force_error_sign_matches_f_est_minus_f_des() -> None:
    positive = NormalAdmittanceController(_config()).preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    ).output
    negative = NormalAdmittanceController(_config()).preview(
        observation=_calibrated(10.0),
        desired_normal_force_n=30.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    ).output

    assert positive.raw_force_error_n == pytest.approx(20.0)
    assert positive.state.correction_velocity_m_per_s > 0.0
    assert positive.correction_position_world_m[2] > 0.0
    assert negative.raw_force_error_n == pytest.approx(-20.0)
    assert negative.state.correction_velocity_m_per_s < 0.0
    assert negative.correction_position_world_m[2] < 0.0


def test_deadband_removes_small_force_bias_without_drift() -> None:
    controller = NormalAdmittanceController(_config(deadband_n=0.5))
    transition = controller.preview(
        observation=_calibrated(10.4),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )

    assert transition.output.raw_force_error_n == pytest.approx(0.4)
    assert transition.output.admittance_force_error_n == pytest.approx(0.0)
    assert transition.output.state.correction_position_m == pytest.approx(0.0)
    assert transition.output.state.correction_velocity_m_per_s == pytest.approx(0.0)


def test_velocity_and_position_hard_limits_apply_anti_windup() -> None:
    controller = NormalAdmittanceController(
        _config(position_limit_m=0.01, velocity_limit_m_per_s=0.02)
    )
    transition = controller.preview(
        observation=_calibrated(1000.0),
        desired_normal_force_n=0.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=1.0,
    )

    assert transition.output.correction_velocity_limited
    assert transition.output.correction_position_limited
    assert transition.output.state.correction_position_m == pytest.approx(0.01)
    assert transition.output.state.correction_velocity_m_per_s == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("policy", "expected_position"),
    [
        (ContactLossPolicy.RESET, 0.0),
        (ContactLossPolicy.FREEZE, 0.02),
    ],
)
def test_contact_loss_reset_or_freeze_without_using_counts_as_force(
    policy: ContactLossPolicy,
    expected_position: float,
) -> None:
    controller = NormalAdmittanceController(
        _config(contact_loss_policy=policy),
        initial_state=NormalAdmittanceState(
            correction_position_m=0.02,
            correction_velocity_m_per_s=0.1,
            contact_seen=True,
        ),
        initial_ground_normal_world=[0.0, 0.0, 1.0],
    )
    transition = controller.preview(
        observation=_counts(2000, contact=False),
        desired_normal_force_n=25.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )
    output = controller.commit(transition)

    assert output.contact_loss_state_handled
    assert output.estimated_normal_force_n is None
    assert output.raw_force_error_n is None
    assert output.admittance_force_error_n is None
    assert output.state.correction_position_m == pytest.approx(expected_position)
    assert output.state.correction_velocity_m_per_s == pytest.approx(0.0)
    assert not output.state.contact_seen


def test_preview_commit_abort_are_transactional() -> None:
    controller = NormalAdmittanceController(_config())
    first = controller.preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )
    assert controller.state().correction_position_m == pytest.approx(0.0)
    controller.abort(first)
    with pytest.raises(NormalAdmittanceError, match="aborted"):
        controller.commit(first)

    second = controller.preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )
    controller.commit(second)
    assert controller.state().correction_position_m > 0.0
    with pytest.raises(NormalAdmittanceError, match="stale"):
        controller.commit(second)


def test_ground_normal_is_bound_only_on_commit_and_change_fails_closed() -> None:
    controller = NormalAdmittanceController(_config())
    transition = controller.preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )

    # preview/abort 不改变积分器或会话方向；commit 才原子地绑定方向和状态。
    assert controller.bound_ground_normal_world() is None
    controller.commit(transition)
    np.testing.assert_allclose(controller.bound_ground_normal_world(), [0.0, 0.0, 1.0])
    committed_state = controller.state()

    with pytest.raises(NormalAdmittanceError, match="ground normal differs"):
        controller.preview(
            observation=_calibrated(30.0),
            desired_normal_force_n=10.0,
            ground_normal_world=[1.0, 0.0, 0.0],
            dt_s=0.01,
        )
    assert controller.state() == committed_state


def test_explicit_reset_clears_normal_identity_and_allows_new_direction() -> None:
    controller = NormalAdmittanceController(_config())
    first = controller.preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    )
    controller.commit(first)
    assert controller.state().correction_position_m > 0.0

    reset_state = controller.reset()
    assert reset_state == NormalAdmittanceState()
    assert controller.bound_ground_normal_world() is None

    second = controller.preview(
        observation=_calibrated(30.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[1.0, 0.0, 0.0],
        dt_s=0.01,
    )
    output = controller.commit(second)
    assert output.correction_position_world_m[0] > 0.0
    np.testing.assert_allclose(output.correction_position_world_m[1:], [0.0, 0.0])
    np.testing.assert_allclose(controller.bound_ground_normal_world(), [1.0, 0.0, 0.0])


def test_directional_initial_state_requires_explicit_ground_normal_identity() -> None:
    initial = NormalAdmittanceState(
        correction_position_m=0.01,
        correction_velocity_m_per_s=0.0,
        contact_seen=True,
    )
    with pytest.raises(NormalAdmittanceError, match="requires initial_ground_normal"):
        NormalAdmittanceController(_config(), initial_state=initial)

    controller = NormalAdmittanceController(
        _config(),
        initial_state=initial,
        initial_ground_normal_world=[0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(controller.bound_ground_normal_world(), [0.0, 1.0, 0.0])


def test_invalid_ground_normal_and_negative_projection_are_rejected() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        resolve_normal_force(_calibrated(10.0), [0.0, 0.0, 2.0])

    observation = ForceObservation(
        mode=ForceObservationMode.INDEPENDENT_3D_WORLD_N,
        contact_detected=True,
        independent_force_world_n=[0.0, 0.0, -1.0],
    )
    with pytest.raises(NormalAdmittanceError, match="negative projection"):
        resolve_normal_force(observation, [0.0, 0.0, 1.0])


def test_positive_stance_stiffness_is_mandatory() -> None:
    with pytest.raises(ValueError, match="stance_stiffness_n_per_m"):
        NormalAdmittanceConfig(
            virtual_mass_kg=1.0,
            damping_n_s_per_m=2.0,
            stance_stiffness_n_per_m=0.0,
            force_error_deadband_n=0.5,
            correction_position_limit_m=0.1,
            correction_velocity_limit_m_per_s=0.5,
        )


def test_hardware_output_permission_is_always_false() -> None:
    controller = NormalAdmittanceController(_config())
    output = controller.preview(
        observation=_calibrated(10.0),
        desired_normal_force_n=10.0,
        ground_normal_world=[0.0, 0.0, 1.0],
        dt_s=0.01,
    ).output

    assert HARDWARE_OUTPUT_PERMITTED is False
    assert controller.hardware_output_permitted is False
    assert output.hardware_output_permitted is False
