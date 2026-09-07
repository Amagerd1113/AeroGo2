from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import List, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from aerogo2.landing.impact_aware.admittance import (
    AdmittanceConfig,
    AdmittanceTransition,
    AxisAlignedWorkspace,
    LegAdmittanceController,
    nominal_foot_position_body,
    scheduled_spd_matrix,
    touchdown_blend,
    world_to_body_force,
)

FloatArray = NDArray[np.float64]
IDENTITY = np.eye(3, dtype=np.float64)
ZERO = np.zeros(3, dtype=np.float64)


def _config(
    *,
    anti_windup_enabled: bool = False,
    stance_stiffness: Optional[FloatArray] = None,
    force_error_deadband_n: Optional[FloatArray] = None,
    correction_position_limit_m: Optional[FloatArray] = None,
    correction_velocity_limit_m_per_s: Optional[FloatArray] = None,
    contact_release_policy: str = "reset",
    joint_lower: Optional[FloatArray] = None,
    joint_upper: Optional[FloatArray] = None,
    joint_rate_limit: Optional[FloatArray] = None,
) -> AdmittanceConfig:
    return AdmittanceConfig(
        transition_duration_s=1.0,
        touchdown_inertia=np.ones(3),
        stance_inertia=np.ones(3),
        touchdown_damping=np.ones(3),
        stance_damping=np.ones(3),
        restoring_stiffness=np.ones(3),
        stance_stiffness=np.ones(3) if stance_stiffness is None else stance_stiffness,
        force_error_deadband_n=(
            np.zeros(3) if force_error_deadband_n is None else force_error_deadband_n
        ),
        correction_position_limit_m=(
            np.full(3, 10.0)
            if correction_position_limit_m is None
            else correction_position_limit_m
        ),
        correction_velocity_limit_m_per_s=(
            np.full(3, 100.0)
            if correction_velocity_limit_m_per_s is None
            else correction_velocity_limit_m_per_s
        ),
        contact_release_policy=contact_release_policy,
        joint_lower=np.full(3, -10.0) if joint_lower is None else joint_lower,
        joint_upper=np.full(3, 10.0) if joint_upper is None else joint_upper,
        joint_rate_limit=np.full(3, 100.0) if joint_rate_limit is None else joint_rate_limit,
        anti_windup_enabled=anti_windup_enabled,
    )


def _controller(
    *,
    config: Optional[AdmittanceConfig] = None,
    workspace: Optional[AxisAlignedWorkspace] = None,
) -> LegAdmittanceController:
    return LegAdmittanceController(
        config if config is not None else _config(),
        lambda foot: foot,
        workspace
        if workspace is not None
        else AxisAlignedWorkspace(np.full(3, -10.0), np.full(3, 10.0)),
        ZERO,
    )


def _step(
    controller: LegAdmittanceController,
    *,
    current_time_s: float = 0.0,
    dt_s: float = 0.1,
    measured_contact: bool = False,
    touchdown_time_s: Optional[float] = None,
    nominal_foot_position_world: FloatArray = ZERO,
    desired_force_world: FloatArray = ZERO,
    estimated_force_world: FloatArray = ZERO,
):
    return controller.step(
        current_time_s=current_time_s,
        dt_s=dt_s,
        measured_contact=measured_contact,
        touchdown_time_s=touchdown_time_s,
        rotation_body_to_world=IDENTITY,
        body_position_world=ZERO,
        nominal_foot_position_world=nominal_foot_position_world,
        desired_force_world=desired_force_world,
        estimated_force_world=estimated_force_world,
    )


def _preview(
    controller: LegAdmittanceController,
    *,
    current_time_s: float = 0.0,
    dt_s: float = 0.1,
    measured_contact: bool = False,
    touchdown_time_s: Optional[float] = None,
    nominal_foot_position_world: FloatArray = ZERO,
    desired_force_world: FloatArray = ZERO,
    estimated_force_world: FloatArray = ZERO,
) -> AdmittanceTransition:
    return controller.preview(
        current_time_s=current_time_s,
        dt_s=dt_s,
        measured_contact=measured_contact,
        touchdown_time_s=touchdown_time_s,
        rotation_body_to_world=IDENTITY,
        body_position_world=ZERO,
        nominal_foot_position_world=nominal_foot_position_world,
        desired_force_world=desired_force_world,
        estimated_force_world=estimated_force_world,
    )


def test_eq51_and_eq52_use_transpose_of_body_to_world_rotation() -> None:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    force_body = world_to_body_force(np.array([0.0, 2.0, 0.0]), rotation)
    foot_body = nominal_foot_position_body(
        np.array([2.0, 3.0, 4.0]),
        np.array([1.0, 1.0, 1.0]),
        rotation,
    )

    np.testing.assert_allclose(force_body, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(foot_body, [2.0, -1.0, 3.0])


@pytest.mark.parametrize(
    "rotation",
    (
        np.diag([1.0, 1.0, -1.0]),
        np.diag([1.0, 1.0, 2.0]),
        np.full((3, 3), np.nan),
        np.eye(2),
    ),
)
def test_transform_rejects_invalid_rotations(rotation: FloatArray) -> None:
    with pytest.raises(ValueError, match="rotation_body_to_world"):
        world_to_body_force(np.ones(3), rotation)


def test_eq53_touchdown_smoothstep_has_clamped_paper_values() -> None:
    assert touchdown_blend(3.0, None, 2.0).xi == 0.0
    assert touchdown_blend(3.0, 4.0, 2.0).eta == 0.0
    assert touchdown_blend(4.0, 4.0, 2.0).eta == 0.0
    assert touchdown_blend(5.0, 4.0, 2.0).xi == pytest.approx(0.5)
    assert touchdown_blend(5.0, 4.0, 2.0).eta == pytest.approx(0.5)
    assert touchdown_blend(6.0, 4.0, 2.0).eta == 1.0
    assert touchdown_blend(7.0, 4.0, 2.0).eta == 1.0

    with pytest.raises(ValueError, match="greater than zero"):
        touchdown_blend(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        touchdown_blend(float("nan"), None, 1.0)


def test_eq54_schedules_diagonal_and_full_spd_endpoints() -> None:
    touchdown = np.diag([1.0, 2.0, 3.0])
    stance = np.array(
        [
            [4.0, 0.5, 0.0],
            [0.5, 5.0, 0.25],
            [0.0, 0.25, 6.0],
        ]
    )

    scheduled = scheduled_spd_matrix(np.array([1.0, 2.0, 3.0]), stance, 0.25)

    np.testing.assert_allclose(scheduled, 0.75 * touchdown + 0.25 * stance)
    assert np.all(np.linalg.eigvalsh(scheduled) > 0.0)
    with pytest.raises(ValueError, match="positive definite"):
        scheduled_spd_matrix(np.diag([1.0, -1.0, 1.0]), stance, 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        scheduled_spd_matrix(touchdown, stance, 1.01)


def test_config_strictly_validates_spd_joint_and_safety_parameters() -> None:
    with pytest.raises(ValueError, match="transition_duration_s"):
        AdmittanceConfig(
            transition_duration_s=0.0,
            touchdown_inertia=1.0,
            stance_inertia=1.0,
            touchdown_damping=1.0,
            stance_damping=1.0,
            restoring_stiffness=1.0,
            joint_lower=[-1.0] * 3,
            joint_upper=[1.0] * 3,
            joint_rate_limit=[1.0] * 3,
        )
    with pytest.raises(ValueError, match="strictly below"):
        AdmittanceConfig(
            transition_duration_s=1.0,
            touchdown_inertia=1.0,
            stance_inertia=1.0,
            touchdown_damping=1.0,
            stance_damping=1.0,
            restoring_stiffness=1.0,
            joint_lower=[-1.0, 2.0, -1.0],
            joint_upper=[1.0, 1.0, 1.0],
            joint_rate_limit=[1.0] * 3,
        )
    with pytest.raises(ValueError, match="bool"):
        AdmittanceConfig(
            transition_duration_s=1.0,
            touchdown_inertia=1.0,
            stance_inertia=1.0,
            touchdown_damping=1.0,
            stance_damping=1.0,
            restoring_stiffness=1.0,
            joint_lower=[-1.0] * 3,
            joint_upper=[1.0] * 3,
            joint_rate_limit=[1.0] * 3,
            anti_windup_enabled=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="stance_stiffness.*greater than zero"):
        _config(stance_stiffness=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="force_error_deadband_n.*non-negative"):
        _config(force_error_deadband_n=np.array([0.0, -0.1, 0.0]))
    with pytest.raises(ValueError, match="correction_position_limit_m.*greater than zero"):
        _config(correction_position_limit_m=np.array([0.1, 0.0, 0.1]))
    with pytest.raises(ValueError, match="contact_release_policy"):
        _config(contact_release_policy="continue_integrating")


def test_eq55_implicit_step_tracks_force_and_eq56_limits_position_then_rate() -> None:
    config = _config(
        joint_lower=np.full(3, -0.2),
        joint_upper=np.full(3, 0.2),
        joint_rate_limit=np.ones(3),
    )
    seen_foot_commands: List[FloatArray] = []

    def inverse_kinematics(foot: FloatArray) -> FloatArray:
        seen_foot_commands.append(foot.copy())
        return foot

    controller = LegAdmittanceController(
        config,
        inverse_kinematics,
        AxisAlignedWorkspace(np.full(3, -2.0), np.full(3, 2.0)),
        ZERO,
    )
    output = _step(
        controller,
        current_time_s=1.0,
        dt_s=0.1,
        measured_contact=True,
        touchdown_time_s=0.0,
        nominal_foot_position_world=np.array([0.5, 0.0, 0.0]),
        desired_force_world=np.array([1.0, 0.0, 0.0]),
        estimated_force_world=np.array([3.0, 0.0, 0.0]),
    )

    # At eta=1 the nonzero stance stiffness remains active:
    # (I + dt*I + dt^2*I)v1 = dt*(3-1), p1=dt*v1.
    expected_velocity = 0.2 / 1.11
    expected_correction = 0.1 * expected_velocity
    np.testing.assert_allclose(output.admittance_force_body, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(
        output.state.correction_velocity_body,
        [expected_velocity, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        output.state.correction_position_body,
        [expected_correction, 0.0, 0.0],
    )
    np.testing.assert_allclose(seen_foot_commands[0], [0.5 + expected_correction, 0.0, 0.0])
    np.testing.assert_allclose(output.raw_joint_position, seen_foot_commands[0])
    np.testing.assert_allclose(output.bounded_joint_position, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(output.joint_position_command, [0.1, 0.0, 0.0])
    assert output.joint_position_limited
    assert output.joint_rate_limited


def test_precontact_eq55_ignores_force_and_restores_correction_toward_zero() -> None:
    controller = _controller()
    controller.reset(
        ZERO,
        correction_position_body=np.array([1.0, 0.0, 0.0]),
        correction_velocity_body=ZERO,
    )

    output = _step(
        controller,
        measured_contact=False,
        touchdown_time_s=None,
        estimated_force_world=np.array([1000.0, 0.0, 0.0]),
    )

    expected_velocity = -0.1 / 1.11
    expected_position = 1.0 + 0.1 * expected_velocity
    np.testing.assert_allclose(output.admittance_force_body, ZERO)
    np.testing.assert_allclose(
        output.state.correction_velocity_body,
        [expected_velocity, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        output.state.correction_position_body,
        [expected_position, 0.0, 0.0],
    )
    assert expected_position < 1.0


def test_implicit_eq55_integration_remains_finite_for_stiff_landing_parameters() -> None:
    config = AdmittanceConfig(
        transition_duration_s=1.0,
        touchdown_inertia=1.0,
        stance_inertia=1.0,
        touchdown_damping=1.0,
        stance_damping=1.0,
        restoring_stiffness=1_000_000.0,
        joint_lower=[-10.0] * 3,
        joint_upper=[10.0] * 3,
        joint_rate_limit=[100.0] * 3,
        correction_position_limit_m=[2.0] * 3,
        correction_velocity_limit_m_per_s=[100.0] * 3,
    )
    controller = _controller(config=config)
    controller.reset(
        ZERO,
        correction_position_body=np.array([1.0, 0.0, 0.0]),
        correction_velocity_body=ZERO,
    )

    output = _step(controller, dt_s=0.1)

    assert np.all(np.isfinite(output.state.correction_position_body))
    assert np.all(np.isfinite(output.state.correction_velocity_body))
    assert abs(output.state.correction_position_body[0]) < 0.001


def test_workspace_clamp_preserves_paper_state_when_anti_windup_is_disabled() -> None:
    workspace = AxisAlignedWorkspace(np.full(3, -0.2), np.full(3, 0.2))
    controller = _controller(config=_config(anti_windup_enabled=False), workspace=workspace)
    controller.reset(
        ZERO,
        correction_position_body=np.array([0.3, 0.0, 0.0]),
        correction_velocity_body=np.array([1.0, 0.0, 0.0]),
    )

    output = _step(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
    )

    assert output.workspace_limited
    assert not output.anti_windup_applied
    assert output.state.correction_position_body[0] > 0.3
    assert output.foot_position_command_body[0] == pytest.approx(0.2)


def test_optional_anti_windup_projects_state_and_stops_outward_velocity() -> None:
    workspace = AxisAlignedWorkspace(np.full(3, -0.2), np.full(3, 0.2))
    controller = _controller(config=_config(anti_windup_enabled=True), workspace=workspace)
    controller.reset(
        ZERO,
        correction_position_body=np.array([0.3, 0.0, 0.0]),
        correction_velocity_body=np.array([1.0, 0.0, 0.0]),
    )

    output = _step(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
    )

    assert output.anti_windup_applied
    np.testing.assert_allclose(output.state.correction_position_body, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(output.state.correction_velocity_body, ZERO)


def test_invalid_callback_output_does_not_mutate_controller_state() -> None:
    def invalid_inverse_kinematics(foot: FloatArray) -> FloatArray:
        del foot
        return np.array([np.nan, 0.0, 0.0])

    controller = LegAdmittanceController(
        _config(),
        invalid_inverse_kinematics,
        AxisAlignedWorkspace(np.full(3, -10.0), np.full(3, 10.0)),
        ZERO,
    )
    before_state = controller.state
    before_joint = controller.previous_joint_command

    with pytest.raises(ValueError, match="finite"):
        _step(
            controller,
            current_time_s=1.0,
            measured_contact=True,
            touchdown_time_s=0.0,
            estimated_force_world=np.ones(3),
        )

    np.testing.assert_array_equal(
        controller.state.correction_position_body,
        before_state.correction_position_body,
    )
    np.testing.assert_array_equal(
        controller.state.correction_velocity_body,
        before_state.correction_velocity_body,
    )
    np.testing.assert_array_equal(controller.previous_joint_command, before_joint)


def test_step_rejects_inconsistent_contact_and_nonfinite_inputs_without_mutation() -> None:
    controller = _controller()

    with pytest.raises(ValueError, match="touchdown_time_s is required"):
        _step(controller, measured_contact=True, touchdown_time_s=None)
    with pytest.raises(ValueError, match="finite"):
        _step(controller, estimated_force_world=np.array([np.inf, 0.0, 0.0]))
    with pytest.raises(ValueError, match="bool"):
        controller.step(
            current_time_s=0.0,
            dt_s=0.1,
            measured_contact=1,  # type: ignore[arg-type]
            touchdown_time_s=None,
            rotation_body_to_world=IDENTITY,
            body_position_world=ZERO,
            nominal_foot_position_world=ZERO,
            desired_force_world=ZERO,
            estimated_force_world=ZERO,
        )

    np.testing.assert_array_equal(controller.state.correction_position_body, ZERO)
    np.testing.assert_array_equal(controller.previous_joint_command, ZERO)


def test_axis_aligned_workspace_validates_and_clamps_each_axis() -> None:
    workspace = AxisAlignedWorkspace([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(workspace.limit(np.array([2.0, -4.0, 1.0])), [1.0, -2.0, 1.0])

    with pytest.raises(ValueError, match="strictly below"):
        AxisAlignedWorkspace([0.0, -1.0, -1.0], [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        workspace.limit(np.ones(2))


def test_preview_is_immutable_and_does_not_mutate_controller_state() -> None:
    controller = _controller()
    state_before = controller.state
    joint_before = controller.previous_joint_command
    generation_before = controller.generation

    transition = _preview(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
        estimated_force_world=np.ones(3),
    )

    assert transition.generation == generation_before
    assert np.any(transition.output.state.correction_position_body != ZERO)
    np.testing.assert_array_equal(
        controller.state.correction_position_body,
        state_before.correction_position_body,
    )
    np.testing.assert_array_equal(
        controller.state.correction_velocity_body,
        state_before.correction_velocity_body,
    )
    np.testing.assert_array_equal(controller.previous_joint_command, joint_before)
    assert controller.generation == generation_before
    assert not transition.output.joint_position_command.flags.writeable
    with pytest.raises(FrozenInstanceError):
        transition.generation = 42  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        transition.output.joint_position_command[0] = 42.0


def test_commit_applies_previewed_state_and_advances_generation_once() -> None:
    controller = _controller()
    transition = _preview(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
        estimated_force_world=np.ones(3),
    )

    output = controller.commit(transition)

    assert output is transition.output
    np.testing.assert_allclose(
        controller.state.correction_position_body,
        transition.output.state.correction_position_body,
    )
    np.testing.assert_allclose(
        controller.state.correction_velocity_body,
        transition.output.state.correction_velocity_body,
    )
    np.testing.assert_allclose(
        controller.previous_joint_command,
        transition.output.joint_position_command,
    )
    assert controller.generation == transition.generation + 1

    with pytest.raises(ValueError, match="stale or already committed"):
        controller.commit(transition)


def test_commit_rejects_transition_from_another_controller() -> None:
    source = _controller()
    destination = _controller()
    transition = _preview(source)

    with pytest.raises(ValueError, match="different controller"):
        destination.commit(transition)

    assert destination.generation == 0
    np.testing.assert_array_equal(destination.state.correction_position_body, ZERO)


def test_validate_transition_is_side_effect_free_and_rejects_foreign_or_stale_tokens() -> None:
    controller = _controller()
    other = _controller()
    transition = _preview(controller)

    controller.validate_transition(
        transition,
        applied_joint_position=np.array([0.25, 0.0, 0.0]),
    )
    assert controller.generation == transition.generation
    np.testing.assert_array_equal(controller.state.correction_position_body, ZERO)
    np.testing.assert_array_equal(controller.previous_joint_command, ZERO)

    with pytest.raises(ValueError, match="different controller"):
        other.validate_transition(transition)

    controller.commit(transition)
    with pytest.raises(ValueError, match="stale or already committed"):
        controller.validate_transition(transition)


def test_reset_invalidates_all_outstanding_preview_transitions() -> None:
    controller = _controller()
    transition_a = _preview(controller)
    transition_b = _preview(controller)

    controller.reset(np.array([0.1, 0.0, 0.0]))

    assert controller.generation == transition_a.generation + 1
    with pytest.raises(ValueError, match="stale or already committed"):
        controller.commit(transition_a)
    with pytest.raises(ValueError, match="stale or already committed"):
        controller.commit(transition_b)
    np.testing.assert_allclose(controller.previous_joint_command, [0.1, 0.0, 0.0])


def test_commit_uses_applied_joint_feedback_as_next_rate_limit_reference() -> None:
    controller = _controller(config=_config(joint_rate_limit=np.ones(3)))
    first = _preview(
        controller,
        nominal_foot_position_world=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(first.output.joint_position_command, [0.1, 0.0, 0.0])

    controller.commit(first, applied_joint_position=np.array([0.05, 0.0, 0.0]))
    np.testing.assert_allclose(controller.previous_joint_command, [0.05, 0.0, 0.0])

    second = _preview(
        controller,
        nominal_foot_position_world=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(second.output.joint_position_command, [0.15, 0.0, 0.0])


def test_invalid_applied_joint_feedback_does_not_consume_transition() -> None:
    controller = _controller(config=_config(joint_lower=np.full(3, -1.0), joint_upper=np.ones(3)))
    transition = _preview(controller)

    with pytest.raises(ValueError, match="joint bounds"):
        controller.commit(transition, applied_joint_position=np.array([2.0, 0.0, 0.0]))

    assert controller.generation == transition.generation
    controller.commit(transition, applied_joint_position=np.array([0.25, 0.0, 0.0]))
    np.testing.assert_allclose(controller.previous_joint_command, [0.25, 0.0, 0.0])


def test_stance_keeps_nonzero_stiffness_and_force_deadband_removes_bias() -> None:
    config = _config(
        stance_stiffness=np.full(3, 4.0),
        force_error_deadband_n=np.full(3, 0.5),
        correction_position_limit_m=np.ones(3),
    )
    controller = _controller(config=config)

    inside_deadband = _step(
        controller,
        current_time_s=1.0,
        dt_s=0.01,
        measured_contact=True,
        touchdown_time_s=0.0,
        estimated_force_world=np.array([0.4, 0.0, 0.0]),
    )
    np.testing.assert_allclose(inside_deadband.raw_force_error_body, [0.4, 0.0, 0.0])
    np.testing.assert_allclose(inside_deadband.admittance_force_body, ZERO)
    np.testing.assert_allclose(inside_deadband.effective_stiffness, np.eye(3) * 4.0)

    output = inside_deadband
    for index in range(5000):
        output = _step(
            controller,
            current_time_s=1.01 + 0.01 * index,
            dt_s=0.01,
            measured_contact=True,
            touchdown_time_s=0.0,
            estimated_force_world=np.array([2.0, 0.0, 0.0]),
        )
    # Effective input is 2.0 - 0.5 = 1.5 N, so K_stance=4 N/m
    # yields a bounded 0.375 m equilibrium instead of constant-velocity drift.
    assert output.state.correction_position_body[0] == pytest.approx(0.375, abs=2.0e-4)
    assert abs(output.state.correction_velocity_body[0]) < 2.0e-4


def test_cartesian_correction_position_and_velocity_have_hard_limits() -> None:
    controller = _controller(
        config=_config(
            correction_position_limit_m=np.full(3, 0.02),
            correction_velocity_limit_m_per_s=np.full(3, 0.05),
        )
    )
    controller.reset(
        ZERO,
        correction_position_body=np.array([0.019, 0.0, 0.0]),
    )

    output = _step(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
        estimated_force_world=np.array([1000.0, 0.0, 0.0]),
    )

    assert output.correction_velocity_limited
    assert output.correction_position_limited
    np.testing.assert_allclose(output.state.correction_position_body, [0.02, 0.0, 0.0])
    np.testing.assert_allclose(output.state.correction_velocity_body, ZERO)


@pytest.mark.parametrize("policy", ("reset", "freeze"))
def test_contact_release_explicitly_resets_or_freezes_state(policy: str) -> None:
    controller = _controller(config=_config(contact_release_policy=policy))
    contacted = _step(
        controller,
        current_time_s=1.0,
        measured_contact=True,
        touchdown_time_s=0.0,
        estimated_force_world=np.array([2.0, 0.0, 0.0]),
    )
    assert contacted.state.contact_seen
    assert contacted.state.correction_position_body[0] > 0.0

    released = _step(controller, current_time_s=1.1, measured_contact=False)
    assert released.contact_release_state_handled
    assert released.state.contact_seen
    np.testing.assert_allclose(released.state.correction_velocity_body, ZERO)
    if policy == "reset":
        np.testing.assert_allclose(released.state.correction_position_body, ZERO)
    else:
        np.testing.assert_allclose(
            released.state.correction_position_body,
            contacted.state.correction_position_body,
        )


def test_joint_rate_limit_is_projected_back_through_forward_kinematics() -> None:
    controller = LegAdmittanceController(
        _config(
            anti_windup_enabled=True,
            correction_position_limit_m=np.full(3, 2.0),
            joint_rate_limit=np.full(3, 0.1),
        ),
        lambda foot: foot,
        AxisAlignedWorkspace(np.full(3, -10.0), np.full(3, 10.0)),
        ZERO,
        forward_kinematics=lambda joint: joint,
    )

    output = _step(
        controller,
        nominal_foot_position_world=np.array([1.0, 0.0, 0.0]),
    )

    assert output.joint_rate_limited
    assert output.joint_chain_state_projected
    assert output.anti_windup_applied
    np.testing.assert_allclose(output.joint_position_command, [0.01, 0.0, 0.0])
    np.testing.assert_allclose(output.state.correction_position_body, [-0.99, 0.0, 0.0])
    np.testing.assert_allclose(output.state.correction_velocity_body, ZERO)


def test_downstream_applied_q_feedback_updates_cartesian_anti_windup_state() -> None:
    controller = LegAdmittanceController(
        _config(anti_windup_enabled=True),
        lambda foot: foot,
        AxisAlignedWorkspace(np.full(3, -10.0), np.full(3, 10.0)),
        ZERO,
        forward_kinematics=lambda joint: joint,
    )
    transition = _preview(
        controller,
        nominal_foot_position_world=np.array([0.5, 0.0, 0.0]),
    )

    committed = controller.commit(
        transition,
        applied_joint_position=np.array([0.25, 0.0, 0.0]),
    )

    assert committed.downstream_feedback_anti_windup_applied
    assert committed.anti_windup_applied
    np.testing.assert_allclose(committed.state.correction_position_body, [-0.25, 0.0, 0.0])
    np.testing.assert_allclose(controller.state.correction_position_body, [-0.25, 0.0, 0.0])
