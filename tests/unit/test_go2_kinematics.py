"""Offline FK/IK regression tests for the hash-pinned Unitree Go2 URDF."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from aerogo2.landing.impact_aware.go2_kinematics import (
    HARDWARE_VALIDATED,
    OFFLINE_PRIOR_ONLY,
    SDK_LEG_ORDER,
    Go2KinematicsError,
    Go2LegKinematics,
    Go2UrdfKinematics,
    load_go2_urdf_kinematics,
)
from aerogo2.landing.impact_aware.go2_urdf import (
    Go2UrdfMassProperties,
    load_go2_urdf_mass_properties,
)
from aerogo2.landing.impact_aware.types import (
    FootLeverArmsFromComBody,
    FootPositionsFromBodyOriginB,
    foot_positions_from_body_origin_B_to_com_lever_arms,
)

VENDORED_SHA256 = "8f4571b49f35ce04b8833d561c403bddeb9cd8f7077e2ddbee82895726de487c"


def _home_pose() -> Dict[str, float]:
    return {
        f"{leg}_{role}_joint": value
        for leg in ("FL", "FR", "RL", "RR")
        for role, value in (("hip", 0.0), ("thigh", 0.9), ("calf", -1.8))
    }


def _load(path: Path, digest: str = VENDORED_SHA256) -> Go2UrdfMassProperties:
    return load_go2_urdf_mass_properties(
        path,
        expected_sha256=digest,
        expected_robot_name="go2_description",
        root_link="base",
        joint_positions_rad=_home_pose(),
    )


@pytest.fixture
def kinematics(project_root: Path) -> Go2UrdfKinematics:
    source = project_root / "configs" / "go2_description.unitree_ros.urdf"
    return load_go2_urdf_kinematics(_load(source))


def _two_branch_test_leg(kinematics: Go2UrdfKinematics) -> Go2LegKinematics:
    """Use Go2 geometry with test-only symmetric knee limits.

    The production URDF permits only the normal bent-knee branch.  Widening
    only the test copy's thigh/calf limits creates the two exact planar IK
    branches needed to regression-test selection, without weakening any
    production-model bounds.
    """

    source = kinematics.leg("FR")
    hip, thigh, calf, foot = source.joints
    return Go2LegKinematics(
        leg_name="FR",
        movable_joints=(
            hip,
            replace(thigh, lower_rad=-3.0, upper_rad=3.0),
            replace(calf, lower_rad=-2.7, upper_rad=2.7),
        ),
        foot_joint=foot,
        home_q_rad=[0.0, 0.9, -1.8],
        urdf_root_from_body_origin_B_m=[0.0, 0.0, 0.0],
    )


def test_model_is_explicitly_offline_and_preserves_sdk_leg_order(
    kinematics: Go2UrdfKinematics,
) -> None:
    assert OFFLINE_PRIOR_ONLY
    assert not HARDWARE_VALIDATED
    assert kinematics.offline_prior_only
    assert not kinematics.hardware_validated
    assert kinematics.leg_order == ("FR", "FL", "RR", "RL") == SDK_LEG_ORDER
    assert kinematics.sdk_joint_order == tuple(
        f"{leg}_{role}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for role in ("hip", "thigh", "calf")
    )


def test_joint_geometry_limits_and_arrays_come_from_pinned_urdf(
    kinematics: Go2UrdfKinematics,
) -> None:
    front_right = kinematics.leg("FR")
    hip, thigh, calf, foot = front_right.joints

    assert hip.translation_parent_m == pytest.approx([0.1934, -0.0465, 0.0])
    assert thigh.translation_parent_m == pytest.approx([0.0, -0.0955, 0.0])
    assert calf.translation_parent_m == pytest.approx([0.0, 0.0, -0.213])
    assert foot.translation_parent_m == pytest.approx([0.0, 0.0, -0.213])
    assert tuple(hip.axis_joint) == pytest.approx([1.0, 0.0, 0.0])
    assert tuple(thigh.axis_joint) == pytest.approx([0.0, 1.0, 0.0])
    assert tuple(calf.axis_joint) == pytest.approx([0.0, 1.0, 0.0])
    assert front_right.lower_rad == pytest.approx([-1.0472, -1.5708, -2.7227])
    assert front_right.upper_rad == pytest.approx([1.0472, 3.4907, -0.83776])

    with pytest.raises(ValueError):
        hip.translation_parent_m[0] = 0.0
    limits = front_right.lower_rad
    with pytest.raises(ValueError):
        limits[0] = 0.0


def test_official_home_pose_round_trips_for_all_four_legs(
    kinematics: Go2UrdfKinematics,
) -> None:
    expected = {
        "FR": np.array([0.1934, -0.1420, -0.26480585]),
        "FL": np.array([0.1934, 0.1420, -0.26480585]),
        "RR": np.array([-0.1934, -0.1420, -0.26480585]),
        "RL": np.array([-0.1934, 0.1420, -0.26480585]),
    }
    for leg_name in SDK_LEG_ORDER:
        leg = kinematics.leg(leg_name)
        foot = leg.forward(leg.home_q_rad)
        recovered = leg.inverse(foot)
        assert foot == pytest.approx(expected[leg_name], abs=1.0e-8)
        assert recovered == pytest.approx(leg.home_q_rad, abs=1.0e-12)
        assert leg.forward(recovered) == pytest.approx(foot, abs=1.0e-12)
        with pytest.raises(ValueError):
            foot[0] = 0.0
        with pytest.raises(ValueError):
            recovered[0] = 0.0


@pytest.mark.parametrize("leg_name", SDK_LEG_ORDER)
def test_nonsingular_pose_round_trip_and_jacobian(
    kinematics: Go2UrdfKinematics,
    leg_name: str,
) -> None:
    leg = kinematics.leg(leg_name)
    q = np.array([0.1, 1.1, -1.9])
    foot = leg.forward(q)
    recovered = leg.inverse(foot)
    assert leg.forward(recovered) == pytest.approx(foot, abs=1.0e-7)

    analytic = leg.jacobian(q)
    numerical = np.column_stack(
        [
            (
                leg.forward(q + np.eye(3)[index] * 1.0e-7)
                - leg.forward(q - np.eye(3)[index] * 1.0e-7)
            )
            / 2.0e-7
            for index in range(3)
        ]
    )
    assert analytic == pytest.approx(numerical, abs=2.0e-8)
    with pytest.raises(ValueError):
        analytic[0, 0] = 0.0


def test_inverse_selects_the_exact_branch_nearest_the_preferred_joint_seed(
    kinematics: Go2UrdfKinematics,
) -> None:
    leg = _two_branch_test_leg(kinematics)
    bent_backward = np.array([0.1, 0.6, -1.4])
    bent_forward = np.array([0.1, -0.8, 1.4])
    target = leg.forward(bent_backward)
    np.testing.assert_allclose(leg.forward(bent_forward), target, atol=1.0e-12)

    selected_backward = leg.inverse(target, preferred_q_rad=bent_backward)
    selected_forward = leg.inverse(target, preferred_q_rad=bent_forward)
    selected_default = leg.inverse(target)

    np.testing.assert_allclose(selected_backward, bent_backward, atol=1.0e-10)
    np.testing.assert_allclose(selected_forward, bent_forward, atol=1.0e-10)
    # Backward compatibility: no explicit seed means the pinned home pose.
    np.testing.assert_allclose(selected_default, bent_backward, atol=1.0e-7)


def test_inverse_does_not_stop_at_first_feasible_branch(
    kinematics: Go2UrdfKinematics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = _two_branch_test_leg(kinematics)
    first_but_far = np.array([0.1, 0.6, -1.4])
    later_but_near = np.array([0.1, -0.8, 1.4])
    target = leg.forward(first_but_far)
    np.testing.assert_allclose(leg.forward(later_but_near), target, atol=1.0e-12)

    first_seed = np.zeros(3)
    second_seed = np.ones(3)
    monkeypatch.setattr(
        leg,
        "_candidate_seeds",
        lambda preferred: (first_seed, second_seed),
    )
    monkeypatch.setattr(
        leg,
        "_refine",
        lambda requested, seed: (
            (first_but_far, 0.0) if seed is first_seed else (later_but_near, 0.0)
        ),
    )

    recovered = leg.inverse(target, preferred_q_rad=later_but_near + [0.0, 0.01, 0.0])
    np.testing.assert_allclose(recovered, later_but_near, atol=1.0e-12)


def test_inverse_tracks_a_continuous_cartesian_trajectory_without_branch_switching(
    kinematics: Go2UrdfKinematics,
) -> None:
    leg = _two_branch_test_leg(kinematics)
    expected_joint_path = [
        np.array([0.1, thigh, -1.4]) for thigh in np.linspace(0.55, 0.75, 11)
    ]
    preferred = expected_joint_path[0]
    recovered_path = []
    for expected in expected_joint_path:
        target = leg.forward(expected)
        recovered = leg.inverse(target, preferred_q_rad=preferred)
        np.testing.assert_allclose(leg.forward(recovered), target, atol=1.0e-7)
        assert recovered[2] < 0.0
        recovered_path.append(recovered)
        preferred = recovered

    step_sizes = np.linalg.norm(np.diff(np.vstack(recovered_path), axis=0), axis=1)
    assert np.max(step_sizes) < 0.03


def test_inverse_wrapper_forwards_preferred_seed_and_rejects_invalid_seeds(
    kinematics: Go2UrdfKinematics,
) -> None:
    q = np.array([0.1, 1.1, -1.9])
    target = kinematics.forward("FR", q)
    recovered = kinematics.inverse("FR", target, preferred_q_rad=q)
    np.testing.assert_allclose(recovered, q, atol=1.0e-10)

    leg = kinematics.leg("FR")
    with pytest.raises(Go2KinematicsError, match="preferred_q_rad.*shape"):
        leg.inverse(target, preferred_q_rad=[0.0, 0.9])
    with pytest.raises(Go2KinematicsError, match="preferred_q_rad.*finite"):
        leg.inverse(target, preferred_q_rad=[0.0, np.nan, -1.8])
    with pytest.raises(Go2KinematicsError, match="preferred_q_rad.*violates URDF joint limits"):
        leg.inverse(target, preferred_q_rad=[leg.upper_rad[0] + 0.01, 0.9, -1.8])


def test_inverse_accepts_verified_near_limit_solution_and_fails_closed_beyond_singularity(
    kinematics: Go2UrdfKinematics,
) -> None:
    bounded_leg = kinematics.leg("FR")
    near_limit = np.array([bounded_leg.upper_rad[0] - 1.0e-8, 0.9, -1.8])
    target_at_limit = bounded_leg.forward(near_limit)
    recovered = bounded_leg.inverse(target_at_limit, preferred_q_rad=near_limit)
    np.testing.assert_allclose(recovered, near_limit, atol=1.0e-10)
    assert np.all(recovered >= bounded_leg.lower_rad)
    assert np.all(recovered <= bounded_leg.upper_rad)

    singular_leg = _two_branch_test_leg(kinematics)
    straight = np.array([0.0, 0.6, 0.0])
    singular_target = singular_leg.forward(straight)
    assert np.linalg.matrix_rank(singular_leg.jacobian(straight), tol=1.0e-10) == 2
    np.testing.assert_allclose(
        singular_leg.inverse(singular_target, preferred_q_rad=straight),
        straight,
        atol=1.0e-12,
    )

    # Extend 0.2 micrometres along the fully stretched leg.  This is outside
    # the reachable set but close enough to exercise the strict 0.1 um FK
    # tolerance instead of only the obvious far-away failure case.
    thigh_origin = np.array([0.1934, -0.1420, 0.0])
    outward = singular_target - thigh_origin
    outward /= np.linalg.norm(outward)
    just_unreachable = singular_target + 2.0e-7 * outward
    with pytest.raises(Go2KinematicsError, match="unreachable"):
        singular_leg.inverse(just_unreachable, preferred_q_rad=straight)


def test_body_origin_offset_is_explicit_translation(project_root: Path) -> None:
    properties = _load(project_root / "configs" / "go2_description.unitree_ros.urdf")
    baseline = Go2UrdfKinematics(properties)
    shifted = Go2UrdfKinematics(
        properties,
        urdf_root_from_body_origin_B_m=(0.01, -0.02, 0.03),
    )
    assert shifted.forward("FR", [0.0, 0.9, -1.8]) - baseline.forward(
        "FR", [0.0, 0.9, -1.8]
    ) == pytest.approx([0.01, -0.02, 0.03])


def test_all_feet_use_labeled_order_and_explicit_B_to_C_lever_arm_conversion(
    kinematics: Go2UrdfKinematics,
) -> None:
    joint_positions = {
        leg_name: kinematics.leg(leg_name).home_q_rad
        for leg_name in reversed(SDK_LEG_ORDER)
    }
    feet_from_B = kinematics.forward_all(joint_positions)

    assert isinstance(feet_from_B, FootPositionsFromBodyOriginB)
    assert feet_from_B.leg_order == SDK_LEG_ORDER
    np.testing.assert_allclose(
        feet_from_B.values_m,
        np.vstack(
            [kinematics.forward(name, joint_positions[name]) for name in SDK_LEG_ORDER]
        ),
    )

    target_order = ("RL", "RR", "FL", "FR")
    offset_BC = np.array([0.1, -0.2, 0.3])
    lever_arms = foot_positions_from_body_origin_B_to_com_lever_arms(
        feet_from_B,
        offset_BC,
        target_leg_order=target_order,
    )
    via_model = kinematics.foot_lever_arms_from_com(
        joint_positions,
        offset_BC,
        output_leg_order=target_order,
    )

    assert isinstance(lever_arms, FootLeverArmsFromComBody)
    assert lever_arms.leg_order == target_order
    np.testing.assert_allclose(
        lever_arms.values_m,
        np.vstack([kinematics.forward(name, joint_positions[name]) for name in target_order])
        - offset_BC,
    )
    np.testing.assert_allclose(via_model.values_m, lever_arms.values_m)
    assert not lever_arms.values_m.flags.writeable


def test_all_feet_reject_missing_or_ambiguous_leg_identity(
    kinematics: Go2UrdfKinematics,
) -> None:
    joint_positions = {
        leg_name: kinematics.leg(leg_name).home_q_rad for leg_name in SDK_LEG_ORDER
    }
    missing = dict(joint_positions)
    missing.pop("RL")
    with pytest.raises(Go2KinematicsError, match="exactly the canonical Go2 legs"):
        kinematics.forward_all(missing)
    with pytest.raises(Go2KinematicsError, match="permutation"):
        kinematics.forward_all(
            joint_positions,
            output_leg_order=("FR", "FL", "RR", "UNKNOWN"),
        )
    with pytest.raises(TypeError, match="FootPositionsFromBodyOriginB"):
        foot_positions_from_body_origin_B_to_com_lever_arms(
            np.zeros((4, 3)),  # type: ignore[arg-type]
            np.zeros(3),
        )


def test_geometry_is_reparsed_from_verified_urdf_not_hard_coded(
    project_root: Path,
    tmp_path: Path,
) -> None:
    original = (project_root / "configs" / "go2_description.unitree_ros.urdf").read_bytes()
    old = b'<origin xyz="0.1934 -0.0465 0" rpy="0 0 0" />'
    new = b'<origin xyz="0.2034 -0.0465 0" rpy="0 0 0" />'
    assert original.count(old) == 1
    modified = original.replace(old, new, 1)
    source = tmp_path / "modified_go2.urdf"
    source.write_bytes(modified)
    digest = hashlib.sha256(modified).hexdigest()

    model = Go2UrdfKinematics(_load(source, digest))
    assert model.leg("FR").joints[0].translation_parent_m == pytest.approx([0.2034, -0.0465, 0.0])
    assert model.forward("FR", [0.0, 0.9, -1.8])[0] == pytest.approx(0.2034)


def test_changed_source_after_verification_is_rejected(
    project_root: Path,
    tmp_path: Path,
) -> None:
    payload = (project_root / "configs" / "go2_description.unitree_ros.urdf").read_bytes()
    source = tmp_path / "go2.urdf"
    source.write_bytes(payload)
    properties = _load(source, hashlib.sha256(payload).hexdigest())
    source.write_bytes(payload + b"\n")

    with pytest.raises(Go2KinematicsError, match="changed after mass-property evaluation"):
        Go2UrdfKinematics(properties)


def test_invalid_joint_or_foot_requests_fail_closed(kinematics: Go2UrdfKinematics) -> None:
    leg = kinematics.leg("FR")
    with pytest.raises(Go2KinematicsError, match="unknown Go2 leg"):
        kinematics.leg("RF")
    with pytest.raises(Go2KinematicsError, match="violates URDF joint limits"):
        leg.forward([leg.lower_rad[0] - 0.01, 0.9, -1.8])
    with pytest.raises(Go2KinematicsError, match="shape"):
        leg.forward([0.0, 0.9])
    with pytest.raises(Go2KinematicsError, match="finite"):
        leg.inverse([0.0, np.nan, -0.2])
    with pytest.raises(Go2KinematicsError, match="unreachable"):
        leg.inverse([10.0, 10.0, 10.0])


def test_invalid_model_inputs_are_rejected(project_root: Path) -> None:
    properties = _load(project_root / "configs" / "go2_description.unitree_ros.urdf")
    with pytest.raises(TypeError, match="Go2UrdfMassProperties"):
        Go2UrdfKinematics(object())  # type: ignore[arg-type]
    with pytest.raises(Go2KinematicsError, match="finite"):
        Go2UrdfKinematics(properties, urdf_root_from_body_origin_B_m=[0.0, 0.0, np.inf])
