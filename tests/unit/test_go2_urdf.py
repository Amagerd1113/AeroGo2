"""Pinned Unitree Go2 URDF provenance and mass-property regression tests."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from aerogo2.landing.impact_aware.go2_urdf import (
    Go2UrdfError,
    Go2UrdfMassProperties,
    combine_with_point_mass,
    load_go2_urdf_mass_properties,
)

VENDORED_SHA256 = "8f4571b49f35ce04b8833d561c403bddeb9cd8f7077e2ddbee82895726de487c"


def _urdf_path(project_root: Path) -> Path:
    return project_root / "configs" / "go2_description.unitree_ros.urdf"


def _standard_standing_pose() -> Dict[str, float]:
    pose: Dict[str, float] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        pose[f"{leg}_hip_joint"] = 0.0
        pose[f"{leg}_thigh_joint"] = 0.9
        pose[f"{leg}_calf_joint"] = -1.8
    return pose


def _load(path: Path, *, sha256: str = VENDORED_SHA256) -> Go2UrdfMassProperties:
    return load_go2_urdf_mass_properties(
        path,
        expected_sha256=sha256,
        expected_robot_name="go2_description",
        root_link="base",
        joint_positions_rad=_standard_standing_pose(),
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_vendored_urdf_hash_mass_and_standard_pose_regression(project_root: Path) -> None:
    properties = _load(_urdf_path(project_root))

    assert properties.source_sha256 == f"sha256:{VENDORED_SHA256}"
    assert properties.robot_name == "go2_description"
    assert properties.root_link == "base"
    assert properties.total_link_count == 42
    assert properties.inertial_link_count == 33
    assert properties.positive_mass_link_count == 31
    assert properties.mass_kg == pytest.approx(16.087, abs=1.0e-12)
    assert properties.com_from_root_m == pytest.approx(
        np.array([-0.00169336104, 0.0, -0.0175892312]),
        abs=1.0e-10,
    )
    assert properties.inertia_about_com_root_axes_kg_m2 == pytest.approx(
        np.array(
            [
                [0.165562021, 0.000121660, -0.0155785415],
                [0.000121660, 0.501245124, -0.000031200],
                [-0.0155785415, -0.000031200, 0.562125768],
            ]
        ),
        abs=1.0e-9,
    )
    assert properties.inertia_about_root_origin_root_axes_kg_m2 == pytest.approx(
        np.array(
            [
                [0.170539034, 0.000121660, -0.0160576915],
                [0.000121660, 0.506268266, -0.000031200],
                [-0.0160576915, -0.000031200, 0.562171897],
            ]
        ),
        abs=1.0e-9,
    )
    with pytest.raises(ValueError):
        properties.com_from_root_m[0] = 0.0


def test_all_twelve_movable_joints_and_vendor_limits_are_preserved(project_root: Path) -> None:
    properties = _load(_urdf_path(project_root))
    limits = {limit.name: limit for limit in properties.movable_joint_limits}

    assert tuple(limits) == tuple(_standard_standing_pose())
    assert len(limits) == 12
    for leg in ("FL", "FR", "RL", "RR"):
        hip = limits[f"{leg}_hip_joint"]
        thigh = limits[f"{leg}_thigh_joint"]
        calf = limits[f"{leg}_calf_joint"]
        assert hip.axis_joint == pytest.approx([1.0, 0.0, 0.0])
        assert thigh.axis_joint == pytest.approx([0.0, 1.0, 0.0])
        assert calf.axis_joint == pytest.approx([0.0, 1.0, 0.0])
        assert (hip.lower_rad, hip.upper_rad, hip.effort_nm, hip.velocity_rad_s) == pytest.approx(
            (-1.0472, 1.0472, 23.7, 30.1)
        )
        expected_thigh_lower = -1.5708 if leg.startswith("F") else -0.5236
        expected_thigh_upper = 3.4907 if leg.startswith("F") else 4.5379
        assert (
            thigh.lower_rad,
            thigh.upper_rad,
            thigh.effort_nm,
            thigh.velocity_rad_s,
        ) == pytest.approx((expected_thigh_lower, expected_thigh_upper, 23.7, 30.1))
        assert (
            calf.lower_rad,
            calf.upper_rad,
            calf.effort_nm,
            calf.velocity_rad_s,
        ) == pytest.approx((-2.7227, -0.83776, 45.43, 15.70))


def test_hash_pin_rejects_a_single_byte_change(project_root: Path, tmp_path: Path) -> None:
    payload = _urdf_path(project_root).read_bytes()
    changed_payload = payload[:-1] + (b" " if payload[-1:] != b" " else b"\n")
    assert changed_payload != payload
    tampered = tmp_path / "tampered.urdf"
    tampered.write_bytes(changed_payload)

    with pytest.raises(Go2UrdfError, match="SHA-256 mismatch"):
        _load(tampered)


def test_expected_hash_must_have_exact_sha256_syntax(project_root: Path) -> None:
    with pytest.raises(Go2UrdfError, match="hexadecimal SHA-256"):
        _load(_urdf_path(project_root), sha256="not-a-digest")


def test_dtd_and_entity_declarations_are_rejected_before_xml_parse(
    project_root: Path,
    tmp_path: Path,
) -> None:
    payload = (
        b'<!DOCTYPE robot [<!ENTITY injected "unsafe">]>\n' + _urdf_path(project_root).read_bytes()
    )
    path = tmp_path / "entity.urdf"
    path.write_bytes(payload)

    with pytest.raises(Go2UrdfError, match="DTD/entity declarations are prohibited"):
        _load(path, sha256=_sha256(payload))


def test_missing_urdf_joint_and_incomplete_pose_fail_closed(
    project_root: Path,
    tmp_path: Path,
) -> None:
    robot = ET.fromstring(_urdf_path(project_root).read_bytes())
    removed = next(
        joint for joint in robot.findall("joint") if joint.attrib.get("name") == "FL_hip_joint"
    )
    robot.remove(removed)
    payload = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    path = tmp_path / "missing-joint.urdf"
    path.write_bytes(payload)
    with pytest.raises(Go2UrdfError, match="joint pose does not match URDF"):
        _load(path, sha256=_sha256(payload))

    incomplete_pose = _standard_standing_pose()
    incomplete_pose.pop("RR_calf_joint")
    with pytest.raises(Go2UrdfError, match=r"missing=\['RR_calf_joint'\]"):
        load_go2_urdf_mass_properties(
            _urdf_path(project_root),
            expected_sha256=VENDORED_SHA256,
            expected_robot_name="go2_description",
            root_link="base",
            joint_positions_rad=incomplete_pose,
        )


def test_pose_outside_vendor_joint_limit_is_rejected(project_root: Path) -> None:
    pose = _standard_standing_pose()
    pose["FL_calf_joint"] = 0.0

    with pytest.raises(Go2UrdfError, match="FL_calf_joint.*above its URDF limit"):
        load_go2_urdf_mass_properties(
            _urdf_path(project_root),
            expected_sha256=VENDORED_SHA256,
            expected_robot_name="go2_description",
            root_link="base",
            joint_positions_rad=pose,
        )


def test_physically_impossible_link_inertia_is_rejected(
    project_root: Path,
    tmp_path: Path,
) -> None:
    robot = ET.fromstring(_urdf_path(project_root).read_bytes())
    base_inertia = robot.find("./link[@name='base']/inertial/inertia")
    assert base_inertia is not None
    base_inertia.attrib.update({"ixx": "1", "iyy": "1", "izz": "3"})
    payload = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    path = tmp_path / "nonphysical-inertia.urdf"
    path.write_bytes(payload)

    with pytest.raises(Go2UrdfError, match="triangle inequality"):
        _load(path, sha256=_sha256(payload))


def test_point_mass_combination_uses_parallel_axis_theorem(project_root: Path) -> None:
    base = _load(_urdf_path(project_root))
    point_position = np.array([0.0, 0.0, 0.0972])
    combined = combine_with_point_mass(
        base,
        point_mass_kg=10.0,
        point_position_from_root_m=point_position,
    )

    expected_com = (base.mass_kg * base.com_from_root_m + 10.0 * point_position) / 26.087
    assert combined.mass_kg == pytest.approx(26.087)
    assert combined.com_from_root_m == pytest.approx(expected_com)
    assert np.all(np.linalg.eigvalsh(combined.inertia_about_com_root_axes_kg_m2) > 0.0)
