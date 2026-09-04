"""The offline parameter report must stay non-authoritative and reproducible."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_parameter_report_exposes_estimates_without_hardware_authority(
    project_root: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "report_impact_aware_parameters.py"),
            "--config",
            str(project_root / "configs" / "impact_aware_preliminary.yaml"),
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    report = json.loads(completed.stdout)

    assert report["hardware_output_permitted"] is False
    assert report["mass_kg"]["go2_urdf"] == pytest.approx(16.087)
    assert report["mass_kg"]["total_provisional"] == pytest.approx(26.087)
    assert report["reference_points"]["frame_center_O_from_body_B_m"] == pytest.approx(
        [0.0, 0.0, 0.0972]
    )
    assert report["reference_points"]["total_com_C_from_body_B_m"] == pytest.approx(
        [0.0, 0.0, 0.05]
    )
    assert report["rotors"]["lever_arms_from_total_com_C_body_m"][1] == pytest.approx(
        [0.470226009, 0.470226009, 0.0672]
    )
    assert report["offline_inertia_estimate"]["usable_for_hardware"] is False
    assert report["go2_urdf"]["kinematics_offline_prior_only"] is True
    assert report["go2_urdf"]["kinematics_hardware_validated"] is False
    assert report["go2_urdf"]["sdk_leg_order"] == ["FR", "FL", "RR", "RL"]
    assert report["go2_urdf"]["legs"]["FR"]["joint_origins_from_parent_m"] == [
        [0.1934, -0.0465, 0.0],
        [0.0, -0.0955, 0.0],
        [0.0, 0.0, -0.213],
    ]
    assert report["go2_urdf"]["legs"]["FR"]["foot_origin_from_calf_m"] == pytest.approx(
        [0.0, 0.0, -0.213]
    )
    assert report["offline_inertia_estimate"][
        "remaining_added_mass_effective_com_from_body_B_m"
    ] == pytest.approx([0.004847170654, 0.0, 0.191098213990])
    assert report["offline_inertia_estimate"]["diagonal_lower_kg_m2"] == pytest.approx(
        [1.339191385297, 1.675052659538, 2.499249435208]
    )
    assert report["must_confirm_before_hardware"]
