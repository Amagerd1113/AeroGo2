"""Safe regeneration tests for the schema-4 preliminary physical prior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from aerogo2.landing.impact_aware.preliminary import (
    PreliminaryModelError,
    load_preliminary_landing_model,
    recompute_preliminary_derived_document,
)


def _fixture_config(project_root: Path, tmp_path: Path) -> Path:
    config_root = project_root / "configs"
    source = tmp_path / "preliminary-source.yaml"
    copyfile(config_root / "impact_aware_preliminary.yaml", source)
    copyfile(
        config_root / "go2_description.unitree_ros.urdf",
        tmp_path / "go2_description.unitree_ros.urdf",
    )
    copyfile(config_root / "UNITREE_ROS_LICENSE.txt", tmp_path / "UNITREE_ROS_LICENSE.txt")
    return source


def _read(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _run(project_root: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recompute_impact_aware_preliminary.py"),
            *(str(value) for value in arguments),
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )


def test_recompute_updates_all_derived_values_without_touching_source(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = _fixture_config(project_root, tmp_path)
    document = _read(source)
    document["mass"]["added_system_nominal_kg"] = 10.5
    document["geometry"]["frame_center_O_from_body_origin_B_m"] = [0.0, 0.0, 0.100]
    document["geometry"]["total_com_C_from_body_origin_B_m"] = [0.0, 0.0, 0.055]
    document["geometry"]["rotor_plane_from_frame_center_O_m"] = [0.0, 0.0, 0.025]
    document["geometry"]["horizontal_radius_from_frame_center_m"] = 0.700
    # Deliberately leave every redundant value stale.  The strict loader must
    # reject this source, while the recomputation tool may repair a candidate.
    source.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    source_before = source.read_bytes()
    with pytest.raises(PreliminaryModelError):
        load_preliminary_landing_model(source, allow_provisional=True)

    completed = _run(project_root, "--config", source)

    assert completed.returncode == 0, completed.stderr
    assert source.read_bytes() == source_before
    candidate = yaml.safe_load(completed.stdout)
    assert isinstance(candidate, dict)
    assert candidate["physical_use_prohibited"] is True
    assert candidate["allow_hardware_output"] is False
    assert candidate["parameters_identified"] is False
    assert candidate["mass"]["total_nominal_kg"] == pytest.approx(26.587)
    assert candidate["geometry"]["total_com_from_frame_center_body_m"] == pytest.approx(
        [0.0, 0.0, -0.045]
    )
    assert candidate["geometry"]["rotor_frame_com_from_body_origin_B_m"] == pytest.approx(
        [0.0, 0.0, 0.100]
    )
    assert candidate["geometry"]["rotor_plane_z_from_com_m"] == pytest.approx(0.070)
    arms = np.asarray(candidate["geometry"]["lever_arms_from_com_body_m"], dtype=float)
    assert np.linalg.norm(arms[:, :2], axis=1) == pytest.approx(np.full(4, 0.700))
    assert arms[:, 2] == pytest.approx(np.full(4, 0.070))
    assert candidate["offline_inertia_estimate"]["remaining_added_mass_kg"] == pytest.approx(6.12)
    assert (
        candidate["offline_inertia_estimate"][
            "remaining_added_mass_effective_com_from_body_origin_B_m"
        ]
        != document["offline_inertia_estimate"][
            "remaining_added_mass_effective_com_from_body_origin_B_m"
        ]
    )
    nominal = np.asarray(
        candidate["offline_inertia_estimate"]["nominal_body_kg_m2"],
        dtype=float,
    )
    lower = np.asarray(
        candidate["offline_inertia_estimate"]["diagonal_lower_body_kg_m2"],
        dtype=float,
    )
    upper = np.asarray(
        candidate["offline_inertia_estimate"]["diagonal_upper_body_kg_m2"],
        dtype=float,
    )
    assert np.all(lower < np.diag(nominal))
    assert np.all(np.diag(nominal) < upper)


def test_output_is_new_same_directory_and_strictly_self_checked(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = _fixture_config(project_root, tmp_path)
    source_before = source.read_bytes()
    destination = tmp_path / "preliminary-recomputed.yaml"

    completed = _run(project_root, "--config", source, "--output", destination)

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == destination
    assert source.read_bytes() == source_before
    checked = load_preliminary_landing_model(destination, allow_provisional=True)
    assert not checked.hardware_output_permitted
    assert checked.schema_version == 4

    second = _run(project_root, "--config", source, "--output", destination)
    assert second.returncode == 2
    assert "already exists" in second.stderr
    assert destination.exists()

    in_place = _run(project_root, "--config", source, "--output", source)
    assert in_place.returncode == 2
    assert "in-place" in in_place.stderr
    assert source.read_bytes() == source_before


def test_output_cannot_escape_source_directory(project_root: Path, tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    source = _fixture_config(project_root, source_directory)
    outside = tmp_path / "outside.yaml"

    completed = _run(project_root, "--config", source, "--output", outside)

    assert completed.returncode == 2
    assert "must be beside" in completed.stderr
    assert not outside.exists()


def test_duplicate_input_key_is_rejected_before_recomputation(
    project_root: Path,
    tmp_path: Path,
) -> None:
    source = _fixture_config(project_root, tmp_path)
    text = source.read_text(encoding="utf-8").replace(
        "schema_version: 4",
        "schema_version: 4\nschema_version: 4",
        1,
    )
    source.write_text(text, encoding="utf-8")

    with pytest.raises(PreliminaryModelError, match="duplicate key"):
        recompute_preliminary_derived_document(source)
    completed = _run(project_root, "--config", source)
    assert completed.returncode == 2
    assert "duplicate key" in completed.stderr
