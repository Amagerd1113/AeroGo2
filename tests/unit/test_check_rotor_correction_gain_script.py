"""CLI regressions for the offline rotor transport-boundary audit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest


def _run(project_root: Path, gain: str = "0.2") -> subprocess.CompletedProcess:
    environment: Dict[str, str] = dict(os.environ)
    source_path = str(project_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "check_rotor_correction_gain.py"),
            "--gain",
            gain,
            "--baseline",
            "4,4,4,4",
            "--transport-target",
            "6,6,6,6",
            "--thrust-min",
            "0,0,0,0",
            "--thrust-max",
            "10,10,10,10",
            "--max-correction",
            "5,5,5,5",
            "--gain-rise-per-s",
            "100",
            "--dt",
            "0.01",
        ],
        cwd=str(project_root),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10.0,
    )


def test_transport_audit_emits_applied_payload_without_double_scaling(
    project_root: Path,
) -> None:
    completed = _run(project_root)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["transport_raw_correction_n"] == pytest.approx([2.0] * 4)
    assert report["applied_residual_thrusts_n"] == pytest.approx([0.4] * 4)
    assert report["applied_total_thrusts_n"] == pytest.approx([4.4] * 4)
    assert report["applied_gain"] == pytest.approx(0.2)


def test_zero_gain_transport_audit_does_not_invent_raw_target(project_root: Path) -> None:
    completed = _run(project_root, gain="0")
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["transport_target_thrusts_n"] is None
    assert report["transport_raw_correction_n"] is None
    assert report["applied_residual_thrusts_n"] == pytest.approx([0.0] * 4)
    assert report["applied_total_thrusts_n"] == pytest.approx([4.0] * 4)
