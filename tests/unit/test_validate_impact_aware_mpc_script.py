from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml


def _run_validator(
    project_root: Path,
    *arguments: str,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    environment: Dict[str, str] = dict(os.environ)
    source_path = str(project_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_impact_aware_mpc.py"),
            *arguments,
        ],
        cwd=str(project_root if cwd is None else cwd),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15.0,
    )


def test_default_validator_outputs_passing_synthetic_json(
    project_root: Path, tmp_path: Path
) -> None:
    completed = _run_validator(project_root, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["overall_pass"] is True
    assert report["synthetic"] is True
    assert report["paper_experiment_reproduction"] is False
    assert report["hardware_output_permitted"] is False
    assert "not a reproduction" in report["disclaimer"]
    required = {
        "config_load",
        "synthetic_profile_safety",
        "strict_configuration_assembly",
        "parameter_validation",
        "static_hover_allocation_and_dynamics",
        "nlp_slsqp",
        "touchdown_nlp_slsqp",
        "impact_momentum_reset_and_sticking",
        "rotor_residual_gain_endpoints",
        "contact_detection_and_admittance_execution",
    }
    assert required == set(report["checks"])
    assert all(report["checks"][name]["pass"] is True for name in required)


def test_output_option_writes_report_instead_of_stdout(project_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "impact_report.json"

    completed = _run_validator(project_root, "--output", str(output))

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_pass"] is True
    assert report["report_type"] == "aerogo2_impact_aware_mpc_synthetic_dry_run"


def test_output_cannot_overwrite_input_config(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "configs" / "impact_aware_mpc_demo.yaml"
    config = tmp_path / "protected.yaml"
    original = source.read_text(encoding="utf-8")
    config.write_text(original, encoding="utf-8")

    completed = _run_validator(
        project_root,
        "--config",
        str(config),
        "--output",
        str(config),
    )

    assert completed.returncode == 2
    assert "must not refer" in completed.stderr
    assert config.read_text(encoding="utf-8") == original


def test_unsafe_profile_fails_closed_with_nonzero_exit(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "configs" / "impact_aware_mpc_demo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["allow_hardware_output"] = True
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = _run_validator(project_root, "--config", str(unsafe))

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["overall_pass"] is False
    assert report["checks"]["synthetic_profile_safety"]["pass"] is False
    assert "allow_hardware_output" in " ".join(
        report["checks"]["synthetic_profile_safety"]["failures"]
    )


def test_validator_has_no_hardware_or_bridge_imports(project_root: Path) -> None:
    script = project_root / "scripts" / "validate_impact_aware_mpc.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    forbidden_prefixes = (
        "aerogo2.hardware",
        "aerogo2.bridges",
        "pymavlink",
        "serial",
        "unitree",
    )
    assert not any(module.startswith(forbidden_prefixes) for module in imported_modules), (
        imported_modules
    )
