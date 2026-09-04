"""AeroGo2 physical-prior regression for the one-dimensional touchdown path."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from aerogo2.landing.impact_aware.aerogo2_normal_only import (
    build_aerogo2_normal_only_landing_fixture,
    solve_aerogo2_normal_only_landing,
)
from aerogo2.landing.impact_aware.aerogo2_offline import (
    AeroGo2OfflinePriorBundle,
    build_aerogo2_offline_prior_bundle,
)


def _bundle(project_root: Path) -> AeroGo2OfflinePriorBundle:
    return build_aerogo2_offline_prior_bundle(
        project_root / "configs" / "impact_aware_preliminary.yaml",
        project_root / "configs" / "impact_aware_mpc_demo.yaml",
        allow_provisional=True,
    )


def test_current_aerogo2_prior_solves_normal_only_touchdown(project_root: Path) -> None:
    fixture = build_aerogo2_normal_only_landing_fixture(_bundle(project_root))
    result = solve_aerogo2_normal_only_landing(fixture)

    assert fixture.hardware_output_permitted is False
    assert result.hardware_output_permitted is False
    assert fixture.foot_lever_arms_from_com.leg_order == ("FR", "FL", "RR", "RL")
    assert fixture.foot_lever_arms_from_com.values_m[:, 2] == pytest.approx(
        np.full(4, -0.314805846483303),
        abs=1.0e-12,
    )
    assert fixture.problem.rotor_force_allocation == pytest.approx(
        np.full_like(fixture.problem.rotor_force_allocation, 0.25)
    )
    assert fixture.problem.contact_force_allocation[0] == pytest.approx(np.zeros(4))
    assert fixture.problem.contact_force_allocation[1:] == pytest.approx(0.25)
    assert fixture.problem.normal_impulse_allocation[0] == pytest.approx(0.25)
    assert fixture.problem.normal_impulse_allocation[1:] == pytest.approx(0.0)
    assert result.success
    touchdown_clearance = (
        result.states[1].height_world_m - fixture.touchdown_com_height_world_m
    )
    assert touchdown_clearance >= -1.0e-9
    assert touchdown_clearance <= fixture.problem.touchdown_position_tolerance_m + 1.0e-9
    assert result.states[1].vertical_velocity_world_m_per_s == pytest.approx(0.0, abs=1.0e-6)
    assert np.all(result.normal_impulses_ns[0] > 0.0)


def test_aerogo2_normal_only_builder_rejects_nonpositive_descent(project_root: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_aerogo2_normal_only_landing_fixture(
            _bundle(project_root),
            descent_speed_m_per_s=0.0,
        )


def test_aerogo2_normal_only_module_is_hardware_isolated(project_root: Path) -> None:
    source = (
        project_root
        / "src"
        / "aerogo2"
        / "landing"
        / "impact_aware"
        / "aerogo2_normal_only.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        name.startswith(("aerogo2.hardware", "aerogo2.bridges", "unitree", "pymavlink"))
        for name in imported
    )
