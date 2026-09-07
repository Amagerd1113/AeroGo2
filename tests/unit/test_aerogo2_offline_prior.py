"""Regression tests for the hardware-prohibited AeroGo2 full-NLP prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerogo2.landing.impact_aware.aerogo2_offline import (
    build_aerogo2_offline_prior_bundle,
    solve_aerogo2_offline_hover,
)
from aerogo2.landing.impact_aware.preliminary import PreliminaryModelError


def _paths(project_root: Path) -> tuple[Path, Path]:
    return (
        project_root / "configs" / "impact_aware_preliminary.yaml",
        project_root / "configs" / "impact_aware_mpc_demo.yaml",
    )


def test_aerogo2_offline_prior_requires_opt_in_and_rejects_hardware(
    project_root: Path,
) -> None:
    preliminary, fixture = _paths(project_root)
    with pytest.raises(PreliminaryModelError, match="allow_provisional=True"):
        build_aerogo2_offline_prior_bundle(preliminary, fixture)
    with pytest.raises(PreliminaryModelError, match="prohibited for hardware"):
        build_aerogo2_offline_prior_bundle(
            preliminary,
            fixture,
            allow_provisional=True,
            for_hardware=True,
        )


def test_aerogo2_offline_prior_uses_physical_mass_inertia_and_geometry(
    project_root: Path,
) -> None:
    merged = build_aerogo2_offline_prior_bundle(
        *_paths(project_root),
        allow_provisional=True,
    )
    controller = merged.controller

    assert not merged.hardware_output_permitted
    assert controller.is_synthetic
    assert not controller.hardware_output_permitted
    assert controller.physical_use_prohibited
    assert not controller.allow_hardware_output
    assert controller.profile == "aerogo2_provisional_offline_hybrid"
    assert controller.dynamics.mass_kg == pytest.approx(26.087)
    assert controller.dynamics.inertia_body_kg_m2 == pytest.approx(
        merged.preliminary.offline_inertia_estimate.nominal_body_kg_m2  # type: ignore[union-attr]
    )
    assert controller.rotor_geometry.lever_arms_from_com_body_m == pytest.approx(
        merged.preliminary.geometry.lever_arms_from_com_body_m
    )
    assert controller.rotor_actuator.thrust_max_n == pytest.approx(np.full(4, 15006.0 * 9.80665e-3))
    assert controller.rotor_correction_safety.target_gain == 0.0
    assert np.array_equal(controller.contact_force_limits.friction_coefficients, np.zeros(4))
    assert controller.go2_foot_force_calibration is None
    assert merged.numerical_only_fields


def test_aerogo2_offline_prior_completes_one_full_hover_nlp(project_root: Path) -> None:
    merged = build_aerogo2_offline_prior_bundle(
        *_paths(project_root),
        allow_provisional=True,
    )

    result = solve_aerogo2_offline_hover(merged)

    assert result.success, result.message
    assert result.max_equality_violation <= merged.controller.solver_settings.constraint_tolerance
    assert result.min_inequality_residual >= -merged.controller.solver_settings.constraint_tolerance
