"""Regression tests for the hardware-isolated normal-only landing MPC."""

from __future__ import annotations

import ast
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

import aerogo2.landing.impact_aware.normal_only_mpc as normal_only_module
from aerogo2.landing.impact_aware.normal_only_mpc import (
    NormalOnlyMPCProblem,
    solve_normal_only_mpc,
)
from aerogo2.landing.impact_aware.preliminary import NormalOnlyVerticalState


def _problem(
    *,
    initial_height_m: float = 0.51,
    initial_velocity_m_per_s: float = -0.1,
    schedule: Optional[np.ndarray] = None,
) -> NormalOnlyMPCProblem:
    horizon = 3
    if schedule is None:
        schedule = np.array(
            [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=np.int8,
        )
    contact_allocation = np.zeros((horizon, 4), dtype=np.float64)
    impulse_allocation = np.zeros((horizon, 4), dtype=np.float64)
    touchdown = (1 - schedule[:-1]) * schedule[1:]
    for step in range(horizon):
        active_contacts = np.flatnonzero(schedule[step])
        if active_contacts.size:
            contact_allocation[step, active_contacts] = 1.0 / float(active_contacts.size)
        touchdown_feet = np.flatnonzero(touchdown[step])
        if touchdown_feet.size:
            impulse_allocation[step, touchdown_feet] = 1.0 / float(touchdown_feet.size)
    return NormalOnlyMPCProblem(
        initial_state=NormalOnlyVerticalState(
            height_world_m=initial_height_m,
            vertical_velocity_world_m_per_s=initial_velocity_m_per_s,
        ),
        dt_s=0.1,
        contact_schedule=schedule,
        leg_order=("FR", "FL", "RR", "RL"),
        rotor_order=("RR", "LF", "LR", "RF"),
        foot_heights_from_com_m=np.full((horizon + 1, 4), -0.5),
        ground_height_world_m=0.0,
        touchdown_position_tolerance_m=1.0e-5,
        minimum_downward_speed_m_per_s=0.05,
        mass_kg=10.0,
        gravity_m_per_s2=10.0,
        previous_rotor_forces_n=np.full(4, 25.0),
        rotor_force_min_n=np.zeros(4),
        rotor_force_max_n=np.full(4, 50.0),
        rotor_force_rate_max_n_per_s=np.full(4, 1000.0),
        contact_force_max_n=np.full(4, 100.0),
        normal_impulse_max_ns=np.full(4, 1.0),
        impact_duration_s=0.02,
        average_impact_force_max_n=np.full(4, 100.0),
        rotor_force_allocation=np.full((horizon, 4), 0.25),
        contact_force_allocation=contact_allocation,
        normal_impulse_allocation=impulse_allocation,
        reference_height_world_m=np.array([initial_height_m, 0.5, 0.5, 0.5]),
        reference_vertical_velocity_world_m_per_s=np.array(
            [initial_velocity_m_per_s, 0.0, 0.0, 0.0]
        ),
        reference_rotor_forces_n=np.array(
            [[25.0] * 4, [12.5] * 4, [12.5] * 4]
        ),
        reference_contact_forces_n=np.array(
            [[0.0] * 4, [12.5] * 4, [12.5] * 4]
        ),
        minimum_com_height_world_m=0.4,
        maximum_com_height_world_m=3.0,
        maximum_abs_vertical_velocity_m_per_s=3.0,
    )


def test_normal_only_mpc_solves_explicit_touchdown_without_hardware() -> None:
    result = solve_normal_only_mpc(_problem(), timeout_s=5.0)

    assert result.success
    assert result.status == "success"
    assert result.hardware_output_permitted is False
    assert result.leg_order == ("FR", "FL", "RR", "RL")
    assert result.max_equality_violation <= 1.0e-6
    assert result.min_inequality_residual >= -1.0e-6
    assert result.min_variable_bound_residual >= -1.0e-6
    assert result.first_rotor_forces_n is not None
    assert result.first_desired_contact_normal_forces_n is not None
    assert result.states[1].vertical_velocity_world_m_per_s == pytest.approx(0.0, abs=1.0e-6)
    assert np.sum(result.normal_impulses_ns[0]) == pytest.approx(1.0, abs=2.1e-3)
    for values, allocation in (
        (result.rotor_forces_n, result.rotor_force_allocation),
        (result.desired_contact_normal_forces_n, result.contact_force_allocation),
        (result.normal_impulses_ns, result.normal_impulse_allocation),
    ):
        assert values == pytest.approx(np.sum(values, axis=1)[:, None] * allocation)


def test_asymmetric_references_cannot_override_fixed_allocations() -> None:
    problem = _problem()
    result = solve_normal_only_mpc(
        replace(
            problem,
            reference_rotor_forces_n=np.array(
                [[49.0, 1.0, 1.0, 1.0], [1.0, 49.0, 1.0, 1.0], [1.0, 1.0, 49.0, 1.0]]
            ),
            reference_contact_forces_n=np.array(
                [[99.0, 0.0, 0.0, 0.0], [99.0, 1.0, 1.0, 1.0], [1.0, 99.0, 1.0, 1.0]]
            ),
        ),
        timeout_s=5.0,
    )

    assert result.success
    assert result.rotor_forces_n == pytest.approx(
        np.sum(result.rotor_forces_n, axis=1)[:, None] * problem.rotor_force_allocation,
        abs=1.0e-7,
    )
    assert result.desired_contact_normal_forces_n == pytest.approx(
        np.sum(result.desired_contact_normal_forces_n, axis=1)[:, None]
        * problem.contact_force_allocation,
        abs=1.0e-7,
    )
    assert result.normal_impulses_ns == pytest.approx(
        np.sum(result.normal_impulses_ns, axis=1)[:, None]
        * problem.normal_impulse_allocation,
        abs=1.0e-7,
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        (
            "rotor_force_allocation",
            np.array([[0.4, 0.2, 0.2, 0.1], [0.25] * 4, [0.25] * 4]),
            "sum to 1",
        ),
        (
            "contact_force_allocation",
            np.array([[0.25] * 4, [0.25] * 4, [0.25] * 4]),
            "inactive",
        ),
        (
            "normal_impulse_allocation",
            np.array([[0.25] * 4, [0.25] * 4, [0.0] * 4]),
            "inactive",
        ),
        (
            "rotor_force_allocation",
            np.array([[-0.1, 0.3, 0.4, 0.4], [0.25] * 4, [0.25] * 4]),
            "nonnegative",
        ),
    ),
)
def test_bad_fixed_allocations_fail_closed(
    field: str,
    bad_value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_problem(), **{field: bad_value})


def test_fixed_allocations_are_deeply_immutable() -> None:
    problem = _problem()
    for allocation in (
        problem.rotor_force_allocation,
        problem.contact_force_allocation,
        problem.normal_impulse_allocation,
    ):
        with pytest.raises(ValueError, match="read-only"):
            allocation[0, 0] = 1.0
        with pytest.raises(ValueError, match="WRITEABLE"):
            allocation.setflags(write=True)


def test_normal_only_landing_rejects_planned_contact_release() -> None:
    schedule = np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )
    with pytest.raises(ValueError, match="monotone"):
        _problem(schedule=schedule)


def test_normal_only_touchdown_cannot_occur_while_foot_is_in_midair() -> None:
    result = solve_normal_only_mpc(
        _problem(initial_height_m=2.0, initial_velocity_m_per_s=0.0),
        timeout_s=5.0,
    )

    assert not result.success
    assert result.first_rotor_forces_n is None
    assert result.first_desired_contact_normal_forces_n is None
    assert result.min_inequality_residual < 0.0 or result.max_equality_violation > 1.0e-6


def test_normal_only_problem_rejects_initial_ground_penetration() -> None:
    with pytest.raises(ValueError, match="penetrates"):
        _problem(initial_height_m=0.49)


def test_normal_only_problem_rejects_motion_after_a_foot_enters_stance() -> None:
    problem = _problem()
    heights = np.array(problem.foot_heights_from_com_m, copy=True)
    heights[2, 0] += 0.001

    with pytest.raises(ValueError, match="remain fixed"):
        replace(problem, foot_heights_from_com_m=heights)


def test_normal_only_problem_rejects_nonzero_velocity_when_already_in_contact() -> None:
    with pytest.raises(ValueError, match="zero vertical COM velocity"):
        _problem(
            initial_height_m=0.5,
            initial_velocity_m_per_s=0.1,
            schedule=np.ones((4, 4), dtype=np.int8),
        )


def test_normal_only_touchdown_enforces_configured_descent_guard() -> None:
    result = solve_normal_only_mpc(
        replace(_problem(), minimum_downward_speed_m_per_s=0.2),
        timeout_s=5.0,
    )

    assert not result.success
    assert result.min_inequality_residual < 0.0


def test_normal_only_timeout_includes_postsolve_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    original_inequality = normal_only_module._NormalOnlyNLP.inequality

    def delayed_inequality(
        self: normal_only_module._NormalOnlyNLP,
        decision: object,
    ) -> np.ndarray:
        time.sleep(0.02)
        return original_inequality(self, decision)

    def fake_minimize(*args: object, **kwargs: object) -> SimpleNamespace:
        del kwargs
        candidate = np.asarray(args[1], dtype=float)
        monkeypatch.setattr(
            normal_only_module._NormalOnlyNLP,
            "inequality",
            delayed_inequality,
        )
        return SimpleNamespace(x=candidate, success=True, message="simulated feasible return")

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
    result = solve_normal_only_mpc(_problem(), timeout_s=0.01)

    assert not result.success
    assert result.status == "timeout"
    assert result.solve_time_s >= 0.02


def test_normal_only_module_does_not_import_hardware_or_bridges(project_root: Path) -> None:
    source = (
        project_root
        / "src"
        / "aerogo2"
        / "landing"
        / "impact_aware"
        / "normal_only_mpc.py"
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
