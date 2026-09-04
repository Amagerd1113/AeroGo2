"""Offline synthetic validation for the impact-aware landing mathematics.

This script never imports AeroGo2 hardware/bridge modules and never transmits
commands.  Its synthetic checks are software verification only; they are not
paper-experiment reproduction or evidence of physical-system performance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from aerogo2.landing.impact_aware.admittance import LegAdmittanceController
from aerogo2.landing.impact_aware.config import (
    ImpactAwareConfigBundle,
    load_impact_aware_config,
)
from aerogo2.landing.impact_aware.dynamics import reduced_continuous_dynamics
from aerogo2.landing.impact_aware.impact import (
    evaluate_impulse_constraints,
    foot_post_impact_velocity,
    momentum_reset,
)
from aerogo2.landing.impact_aware.nlp import (
    CONTROL_DIM,
    STATE_DIM,
    ContactForceLimits,
    ImpactAwareMPCProblem,
    ImpactEvent,
    LandingContactGeometry,
    MPCReferences,
    MPCWeights,
    RotorExecutionPlan,
    SLSQPSettings,
    StateBounds,
    reconstruct_transport_target,
    solve_impact_aware_mpc,
)
from aerogo2.landing.impact_aware.rotor import (
    build_fixed_deployed_allocation_matrix,
    evaluate_rotor_constraints,
    first_order_thrust_rate,
    thrust_and_reaction_torque,
)
from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FixedDeployedRotorGeometry,
    FloatArray,
    FootLeverArmsFromComBody,
    FootLeverArmsFromComBodyHorizon,
    ImpactLimits,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
    RotorAerodynamics,
)

JSONDict = Dict[str, Any]

REPORT_TYPE = "aerogo2_impact_aware_mpc_synthetic_dry_run"
DISCLAIMER = (
    "Synthetic offline software verification only. This report is not a reproduction "
    "of the paper experiments, does not validate identified AeroGo2 parameters, and "
    "must not be used as evidence of hardware safety or physical performance."
)


@dataclass(frozen=True)
class _ValidationContext:
    geometry: FixedDeployedRotorGeometry
    aerodynamics: RotorAerodynamics
    actuator: RotorActuatorConfig
    dynamics: ReducedDynamicsConfig
    contact_limits: ContactForceLimits
    impact_limits: ImpactLimits
    correction_safety: RotorCorrectionSafetyConfig
    horizon: int
    dt_s: float
    solver: SLSQPSettings
    weights: MPCWeights
    state_lower: FloatArray
    state_upper: FloatArray
    state_soft_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class _HoverFixture:
    thrusts_n: FloatArray
    state: ReducedState
    control: ReducedInput
    report: JSONDict


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _section(root: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current = root
    traversed = []
    for key in path:
        traversed.append(key)
        if key not in current:
            raise KeyError("missing configuration key: {}".format(".".join(traversed)))
        current = _mapping(current[key], ".".join(traversed))
    return current


def _value(root: Mapping[str, Any], *path: str) -> object:
    if not path:
        raise ValueError("configuration path cannot be empty")
    parent = _section(root, *path[:-1]) if len(path) > 1 else root
    key = path[-1]
    if key not in parent:
        raise KeyError("missing configuration key: {}".format(".".join(path)))
    return parent[key]


def _float_value(root: Mapping[str, Any], *path: str, positive: bool = False) -> float:
    raw = _value(root, *path)
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise TypeError("{} must be a real scalar".format(".".join(path)))
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(".".join(path)))
    if positive and result <= 0.0:
        raise ValueError("{} must be positive".format(".".join(path)))
    return result


def _int_value(root: Mapping[str, Any], *path: str) -> int:
    raw = _value(root, *path)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError("{} must be an integer".format(".".join(path)))
    if raw < 1:
        raise ValueError("{} must be at least 1".format(".".join(path)))
    return raw


def _bool_value(root: Mapping[str, Any], *path: str) -> bool:
    raw = _value(root, *path)
    if not isinstance(raw, bool):
        raise TypeError("{} must be a boolean".format(".".join(path)))
    return raw


def _float_array(
    root: Mapping[str, Any], path: Tuple[str, ...], shape: Tuple[int, ...]
) -> FloatArray:
    raw = np.asarray(_value(root, *path))
    if raw.dtype.kind not in "fiu":
        raise TypeError("{} must contain real values".format(".".join(path)))
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(".".join(path), shape, result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must contain only finite values".format(".".join(path)))
    return cast(FloatArray, result)


def _bool_array(
    root: Mapping[str, Any], path: Tuple[str, ...], shape: Tuple[int, ...]
) -> NDArray[np.bool_]:
    raw = np.asarray(_value(root, *path))
    if raw.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(".".join(path), shape, raw.shape))
    if raw.dtype.kind != "b":
        raise TypeError("{} must contain booleans".format(".".join(path)))
    return np.array(raw, dtype=bool, copy=True)


def _diagonal(root: Mapping[str, Any], path: Tuple[str, ...], size: int) -> FloatArray:
    return cast(FloatArray, np.diag(_float_array(root, path, (size,))))


def _load_config(path: Path) -> Mapping[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(loaded, "configuration root")


def _safe_profile_check(config: Mapping[str, Any]) -> JSONDict:
    expected = {
        "schema_version": 3,
        "parameters_identified": False,
        "physical_use_prohibited": True,
        "allow_hardware_output": False,
    }
    actual: Dict[str, object] = {}
    failures = []
    for key, expected_value in expected.items():
        try:
            value = _value(config, key)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
            continue
        actual[key] = value
        if type(value) is not type(expected_value) or value != expected_value:
            failures.append(f"{key} must be {expected_value!r} for synthetic validation")
    profile = _value(config, "profile") if "profile" in config else None
    actual["profile"] = profile
    if profile != "synthetic_demo":
        failures.append("profile must be 'synthetic_demo'")
    try:
        world_frame = _value(config, "frames", "world")
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(str(exc))
        world_frame = None
    actual["world_frame"] = world_frame
    if world_frame != "ENU_Z_UP":
        failures.append("frames.world must be ENU_Z_UP for horizontal z-normal checks")
    return {
        "pass": not failures,
        "values": actual,
        "failures": failures,
    }


def _build_context(config: Mapping[str, Any]) -> _ValidationContext:
    configuration = _value(config, "rotor", "geometry", "configuration")
    if configuration != "FIXED_DEPLOYED_LOCKED":
        raise ValueError("rotor.geometry.configuration must be FIXED_DEPLOYED_LOCKED")
    reference_origin = _value(config, "rotor", "geometry", "reference_origin")
    if reference_origin != "TOTAL_SYSTEM_COM_C":
        raise ValueError("rotor.geometry.reference_origin must be TOTAL_SYSTEM_COM_C")
    geometry = FixedDeployedRotorGeometry(
        lever_arms_from_com_body_m=_float_array(
            config,
            ("rotor", "geometry", "lever_arms_from_com_body_m"),
            (4, 3),
        ),
        thrust_directions_body=_float_array(
            config, ("rotor", "geometry", "thrust_directions_body"), (4, 3)
        ),
    )
    aerodynamics = RotorAerodynamics(
        thrust_coefficient_n_per_rad_s_squared=_float_value(
            config,
            "rotor",
            "aerodynamics",
            "thrust_coefficient_n_per_rad_s_squared",
            positive=True,
        ),
        drag_torque_coefficient_nm_per_rad_s_squared=_float_value(
            config,
            "rotor",
            "aerodynamics",
            "drag_torque_coefficient_nm_per_rad_s_squared",
        ),
        spin_directions=_float_array(config, ("rotor", "aerodynamics", "spin_directions"), (4,)),
    )
    actuator = RotorActuatorConfig(
        time_constants_s=_float_array(config, ("rotor", "actuator", "time_constants_s"), (4,)),
        thrust_min_n=_float_array(config, ("rotor", "actuator", "thrust_min_n"), (4,)),
        thrust_max_n=_float_array(config, ("rotor", "actuator", "thrust_max_n"), (4,)),
        thrust_rate_min_n_per_s=_float_array(
            config, ("rotor", "actuator", "thrust_rate_min_n_per_s"), (4,)
        ),
        thrust_rate_max_n_per_s=_float_array(
            config, ("rotor", "actuator", "thrust_rate_max_n_per_s"), (4,)
        ),
    )
    allocation = build_fixed_deployed_allocation_matrix(geometry, aerodynamics)
    dynamics = ReducedDynamicsConfig(
        mass_kg=_float_value(config, "dynamics", "mass_kg", positive=True),
        inertia_body_kg_m2=_float_array(config, ("dynamics", "inertia_body_kg_m2"), (3, 3)),
        gravity_world_m_per_s2=_float_array(config, ("dynamics", "gravity_world_m_per_s2"), (3,)),
        rotor_allocation_body=allocation,
    )
    contact_friction = _float_array(config, ("contact", "friction_coefficients"), (4,))
    contact_limits = ContactForceLimits(
        friction_coefficients=contact_friction,
        maximum_normal_force_n=_float_array(config, ("contact", "normal_force_max_n"), (4,)),
    )
    impact_limits = ImpactLimits(
        friction_coefficients=contact_friction,
        maximum_normal_impulse_ns=_float_value(config, "impact", "maximum_normal_impulse_ns"),
        impact_duration_s=_float_value(config, "impact", "impact_duration_s", positive=True),
        maximum_average_normal_force_n=_float_value(
            config, "impact", "maximum_average_normal_force_n"
        ),
    )
    correction_safety = RotorCorrectionSafetyConfig(
        target_gain=_float_value(config, "rotor", "correction_safety", "target_gain"),
        thrust_min_n=actuator.thrust_min_n,
        thrust_max_n=actuator.thrust_max_n,
        maximum_correction_n=_float_array(
            config, ("rotor", "correction_safety", "maximum_correction_n"), (4,)
        ),
        maximum_gain_rise_per_s=_float_value(
            config,
            "rotor",
            "correction_safety",
            "maximum_gain_rise_per_s",
            positive=True,
        ),
    )
    horizon = _int_value(config, "mpc", "horizon_steps")
    solver = SLSQPSettings(
        max_iterations=_int_value(config, "mpc", "solver", "max_iterations"),
        ftol=_float_value(config, "mpc", "solver", "ftol", positive=True),
        constraint_tolerance=_float_value(config, "mpc", "solver", "constraint_tolerance"),
        timeout_s=_float_value(config, "mpc", "solver", "timeout_s", positive=True),
        display=_bool_value(config, "mpc", "solver", "display"),
    )
    weights = MPCWeights(
        tracking=_diagonal(config, ("mpc", "weights", "tracking"), 12),
        input=_diagonal(config, ("mpc", "weights", "input"), CONTROL_DIM),
        input_rate=_diagonal(config, ("mpc", "weights", "input_rate"), CONTROL_DIM),
        slack=_diagonal(config, ("mpc", "weights", "slack"), STATE_DIM),
        terminal_tracking=_diagonal(config, ("mpc", "weights", "terminal_tracking"), 12),
        impulse=_diagonal(config, ("mpc", "weights", "impulse"), 3),
        touchdown_velocity=_diagonal(config, ("mpc", "weights", "touchdown_velocity"), 3),
    )
    return _ValidationContext(
        geometry=geometry,
        aerodynamics=aerodynamics,
        actuator=actuator,
        dynamics=dynamics,
        contact_limits=contact_limits,
        impact_limits=impact_limits,
        correction_safety=correction_safety,
        horizon=horizon,
        dt_s=_float_value(config, "mpc", "dt_s", positive=True),
        solver=solver,
        weights=weights,
        state_lower=_float_array(config, ("mpc", "state_bounds", "lower"), (STATE_DIM,)),
        state_upper=_float_array(config, ("mpc", "state_bounds", "upper"), (STATE_DIM,)),
        state_soft_mask=_bool_array(config, ("mpc", "state_bounds", "soft_mask"), (STATE_DIM,)),
    )


def _hover_fixture(context: _ValidationContext) -> _HoverFixture:
    desired_wrench = np.concatenate(
        (-context.dynamics.mass_kg * context.dynamics.gravity_world_m_per_s2, np.zeros(3))
    )
    thrusts, _, _, _ = np.linalg.lstsq(
        context.dynamics.rotor_allocation_body, desired_wrench, rcond=None
    )
    thrusts = cast(FloatArray, np.asarray(thrusts, dtype=float))
    state = ReducedState(
        position_world_m=np.zeros(3),
        linear_velocity_world_m_per_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=thrusts,
    )
    control = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=thrusts,
    )
    derivative = reduced_continuous_dynamics(
        state,
        control,
        np.zeros(4, dtype=int),
        FootLeverArmsFromComBody(
            values_m=np.zeros((4, 3)),
            leg_order=GO2_SDK_LEG_ORDER,
        ),
        context.dynamics,
        context.actuator,
        contact_force_leg_order=GO2_SDK_LEG_ORDER,
    )
    rates = first_order_thrust_rate(
        state.rotor_thrusts_n, control.rotor_thrust_commands_n, context.actuator
    )
    residuals = evaluate_rotor_constraints(
        state.rotor_thrusts_n,
        rates,
        control.rotor_thrust_commands_n,
        context.actuator,
    )
    angular_speeds = np.sqrt(thrusts / context.aerodynamics.thrust_coefficient_n_per_rad_s_squared)
    thrusts_from_speed, reaction_torques = thrust_and_reaction_torque(
        angular_speeds,
        context.aerodynamics,
    )
    wrench_error = context.dynamics.rotor_allocation_body @ thrusts - desired_wrench
    minimum_margin = min(
        float(np.min(residuals.thrust_lower_margin_n)),
        float(np.min(residuals.thrust_upper_margin_n)),
        float(np.min(residuals.thrust_rate_lower_margin_n_per_s)),
        float(np.min(residuals.thrust_rate_upper_margin_n_per_s)),
        float(np.min(residuals.command_lower_margin_n)),
        float(np.min(residuals.command_upper_margin_n)),
    )
    tolerance = context.solver.constraint_tolerance
    metrics = {
        "wrench_error_norm": float(np.linalg.norm(wrench_error)),
        "linear_acceleration_norm_m_per_s2": float(
            np.linalg.norm(derivative.linear_acceleration_world_m_per_s2)
        ),
        "angular_acceleration_norm_rad_per_s2": float(
            np.linalg.norm(derivative.angular_acceleration_body_rad_per_s2)
        ),
        "thrust_rate_norm_n_per_s": float(np.linalg.norm(derivative.rotor_thrust_rates_n_per_s)),
        "minimum_rotor_constraint_margin": minimum_margin,
        "hover_thrusts_n": thrusts.tolist(),
        "thrust_law_error_norm_n": float(np.linalg.norm(thrusts_from_speed - thrusts)),
        "reaction_torques_nm": reaction_torques.tolist(),
    }
    passed = (
        metrics["wrench_error_norm"] <= tolerance
        and metrics["linear_acceleration_norm_m_per_s2"] <= tolerance
        and metrics["angular_acceleration_norm_rad_per_s2"] <= tolerance
        and metrics["thrust_rate_norm_n_per_s"] <= tolerance
        and metrics["thrust_law_error_norm_n"] <= tolerance
        and minimum_margin >= -tolerance
    )
    return _HoverFixture(
        thrusts_n=thrusts,
        state=state,
        control=control,
        report={
            "pass": bool(passed),
            "tolerance": tolerance,
            "metrics": metrics,
        },
    )


def _validate_nlp(context: _ValidationContext, hover: _HoverFixture) -> JSONDict:
    horizon = context.horizon
    references = MPCReferences(
        position_world_m=np.zeros((horizon + 1, 3)),
        linear_velocity_world_m_per_s=np.zeros((horizon + 1, 3)),
        rotation_body_to_world=np.repeat(np.eye(3)[None, :, :], horizon + 1, axis=0),
        angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
        contact_forces_world_n=np.zeros((horizon, 4, 3)),
        rotor_thrust_commands_n=np.repeat(hover.thrusts_n[None, :], horizon, axis=0),
    )
    problem = ImpactAwareMPCProblem(
        initial_state=hover.state,
        previous_input=hover.control,
        dt_s=context.dt_s,
        contact_schedule=np.zeros((horizon + 1, 4), dtype=int),
        foot_leg_order=GO2_SDK_LEG_ORDER,
        # Foot terms are exactly masked by the known no-contact schedule in
        # this synthetic hover case; zeros make that irrelevance explicit.
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            values_m=np.zeros((horizon + 1, 4, 3)),
            leg_order=GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=references,
        state_bounds=StateBounds(
            lower=np.repeat(context.state_lower[None, :], horizon + 1, axis=0),
            upper=np.repeat(context.state_upper[None, :], horizon + 1, axis=0),
            soft_mask=np.repeat(context.state_soft_mask[None, :], horizon, axis=0),
        ),
        contact_limits=context.contact_limits,
        impact_events=(),
        dynamics_config=context.dynamics,
        rotor_actuator_config=context.actuator,
        weights=context.weights,
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=-1.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=math.radians(30.0),
        ),
    )
    result = solve_impact_aware_mpc(problem, context.solver)
    return {
        "pass": result.success,
        "status": result.status,
        "message": result.message,
        "objective": result.objective,
        "iterations": result.iterations,
        "solve_time_s": result.solve_time_s,
        "max_equality_violation": result.max_equality_violation,
        "min_inequality_residual": result.min_inequality_residual,
        "horizon_steps": horizon,
    }


def _validate_touchdown_nlp(
    context: _ValidationContext,
    hover: _HoverFixture,
) -> JSONDict:
    """Solve one full flight-to-contact reset, not only standalone formulas."""

    horizon = 1
    feasible_normal_impulse = 0.5 * min(
        context.impact_limits.maximum_normal_impulse_ns,
        context.impact_limits.maximum_average_normal_force_n
        * context.impact_limits.impact_duration_s,
    )
    if feasible_normal_impulse <= 0.0:
        raise ValueError("impact limits do not permit a positive touchdown impulse")
    preimpact_speed_m_per_s = feasible_normal_impulse / context.dynamics.mass_kg
    initial = ReducedState(
        # Hover thrust cancels gravity in this synthetic case.  Starting one
        # Euler step above the plane therefore places the touchdown foot on it.
        position_world_m=np.array([0.0, 0.0, context.dt_s * preimpact_speed_m_per_s]),
        linear_velocity_world_m_per_s=np.array(
            [0.0, 0.0, -preimpact_speed_m_per_s]
        ),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=hover.thrusts_n,
    )
    event = ImpactEvent(
        step=1,
        touchdown=np.array([1, 0, 0, 0]),
        participation=np.array([1, 0, 0, 0]),
        post_impact_joint_velocities_rad_per_s=np.zeros((4, 3)),
        impulse_limits=context.impact_limits,
    )
    zero_weights = MPCWeights(
        tracking=np.zeros((12, 12)),
        input=np.zeros((CONTROL_DIM, CONTROL_DIM)),
        input_rate=np.zeros((CONTROL_DIM, CONTROL_DIM)),
        slack=np.zeros((STATE_DIM, STATE_DIM)),
        terminal_tracking=np.zeros((12, 12)),
        impulse=np.zeros((3, 3)),
        touchdown_velocity=np.zeros((3, 3)),
    )
    problem = ImpactAwareMPCProblem(
        initial_state=initial,
        previous_input=hover.control,
        dt_s=context.dt_s,
        contact_schedule=np.array([[0, 0, 0, 0], [1, 0, 0, 0]]),
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            values_m=np.zeros((horizon + 1, 4, 3)),
            leg_order=GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=MPCReferences(
            position_world_m=np.zeros((horizon + 1, 3)),
            linear_velocity_world_m_per_s=np.zeros((horizon + 1, 3)),
            rotation_body_to_world=np.repeat(np.eye(3)[None, :, :], horizon + 1, axis=0),
            angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
            contact_forces_world_n=np.zeros((horizon, 4, 3)),
            rotor_thrust_commands_n=np.repeat(hover.thrusts_n[None, :], horizon, axis=0),
        ),
        state_bounds=StateBounds(
            lower=np.full((horizon + 1, STATE_DIM), -np.inf),
            upper=np.full((horizon + 1, STATE_DIM), np.inf),
            soft_mask=np.zeros((horizon, STATE_DIM), dtype=bool),
        ),
        contact_limits=context.contact_limits,
        impact_events=(event,),
        dynamics_config=context.dynamics,
        rotor_actuator_config=context.actuator,
        weights=zero_weights,
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=0.0,
            touchdown_position_tolerance_m=1.0e-8,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=math.radians(30.0),
        ),
    )
    result = solve_impact_aware_mpc(problem, context.solver)
    if not result.states:
        raise ValueError("touchdown solve returned no state trajectory")
    sticking = foot_post_impact_velocity(
        result.states[1],
        problem.foot_lever_arms_from_com_body_m.at_step(1),
        problem.leg_jacobians_body[1],
        event.post_impact_joint_velocities_rad_per_s,
        leg_order=problem.foot_leg_order,
    )[0]
    impulse = result.impulses_by_step.get(1, np.zeros((4, 3)))[0]
    return {
        "pass": bool(result.success),
        "status": result.status,
        "message": result.message,
        "iterations": result.iterations,
        "solve_time_s": result.solve_time_s,
        "max_equality_violation": result.max_equality_violation,
        "min_inequality_residual": result.min_inequality_residual,
        "postimpact_sticking_velocity_norm_m_per_s": float(np.linalg.norm(sticking)),
        "solved_touchdown_impulse_ns": np.asarray(impulse).tolist(),
        "preimpact_vertical_velocity_m_per_s": float(initial.linear_velocity_world_m_per_s[2]),
    }


def _validate_impact(context: _ValidationContext, hover: _HoverFixture) -> JSONDict:
    feasible_normal_impulse = min(
        context.impact_limits.maximum_normal_impulse_ns,
        context.impact_limits.maximum_average_normal_force_n
        * context.impact_limits.impact_duration_s,
    )
    if feasible_normal_impulse <= 0.0:
        raise ValueError("impact limits do not permit a nonzero synthetic stopping impulse")
    pre_impact = ReducedState(
        position_world_m=np.zeros(3),
        linear_velocity_world_m_per_s=np.array(
            [0.0, 0.0, -feasible_normal_impulse / context.dynamics.mass_kg]
        ),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=hover.thrusts_n,
    )
    impulses = np.zeros((4, 3))
    impulses[0, 2] = feasible_normal_impulse
    participation = np.array([1, 0, 0, 0])
    post_impact = momentum_reset(
        pre_impact,
        impulses,
        participation,
        FootLeverArmsFromComBody(
            values_m=np.zeros((4, 3)),
            leg_order=GO2_SDK_LEG_ORDER,
        ),
        context.dynamics,
        impulse_leg_order=GO2_SDK_LEG_ORDER,
    )
    foot_velocities = foot_post_impact_velocity(
        post_impact,
        FootLeverArmsFromComBody(
            values_m=np.zeros((4, 3)),
            leg_order=GO2_SDK_LEG_ORDER,
        ),
        np.zeros((4, 3, 3)),
        np.zeros((4, 3)),
        leg_order=GO2_SDK_LEG_ORDER,
    )
    impulse_residuals = evaluate_impulse_constraints(impulses, participation, context.impact_limits)
    linear_momentum_error = context.dynamics.mass_kg * (
        post_impact.linear_velocity_world_m_per_s - pre_impact.linear_velocity_world_m_per_s
    ) - np.sum(impulses * participation[:, None], axis=0)
    angular_momentum_error = context.dynamics.inertia_body_kg_m2 @ (
        post_impact.angular_velocity_body_rad_per_s - pre_impact.angular_velocity_body_rad_per_s
    )
    tolerance = context.solver.constraint_tolerance
    linear_error_norm = float(np.linalg.norm(linear_momentum_error))
    angular_error_norm = float(np.linalg.norm(angular_momentum_error))
    sticking_error_norm = float(np.linalg.norm(foot_velocities[0]))
    passed = (
        linear_error_norm <= tolerance
        and angular_error_norm <= tolerance
        and sticking_error_norm <= tolerance
        and impulse_residuals.is_feasible(atol=tolerance)
    )
    return {
        "pass": bool(passed),
        "tolerance": tolerance,
        "synthetic_preimpact_vertical_velocity_m_per_s": float(
            pre_impact.linear_velocity_world_m_per_s[2]
        ),
        "normal_impulse_ns": feasible_normal_impulse,
        "linear_momentum_error_norm_ns": linear_error_norm,
        "angular_momentum_error_norm_nms": angular_error_norm,
        "postimpact_sticking_velocity_norm_m_per_s": sticking_error_norm,
        "minimum_impulse_constraint_margin": min(
            float(np.min(impulse_residuals.normal_lower_margin_ns)),
            float(np.min(impulse_residuals.normal_upper_margin_ns)),
            float(np.min(impulse_residuals.friction_cone_margin_ns)),
            float(np.min(impulse_residuals.average_force_upper_margin_n)),
        ),
    }


def _validate_gain_endpoints(context: _ValidationContext, hover: _HoverFixture) -> JSONDict:
    baseline = hover.thrusts_n
    correction = np.minimum(
        np.asarray(context.correction_safety.maximum_correction_n),
        context.actuator.thrust_max_n - baseline,
    )
    mpc_total = baseline + correction
    ramp_duration_s = 1.0 / context.correction_safety.maximum_gain_rise_per_s
    maximum_raw = cast(
        FloatArray,
        np.asarray(context.correction_safety.maximum_correction_n, dtype=np.float64),
    )
    gain_zero = RotorCorrectionBlender(replace(context.correction_safety, target_gain=0.0)).blend(
        baseline, mpc_total, ramp_duration_s, healthy=True
    )
    gain_one = RotorCorrectionBlender(replace(context.correction_safety, target_gain=1.0)).blend(
        baseline, mpc_total, ramp_duration_s, healthy=True
    )
    zero_plan = RotorExecutionPlan(
        baseline_thrusts_n=baseline[None, :],
        correction_gains=np.zeros(1),
        maximum_raw_correction_n=maximum_raw,
    )
    one_plan = RotorExecutionPlan(
        baseline_thrusts_n=baseline[None, :],
        correction_gains=np.ones(1),
        maximum_raw_correction_n=maximum_raw,
    )
    zero_transport = reconstruct_transport_target(zero_plan, 0, baseline)
    one_transport = reconstruct_transport_target(one_plan, 0, mpc_total)
    gain_zero_command = np.asarray(gain_zero.applied_total_thrusts_n)
    gain_one_command = np.asarray(gain_one.applied_total_thrusts_n)
    tolerance = context.solver.constraint_tolerance
    zero_error = float(np.linalg.norm(gain_zero_command - baseline))
    one_error = float(np.linalg.norm(gain_one_command - mpc_total))
    passed = (
        gain_zero.valid
        and gain_one.valid
        and abs(gain_zero.applied_gain) <= tolerance
        and abs(gain_one.applied_gain - 1.0) <= tolerance
        and zero_error <= tolerance
        and one_error <= tolerance
        and zero_transport is None
        and one_transport is not None
        and np.linalg.norm(one_transport.target_thrusts_n - mpc_total) <= tolerance
    )
    return {
        "pass": bool(passed),
        "tolerance": tolerance,
        "raw_residual_correction_n": correction.tolist(),
        "gain_zero_applied": gain_zero.applied_gain,
        "gain_zero_command_error_norm_n": zero_error,
        "gain_one_applied": gain_one.applied_gain,
        "gain_one_command_error_norm_n": one_error,
        "gain_zero_transport_target_defined": zero_transport is not None,
        "gain_one_transport_target_error_norm_n": (
            None
            if one_transport is None
            else float(np.linalg.norm(one_transport.target_thrusts_n - mpc_total))
        ),
        "ramp_duration_s": ramp_duration_s,
    }


def _validate_contact_and_admittance(bundle: ImpactAwareConfigBundle) -> JSONDict:
    detector = bundle.new_contact_detector()
    detector.update(np.zeros(4), 0.0)
    filter_settle_s = max(0.1, 20.0 * bundle.contact_detector.filter_time_constant_s)
    high_forces = 2.0 * np.asarray(bundle.contact_detector.contact_on_threshold_n)
    detector.update(high_forces, filter_settle_s)
    detection = detector.update(
        high_forces,
        filter_settle_s + bundle.contact_detector.contact_confirm_s + 1.0e-6,
    )

    config = bundle.admittance_configs[0]
    workspace = bundle.admittance_workspaces[0]
    initial_joint_command = 0.5 * (np.asarray(config.joint_lower) + np.asarray(config.joint_upper))
    nominal_foot_body = 0.5 * (workspace.lower + workspace.upper)

    def fixed_inverse_kinematics(foot_position_body: FloatArray) -> FloatArray:
        if not np.all(np.isfinite(foot_position_body)):
            raise ValueError("synthetic inverse-kinematics input became nonfinite")
        return cast(FloatArray, initial_joint_command.copy())

    controller = LegAdmittanceController(
        config,
        fixed_inverse_kinematics,
        workspace,
        initial_joint_command,
    )
    estimated_force = np.array([0.0, 0.0, 1.0])
    touchdown_time = detection.touchdown_times_s[0]
    if touchdown_time is None:
        raise ValueError("synthetic contact detector did not provide a touchdown time")
    touchdown = controller.step(
        current_time_s=detection.timestamp_s,
        dt_s=bundle.mpc_dt_s,
        measured_contact=detection.contacts[0],
        touchdown_time_s=touchdown_time,
        rotation_body_to_world=np.eye(3),
        body_position_world=np.zeros(3),
        nominal_foot_position_world=nominal_foot_body,
        desired_force_world=np.zeros(3),
        estimated_force_world=estimated_force,
    )
    stance_time = touchdown_time + config.transition_duration_s
    stance = controller.step(
        current_time_s=stance_time,
        dt_s=bundle.mpc_dt_s,
        measured_contact=True,
        touchdown_time_s=touchdown_time,
        rotation_body_to_world=np.eye(3),
        body_position_world=np.zeros(3),
        nominal_foot_position_world=nominal_foot_body,
        desired_force_world=estimated_force,
        estimated_force_world=estimated_force,
    )
    expected_touchdown_force = np.sign(estimated_force) * np.maximum(
        np.abs(estimated_force) - np.asarray(config.force_error_deadband_n),
        0.0,
    )
    touchdown_force_error = float(
        np.linalg.norm(touchdown.admittance_force_body - expected_touchdown_force)
    )
    stance_force_error = float(np.linalg.norm(stance.admittance_force_body))
    finite_outputs = bool(
        np.all(np.isfinite(touchdown.joint_position_command))
        and np.all(np.isfinite(stance.joint_position_command))
        and np.all(np.isfinite(stance.state.correction_position_body))
    )
    passed = (
        all(detection.contacts)
        and all(detection.touchdown_events)
        and touchdown.blend.eta == 0.0
        and stance.blend.eta == 1.0
        and touchdown_force_error <= 1.0e-12
        and stance_force_error <= 1.0e-12
        and finite_outputs
    )
    return {
        "pass": bool(passed),
        "contacts_confirmed": list(detection.contacts),
        "touchdown_events": list(detection.touchdown_events),
        "touchdown_blend_eta": touchdown.blend.eta,
        "stance_blend_eta": stance.blend.eta,
        "touchdown_force_equation_error_n": touchdown_force_error,
        "stance_force_equation_error_n": stance_force_error,
        "joint_and_admittance_outputs_finite": finite_outputs,
    }


def _failed_check(exc: Exception) -> JSONDict:
    return {
        "pass": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_validation(config_path: Path) -> JSONDict:
    """Run all offline checks and return a JSON-serializable report."""

    report: JSONDict = {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "synthetic": True,
        "paper_experiment_reproduction": False,
        "hardware_output_permitted": False,
        "disclaimer": DISCLAIMER,
        "config_path": str(config_path.resolve()),
        "checks": {},
        "overall_pass": False,
    }
    checks = cast(Dict[str, JSONDict], report["checks"])
    try:
        config = _load_config(config_path)
        checks["config_load"] = {"pass": True}
    except Exception as exc:
        checks["config_load"] = _failed_check(exc)
        return report

    safety = _safe_profile_check(config)
    checks["synthetic_profile_safety"] = safety
    if not bool(safety["pass"]):
        return report

    try:
        bundle = load_impact_aware_config(config_path, allow_synthetic=True)
        checks["strict_configuration_assembly"] = {"pass": True}
    except Exception as exc:
        checks["strict_configuration_assembly"] = _failed_check(exc)
        return report

    try:
        context = _build_context(config)
        checks["parameter_validation"] = {"pass": True}
    except Exception as exc:
        checks["parameter_validation"] = _failed_check(exc)
        return report

    try:
        hover = _hover_fixture(context)
        checks["static_hover_allocation_and_dynamics"] = hover.report
    except Exception as exc:
        checks["static_hover_allocation_and_dynamics"] = _failed_check(exc)
        return report

    validations: Tuple[Tuple[str, Callable[[], JSONDict]], ...] = (
        ("nlp_slsqp", lambda: _validate_nlp(context, hover)),
        ("touchdown_nlp_slsqp", lambda: _validate_touchdown_nlp(context, hover)),
        ("impact_momentum_reset_and_sticking", lambda: _validate_impact(context, hover)),
        (
            "rotor_residual_gain_endpoints",
            lambda: _validate_gain_endpoints(context, hover),
        ),
        (
            "contact_detection_and_admittance_execution",
            lambda: _validate_contact_and_admittance(bundle),
        ),
    )
    for name, check in validations:
        try:
            checks[name] = check()
        except Exception as exc:
            checks[name] = _failed_check(exc)

    report["overall_pass"] = all(bool(check.get("pass", False)) for check in checks.values())
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "impact_aware_mpc_demo.yaml"
    parser = argparse.ArgumentParser(
        description="Run synthetic offline validation of impact-aware MPC mathematics."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, help="write JSON report to this file")
    args = parser.parse_args(argv)

    if args.output is not None:
        config_path = args.config.resolve()
        output_path = args.output.resolve()
        same_file = config_path == output_path
        if not same_file and config_path.exists() and output_path.exists():
            try:
                same_file = config_path.samefile(output_path)
            except OSError:
                same_file = False
        if same_file:
            parser.error("--output must not refer to the --config input file")

    report = run_validation(args.config)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(payload)
    else:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            report["overall_pass"] = False
            report["output_error"] = str(exc)
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
    return 0 if bool(report["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
