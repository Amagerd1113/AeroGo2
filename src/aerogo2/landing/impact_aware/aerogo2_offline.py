"""Build and exercise an explicitly non-hardware AeroGo2 MPC prior.

The preliminary YAML is the single source for the known mass, CoM, inertia
estimate and deployed rotor geometry.  Parameters that have not been measured
are borrowed from the synthetic numerical fixture only so the complete NLP can
be exercised offline; their provenance is exposed and they confer no hardware
authority.

中文说明：此适配器把已知的 Go2 URDF 质量属性、暂定上装质量和固定旋翼几何注入
完整 NLP；尚未实测的气动、接触、时序和权重参数仅借用合成样例以验证代码通路。
生成的 bundle 永远标记为离线且 κ=0，悬停求解成功只证明方程和约束在该工况可解，
不证明触地模型、推力映射或真机着陆安全。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple, Union, cast

import numpy as np

from aerogo2.landing.impact_aware.config import (
    ImpactAwareConfigBundle,
    load_impact_aware_config,
)
from aerogo2.landing.impact_aware.nlp import (
    ImpactAwareMPCProblem,
    LandingContactGeometry,
    MPCReferences,
    MPCSolveResult,
    StateBounds,
    solve_impact_aware_mpc,
)
from aerogo2.landing.impact_aware.preliminary import (
    PreliminaryLandingModelConfig,
    PreliminaryModelError,
    load_preliminary_landing_model,
)
from aerogo2.landing.impact_aware.rotor import build_fixed_deployed_allocation_matrix
from aerogo2.landing.impact_aware.rotor_safety import RotorCorrectionSafetyConfig
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FixedDeployedRotorGeometry,
    FloatArray,
    FootLeverArmsFromComBodyHorizon,
    ImpactLimits,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
)

PathLike = Union[str, Path]
_PROFILE = "aerogo2_provisional_offline_hybrid"


@dataclass(frozen=True)
class AeroGo2OfflinePriorBundle:
    """Complete numerical bundle plus an audit of every borrowed assumption."""

    controller: ImpactAwareConfigBundle
    preliminary: PreliminaryLandingModelConfig
    physical_prior_fields: Tuple[str, ...]
    numerical_only_fields: Tuple[str, ...]
    rotor_thrust_ceiling_source: str

    @property
    def hardware_output_permitted(self) -> bool:
        return self.controller.hardware_output_permitted


def build_aerogo2_offline_prior_bundle(
    preliminary_path: PathLike,
    synthetic_fixture_path: PathLike,
    *,
    allow_provisional: bool = False,
    for_hardware: bool = False,
) -> AeroGo2OfflinePriorBundle:
    """Merge physical priors into the full MPC strictly for offline execution."""

    if type(allow_provisional) is not bool or type(for_hardware) is not bool:
        raise TypeError("allow_provisional and for_hardware must be bool")
    if for_hardware:
        raise PreliminaryModelError(
            "the AeroGo2 provisional MPC hybrid is permanently prohibited for hardware"
        )
    if not allow_provisional:
        raise PreliminaryModelError("set allow_provisional=True for the offline MPC hybrid")

    preliminary = load_preliminary_landing_model(
        preliminary_path,
        allow_provisional=True,
        for_hardware=False,
    )
    fixture = load_impact_aware_config(
        synthetic_fixture_path,
        allow_synthetic=True,
        for_hardware=False,
    )
    estimate = preliminary.offline_inertia_estimate
    if estimate is None:
        raise PreliminaryModelError("offline_inertia_estimate is required for the MPC hybrid")

    geometry = FixedDeployedRotorGeometry(
        lever_arms_from_com_body_m=preliminary.geometry.lever_arms_from_com_body_m,
        thrust_directions_body=preliminary.geometry.thrust_directions_body,
    )
    allocation = build_fixed_deployed_allocation_matrix(
        geometry,
        fixture.rotor_aerodynamics,
    )
    dynamics = ReducedDynamicsConfig(
        mass_kg=preliminary.mass.total_nominal_kg,
        inertia_body_kg_m2=estimate.nominal_body_kg_m2,
        gravity_world_m_per_s2=np.array(
            [0.0, 0.0, -preliminary.gravity_m_per_s2], dtype=np.float64
        ),
        rotor_allocation_body=allocation,
    )

    # Battery S-count is not confirmed.  The lower of the two reviewed table
    # endpoints is only an offline feasibility ceiling, never an installed
    # thrust limit or a command conversion.
    manufacturer_ceiling_n = min(
        float(curve.thrust_n[-1]) for curve in preliminary.rotor_prior.curves
    )
    ceiling = np.full(4, manufacturer_ceiling_n, dtype=np.float64)
    actuator = RotorActuatorConfig(
        time_constants_s=fixture.rotor_actuator.time_constants_s,
        thrust_min_n=np.zeros(4, dtype=np.float64),
        thrust_max_n=ceiling,
        thrust_rate_min_n_per_s=fixture.rotor_actuator.thrust_rate_min_n_per_s,
        thrust_rate_max_n_per_s=fixture.rotor_actuator.thrust_rate_max_n_per_s,
    )
    correction_safety = RotorCorrectionSafetyConfig(
        target_gain=0.0,
        thrust_min_n=actuator.thrust_min_n,
        thrust_max_n=actuator.thrust_max_n,
        maximum_correction_n=fixture.rotor_correction_safety.maximum_correction_n,
        maximum_gain_rise_per_s=fixture.rotor_correction_safety.maximum_gain_rise_per_s,
    )

    lower = np.array(fixture.mpc_state_bounds.lower, dtype=np.float64, copy=True)
    upper = np.array(fixture.mpc_state_bounds.upper, dtype=np.float64, copy=True)
    lower[:, 12:16] = 0.0
    upper[:, 12:16] = manufacturer_ceiling_n
    state_bounds = StateBounds(
        lower=lower,
        upper=upper,
        soft_mask=fixture.mpc_state_bounds.soft_mask,
    )
    # The preliminary model is normal-only.  A zero coefficient removes
    # tangential force/impulse authority while retaining the full NLP shape.
    contact_limits = replace(
        fixture.contact_force_limits,
        friction_coefficients=np.zeros(4, dtype=np.float64),
    )
    impact_limits = ImpactLimits(
        friction_coefficients=np.zeros(4, dtype=np.float64),
        maximum_normal_impulse_ns=fixture.impact_limits.maximum_normal_impulse_ns,
        impact_duration_s=fixture.impact_limits.impact_duration_s,
        maximum_average_normal_force_n=fixture.impact_limits.maximum_average_normal_force_n,
    )

    controller = replace(
        fixture,
        source_path=Path(preliminary_path).expanduser().resolve(),
        profile=_PROFILE,
        parameters_identified=False,
        allow_hardware_output=False,
        physical_use_prohibited=True,
        rotor_geometry=geometry,
        rotor_actuator=actuator,
        dynamics=dynamics,
        impact_limits=impact_limits,
        rotor_correction_safety=correction_safety,
        go2_foot_force_calibration=None,
        contact_force_limits=contact_limits,
        mpc_state_bounds=state_bounds,
    )
    return AeroGo2OfflinePriorBundle(
        controller=controller,
        preliminary=preliminary,
        physical_prior_fields=(
            "mass.total_nominal_kg",
            "offline_inertia_estimate.nominal_body_kg_m2",
            "geometry.lever_arms_from_com_body_m",
            "geometry.thrust_directions_body",
            "model.gravity_m_per_s2",
        ),
        numerical_only_fields=(
            "rotor.aerodynamics and spin directions",
            "rotor actuator time constants and thrust-rate limits",
            "contact/impact force limits and detector thresholds",
            "MPC horizon, weights, bounds and solver settings",
            "admittance gains, workspaces and joint limits",
        ),
        rotor_thrust_ceiling_source=(
            "minimum 100%-throttle endpoint of reviewed X8 G2 12S/14S static tables; "
            "offline feasibility only"
        ),
    )


def solve_aerogo2_offline_hover(
    bundle: AeroGo2OfflinePriorBundle,
) -> MPCSolveResult:
    """Run one full no-contact hover NLP using the provisional AeroGo2 prior."""

    if not isinstance(bundle, AeroGo2OfflinePriorBundle):
        raise TypeError("bundle must be an AeroGo2OfflinePriorBundle")
    if bundle.hardware_output_permitted:
        raise PreliminaryModelError("offline validation cannot accept hardware authority")
    config = bundle.controller
    desired_wrench = np.concatenate(
        (-config.dynamics.mass_kg * config.dynamics.gravity_world_m_per_s2, np.zeros(3))
    )
    thrusts, _, _, _ = np.linalg.lstsq(
        config.dynamics.rotor_allocation_body,
        desired_wrench,
        rcond=None,
    )
    thrusts = cast(FloatArray, np.asarray(thrusts, dtype=np.float64))
    if np.any(thrusts < config.rotor_actuator.thrust_min_n) or np.any(
        thrusts > config.rotor_actuator.thrust_max_n
    ):
        raise PreliminaryModelError("provisional rotor ceiling cannot support level hover")

    initial = ReducedState(
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
    horizon = config.mpc_horizon_steps
    references = MPCReferences(
        position_world_m=np.zeros((horizon + 1, 3)),
        linear_velocity_world_m_per_s=np.zeros((horizon + 1, 3)),
        rotation_body_to_world=np.repeat(np.eye(3)[None, :, :], horizon + 1, axis=0),
        angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
        contact_forces_world_n=np.zeros((horizon, 4, 3)),
        rotor_thrust_commands_n=np.repeat(thrusts[None, :], horizon, axis=0),
    )
    problem = ImpactAwareMPCProblem(
        initial_state=initial,
        previous_input=control,
        dt_s=config.mpc_dt_s,
        contact_schedule=np.zeros((horizon + 1, 4), dtype=np.int8),
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            np.zeros((horizon + 1, 4, 3)),
            GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=references,
        state_bounds=config.mpc_state_bounds,
        contact_limits=config.contact_force_limits,
        impact_events=(),
        dynamics_config=config.dynamics,
        rotor_actuator_config=config.rotor_actuator,
        weights=config.mpc_weights,
        # Even an all-flight horizon keeps an explicit plane so the landing
        # NLP cannot silently disable nonpenetration before touchdown.
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=-1.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
    )
    return solve_impact_aware_mpc(problem, config.solver_settings)


__all__ = [
    "AeroGo2OfflinePriorBundle",
    "build_aerogo2_offline_prior_bundle",
    "solve_aerogo2_offline_hover",
]
