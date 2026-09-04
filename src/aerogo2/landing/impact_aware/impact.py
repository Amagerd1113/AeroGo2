"""Touchdown impulse reset and landing-safety residuals (Eqs. 38-44).

中文说明：该模块实现离散触地瞬间的动量跃迁，而不是有限时长接触力积分器的替代品。
冲量先统一到机体系，再更新整机线动量和角动量；粘着残差检查触地点冲击后速度是否
接近零。当前论文初步版本在真实 Go2 上只能可靠获得标量法向接触趋势，因此三维冲量、
切向摩擦锥和粘着约束仅保留为离线模型，不能据此宣称已经完成实机验证。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from aerogo2.landing.impact_aware.math_utils import (
    _as_binary_vector,
    _as_finite_array,
    _finite_scalar,
    _readonly,
    require_rotation_matrix,
)
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    FloatArray,
    FootLeverArmsFromComBody,
    ImpactLimits,
    ImpulseConstraintResiduals,
    ReducedDynamicsConfig,
    ReducedState,
    require_com_foot_lever_arms,
)


def integrate_contact_impulse(
    force_samples_world_n: object,
    sample_times_s: object,
) -> FloatArray:
    """Trapezoidally integrate sampled contact force for paper Eq. (38).

    ``force_samples_world_n`` has shape ``(sample_count, 4, 3)`` and each row
    corresponds to the strictly increasing timestamp at the same index.  The
    returned impulse has shape ``(4, 3)`` and units N*s in world frame.
    """

    raw_forces = np.asarray(force_samples_world_n)
    if raw_forces.dtype.kind not in "fiu":
        raise TypeError("force_samples_world_n must contain real numeric values")
    forces = np.array(raw_forces, dtype=float, copy=True)
    if forces.ndim != 3 or forces.shape[1:] != (FOOT_COUNT, 3):
        raise ValueError("force_samples_world_n must have shape (sample_count, 4, 3)")
    if forces.shape[0] < 2:
        raise ValueError("at least two force samples are required")
    if not np.all(np.isfinite(forces)):
        raise ValueError("force_samples_world_n must contain only finite values")

    times = _as_finite_array(
        sample_times_s,
        (forces.shape[0],),
        "sample_times_s",
    )
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("sample_times_s must be strictly increasing")
    trapezoids = 0.5 * (forces[:-1] + forces[1:])
    impulse = np.sum(trapezoids * intervals[:, np.newaxis, np.newaxis], axis=0)
    return _readonly(impulse)


def contact_transition_indicators(
    contact_before: object,
    contact_after: object,
) -> Tuple[FloatArray, FloatArray]:
    """Return touchdown and impact-participation indicators from Eq. (39)."""

    before = _as_binary_vector(contact_before, FOOT_COUNT, "contact_before")
    after = _as_binary_vector(contact_after, FOOT_COUNT, "contact_after")
    touchdown = (1 - before) * after
    participation = after.copy()
    touchdown.setflags(write=False)
    participation.setflags(write=False)
    return touchdown, participation


def impulses_world_to_body(
    impulses_world_ns: object,
    rotation_body_to_world: object,
) -> FloatArray:
    """Transform four world-frame impulses to body frame (paper Eq. 41)."""

    impulses = _as_finite_array(
        impulses_world_ns,
        (FOOT_COUNT, 3),
        "impulses_world_ns",
    )
    rotation = require_rotation_matrix(
        rotation_body_to_world,
        name="rotation_body_to_world",
    )
    # Row-vector form of B_Lambda = R_B.T @ Lambda.
    return _readonly(impulses @ rotation)


def momentum_reset(
    pre_impact_state: ReducedState,
    impulses_world_ns: object,
    impact_participation: object,
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBody,
    config: ReducedDynamicsConfig,
    *,
    impulse_leg_order: object,
) -> ReducedState:
    """Apply the linear/angular momentum jump in paper Eqs. (40)-(41).

    Position, attitude, and actual rotor thrust remain continuous.  Gravity and
    rotor impulses are intentionally absent, matching the paper's short-impact
    assumptions.  Inputs for inactive feet are masked by ``impact_participation``.
    """

    if not isinstance(pre_impact_state, ReducedState):
        raise TypeError("pre_impact_state must be a ReducedState")
    if not isinstance(config, ReducedDynamicsConfig):
        raise TypeError("config must be a ReducedDynamicsConfig")

    impulses_world = _as_finite_array(
        impulses_world_ns,
        (FOOT_COUNT, 3),
        "impulses_world_ns",
    )
    participation = _as_binary_vector(
        impact_participation,
        FOOT_COUNT,
        "impact_participation",
    )
    lever_arms = require_com_foot_lever_arms(
        foot_lever_arms_from_com_body_m,
        data_leg_order=impulse_leg_order,
    )
    foot_lever_arms = lever_arms.values_m
    active = participation[:, np.newaxis]
    active_impulses_world = active * impulses_world
    active_impulses_body = impulses_world_to_body(
        active_impulses_world,
        pre_impact_state.rotation_body_to_world,
    )

    linear_velocity_after = (
        pre_impact_state.linear_velocity_world_m_per_s
        + np.sum(active_impulses_world, axis=0) / config.mass_kg
    )
    angular_impulse_body = np.sum(
        np.cross(foot_lever_arms, active_impulses_body),
        axis=0,
    )
    angular_velocity_after = pre_impact_state.angular_velocity_body_rad_per_s + np.linalg.solve(
        config.inertia_body_kg_m2, angular_impulse_body
    )

    return ReducedState(
        position_world_m=pre_impact_state.position_world_m,
        linear_velocity_world_m_per_s=linear_velocity_after,
        rotation_body_to_world=pre_impact_state.rotation_body_to_world,
        angular_velocity_body_rad_per_s=angular_velocity_after,
        rotor_thrusts_n=pre_impact_state.rotor_thrusts_n,
    )


def foot_post_impact_velocity(
    state: ReducedState,
    foot_lever_arms_from_com_body_m: FootLeverArmsFromComBody,
    leg_jacobians_body: object,
    joint_velocities_rad_per_s: object,
    *,
    leg_order: object,
) -> FloatArray:
    """Evaluate four world-frame foot velocities using paper Eq. (42).

    The prescribed leg Jacobians have shape ``(4, 3, 3)`` and map each leg's
    three joint rates to body-frame foot-relative velocity.  Pass zero joint
    rates to represent a fixed leg configuration during the impact.
    """

    if not isinstance(state, ReducedState):
        raise TypeError("state must be a ReducedState")
    lever_arms = require_com_foot_lever_arms(
        foot_lever_arms_from_com_body_m,
        data_leg_order=leg_order,
    )
    foot_lever_arms = lever_arms.values_m
    jacobians = _as_finite_array(
        leg_jacobians_body,
        (FOOT_COUNT, 3, 3),
        "leg_jacobians_body",
    )
    joint_velocities = _as_finite_array(
        joint_velocities_rad_per_s,
        (FOOT_COUNT, 3),
        "joint_velocities_rad_per_s",
    )

    rotational_velocity_body = np.cross(
        np.broadcast_to(state.angular_velocity_body_rad_per_s, (FOOT_COUNT, 3)),
        foot_lever_arms,
    )
    joint_velocity_body = np.einsum(
        "fij,fj->fi",
        jacobians,
        joint_velocities,
    )
    relative_velocity_world = (
        rotational_velocity_body + joint_velocity_body
    ) @ state.rotation_body_to_world.T
    velocities_world = state.linear_velocity_world_m_per_s[np.newaxis, :] + relative_velocity_world
    return _readonly(velocities_world)


def sticking_velocity_residual(
    post_impact_foot_velocities_world_m_per_s: object,
    impact_participation: object,
) -> FloatArray:
    """Return the masked equality residual for paper Eq. (43)."""

    velocities = _as_finite_array(
        post_impact_foot_velocities_world_m_per_s,
        (FOOT_COUNT, 3),
        "post_impact_foot_velocities_world_m_per_s",
    )
    participation = _as_binary_vector(
        impact_participation,
        FOOT_COUNT,
        "impact_participation",
    )
    return _readonly(participation[:, np.newaxis] * velocities)


def sticking_constraint_satisfied(
    post_impact_foot_velocities_world_m_per_s: object,
    impact_participation: object,
    *,
    atol_m_per_s: object = 0.0,
) -> bool:
    """Validate Eq. (43) against an explicit caller-selected tolerance."""

    tolerance = _finite_scalar(atol_m_per_s, "atol_m_per_s", minimum=0.0)
    residual = sticking_velocity_residual(
        post_impact_foot_velocities_world_m_per_s,
        impact_participation,
    )
    return bool(np.all(np.abs(residual) <= tolerance))


def evaluate_impulse_constraints(
    impulses_world_ns: object,
    impact_participation: object,
    limits: ImpactLimits,
) -> ImpulseConstraintResiduals:
    """Return nonnegative margins for horizontal-surface paper Eq. (44).

    The world ``z`` component is the normal impulse.  For inactive feet, the
    upper normal margin together with the lower and friction margins forces all
    impulse components to zero.  The equivalent force is an interval average,
    not an estimate of peak force.
    """

    if not isinstance(limits, ImpactLimits):
        raise TypeError("limits must be ImpactLimits")
    impulses = _as_finite_array(
        impulses_world_ns,
        (FOOT_COUNT, 3),
        "impulses_world_ns",
    )
    participation = _as_binary_vector(
        impact_participation,
        FOOT_COUNT,
        "impact_participation",
    )

    normal = impulses[:, 2]
    tangential_norm = np.linalg.norm(impulses[:, :2], axis=1)
    average_force = normal / limits.impact_duration_s
    return ImpulseConstraintResiduals(
        normal_lower_margin_ns=normal,
        normal_upper_margin_ns=(participation * limits.maximum_normal_impulse_ns - normal),
        friction_cone_margin_ns=(limits.friction_coefficients * normal - tangential_norm),
        average_force_upper_margin_n=(limits.maximum_average_normal_force_n - average_force),
        equivalent_average_normal_force_n=average_force,
    )


__all__ = [
    "contact_transition_indicators",
    "evaluate_impulse_constraints",
    "foot_post_impact_velocity",
    "impulses_world_to_body",
    "integrate_contact_impulse",
    "momentum_reset",
    "sticking_constraint_satisfied",
    "sticking_velocity_residual",
]
