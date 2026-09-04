"""Build a reproducible AeroGo2 normal-only touchdown reference problem.

Only hash-pinned URDF geometry and the provisional offline parameter bundle are
used.  No hardware module is imported and the returned problem/result cannot
authorize Go2 or flight-controller output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, cast

import numpy as np
from numpy.typing import NDArray

from aerogo2.landing.impact_aware.aerogo2_offline import AeroGo2OfflinePriorBundle
from aerogo2.landing.impact_aware.go2_kinematics import (
    SDK_LEG_ORDER,
    load_go2_urdf_kinematics,
)
from aerogo2.landing.impact_aware.normal_only_mpc import (
    NormalOnlyMPCProblem,
    NormalOnlyMPCResult,
    solve_normal_only_mpc,
)
from aerogo2.landing.impact_aware.preliminary import NormalOnlyVerticalState
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    FootLeverArmsFromComBody,
    foot_positions_from_body_origin_B_to_com_lever_arms,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AeroGo2NormalOnlyLandingFixture:
    """Offline touchdown fixture and the exact B-to-C geometry used to build it."""

    problem: NormalOnlyMPCProblem
    foot_lever_arms_from_com: FootLeverArmsFromComBody
    touchdown_com_height_world_m: float

    @property
    def hardware_output_permitted(self) -> bool:
        return False


def build_aerogo2_normal_only_landing_fixture(
    bundle: AeroGo2OfflinePriorBundle,
    *,
    descent_speed_m_per_s: float = 0.2,
    ground_height_world_m: float = 0.0,
    touchdown_position_tolerance_m: float = 1.0e-5,
) -> AeroGo2NormalOnlyLandingFixture:
    """Create a simultaneous four-foot touchdown case from current priors.

    The initial clearance is chosen so level-hover acceleration reaches the
    plane after exactly one MPC interval.  This is a deterministic regression
    fixture, not a claim that the real landing planner will know touchdown time
    to that precision.
    """

    if not isinstance(bundle, AeroGo2OfflinePriorBundle):
        raise TypeError("bundle must be AeroGo2OfflinePriorBundle")
    if bundle.hardware_output_permitted:
        raise ValueError("normal-only offline fixture refuses hardware-capable bundles")
    speed = float(descent_speed_m_per_s)
    ground = float(ground_height_world_m)
    position_tolerance = float(touchdown_position_tolerance_m)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("descent_speed_m_per_s must be finite and positive")
    if not math.isfinite(ground):
        raise ValueError("ground_height_world_m must be finite")
    if not math.isfinite(position_tolerance) or position_tolerance < 0.0:
        raise ValueError("touchdown_position_tolerance_m must be finite and nonnegative")

    preliminary = bundle.preliminary
    urdf = preliminary.go2_urdf
    com_from_B = preliminary.geometry.total_com_C_from_body_origin_B_m
    if urdf is None or com_from_B is None:
        raise ValueError("AeroGo2 normal-only fixture requires pinned URDF and explicit B-to-C offset")
    kinematics = load_go2_urdf_kinematics(
        urdf.mass_properties,
        urdf_root_from_body_origin_B_m=urdf.urdf_root_from_body_origin_B_m,
    )
    q_by_leg_values = np.asarray(urdf.sdk_joint_positions_rad, dtype=np.float64).reshape(
        FOOT_COUNT,
        3,
    )
    q_by_leg: Dict[str, FloatArray] = {
        leg: cast(FloatArray, np.array(q_by_leg_values[index], copy=True))
        for index, leg in enumerate(SDK_LEG_ORDER)
    }
    feet_from_B = kinematics.forward_all(q_by_leg, output_leg_order=SDK_LEG_ORDER)
    feet_from_C = foot_positions_from_body_origin_B_to_com_lever_arms(
        feet_from_B,
        com_from_B,
        target_leg_order=SDK_LEG_ORDER,
    )
    foot_z = np.asarray(feet_from_C.values_m[:, 2], dtype=np.float64)
    touchdown_heights = ground - foot_z
    touchdown_height = float(np.mean(touchdown_heights))
    if np.max(np.abs(touchdown_heights - touchdown_height)) > position_tolerance:
        raise ValueError(
            "standard-pose feet do not share one horizontal touchdown plane within tolerance"
        )

    controller = bundle.controller
    horizon = controller.mpc_horizon_steps
    if horizon < 2:
        raise ValueError("normal-only touchdown regression requires at least two MPC intervals")
    dt = controller.mpc_dt_s
    mass = controller.dynamics.mass_kg
    gravity = -float(controller.dynamics.gravity_world_m_per_s2[2])
    if gravity <= 0.0:
        raise ValueError("normal-only fixture requires world-Z-up gravity")
    hover_per_rotor = mass * gravity / 4.0
    rotor_minimum = np.asarray(controller.rotor_actuator.thrust_min_n, dtype=np.float64)
    rotor_maximum = np.asarray(controller.rotor_actuator.thrust_max_n, dtype=np.float64)
    if np.any(hover_per_rotor < rotor_minimum) or np.any(hover_per_rotor > rotor_maximum):
        raise ValueError("configured rotor bounds cannot support the normal-only hover reference")
    symmetric_rate = np.minimum(
        np.asarray(controller.rotor_actuator.thrust_rate_max_n_per_s, dtype=np.float64),
        -np.asarray(controller.rotor_actuator.thrust_rate_min_n_per_s, dtype=np.float64),
    )
    if np.any(symmetric_rate <= 0.0):
        raise ValueError("normal-only fixture requires bidirectional finite rotor-force rate limits")

    schedule = np.ones((horizon + 1, FOOT_COUNT), dtype=np.int8)
    schedule[0] = 0
    foot_heights = np.repeat(foot_z[np.newaxis, :], horizon + 1, axis=0)
    initial_height = touchdown_height + speed * dt
    reference_height = np.full(horizon + 1, touchdown_height, dtype=np.float64)
    reference_height[0] = initial_height
    reference_velocity = np.zeros(horizon + 1, dtype=np.float64)
    reference_velocity[0] = -speed

    rotor_reference = np.empty((horizon, 4), dtype=np.float64)
    contact_reference = np.empty((horizon, 4), dtype=np.float64)
    rotor_reference[0] = hover_per_rotor
    contact_reference[0] = 0.0
    for step in range(1, horizon):
        # Use at most half of the configured downward rate per interval so the
        # reference has numerical headroom.  Contact force supplies the exact
        # complement required for static vertical balance.
        reduction = np.minimum(
            0.5 * symmetric_rate * dt * float(step),
            np.full(4, 0.5 * hover_per_rotor),
        )
        rotor_reference[step] = hover_per_rotor - reduction
        contact_reference[step] = reduction
    if np.any(contact_reference > controller.contact_force_limits.maximum_normal_force_n):
        raise ValueError("configured contact-force limits cannot support the touchdown reference")

    impact = controller.impact_limits
    # 一维模型没有姿态动力学，所以不能让求解器任意决定四通道分配。当前物理
    # 回归夹具为对称布局，采用四旋翼/四足等分；无接触或非触地步严格为零。
    rotor_allocation = np.full((horizon, FOOT_COUNT), 0.25, dtype=np.float64)
    contact_allocation = 0.25 * schedule[:-1].astype(np.float64)
    touchdown = (1 - schedule[:-1]) * schedule[1:]
    impulse_allocation = 0.25 * touchdown.astype(np.float64)
    problem = NormalOnlyMPCProblem(
        initial_state=NormalOnlyVerticalState(
            height_world_m=initial_height,
            vertical_velocity_world_m_per_s=-speed,
        ),
        dt_s=dt,
        contact_schedule=schedule,
        leg_order=SDK_LEG_ORDER,
        rotor_order=controller.rotor_order,
        foot_heights_from_com_m=foot_heights,
        ground_height_world_m=ground,
        touchdown_position_tolerance_m=position_tolerance,
        minimum_downward_speed_m_per_s=0.5 * speed,
        mass_kg=mass,
        gravity_m_per_s2=gravity,
        previous_rotor_forces_n=np.full(4, hover_per_rotor),
        rotor_force_min_n=rotor_minimum,
        rotor_force_max_n=rotor_maximum,
        rotor_force_rate_max_n_per_s=symmetric_rate,
        contact_force_max_n=controller.contact_force_limits.maximum_normal_force_n,
        normal_impulse_max_ns=np.full(4, impact.maximum_normal_impulse_ns),
        impact_duration_s=impact.impact_duration_s,
        average_impact_force_max_n=np.full(4, impact.maximum_average_normal_force_n),
        rotor_force_allocation=rotor_allocation,
        contact_force_allocation=contact_allocation,
        normal_impulse_allocation=impulse_allocation,
        reference_height_world_m=reference_height,
        reference_vertical_velocity_world_m_per_s=reference_velocity,
        reference_rotor_forces_n=rotor_reference,
        reference_contact_forces_n=contact_reference,
        minimum_com_height_world_m=touchdown_height - position_tolerance,
        maximum_com_height_world_m=initial_height + 0.25,
        maximum_abs_vertical_velocity_m_per_s=max(1.0, 2.0 * speed),
    )
    return AeroGo2NormalOnlyLandingFixture(
        problem=problem,
        foot_lever_arms_from_com=feet_from_C,
        touchdown_com_height_world_m=touchdown_height,
    )


def solve_aerogo2_normal_only_landing(
    fixture: AeroGo2NormalOnlyLandingFixture,
) -> NormalOnlyMPCResult:
    """Solve the fixture with the bundle-independent offline reference solver."""

    if not isinstance(fixture, AeroGo2NormalOnlyLandingFixture):
        raise TypeError("fixture must be AeroGo2NormalOnlyLandingFixture")
    return solve_normal_only_mpc(fixture.problem)


__all__ = [
    "AeroGo2NormalOnlyLandingFixture",
    "build_aerogo2_normal_only_landing_fixture",
    "solve_aerogo2_normal_only_landing",
]
