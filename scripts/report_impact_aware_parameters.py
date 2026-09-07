#!/usr/bin/env python3
"""Print the reviewed offline AeroGo2 physical-parameter prior as JSON.

This command never opens a hardware transport and cannot authorize output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aerogo2.landing.impact_aware.go2_kinematics import load_go2_urdf_kinematics
from aerogo2.landing.impact_aware.preliminary import (
    PreliminaryLandingModelConfig,
    ideal_level_hover_thrust_per_rotor_n,
    load_preliminary_landing_model,
)


def build_report(config: PreliminaryLandingModelConfig) -> Dict[str, Any]:
    """Build a serializable report without changing the immutable model."""

    geometry = config.geometry
    urdf = config.go2_urdf
    estimate = config.offline_inertia_estimate
    if urdf is None or estimate is None:
        raise ValueError("current-schema Go2 URDF and offline inertia estimate are required")
    kinematics = load_go2_urdf_kinematics(
        urdf.mass_properties,
        urdf_root_from_body_origin_B_m=urdf.urdf_root_from_body_origin_B_m,
    )
    leg_kinematics = {}
    for leg_name, leg in kinematics.legs.items():
        movable = leg.joints[:3]
        leg_kinematics[leg_name] = {
            "joint_names": list(leg.joint_names),
            "joint_origins_from_parent_m": [
                joint.translation_parent_m.tolist() for joint in movable
            ],
            "joint_axes": [joint.axis_joint.tolist() for joint in movable],
            "foot_origin_from_calf_m": leg.joints[3].translation_parent_m.tolist(),
            "urdf_joint_lower_rad": leg.lower_rad.tolist(),
            "urdf_joint_upper_rad": leg.upper_rad.tolist(),
            "home_joint_positions_rad": leg.home_q_rad.tolist(),
            "home_foot_from_body_origin_B_m": leg.forward(leg.home_q_rad).tolist(),
        }
    return {
        "source_config": str(config.source_path),
        "schema_version": config.schema_version,
        "profile": config.profile,
        "hardware_output_permitted": config.hardware_output_permitted,
        "reference_points": {
            "dynamics": config.dynamics_reference_point,
            "leg_kinematics": config.leg_kinematics_reference_point,
            "frame_center_O_from_body_B_m": geometry.frame_center_O_from_body_origin_B_m.tolist(),
            "total_com_C_from_body_B_m": geometry.total_com_C_from_body_origin_B_m.tolist(),
            "rotor_plane_from_frame_O_m": geometry.rotor_plane_from_frame_center_O_m.tolist(),
        },
        "mass_kg": {
            "go2_urdf": urdf.mass_properties.mass_kg,
            "added_system_provisional": config.mass.added_system_nominal_kg,
            "total_provisional": config.mass.total_nominal_kg,
            "measured_uncertainty": config.mass.total_uncertainty_kg,
        },
        "go2_urdf": {
            "path": str(urdf.bundled_path),
            "sha256": urdf.bundled_sha256,
            "quality": urdf.model_quality,
            "reference_pose": urdf.reference_pose,
            "links": urdf.mass_properties.total_link_count,
            "inertial_links": urdf.mass_properties.inertial_link_count,
            "positive_mass_links": urdf.mass_properties.positive_mass_link_count,
            "com_from_urdf_root_m": urdf.mass_properties.com_from_root_m.tolist(),
            "inertia_about_go2_com_kg_m2": (
                urdf.mass_properties.inertia_about_com_root_axes_kg_m2.tolist()
            ),
            "kinematics_offline_prior_only": kinematics.offline_prior_only,
            "kinematics_hardware_validated": kinematics.hardware_validated,
            "sdk_leg_order": list(kinematics.leg_order),
            "legs": leg_kinematics,
        },
        "rotors": {
            "order": list(config.rotor_order),
            "lever_arms_from_total_com_C_body_m": (geometry.lever_arms_from_com_body_m.tolist()),
            "ideal_hover_thrust_each_n": ideal_level_hover_thrust_per_rotor_n(config),
        },
        "offline_inertia_estimate": {
            "quality": estimate.quality,
            "method": estimate.method,
            "remaining_added_mass_effective_com_from_body_B_m": (
                estimate.remaining_added_mass_effective_com_from_body_origin_B_m.tolist()
            ),
            "remaining_mass_distribution_interval": (estimate.remaining_mass_distribution_interval),
            "nominal_about_C_body_kg_m2": estimate.nominal_body_kg_m2.tolist(),
            "diagonal_lower_kg_m2": estimate.diagonal_lower_body_kg_m2.tolist(),
            "diagonal_upper_kg_m2": estimate.diagonal_upper_body_kg_m2.tolist(),
            "cross_terms_unbounded": estimate.cross_terms_unbounded,
            "usable_for_hardware": estimate.usable_for_hardware,
        },
        "must_confirm_before_hardware": [
            "assembled mass and uncertainty",
            "B-to-URDF-root transform",
            "total CoM C and complete CAD/BOM inertia",
            "four installed thrust axes and lever-arm uncertainty",
            "Go2 joint mapping/zero/direction/safe limits and standard-pose LowState",
            "Pixhawk firmware/frame/mount/quaternion/IMU-to-C transform",
            "contact-count mapping, thresholds and timing",
            "real flight-controller residual transport and executed-command feedback",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/impact_aware_preliminary.yaml"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = load_preliminary_landing_model(args.config, allow_provisional=True)
    print(json.dumps(build_report(config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
