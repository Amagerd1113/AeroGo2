#!/usr/bin/env python3
"""Validate one explicit normal-only AeroGo2 touchdown with no hardware I/O."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from aerogo2.landing.impact_aware.aerogo2_normal_only import (
    build_aerogo2_normal_only_landing_fixture,
    solve_aerogo2_normal_only_landing,
)
from aerogo2.landing.impact_aware.aerogo2_offline import (
    build_aerogo2_offline_prior_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preliminary",
        type=Path,
        default=Path("configs/impact_aware_preliminary.yaml"),
    )
    parser.add_argument(
        "--synthetic-fixture",
        type=Path,
        default=Path("configs/impact_aware_mpc_demo.yaml"),
        help="Numerical-only actuator/solver bounds for still-unidentified parameters",
    )
    parser.add_argument("--descent-speed-m-per-s", type=float, default=0.2)
    parser.add_argument("--ground-height-world-m", type=float, default=0.0)
    parser.add_argument("--touchdown-position-tolerance-m", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    bundle = build_aerogo2_offline_prior_bundle(
        args.preliminary,
        args.synthetic_fixture,
        allow_provisional=True,
    )
    fixture = build_aerogo2_normal_only_landing_fixture(
        bundle,
        descent_speed_m_per_s=args.descent_speed_m_per_s,
        ground_height_world_m=args.ground_height_world_m,
        touchdown_position_tolerance_m=args.touchdown_position_tolerance_m,
    )
    result = solve_aerogo2_normal_only_landing(fixture)
    report = {
        "pass": bool(result.success),
        "hardware_output_permitted": False,
        "model": "NORMAL_ONLY_VERTICAL",
        "leg_order": list(result.leg_order),
        "rotor_order": list(result.rotor_order),
        "touchdown_com_height_world_m": fixture.touchdown_com_height_world_m,
        "foot_lever_arms_from_com_body_m": (
            fixture.foot_lever_arms_from_com.values_m.tolist()
        ),
        "assumptions": {
            "ground": "horizontal plane in ENU world frame",
            "touchdown": "simultaneous four-foot contact after one MPC interval",
            "rotor_force": "quasi-static world-Z force; no PWM/throttle conversion",
            "contact_force": "desired normal force only; not raw Go2 SDK counts",
        },
        "solver": {
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "objective": result.objective,
            "solve_time_s": result.solve_time_s,
            "max_equality_violation": result.max_equality_violation,
            "min_inequality_residual": result.min_inequality_residual,
            "min_variable_bound_residual": result.min_variable_bound_residual,
        },
        "first_stage": {
            "rotor_forces_n": (
                None
                if result.first_rotor_forces_n is None
                else result.first_rotor_forces_n.tolist()
            ),
            "desired_contact_normal_forces_n": (
                None
                if result.first_desired_contact_normal_forces_n is None
                else result.first_desired_contact_normal_forces_n.tolist()
            ),
        },
        "normal_impulses_ns": result.normal_impulses_ns.tolist(),
    }
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    return 0 if result.success and math.isfinite(result.objective) else 2


if __name__ == "__main__":
    raise SystemExit(main())
