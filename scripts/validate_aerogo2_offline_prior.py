#!/usr/bin/env python3
"""Validate the full NLP with AeroGo2 physical priors and no hardware I/O."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from aerogo2.landing.impact_aware.aerogo2_offline import (
    build_aerogo2_offline_prior_bundle,
    solve_aerogo2_offline_hover,
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
        help="Numerical-only defaults for parameters not yet identified",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    merged = build_aerogo2_offline_prior_bundle(
        args.preliminary,
        args.synthetic_fixture,
        allow_provisional=True,
    )
    result = solve_aerogo2_offline_hover(merged)
    controller = merged.controller
    hover = -controller.dynamics.mass_kg * controller.dynamics.gravity_world_m_per_s2[2] / 4.0
    report = {
        "pass": bool(result.success),
        "hardware_output_permitted": merged.hardware_output_permitted,
        "profile": controller.profile,
        "mass_kg": controller.dynamics.mass_kg,
        "inertia_body_kg_m2": controller.dynamics.inertia_body_kg_m2.tolist(),
        "lever_arms_from_com_body_m": controller.rotor_geometry.lever_arms_from_com_body_m.tolist(),
        "level_hover_thrust_per_rotor_n": float(hover),
        "offline_rotor_thrust_ceiling_n": float(controller.rotor_actuator.thrust_max_n[0]),
        "rotor_thrust_ceiling_source": merged.rotor_thrust_ceiling_source,
        "physical_prior_fields": list(merged.physical_prior_fields),
        "numerical_only_fields": list(merged.numerical_only_fields),
        "solver": {
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "iterations": result.iterations,
            "solve_time_s": result.solve_time_s,
            "max_equality_violation": result.max_equality_violation,
            "min_inequality_residual": result.min_inequality_residual,
        },
    }
    # Refuse accidental serialization of NaN/Inf as non-standard JSON.
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    return 0 if result.success and math.isfinite(result.objective) else 2


if __name__ == "__main__":
    raise SystemExit(main())
