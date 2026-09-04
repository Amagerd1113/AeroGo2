"""Offline audit of the algebraic rotor transport boundary.

This script performs no device discovery and has no hardware transport.  It is
intended for replaying identified thrust-domain values before an FC residual
adapter is enabled.  It does not solve the MPC: at partial gain its target
argument is transport provenance reconstructed from an already optimized
applied-total command.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)


def _vector4(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected four comma-separated numbers") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected exactly four comma-separated numbers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit u_fc + kappa*(u_transport_target-u_fc) without hardware",
    )
    parser.add_argument("--gain", type=float, default=0.0, help="kappa in the hard range 0..1")
    parser.add_argument("--baseline", type=_vector4, required=True, metavar="T0,T1,T2,T3")
    parser.add_argument(
        "--transport-target",
        type=_vector4,
        required=True,
        metavar="T0,T1,T2,T3",
        help="algebraic transport target, not a partial-gain MPC solution",
    )
    parser.add_argument("--thrust-min", type=_vector4, required=True, metavar="T0,T1,T2,T3")
    parser.add_argument("--thrust-max", type=_vector4, required=True, metavar="T0,T1,T2,T3")
    parser.add_argument(
        "--max-correction",
        type=_vector4,
        required=True,
        metavar="D0,D1,D2,D3",
    )
    parser.add_argument("--gain-rise-per-s", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True, help="control interval in seconds")
    parser.add_argument(
        "--unhealthy",
        action="store_true",
        help="show fail-closed removal of the MPC residual",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = RotorCorrectionSafetyConfig(
        target_gain=args.gain,
        thrust_min_n=args.thrust_min,
        thrust_max_n=args.thrust_max,
        maximum_correction_n=args.max_correction,
        maximum_gain_rise_per_s=args.gain_rise_per_s,
    )
    result = RotorCorrectionBlender(config).blend(
        args.baseline,
        args.transport_target,
        args.dt,
        healthy=not args.unhealthy,
    )
    print(
        json.dumps(
            {
                "valid": result.valid,
                "reason": result.reason,
                "baseline_thrusts_n": result.baseline_thrusts_n,
                "transport_target_thrusts_n": result.transport_target_thrusts_n,
                "transport_raw_correction_n": result.transport_raw_correction_n,
                "applied_residual_thrusts_n": result.applied_residual_thrusts_n,
                "applied_total_thrusts_n": result.applied_total_thrusts_n,
                "transport_target_semantics": result.transport_target_semantics,
                "requested_gain": result.requested_gain,
                "applied_gain": result.applied_gain,
                "headroom_gain": result.headroom_gain,
                "headroom_limited": result.headroom_limited,
                "notice": (
                    "0 means FC baseline; 1 removes gain attenuation but other "
                    "identified safety constraints still apply. "
                    "No nonzero gain is universally safe."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
