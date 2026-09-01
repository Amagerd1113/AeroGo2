"""Atomic configuration tool for calibrated F446 motion parameters."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.exceptions import AeroGo2Error


def configure_f446(
    config_path: Path,
    *,
    walk_duty: Optional[int] = None,
    flight_duty: Optional[int] = None,
    transform_timeout_s: Optional[float] = None,
    firmware_timeout_ms: Optional[int] = None,
    threshold_adc: Optional[int] = None,
    blanking_ms: Optional[int] = None,
    overcurrent_ms: Optional[int] = None,
) -> tuple[Path, AppConfig]:
    """Validate and atomically replace an F446 configuration overlay."""

    changes = {
        "walk_duty": walk_duty,
        "flight_duty": flight_duty,
        "transform_timeout_s": transform_timeout_s,
        "firmware_timeout_ms": firmware_timeout_ms,
        "automatic_stall_threshold_adc": threshold_adc,
        "stall_blanking_ms": blanking_ms,
        "stall_overcurrent_ms": overcurrent_ms,
    }
    if all(item is None for item in changes.values()):
        raise ValueError("At least one F446 parameter must be supplied")
    for label, duty in (("walk_duty", walk_duty), ("flight_duty", flight_duty)):
        if duty is not None and (isinstance(duty, bool) or not 1 <= duty <= 350):
            raise ValueError(f"{label} must be within 1..350")
    if transform_timeout_s is not None and not 0.1 <= transform_timeout_s <= 60.0:
        raise ValueError("transform_timeout_s must be within 0.1..60.0")
    if firmware_timeout_ms is not None and not 100 <= firmware_timeout_ms <= 60000:
        raise ValueError("firmware_timeout_ms must be within 100..60000")
    if threshold_adc is not None and not 0 <= threshold_adc <= 4095:
        raise ValueError("threshold_adc must be within 0..4095")
    if blanking_ms is not None and not 0 <= blanking_ms <= 5000:
        raise ValueError("blanking_ms must be within 0..5000")
    if overcurrent_ms is not None and not 10 <= overcurrent_ms <= 3000:
        raise ValueError("overcurrent_ms must be within 10..3000")

    path = config_path.resolve()
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        document: Dict[str, Any] = {}
    elif isinstance(loaded, Mapping):
        document = dict(loaded)
    else:
        raise ValueError("Top-level hardware configuration must be a mapping")
    existing_f446 = document.get("f446", {})
    if not isinstance(existing_f446, Mapping):
        raise ValueError("The f446 configuration section must be a mapping")
    f446 = dict(existing_f446)
    for key, item in changes.items():
        if item is not None:
            f446[key] = item
    # A host timeout update normally means one total 15 s limit. Keep the
    # local firmware timeout synchronized unless the operator supplied a
    # distinct, shorter local value explicitly.
    if transform_timeout_s is not None and firmware_timeout_ms is None:
        f446["firmware_timeout_ms"] = int(round(transform_timeout_s * 1000.0))
    document["f446"] = f446

    candidate = path.parent / f".{path.name}.aerogo2-f446-candidate"
    backup = path.parent / f"{path.name}.bak"
    validated: Optional[AppConfig] = None
    try:
        candidate.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        validated = load_config(candidate)
        shutil.copy2(path, backup)
        os.replace(candidate, path)
    finally:
        if candidate.exists():
            candidate.unlink()
    assert validated is not None
    return backup, validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically configure calibrated AeroGo2 F446 parameters",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--walk-duty", type=int)
    parser.add_argument("--flight-duty", type=int)
    parser.add_argument("--transform-timeout-s", type=float)
    parser.add_argument("--firmware-timeout-ms", type=int)
    parser.add_argument(
        "--threshold-adc",
        type=int,
        help="0 preserves the board value; nonzero is reapplied on every write-enabled connect",
    )
    parser.add_argument("--blanking-ms", type=int)
    parser.add_argument("--overcurrent-ms", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        backup, config = configure_f446(
            args.config,
            walk_duty=args.walk_duty,
            flight_duty=args.flight_duty,
            transform_timeout_s=args.transform_timeout_s,
            firmware_timeout_ms=args.firmware_timeout_ms,
            threshold_adc=args.threshold_adc,
            blanking_ms=args.blanking_ms,
            overcurrent_ms=args.overcurrent_ms,
        )
    except (AeroGo2Error, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration not changed: {exc}", file=sys.stderr)
        return 2
    f446 = config.f446
    print(
        "F446 configuration updated; "
        f"walk_duty={f446.walk_duty}; flight_duty={f446.flight_duty}; "
        f"host_timeout_s={f446.transform_timeout_s:g}; "
        f"firmware_timeout_ms={f446.firmware_timeout_ms}; "
        f"threshold_adc={f446.automatic_stall_threshold_adc}; "
        f"blanking_ms={f446.stall_blanking_ms}; "
        f"overcurrent_ms={f446.stall_overcurrent_ms}; backup={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
