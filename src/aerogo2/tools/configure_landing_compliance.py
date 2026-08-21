"""Atomically configure calibrated Go2 landing-compliance thresholds."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from aerogo2.common.config import load_config
from aerogo2.common.exceptions import AeroGo2Error


def configure_landing_compliance(
    config_path: Path,
    *,
    enabled: bool,
    thresholds: Optional[Tuple[int, int, int, int]] = None,
    minimum_contact_feet: int = 3,
    contact_confirm_s: float = 0.5,
    settle_s: float = 1.5,
) -> Path:
    """Validate and atomically replace one YAML overlay, returning its backup path."""

    path = config_path.resolve()
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    if enabled and thresholds is None:
        raise ValueError("Four calibrated thresholds are required when enabling")
    if thresholds is not None and any(item <= 0 or item > 32767 for item in thresholds):
        raise ValueError("Every calibrated threshold must be within 1..32767")
    if minimum_contact_feet < 1 or minimum_contact_feet > 4:
        raise ValueError("minimum_contact_feet must be within 1..4")
    if contact_confirm_s <= 0.0 or settle_s <= 0.0:
        raise ValueError("Confirmation and settle times must be positive")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        document: Dict[str, Any] = {}
    elif isinstance(loaded, Mapping):
        document = dict(loaded)
    else:
        raise ValueError("Top-level hardware configuration must be a mapping")
    existing_go2 = document.get("go2", {})
    if not isinstance(existing_go2, Mapping):
        raise ValueError("The go2 configuration section must be a mapping")
    go2 = dict(existing_go2)
    go2["landing_compliance_enabled"] = enabled
    if thresholds is not None:
        go2["foot_force_contact_thresholds"] = list(thresholds)
        go2["landing_contact_min_feet"] = minimum_contact_feet
        go2["landing_contact_confirm_s"] = contact_confirm_s
        go2["landing_compliance_settle_s"] = settle_s
    document["go2"] = go2

    candidate = path.parent / f".{path.name}.aerogo2-candidate"
    backup = path.parent / f"{path.name}.bak"
    try:
        candidate.write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
        load_config(candidate)
        shutil.copy2(path, backup)
        os.replace(candidate, path)
    finally:
        if candidate.exists():
            candidate.unlink()
    return backup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure calibrated AeroGo2 landing compliance",
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enable", action="store_true")
    mode.add_argument("--disable", action="store_true")
    parser.add_argument("--thresholds", nargs=4, type=int, metavar=("F0", "F1", "F2", "F3"))
    parser.add_argument("--minimum-contact-feet", type=int, default=3)
    parser.add_argument("--contact-confirm-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    raw_thresholds = args.thresholds
    thresholds = (
        None
        if raw_thresholds is None
        else (
            raw_thresholds[0],
            raw_thresholds[1],
            raw_thresholds[2],
            raw_thresholds[3],
        )
    )
    try:
        backup = configure_landing_compliance(
            args.config,
            enabled=bool(args.enable),
            thresholds=thresholds,
            minimum_contact_feet=args.minimum_contact_feet,
            contact_confirm_s=args.contact_confirm_s,
            settle_s=args.settle_s,
        )
    except (AeroGo2Error, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration not changed: {exc}", file=sys.stderr)
        return 2
    state = "enabled" if args.enable else "disabled"
    print(f"Landing compliance {state}; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
