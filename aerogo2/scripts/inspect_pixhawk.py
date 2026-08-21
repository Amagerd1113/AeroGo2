"""Offline-only Pixhawk fixture inspection for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args(argv)
    if args.fixture is None:
        print(
            "Phase 1 refuses live MAVLink access. Provide --fixture PATH "
            "to inspect captured JSON offline."
        )
        return 2
    data = json.loads(args.fixture.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
