"""Validate and copy a JSONL event log."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    lines = args.source.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON at line {number}: {exc}")
            return 2
        if not isinstance(value, dict):
            print(f"Line {number} is not a JSON object")
            return 2
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(args.source), str(args.destination))
    print(f"Exported {len(lines)} records to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
