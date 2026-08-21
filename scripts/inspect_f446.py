"""Offline-only F446 parser inspection for Phase 1 fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from aerogo2.bridges.f446_parser import F446TextParser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args(argv)
    if args.fixture is None:
        print(
            "Phase 1 refuses live F446 serial access. Provide --fixture PATH "
            "to inspect captured text offline."
        )
        return 2
    parser_impl = F446TextParser()
    events = parser_impl.feed(args.fixture.read_bytes())
    events += parser_impl.flush_eof()
    for event in events:
        print(f"{event.event_type.value}: {event.line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
