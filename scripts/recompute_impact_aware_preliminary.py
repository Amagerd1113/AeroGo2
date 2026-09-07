#!/usr/bin/env python3
"""Recompute an offline schema-4 preliminary candidate without hardware I/O.

The source is never modified.  With no ``--output`` the validated candidate
is printed to stdout.  ``--output`` must name a new YAML file beside the
source so config-relative pinned assets retain the same path meaning.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import yaml

from aerogo2.landing.impact_aware.preliminary import (
    PreliminaryModelError,
    load_preliminary_landing_model,
    recompute_preliminary_derived_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/impact_aware_preliminary.yaml"),
        help="Existing schema-4 source YAML (read-only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New candidate .yaml/.yml beside --config; existing files are refused",
    )
    return parser


def _encode_candidate(document: object) -> str:
    body = yaml.safe_dump(
        document,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return (
        "# 由 recompute_impact_aware_preliminary.py 生成的离线候选配置。\n"
        "# 已重算派生量，但不会解除任何硬件门禁；请人工审阅后再决定是否采用。\n" + body
    )


def _strict_self_check(encoded: str, source_directory: Path) -> None:
    """Validate serialized bytes from the same directory as pinned assets."""

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".aerogo2-preliminary-candidate-",
            suffix=".yaml",
            dir=source_directory,
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary_path = Path(handle.name)
        checked = load_preliminary_landing_model(
            temporary_path,
            allow_provisional=True,
            for_hardware=False,
        )
        if checked.hardware_output_permitted:
            raise PreliminaryModelError("self-check unexpectedly granted hardware output")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _resolve_new_output(source: Path, requested: Path) -> Path:
    if requested.suffix.lower() not in {".yaml", ".yml"}:
        raise PreliminaryModelError("--output must use a .yaml or .yml suffix")
    try:
        parent = requested.expanduser().parent.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise PreliminaryModelError("--output parent directory must already exist") from exc
    if parent != source.parent:
        raise PreliminaryModelError(
            "--output must be beside --config so pinned relative asset paths cannot change"
        )
    destination = parent / requested.name
    if destination == source:
        raise PreliminaryModelError("in-place overwrite of --config is prohibited")
    if destination.exists() or destination.is_symlink():
        raise PreliminaryModelError("--output already exists; overwrite is prohibited")
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = args.config.expanduser().resolve(strict=True)
        if not source.is_file():
            raise PreliminaryModelError("--config must resolve to a regular file")
        document = recompute_preliminary_derived_document(source)
        encoded = _encode_candidate(document)
        # Validate exactly the serialized representation before exposing it.
        _strict_self_check(encoded, source.parent)
        if args.output is None:
            print(encoded, end="")
            return 0
        destination = _resolve_new_output(source, args.output)
        created = False
        try:
            with destination.open("x", encoding="utf-8", newline="\n") as handle:
                created = True
                handle.write(encoded)
            # Re-read the final path through the strict loader as the final
            # commit check.  On failure, remove only the file this invocation
            # created; pre-existing paths can never reach this point.
            load_preliminary_landing_model(
                destination,
                allow_provisional=True,
                for_hardware=False,
            )
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        print(str(destination))
        return 0
    except (OSError, TypeError, ValueError, PreliminaryModelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
