"""Audit and preflight commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandSpec
from aerogo2.cli.commands._helpers import readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    paths = (
        "audit",
        "audit pixhawk",
        "audit f446",
        "audit rc",
        "audit configuration",
        "preflight",
        "preflight flight",
        "preflight home-walk",
        "preflight manual-position",
        "preflight transform-flight",
        "preflight transform-walk",
        "preflight autoland",
        "check invariant",
        "check communication",
        "check sensors",
    )
    return tuple(
        readonly(path, f"Run {path} safety checks", "preflight", "query_guards") for path in paths
    )
