"""Fault inspection and recovery commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("faults", "Show active manager faults", "faults", "query_faults"),
        readonly("faults active", "Show active manager faults", "faults", "query_faults"),
        readonly("faults history", "Show fault history", "faults", "query_faults"),
        readonly(
            "faults explain",
            "Explain a safety violation code",
            "faults",
            "query_faults",
            usage="faults explain CODE",
        ),
        command(
            "clear-fault",
            "Clear only inactive manager-level fault latches",
            "faults",
            "clear_fault",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        command(
            "stop",
            "Supervised stop of F446, Go2, and automatic setpoints; never rotors",
            "faults",
            "stop",
            aliases=("s",),
            capability=CommandPermission.SAFETY_STOP,
            requires_hardware_write=True,
        ),
    )
