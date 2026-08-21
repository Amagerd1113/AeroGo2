"""Go2 walking permission commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("walk status", "Show walking permission", "walking", "query_go2"),
        readonly("walk permit", "Audit whether walking is permitted", "walking", "query_guards"),
        command(
            "walk stop",
            "Request a high-level Go2 stop",
            "walking",
            "walk_stop",
            capability=CommandPermission.SAFETY_STOP,
            requires_hardware_write=True,
        ),
        command(
            "walk stand",
            "Request Go2 stand in WALK",
            "walking",
            "walk_stand",
            capability=CommandPermission.SAFE_CONTROL,
            requires_hardware_write=True,
        ),
    )
