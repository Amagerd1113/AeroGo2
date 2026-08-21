"""General console commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly(
            "help",
            "List commands or show help for a command",
            "general",
            "help",
            usage="help [COMMAND]",
        ),
        readonly("version", "Show the AeroGo2 version", "general", "version"),
        readonly("clear", "Clear the terminal display", "general", "clear_screen"),
        readonly("history", "Show command history", "general", "history"),
        command(
            "exit",
            "Exit the console without arming/disarming anything",
            "general",
            "exit",
            aliases=("quit",),
        ),
    )
