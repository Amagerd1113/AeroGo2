"""JSONL logging commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("log status", "Show JSONL logger status", "logging", "log_status"),
        command(
            "log start",
            "Start JSONL logging",
            "logging",
            "log_start",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        command(
            "log stop",
            "Stop JSONL logging",
            "logging",
            "log_stop",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        command(
            "log mark",
            "Add an operator marker",
            "logging",
            "log_mark",
            usage="log mark TEXT",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        readonly("log tail", "Show recent JSONL records", "logging", "log_tail"),
        command(
            "log export",
            "Export the current JSONL file",
            "logging",
            "log_export",
            usage="log export PATH",
            capability=CommandPermission.SAFE_CONTROL,
        ),
    )
