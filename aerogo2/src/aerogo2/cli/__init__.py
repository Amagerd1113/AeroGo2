"""Interactive console building blocks.

Command modules and the shell depend on these abstractions; none of them expose
device bridges directly.
"""

from aerogo2.cli.command_models import (
    CommandContext,
    CommandDefinition,
    CommandInvocation,
    CommandMatch,
    CommandPermission,
    CommandSpec,
    ConfirmationPolicy,
    ParsedCommand,
    PermissionDecision,
    PermissionPolicy,
)
from aerogo2.cli.completer import AeroGo2Completer
from aerogo2.cli.confirmation import ConfirmationDecision, ConfirmationService
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.parser import CommandParser, ParsedArguments
from aerogo2.cli.registry import CommandNotFoundError, CommandRegistry
from aerogo2.cli.renderer import ConsoleRenderer

__all__ = [
    "AeroGo2Completer",
    "CommandContext",
    "CommandDefinition",
    "CommandHistory",
    "CommandInvocation",
    "CommandMatch",
    "CommandNotFoundError",
    "CommandParser",
    "CommandPermission",
    "CommandRegistry",
    "CommandSpec",
    "ConfirmationDecision",
    "ConfirmationPolicy",
    "ConfirmationService",
    "ConsoleRenderer",
    "ParsedArguments",
    "ParsedCommand",
    "PermissionDecision",
    "PermissionPolicy",
]
