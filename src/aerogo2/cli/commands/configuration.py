"""Configuration inspection commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("config show", "Show merged configuration", "configuration", "query_config"),
        readonly(
            "config get",
            "Get a dotted configuration key",
            "configuration",
            "config_get",
            usage="config get KEY",
        ),
        readonly("config validate", "Validate configuration", "configuration", "config_validate"),
        command(
            "config reload",
            "Reload configuration only when safe",
            "configuration",
            "config_reload",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        readonly(
            "config diff",
            "Show disk/runtime configuration differences",
            "configuration",
            "config_diff",
        ),
    )
