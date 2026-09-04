"""Build the complete, phase-gated AeroGo2 command tree."""

from aerogo2.cli.commands import (
    configuration,
    devices,
    faults,
    flight,
    general,
    go2_lowcmd,
    landing,
    logging_commands,
    monitoring,
    motor,
    preflight,
    simulation,
    transition,
    walking,
)
from aerogo2.cli.registry import CommandRegistry


def build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for module in (
        general,
        devices,
        monitoring,
        go2_lowcmd,
        transition,
        motor,
        walking,
        preflight,
        flight,
        landing,
        faults,
        configuration,
        logging_commands,
        simulation,
    ):
        registry.register_many(module.command_specs())
    return registry


__all__ = ["build_registry"]
