"""Dry-run simulation and fault injection commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    specs = [
        readonly("sim status", "Show simulation world status", "simulation", "sim_status"),
    ]
    for suffix in ("reset", "run", "pause", "step", "clear"):
        specs.append(
            command(
                f"sim {suffix}",
                f"{suffix.capitalize()} the Phase 1 simulation",
                "simulation",
                "sim_{}".format(suffix.replace("-", "_")),
                usage="sim step [SECONDS]" if suffix == "step" else "",
                capability=CommandPermission.SIMULATION,
                dry_run_only=True,
            )
        )
    for name in (
        "nominal",
        "transform-failure",
        "rc-loss",
        "pixhawk-timeout",
        "f446-overcurrent",
        "landing",
    ):
        specs.append(
            command(
                f"sim scenario {name}",
                f"Select the {name} scenario",
                "simulation",
                "sim_scenario",
                capability=CommandPermission.SIMULATION,
                dry_run_only=True,
            )
        )
    specs.append(
        command(
            "sim inject",
            "Inject a named simulated fault",
            "simulation",
            "sim_inject",
            usage="sim inject FAULT",
            capability=CommandPermission.SIMULATION,
            dry_run_only=True,
        )
    )
    return tuple(specs)
