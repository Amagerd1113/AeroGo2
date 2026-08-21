"""Device and health commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    specs = [
        readonly("devices", "Show all device connections", "devices", "query_status"),
        command(
            "connect all",
            "Connect all injected Phase 1 fake devices",
            "devices",
            "connect_all",
            capability=CommandPermission.SAFE_CONTROL,
        ),
        command(
            "disconnect all",
            "Disconnect every simulated device",
            "devices",
            "disconnect_all",
            capability=CommandPermission.SAFETY_STOP,
        ),
        readonly(
            "health",
            "Evaluate subsystem health",
            "devices",
            "health",
            usage="health [--watch SECONDS]",
            options=("--watch",),
        ),
    ]
    for device in ("pixhawk", "f446", "go2"):
        specs.append(
            command(
                f"connect {device}",
                f"Connect only the {device} fake",
                "devices",
                f"connect_{device}",
                capability=CommandPermission.SAFE_CONTROL,
            )
        )
        specs.append(
            command(
                f"disconnect {device}",
                f"Disconnect the {device} fake",
                "devices",
                f"disconnect_{device}",
                capability=CommandPermission.SAFETY_STOP,
            )
        )
    return tuple(specs)
