"""Morphology transition commands."""

from typing import Tuple

from aerogo2.cli.command_models import (
    CommandPermission,
    CommandSpec,
    ConfirmationPolicy,
)
from aerogo2.cli.commands._helpers import command, readonly
from aerogo2.common.enums import SystemState


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("transform status", "Show morphology state", "transition", "query_status"),
        command(
            "transform home-walk",
            "Home UNKNOWN F446 morphology to the verified WALK limit",
            "transition",
            "home_walk",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.BOOT_SAFE}),
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.two_stage(
                "HOME_F446_TO_WALK",
                prompt="Confirm props are removed, ESC telemetry is online at zero RPM, and the linkage path is clear",
                warning="F446 will move toward the configured WALK limit and stop only on verified current limit or timeout.",
            ),
        ),
        command(
            "transform flight",
            "Run guarded WALK-to-FLIGHT transformation",
            "transition",
            "transform_flight",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.WALK}),
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.exact(
                "TRANSFORM_TO_FLIGHT",
                prompt="Confirm the Go2 original remote is no longer in use; type the exact phrase",
                warning="Before FLIGHT morphology, confirm the Go2 original remote has stopped being used and the robot is stationary.",
            ),
        ),
        command(
            "transform walk",
            "Run guarded FLIGHT-to-WALK transformation",
            "transition",
            "transform_walk",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset(
                {
                    SystemState.FLIGHT_READY,
                    SystemState.FLIGHT_MANUAL,
                    SystemState.TOUCHDOWN_VERIFY,
                    SystemState.LANDING_COMPLIANT,
                }
            ),
            confirmation=ConfirmationPolicy.exact("TRANSFORM_TO_WALK"),
            requires_hardware_write=True,
        ),
        command(
            "transform stop",
            "Stop F446 motion and external setpoints, never rotors",
            "transition",
            "transform_stop",
            capability=CommandPermission.SAFETY_STOP,
            requires_hardware_write=True,
        ),
    )
