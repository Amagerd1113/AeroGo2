"""Flight readiness and one-shot ground authorization commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec, ConfirmationPolicy
from aerogo2.cli.commands._helpers import command, readonly
from aerogo2.common.enums import SystemState


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("flight status", "Show flight readiness", "flight", "query_pixhawk"),
        readonly("flight enable-check", "Audit CH5 flight enable", "flight", "query_guards"),
        readonly("flight ready", "Audit FLIGHT_READY requirements", "flight", "query_guards"),
        readonly(
            "flight auth-status",
            "Show the one-shot ground Arm authorization",
            "flight",
            "query_ground_arm_authorization",
        ),
        command(
            "flight authorize",
            "Authorize one RadioMaster CH5 Arm attempt for 30 seconds",
            "flight",
            "ground_arm_authorize",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.FLIGHT_READY}),
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.exact(
                "AUTHORIZE_FLIGHT",
                prompt="Keep RadioMaster CH5 LOW, clear the propellers, then type the exact phrase",
                warning=(
                    "This does not Arm. It opens a 30-second one-shot gate; only a later "
                    "RadioMaster CH5 LOW-to-HIGH edge may request normal ArduPilot Arm."
                ),
            ),
        ),
        command(
            "flight revoke",
            "Cancel an unused ground Arm authorization",
            "flight",
            "ground_arm_revoke",
            capability=CommandPermission.SAFETY_STOP,
            allowed_states=frozenset({SystemState.FLIGHT_READY}),
            requires_hardware_write=True,
        ),
    )
