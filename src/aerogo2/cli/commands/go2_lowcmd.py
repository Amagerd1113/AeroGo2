"""Guarded Go2 LowCmd ownership commands.

These commands transfer ownership only.  They intentionally expose no route
for submitting or activating MPC joint targets.
"""

from typing import Tuple

from aerogo2.cli.command_models import (
    CommandPermission,
    CommandSpec,
    ConfirmationPolicy,
)
from aerogo2.cli.commands._helpers import command, readonly
from aerogo2.common.enums import SystemState

_GROUND_TRANSFER_PROMPT = (
    "Confirm the whole vehicle is mechanically supported, all propellers are "
    "removed, Pixhawk reports DISARMED and LANDED, and all four ESCs report "
    "exactly 0 RPM"
)

_GROUND_TRANSFER_WARNING = (
    "DANGER: Go2 LowCmd authority transfer can move joints or let the vehicle "
    "fall. Keep the whole vehicle mechanically supported and personnel clear; "
    "remove every propeller; verify Pixhawk DISARMED/LANDED and all four ESCs "
    "at exactly 0 RPM. The manager will independently re-check these facts."
)


def command_specs() -> Tuple[CommandSpec, ...]:
    acquire_states = frozenset({SystemState.FLIGHT_READY})
    release_states = frozenset(
        {
            SystemState.BOOT_SAFE,
            SystemState.FLIGHT_READY,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.LANDING_COMPLIANT,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        }
    )
    return (
        readonly(
            "go2 lowcmd status",
            "Show the exclusive Go2 LowCmd owner and watchdog status",
            "go2",
            "go2_lowcmd_status",
        ),
        command(
            "go2 lowcmd acquire",
            "Acquire the sole continuous Go2 LowCmd writer in ground safe-hold",
            "go2",
            "go2_lowcmd_acquire",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=acquire_states,
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.two_stage(
                "ACQUIRE_GO2_LOWCMD_SUPPORTED_DISARMED_LANDED_4ESC_ZERO_PROPS_REMOVED",
                prompt=_GROUND_TRANSFER_PROMPT,
                warning=_GROUND_TRANSFER_WARNING,
            ),
        ),
        command(
            "go2 lowcmd release",
            "Release Go2 LowCmd only through a verified ground authority handback",
            "go2",
            "go2_lowcmd_release",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=release_states,
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.two_stage(
                "RELEASE_GO2_LOWCMD_SUPPORTED_DISARMED_LANDED_4ESC_ZERO_PROPS_REMOVED",
                prompt=_GROUND_TRANSFER_PROMPT,
                warning=_GROUND_TRANSFER_WARNING,
            ),
        ),
    )


__all__ = ["command_specs"]
