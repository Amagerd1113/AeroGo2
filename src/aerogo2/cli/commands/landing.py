"""Automatic landing commands."""

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
        readonly("autoland status", "Show automatic landing state", "landing", "query_controller"),
        readonly(
            "autoland hardware",
            "Show real-hardware autoland and MPC capability gates",
            "landing",
            "query_controller",
        ),
        readonly(
            "touchdown status",
            "Show airborne latch and touchdown confirmation progress",
            "landing",
            "query_controller",
        ),
        readonly(
            "landing impact",
            "Show current force, short-window impulse, average, and peaks",
            "landing",
            "query_controller",
        ),
        readonly(
            "landing impact history",
            "Show recent force samples and peak history",
            "landing",
            "query_controller",
        ),
        readonly(
            "landing compliance",
            "Show calibrated foot contact and landing compliance progress",
            "landing",
            "query_controller",
        ),
        command(
            "autoland prepare",
            "Prepare the legacy automatic landing for this flight",
            "landing",
            "autoland_prepare",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.FLIGHT_MANUAL}),
            requires_hardware_write=True,
        ),
        command(
            "autoland prepare mpc",
            "Confirm and prepare Impact-Aware/MPC landing for this flight",
            "landing",
            "autoland_prepare_mpc",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.FLIGHT_MANUAL}),
            confirmation=ConfirmationPolicy.exact(
                "CONFIRM_MPC_AUTOLAND",
                prompt="Confirm Impact-Aware/MPC landing for this flight",
                warning=(
                    "This selects the guarded Impact-Aware/MPC recovery path only for "
                    "the current automatic-landing attempt."
                ),
            ),
            requires_hardware_write=True,
        ),
        command(
            "autoland start",
            "Start guarded automatic landing output",
            "landing",
            "autoland_start",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.AUTO_LANDING_READY}),
            requires_hardware_write=True,
        ),
        command(
            "autoland abort",
            "Stop external setpoints and return to RadioMaster",
            "landing",
            "autoland_abort",
            aliases=("abort",),
            capability=CommandPermission.SAFETY_STOP,
        ),
        readonly("controller status", "Show controller status", "landing", "query_controller"),
        readonly("controller timing", "Show controller timing", "landing", "query_controller"),
        readonly("controller inputs", "Show controller inputs", "landing", "query_controller"),
        readonly("controller output", "Show last controller output", "landing", "query_controller"),
        command(
            "controller reset",
            "Reset the inactive Phase 1 controller",
            "landing",
            "controller_reset",
            capability=CommandPermission.SAFE_CONTROL,
            dry_run_only=True,
        ),
    )
