"""Automatic landing commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandPermission, CommandSpec
from aerogo2.cli.commands._helpers import command, readonly
from aerogo2.common.enums import SystemState


def command_specs() -> Tuple[CommandSpec, ...]:
    return (
        readonly("autoland status", "Show automatic landing state", "landing", "query_controller"),
        readonly(
            "landing compliance",
            "Show calibrated foot contact and landing compliance progress",
            "landing",
            "query_controller",
        ),
        command(
            "autoland prepare",
            "Initialize controller and estimator without sending setpoints",
            "landing",
            "autoland_prepare",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.FLIGHT_MANUAL}),
            dry_run_only=True,
        ),
        command(
            "autoland start",
            "Start FakePixhawk-only automatic landing output",
            "landing",
            "autoland_start",
            capability=CommandPermission.SAFE_CONTROL,
            allowed_states=frozenset({SystemState.AUTO_LANDING_READY}),
            dry_run_only=True,
        ),
        command(
            "autoland abort",
            "Stop external setpoints and return to RadioMaster",
            "landing",
            "autoland_abort",
            aliases=("abort",),
            capability=CommandPermission.SAFETY_STOP,
            dry_run_only=True,
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
