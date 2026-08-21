"""F446 status, guarded manual positioning, and automatic limit commands."""

from typing import Tuple

from aerogo2.cli.command_models import (
    CommandPermission,
    CommandSpec,
    ConfirmationPolicy,
)
from aerogo2.cli.commands._helpers import command, readonly
from aerogo2.common.enums import SystemState


def command_specs() -> Tuple[CommandSpec, ...]:
    specs = [
        readonly("motor status", "Show F446 status", "motor", "query_f446"),
        readonly("motor current", "Show F446 current channels", "motor", "query_f446"),
        readonly("motor parameters", "Show F446 configured parameters", "motor", "query_f446"),
        command(
            "motor to-flight",
            "Alias for guarded FLIGHT transformation",
            "motor",
            "transform_flight",
            capability=CommandPermission.SAFE_CONTROL,
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.exact(
                "TRANSFORM_TO_FLIGHT",
                prompt="Confirm the Go2 original remote is no longer in use; type the exact phrase",
                warning="Before FLIGHT morphology, confirm the Go2 original remote has stopped being used and the robot is stationary.",
            ),
        ),
        command(
            "motor to-walk",
            "Alias for guarded WALK transformation",
            "motor",
            "transform_walk",
            capability=CommandPermission.SAFE_CONTROL,
            requires_hardware_write=True,
            confirmation=ConfirmationPolicy.exact("TRANSFORM_TO_WALK"),
        ),
        command(
            "motor stop",
            "Stop the morphology mechanism",
            "motor",
            "transform_stop",
            capability=CommandPermission.SAFETY_STOP,
            requires_hardware_write=True,
        ),
    ]
    manual_state = frozenset({SystemState.MANUAL_POSITIONING})
    entry_states = frozenset(
        {
            SystemState.BOOT_SAFE,
            SystemState.WALK,
            SystemState.FLIGHT_READY,
        }
    )
    specs.extend(
        (
            command(
                "motor maintenance enter",
                "Enter guarded F446 manual positioning",
                "motor",
                "manual_enter",
                aliases=("manual enter",),
                allowed_states=entry_states,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.two_stage(
                    "ENTER_F446_MANUAL",
                    warning=(
                        "Propellers must be removed; Pixhawk must be disarmed; "
                        "Go2 and every ESC must be stationary."
                    ),
                ),
            ),
            command(
                "motor maintenance exit",
                "Stop F446 and leave without accepting a configuration",
                "motor",
                "manual_exit",
                aliases=("manual exit",),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
            ),
            command(
                "motor confirm walk",
                "Accept the stopped manual position as contracted WALK",
                "motor",
                "manual_confirm_walk",
                aliases=("confirm walk",),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.exact("CONFIRM_MANUAL_WALK"),
            ),
            command(
                "motor confirm flight",
                "Accept the stopped manual position as expanded FLIGHT",
                "motor",
                "manual_confirm_flight",
                aliases=("confirm flight",),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.exact("CONFIRM_MANUAL_FLIGHT"),
            ),
            command(
                "motor threshold",
                "Set and read back the automatic stall threshold in ADC counts",
                "motor",
                "f446_threshold_adc",
                usage="motor threshold ADC",
                aliases=("thr",),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.exact("CHANGE_F446_THRESHOLD"),
            ),
            command(
                "motor threshold-mv",
                "Set and read back the automatic stall threshold in millivolts",
                "motor",
                "f446_threshold_mv",
                usage="motor threshold-mv MV",
                aliases=("thrmv",),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.exact("CHANGE_F446_THRESHOLD"),
            ),
        )
    )
    for operation in ("mf", "mr", "limf", "limr"):
        automatic = operation.startswith("lim")
        warning = (
            "F446 local stall-current detection is active."
            if automatic
            else (
                "No F446 local limit stop; AeroGo2 applies host timeout and "
                "absolute current interlocks. Use stop at the observed position."
            )
        )
        specs.append(
            command(
                f"motor {operation}",
                warning,
                "motor",
                f"f446_{operation}",
                usage=f"motor {operation} DUTY",
                aliases=(operation,),
                allowed_states=manual_state,
                capability=CommandPermission.F446_MAINTENANCE,
                requires_maintenance=True,
                requires_hardware_write=True,
                confirmation=ConfirmationPolicy.two_stage(
                    "RUN_MANUAL_MOTOR",
                    warning=warning,
                ),
            )
        )
    return tuple(specs)
