"""Centralized command permissions.

The shell never decides safety permissions itself; it asks this policy through
SystemManager/CommandService.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, FrozenSet, Mapping

from aerogo2.common.enums import SystemState


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


READ_ONLY_ACTIONS: Final[FrozenSet[str]] = frozenset(
    {
        "status",
        "devices",
        "health",
        "rc",
        "pixhawk_status",
        "f446_status",
        "go2_status",
        "esc_status",
        "audit",
        "preflight",
        "config_show",
        "log_status",
        "sim_status",
    }
)

STATE_ACTIONS: Final[Mapping[str, FrozenSet[SystemState]]] = MappingProxyType(
    {
        "home_walk": frozenset({SystemState.BOOT_SAFE}),
        "manual_enter": frozenset(
            {
                SystemState.BOOT_SAFE,
                SystemState.WALK,
                SystemState.FLIGHT_READY,
                SystemState.TOUCHDOWN_VERIFY,
                SystemState.LANDING_COMPLIANT,
            }
        ),
        "manual_exit": frozenset({SystemState.MANUAL_POSITIONING}),
        "manual_confirm_walk": frozenset({SystemState.MANUAL_POSITIONING}),
        "manual_confirm_flight": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_threshold_adc": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_threshold_mv": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_timeout": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_blank": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_overms": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_mf": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_mr": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_limf": frozenset({SystemState.MANUAL_POSITIONING}),
        "f446_limr": frozenset({SystemState.MANUAL_POSITIONING}),
        "transform_flight": frozenset({SystemState.WALK}),
        "transform_walk": frozenset(
            {
                SystemState.FLIGHT_READY,
                SystemState.TOUCHDOWN_VERIFY,
                SystemState.LANDING_COMPLIANT,
            }
        ),
        "autoland_prepare": frozenset({SystemState.FLIGHT_MANUAL}),
        "autoland_start": frozenset({SystemState.AUTO_LANDING_READY}),
        "autoland_abort": frozenset(
            {
                SystemState.AUTO_LANDING_READY,
                SystemState.AUTO_LANDING,
            }
        ),
        "walk_permit": frozenset({SystemState.WALK}),
        "go2_confirm_lock": frozenset(
            {SystemState.GO2_JOINT_LOCK_WAIT, SystemState.FLIGHT_READY}
        ),
        "ground_arm_authorize": frozenset({SystemState.FLIGHT_READY}),
        "ground_arm_revoke": frozenset({SystemState.FLIGHT_READY}),
    }
)

PHASE_1_DISABLED_ACTIONS: Final[FrozenSet[str]] = frozenset(
    {
        "pixhawk_arm",
        "pixhawk_disarm",
        "pixhawk_motor_test",
        "f446_manual",
        "f446_configuration_write",
        "f446_disable",
        "f446_clear",
        "hardware_connect",
        "real_landing_setpoint",
    }
)


class PermissionPolicy:
    def decide(
        self,
        action: str,
        state: SystemState,
        maintenance_mode: bool = False,
    ) -> PermissionDecision:
        if action in READ_ONLY_ACTIONS:
            return PermissionDecision(True, "read-only")
        if action in PHASE_1_DISABLED_ACTIONS:
            return PermissionDecision(False, "not available in Phase 1")
        if action.startswith("maintenance_"):
            return PermissionDecision(False, "F446 maintenance is Phase 3")
        allowed_states = STATE_ACTIONS.get(action)
        if allowed_states is not None and state not in allowed_states:
            return PermissionDecision(
                False,
                f"{action} is not allowed in {state.name}",
            )
        if (
            action
            in {
                "manual_exit",
                "manual_confirm_walk",
                "manual_confirm_flight",
                "f446_threshold_adc",
                "f446_threshold_mv",
                "f446_timeout",
                "f446_blank",
                "f446_overms",
                "f446_mf",
                "f446_mr",
                "f446_limf",
                "f446_limr",
            }
            and not maintenance_mode
        ):
            return PermissionDecision(False, "F446 maintenance mode is required")
        return PermissionDecision(True, "allowed")
