"""Command-facing facade around SystemManager.

All mutating console commands terminate here before reaching any device bridge.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from aerogo2.common.enums import CommandStatus, Configuration
from aerogo2.common.results import CommandResult, OperationResult
from aerogo2.manager.permissions import PermissionPolicy


class CommandService:
    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._permissions = PermissionPolicy()

    async def run(self, action: str, args: Sequence[str]) -> CommandResult:
        decision = self._permissions.decide(
            action,
            self._manager.state,
            self._manager.snapshot.maintenance_mode,
        )
        if not decision.allowed:
            return CommandResult(CommandStatus.UNAVAILABLE, decision.reason)

        f446_actions = {
            "manual_enter",
            "manual_exit",
            "manual_confirm_walk",
            "manual_confirm_flight",
            "f446_threshold_adc",
            "f446_threshold_mv",
            "f446_mf",
            "f446_mr",
            "f446_limf",
            "f446_limr",
        }
        if action in f446_actions:
            result = await self._run_f446_action(action, args)
            return CommandResult(
                CommandStatus.SUCCESS if result.ok else CommandStatus.REJECTED,
                result.message,
                result.data,
            )

        device_actions = {
            "connect_pixhawk": ("connect_device", "pixhawk"),
            "connect_f446": ("connect_device", "f446"),
            "connect_go2": ("connect_device", "go2"),
            "disconnect_pixhawk": ("disconnect_device", "pixhawk"),
            "disconnect_f446": ("disconnect_device", "f446"),
            "disconnect_go2": ("disconnect_device", "go2"),
        }
        device_action = device_actions.get(action)
        if device_action is not None:
            method_name, device_name = device_action
            handler = getattr(self._manager, method_name, None)
            if handler is None:
                return self._unavailable(action)
            result = cast(OperationResult, await handler(device_name))
        else:
            handler_names = {
                "connect_all": "connect_all",
                "disconnect_all": "disconnect_all",
                "transform_flight": "_transform_flight",
                "home_walk": "_home_walk",
                "transform_walk": "_transform_walk",
                "transform_stop": "stop_transform_motion",
                "stop": "stop_supervised",
                "walk_stop": "walk_stop",
                "walk_stand": "walk_stand",
                "autoland_prepare": "prepare_autoland",
                "autoland_start": "start_autoland",
                "autoland_abort": "abort_autoland",
                "controller_reset": "reset_controller",
                "clear_fault": "clear_fault",
                "config_diff": "config_diff",
                "config_reload": "reload_config",
                "ground_arm_authorize": "authorize_ground_arm",
                "ground_arm_revoke": "revoke_ground_arm",
            }
            handler_name = handler_names.get(action)
            if handler_name is None:
                return self._unavailable(action)
            handler = (
                getattr(self, handler_name)
                if handler_name.startswith("_")
                else getattr(self._manager, handler_name, None)
            )
            if handler is None:
                return self._unavailable(action)
            result = cast(OperationResult, await handler())
        return CommandResult(
            CommandStatus.SUCCESS if result.ok else CommandStatus.REJECTED,
            result.message,
            result.data,
        )

    async def _run_f446_action(
        self,
        action: str,
        args: Sequence[str],
    ) -> OperationResult:
        if action == "manual_enter":
            return cast(
                OperationResult,
                await self._manager.enter_manual_positioning(operator_confirmed=True),
            )
        if action == "manual_exit":
            return cast(OperationResult, await self._manager.exit_manual_positioning())
        if action == "manual_confirm_walk":
            return cast(
                OperationResult,
                await self._manager.confirm_manual_configuration(
                    Configuration.WALK,
                    operator_confirmed=True,
                ),
            )
        if action == "manual_confirm_flight":
            return cast(
                OperationResult,
                await self._manager.confirm_manual_configuration(
                    Configuration.FLIGHT,
                    operator_confirmed=True,
                ),
            )
        if len(args) != 1:
            return OperationResult.failure(
                "INVALID_ARGUMENTS",
                f"{action} requires one integer argument",
            )
        try:
            value = int(args[0], 10)
        except ValueError:
            return OperationResult.failure(
                "INVALID_ARGUMENTS",
                f"{action} requires one integer argument",
            )
        if action == "f446_threshold_adc":
            return cast(
                OperationResult,
                await self._manager.set_f446_current_threshold(value),
            )
        if action == "f446_threshold_mv":
            return cast(
                OperationResult,
                await self._manager.set_f446_current_threshold(
                    value,
                    millivolts=True,
                ),
            )
        operation_by_action = {
            "f446_mf": "mf",
            "f446_mr": "mr",
            "f446_limf": "limf",
            "f446_limr": "limr",
        }
        operation = operation_by_action.get(action)
        if operation is None:
            return OperationResult.failure(
                "UNKNOWN_F446_ACTION",
                f"Unsupported action {action}",
            )
        return cast(
            OperationResult,
            await self._manager.start_f446_maintenance_motion(operation, value),
        )

    @staticmethod
    def _unavailable(action: str) -> CommandResult:
        return CommandResult(
            CommandStatus.UNAVAILABLE,
            f"Command action '{action}' is registered but not executable in Phase 1",
        )

    async def _home_walk(self) -> OperationResult:
        return cast(
            OperationResult,
            await self._manager.request_home_walk(operator_confirmed=True),
        )

    async def _transform_flight(self) -> OperationResult:
        return cast(
            OperationResult,
            await self._manager.request_transform_flight(operator_confirmed=True),
        )

    async def _transform_walk(self) -> OperationResult:
        return cast(
            OperationResult,
            await self._manager.request_transform_walk(operator_confirmed=True),
        )

    def query(self, name: str) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self._manager.query(name))
