"""CLI reachability and safety gates for Go2 LowCmd ownership transfer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from aerogo2.cli.commands import build_registry
from aerogo2.cli.confirmation import ConfirmationService, ScriptedConfirmationReader
from aerogo2.cli.dispatcher import CommandDispatcher
from aerogo2.cli.registry import CommandNotFoundError
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    CommandStatus,
    ConfirmationLevel,
    RuntimeMode,
    SystemState,
)
from aerogo2.common.models import LandingCommand, SystemSnapshot
from aerogo2.common.results import OperationResult
from aerogo2.manager.command_service import CommandService

ACQUIRE_PHRASE = "ACQUIRE_GO2_LOWCMD_SUPPORTED_DISARMED_LANDED_4ESC_ZERO_PROPS_REMOVED"
RELEASE_PHRASE = "RELEASE_GO2_LOWCMD_SUPPORTED_DISARMED_LANDED_4ESC_ZERO_PROPS_REMOVED"


class _Manager:
    def __init__(
        self,
        config: AppConfig,
        *,
        state: SystemState = SystemState.FLIGHT_READY,
        runtime_mode: RuntimeMode = RuntimeMode.DRY_RUN,
    ) -> None:
        self.config = config
        self.runtime_mode = runtime_mode
        self.snapshot = SystemSnapshot(timestamp=1.0, state=state)
        self.calls: List[Tuple[str, Mapping[str, Any]]] = []

    @property
    def state(self) -> SystemState:
        return self.snapshot.state

    @property
    def violations(self) -> Tuple[Any, ...]:
        return ()

    @property
    def last_landing_command(self) -> LandingCommand:
        return LandingCommand(timestamp=self.snapshot.timestamp)

    def query(self, name: str) -> Dict[str, Any]:
        assert name == "go2 status"
        return {
            "go2": {
                "low_level_status": {
                    "ownership_state": "HOLDING",
                    "owner_epoch": 7,
                    "writer_alive": True,
                }
            }
        }

    async def acquire_go2_low_level_control(
        self,
        *,
        operator_confirmed: bool,
        robot_supported: bool,
    ) -> OperationResult:
        self.calls.append(
            (
                "acquire",
                {
                    "operator_confirmed": operator_confirmed,
                    "robot_supported": robot_supported,
                },
            )
        )
        return OperationResult.success("acquired")

    async def release_go2_low_level_control(
        self,
        *,
        operator_confirmed: bool,
        robot_supported: bool,
        reason: str,
    ) -> OperationResult:
        self.calls.append(
            (
                "release",
                {
                    "operator_confirmed": operator_confirmed,
                    "robot_supported": robot_supported,
                    "reason": reason,
                },
            )
        )
        return OperationResult.success("released")


def _dispatcher(
    manager: _Manager,
    responses: Tuple[str, ...] = (),
    warnings: Optional[List[str]] = None,
) -> Tuple[CommandDispatcher, ScriptedConfirmationReader]:
    reader = ScriptedConfirmationReader(responses)
    warning_sink: List[str] = [] if warnings is None else warnings
    confirmation = ConfirmationService(reader, warning_sink.append)
    dispatcher = CommandDispatcher(
        build_registry(),
        SimpleNamespace(manager=manager),
        confirmation=confirmation,
    )
    return dispatcher, reader


def test_lowcmd_paths_resolve_but_no_activation_command_is_registered() -> None:
    registry = build_registry()

    assert registry.resolve("go2 lowcmd status").spec.action == "go2_lowcmd_status"
    assert registry.resolve("go2 lowcmd acquire").spec.action == "go2_lowcmd_acquire"
    assert registry.resolve("go2 lowcmd release").spec.action == "go2_lowcmd_release"
    with pytest.raises(CommandNotFoundError):
        registry.resolve("go2 lowcmd activate")


@pytest.mark.parametrize(
    ("name", "phrase"),
    [
        ("go2 lowcmd acquire", ACQUIRE_PHRASE),
        ("go2 lowcmd release", RELEASE_PHRASE),
    ],
)
def test_lowcmd_transfer_metadata_requires_explicit_two_stage_ground_acknowledgement(
    name: str,
    phrase: str,
) -> None:
    spec = build_registry().get(name)

    assert spec.permission.requires_hardware_write
    assert spec.confirmation.level is ConfirmationLevel.TWO_STAGE
    assert spec.confirmation.exact_phrase == phrase
    combined = f"{spec.confirmation.prompt} {spec.confirmation.warning}".lower()
    for required in (
        "mechanically supported",
        "propeller",
        "disarmed",
        "landed",
        "four esc",
        "exactly 0 rpm",
        "danger",
    ):
        assert required in combined


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "phrase", "expected_call"),
    [
        ("go2 lowcmd acquire", ACQUIRE_PHRASE, "acquire"),
        ("go2 lowcmd release", RELEASE_PHRASE, "release"),
    ],
)
async def test_lowcmd_transfer_dispatches_only_after_both_confirmations(
    app_config: AppConfig,
    line: str,
    phrase: str,
    expected_call: str,
) -> None:
    manager = _Manager(app_config)
    warnings: List[str] = []
    dispatcher, reader = _dispatcher(manager, ("yes", phrase), warnings)

    outcome = await dispatcher.dispatch(line, render=False)

    assert outcome.result.status is CommandStatus.SUCCESS
    assert manager.calls[0][0] == expected_call
    assert manager.calls[0][1]["operator_confirmed"] is True
    assert manager.calls[0][1]["robot_supported"] is True
    assert len(reader.prompts) == 2
    assert phrase in reader.prompts[1]
    assert len(warnings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [("no",), ("yes", "WRONG_EXACT_PHRASE")])
async def test_lowcmd_acquire_rejection_never_reaches_manager(
    app_config: AppConfig,
    responses: Tuple[str, ...],
) -> None:
    manager = _Manager(app_config)
    dispatcher, _ = _dispatcher(manager, responses)

    outcome = await dispatcher.dispatch("go2 lowcmd acquire", render=False)

    assert outcome.result.status is CommandStatus.REJECTED
    assert manager.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_mode",
    [RuntimeMode.HARDWARE_READONLY, RuntimeMode.HARDWARE],
)
async def test_lowcmd_transfer_is_denied_when_hardware_writes_are_locked(
    app_config: AppConfig,
    runtime_mode: RuntimeMode,
) -> None:
    manager = _Manager(app_config, runtime_mode=runtime_mode)
    dispatcher, reader = _dispatcher(manager, ("yes", ACQUIRE_PHRASE))

    outcome = await dispatcher.dispatch("go2 lowcmd acquire", render=False)

    assert outcome.result.status is CommandStatus.UNAVAILABLE
    assert outcome.result.data["code"] == "HARDWARE_WRITE_DISABLED"
    assert reader.prompts == []
    assert manager.calls == []


@pytest.mark.asyncio
async def test_lowcmd_status_is_read_only_and_scoped_to_owner_telemetry(
    app_config: AppConfig,
) -> None:
    manager = _Manager(app_config, state=SystemState.BOOT_SAFE)
    dispatcher, reader = _dispatcher(manager)

    outcome = await dispatcher.dispatch("go2 lowcmd status", render=False)

    assert outcome.result.status is CommandStatus.SUCCESS
    assert outcome.result.data == {
        "low_level_status": {
            "ownership_state": "HOLDING",
            "owner_epoch": 7,
            "writer_alive": True,
        }
    }
    assert reader.prompts == []
    assert manager.calls == []


@pytest.mark.asyncio
async def test_command_service_rejects_surplus_lowcmd_arguments_before_manager_call(
    app_config: AppConfig,
) -> None:
    manager = _Manager(app_config)

    result = await CommandService(manager).run("go2_lowcmd_acquire", ("now",))

    assert result.status is CommandStatus.REJECTED
    assert result.data == {"code": "INVALID_ARGUMENTS"}
    assert manager.calls == []
