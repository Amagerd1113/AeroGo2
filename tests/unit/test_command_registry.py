"""Command-tree, completion, alias, and permission metadata tests."""

from __future__ import annotations

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from aerogo2.cli.command_models import CommandContext, CommandSpec
from aerogo2.cli.commands import build_registry
from aerogo2.cli.completer import AeroGo2Completer
from aerogo2.cli.registry import (
    CommandNotFoundError,
    CommandRegistry,
    DuplicateCommandError,
)
from aerogo2.common.enums import ConfirmationLevel, RuntimeMode, SystemState

EXPECTED_CANONICAL_COMMANDS = frozenset(
    line.strip()
    for line in """
audit
audit configuration
audit f446
audit pixhawk
audit rc
autoland abort
autoland prepare
autoland start
autoland status
check communication
check invariant
check sensors
clear
clear-fault
config diff
config get
config reload
config show
config validate
connect all
connect f446
connect go2
connect pixhawk
controller inputs
controller output
controller reset
controller status
controller timing
devices
disconnect all
disconnect f446
disconnect go2
disconnect pixhawk
esc
esc 1
esc 2
esc 3
esc 4
esc health
esc mapping
exit
faults
faults active
faults explain
faults history
flight enable-check
flight auth-status
flight authorize
flight ready
flight revoke
flight status
go2 controller
go2 motion
go2 status
health
help
history
landing compliance
log export
log mark
log start
log status
log stop
log tail
motor current
motor confirm flight
motor confirm walk
motor limf
motor limr
motor maintenance enter
motor maintenance exit
motor mf
motor mr
motor parameters
motor status
motor stop
motor threshold
motor threshold-mv
motor to-flight
motor to-walk
pixhawk messages
pixhawk params
pixhawk status
pixhawk statustext
preflight
preflight autoland
preflight flight
preflight home-walk
preflight transform-flight
preflight manual-position
preflight transform-walk
rc
rc check
rc mapping
rc raw
sim clear
sim inject
sim pause
sim reset
sim run
sim scenario f446-overcurrent
sim scenario landing
sim scenario nominal
sim scenario pixhawk-timeout
sim scenario rc-loss
sim scenario transform-failure
sim status
sim step
state
state guards
state transitions
status
stop
transform flight
transform home-walk
transform status
transform stop
transform walk
version
walk permit
walk stand
walk status
walk stop
watch esc
watch f446
watch faults
watch rc
watch status
""".splitlines()
    if line.strip()
)


def test_complete_canonical_command_tree_is_registered() -> None:
    registry = build_registry()

    assert set(registry.command_names()) == EXPECTED_CANONICAL_COMMANDS
    assert len(registry) == len(EXPECTED_CANONICAL_COMMANDS)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("quit", "exit"),
        ("abort", "autoland abort"),
        ("s", "stop"),
    ],
)
def test_command_aliases_resolve_to_the_canonical_spec(alias: str, canonical: str) -> None:
    match = build_registry().resolve(alias)

    assert match.spec.name == canonical
    assert match.used_alias


@pytest.mark.parametrize(
    ("line", "command", "arguments"),
    [
        ("help transform flight", "help", ("transform", "flight")),
        ("health --watch 1", "health", ("--watch", "1")),
        ("status --full", "status", ("--full",)),
        ("status --json", "status", ("--json",)),
        ("status --watch 0.5", "status", ("--watch", "0.5")),
    ],
)
def test_usage_forms_resolve_without_becoming_separate_commands(
    line: str, command: str, arguments: tuple[str, ...]
) -> None:
    match = build_registry().resolve(line)

    assert match.spec.name == command
    assert match.arguments == arguments


def test_longest_hierarchical_command_path_wins() -> None:
    match = build_registry().resolve("state transitions newest")

    assert match.spec.name == "state transitions"
    assert match.arguments == ("newest",)


def test_every_help_entry_has_action_description_and_safety_metadata() -> None:
    registry = build_registry()

    for spec in registry.specs():
        assert spec.description.strip(), spec.name
        assert spec.action.strip(), spec.name
        assert spec.effective_usage.startswith(spec.name), spec.name
        assert spec.permission is not None
        assert spec.confirmation is not None


def test_argument_commands_publish_spec_accurate_usage() -> None:
    registry = build_registry()
    expected = {
        "help": "help [COMMAND]",
        "health": "health [--watch SECONDS]",
        "status": "status [--full|--json|--watch SECONDS]",
        "faults explain": "faults explain CODE",
        "config get": "config get KEY",
        "log mark": "log mark TEXT",
        "log export": "log export PATH",
        "sim step": "sim step [SECONDS]",
        "sim inject": "sim inject FAULT",
        "motor mf": "motor mf DUTY",
        "motor mr": "motor mr DUTY",
        "motor limf": "motor limf DUTY",
        "motor limr": "motor limr DUTY",
        "motor threshold": "motor threshold ADC",
        "motor threshold-mv": "motor threshold-mv MV",
        "watch status": "watch status [SECONDS]",
        "watch rc": "watch rc [SECONDS]",
        "watch f446": "watch f446 [SECONDS]",
        "watch esc": "watch esc [SECONDS]",
        "watch faults": "watch faults [SECONDS]",
    }

    assert {name: registry.get(name).effective_usage for name in expected} == expected


def test_stop_is_a_distinct_canonical_supervised_stop() -> None:
    match = build_registry().resolve("stop")

    assert not match.used_alias
    assert match.spec.action == "stop"
    assert "Go2" in match.spec.description


def test_transform_flight_confirmation_warns_about_go2_original_remote() -> None:
    for name in ("transform flight", "motor to-flight"):
        confirmation = build_registry().get(name).confirmation
        assert confirmation.exact_phrase == "TRANSFORM_TO_FLIGHT"
        assert confirmation.warning is not None
        assert "Go2 original remote" in confirmation.warning


def test_help_command_can_lookup_hierarchical_target_metadata() -> None:
    registry = build_registry()
    help_match = registry.resolve("help transform flight")
    target = registry.get(help_match.arguments)

    assert help_match.spec.action == "help"
    assert target.name == "transform flight"
    assert target.confirmation.level is ConfirmationLevel.EXACT_PHRASE
    assert target.confirmation.exact_phrase == "TRANSFORM_TO_FLIGHT"


def test_unknown_command_contains_spelling_suggestion() -> None:
    registry = build_registry()

    with pytest.raises(CommandNotFoundError) as error:
        registry.resolve("statsu")

    assert "status" in error.value.suggestions
    assert "Did you mean" in str(error.value)


def test_completer_suggests_next_hierarchical_word() -> None:
    completer = AeroGo2Completer(build_registry())

    completions = tuple(completer.get_completions(Document("motor to-f"), CompleteEvent()))

    assert "to-flight" in {completion.text for completion in completions}


def test_completer_includes_declared_options() -> None:
    registry = CommandRegistry()
    registry.register(
        CommandSpec(
            path=("status",),
            description="status",
            options=("--full", "--json"),
        )
    )
    completer = AeroGo2Completer(registry)

    completions = tuple(completer.get_completions(Document("status --j"), CompleteEvent()))

    assert [completion.text for completion in completions] == ["--json"]


def test_duplicate_alias_or_command_path_is_rejected() -> None:
    registry = CommandRegistry()
    registry.register(CommandSpec(("status",), "status", aliases=(("st",),)))

    with pytest.raises(DuplicateCommandError):
        registry.register(CommandSpec(("st",), "duplicate"))


@pytest.mark.parametrize("name", ["status", "rc raw", "motor current", "preflight"])
def test_read_only_commands_require_no_confirmation(name: str) -> None:
    spec = build_registry().get(name)

    assert spec.confirmation.level is ConfirmationLevel.NONE


@pytest.mark.parametrize(
    ("name", "phrase"),
    [
        ("transform flight", "TRANSFORM_TO_FLIGHT"),
        ("transform walk", "TRANSFORM_TO_WALK"),
    ],
)
def test_dangerous_commands_publish_exact_confirmation_metadata(name: str, phrase: str) -> None:
    confirmation = build_registry().get(name).confirmation

    assert confirmation.level is ConfirmationLevel.EXACT_PHRASE
    assert confirmation.exact_phrase == phrase


def test_home_walk_uses_two_stage_confirmation() -> None:
    confirmation = build_registry().get("transform home-walk").confirmation

    assert confirmation.level is ConfirmationLevel.TWO_STAGE
    assert confirmation.exact_phrase == "HOME_F446_TO_WALK"
    assert confirmation.warning is not None


def test_phase_one_rejects_manual_motor_command_before_maintenance_check() -> None:
    policy = build_registry().get("motor mr").permission

    decision = policy.evaluate(
        CommandContext(
            runtime_mode=RuntimeMode.DRY_RUN,
            state=SystemState.BOOT_SAFE,
            phase=1,
            maintenance_mode=False,
        )
    )

    assert not decision.allowed
    assert decision.code == "STATE_DENIED"


def test_manual_motor_command_is_still_rejected_outside_maintenance_in_phase_three() -> None:
    policy = build_registry().get("motor mr").permission

    decision = policy.evaluate(
        CommandContext(
            runtime_mode=RuntimeMode.HARDWARE,
            state=SystemState.MANUAL_POSITIONING,
            phase=3,
            maintenance_mode=False,
            hardware_write_enabled=True,
        )
    )

    assert not decision.allowed
    assert decision.code == "MAINTENANCE_REQUIRED"


def test_manual_motor_command_has_two_stage_warning_metadata() -> None:
    confirmation = build_registry().get("motor mr").confirmation

    assert confirmation.level is ConfirmationLevel.TWO_STAGE
    assert confirmation.exact_phrase == "RUN_MANUAL_MOTOR"
    assert confirmation.warning is not None
    assert "No F446 local limit stop" in confirmation.warning


HARDWARE_ACTUATOR_COMMANDS = (
    "transform flight",
    "transform walk",
    "transform home-walk",
    "transform stop",
    "motor to-flight",
    "motor to-walk",
    "motor stop",
    "walk stop",
    "walk stand",
    "stop",
    "flight authorize",
    "flight revoke",
)

DRY_RUN_ONLY_ACTUATOR_COMMANDS = (
    "autoland prepare",
    "autoland start",
    "autoland abort",
    "controller reset",
)

PERMITTED_STATE = {
    "transform flight": SystemState.WALK,
    "transform home-walk": SystemState.BOOT_SAFE,
    "transform walk": SystemState.FLIGHT_READY,
    "flight authorize": SystemState.FLIGHT_READY,
    "flight revoke": SystemState.FLIGHT_READY,
}


@pytest.mark.parametrize("name", HARDWARE_ACTUATOR_COMMANDS)
def test_real_actuator_commands_require_explicit_hardware_write(name: str) -> None:
    policy = build_registry().get(name).permission

    assert policy.allowed_modes == frozenset(RuntimeMode)
    assert policy.requires_hardware_write


@pytest.mark.parametrize("name", HARDWARE_ACTUATOR_COMMANDS)
@pytest.mark.parametrize(
    "runtime_mode",
    [RuntimeMode.HARDWARE_READONLY, RuntimeMode.HARDWARE],
)
def test_real_actuator_commands_fail_closed_without_process_unlock(
    name: str,
    runtime_mode: RuntimeMode,
) -> None:
    policy = build_registry().get(name).permission
    decision = policy.evaluate(
        CommandContext(
            runtime_mode=runtime_mode,
            state=PERMITTED_STATE.get(name, SystemState.BOOT_SAFE),
            phase=1,
            hardware_write_enabled=runtime_mode is RuntimeMode.HARDWARE_READONLY,
        )
    )

    assert not decision.allowed
    assert decision.code == "HARDWARE_WRITE_DISABLED"


@pytest.mark.parametrize("name", HARDWARE_ACTUATOR_COMMANDS)
def test_real_actuator_commands_allow_explicit_hardware_process_unlock(name: str) -> None:
    policy = build_registry().get(name).permission
    decision = policy.evaluate(
        CommandContext(
            runtime_mode=RuntimeMode.HARDWARE,
            state=PERMITTED_STATE.get(name, SystemState.BOOT_SAFE),
            phase=1,
            hardware_write_enabled=True,
        )
    )

    assert decision.allowed


@pytest.mark.parametrize("name", DRY_RUN_ONLY_ACTUATOR_COMMANDS)
def test_unimplemented_autoland_outputs_remain_dry_run_only(name: str) -> None:
    policy = build_registry().get(name).permission

    assert policy.allowed_modes == frozenset({RuntimeMode.DRY_RUN})


@pytest.mark.parametrize("name", ["stop", "motor stop", "transform stop", "abort"])
def test_safety_stop_commands_never_require_confirmation(name: str) -> None:
    match = build_registry().resolve(name)

    assert match.spec.confirmation.level is ConfirmationLevel.NONE


def test_simulation_mutation_is_denied_outside_dry_run() -> None:
    policy = build_registry().get("sim inject").permission

    decision = policy.evaluate(
        CommandContext(
            runtime_mode=RuntimeMode.HARDWARE_READONLY,
            state=SystemState.BOOT_SAFE,
        )
    )

    assert not decision.allowed
    assert decision.code == "RUNTIME_MODE_DENIED"
