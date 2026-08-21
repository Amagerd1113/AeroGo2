from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, SystemState
from aerogo2.common.exceptions import TransitionRejected
from aerogo2.common.models import SystemSnapshot
from aerogo2.manager.state_machine import StateMachine
from aerogo2.manager.transition_guards import TransitionGuards


def make_machine(config: AppConfig, clock: ManualClock) -> StateMachine:
    return StateMachine(TransitionGuards(config), clock)


def test_initial_state_is_always_boot_safe(app_config: AppConfig, clock: ManualClock) -> None:
    assert make_machine(app_config, clock).state is SystemState.BOOT_SAFE


def test_state_has_no_public_setter(app_config: AppConfig, clock: ManualClock) -> None:
    machine = make_machine(app_config, clock)
    with pytest.raises(AttributeError):
        machine.state = SystemState.WALK  # type: ignore[misc]


@pytest.mark.asyncio
async def test_boot_to_flight_requires_confirmed_flight_configuration(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    machine = make_machine(app_config, clock)
    with pytest.raises(TransitionRejected, match="confirmed"):
        await machine.transition_to(
            SystemState.FLIGHT_READY,
            reason="boot recovery",
            snapshot=safe_walk_snapshot,
        )
    assert machine.state is SystemState.BOOT_SAFE
    assert machine.history[-1].permitted is False
    assert machine.history[-1].guard_codes == ("FLIGHT_CONFIGURATION_NOT_CONFIRMED",)


@pytest.mark.asyncio
async def test_boot_recovers_confirmed_flight_configuration(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    machine = make_machine(app_config, clock)
    flight_snapshot = replace(
        safe_walk_snapshot,
        configuration=Configuration.FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=app_config.f446.expected_state_for(Configuration.FLIGHT.value),
            duty=0,
        ),
        go2=replace(
            safe_walk_snapshot.go2,
            joints_locked=True,
            locomotion_mode="JOINT_LOCK",
        ),
    )

    await machine.transition_to(
        SystemState.FLIGHT_READY,
        reason="boot recovery",
        snapshot=flight_snapshot,
    )

    assert machine.state is SystemState.FLIGHT_READY


@pytest.mark.asyncio
async def test_boot_to_walk_requires_confirmed_walk_configuration(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    machine = make_machine(app_config, clock)
    unknown = replace(
        safe_walk_snapshot,
        state=SystemState.BOOT_SAFE,
        configuration=Configuration.UNKNOWN,
    )
    with pytest.raises(TransitionRejected, match="confirmed"):
        await machine.transition_to(SystemState.WALK, reason="boot", snapshot=unknown)


@pytest.mark.asyncio
async def test_rejected_transition_is_recorded_and_logged(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    logged = []
    machine = StateMachine(TransitionGuards(app_config), clock, logged.append)
    with pytest.raises(TransitionRejected):
        await machine.transition_to(
            SystemState.AUTO_LANDING,
            reason="illegal",
            snapshot=safe_walk_snapshot,
        )
    assert logged == list(machine.history)
