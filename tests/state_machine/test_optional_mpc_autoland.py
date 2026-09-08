from __future__ import annotations

from dataclasses import replace

import pytest

from aerogo2.cli.commands import build_registry
from aerogo2.cli.confirmation import ConfirmationService, ScriptedConfirmationReader
from aerogo2.cli.dispatcher import CommandDispatcher
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    CommandStatus,
    ConfirmationLevel,
    RuntimeMode,
    SystemState,
)
from aerogo2.common.models import LandingEstimate
from aerogo2.manager.permissions import PermissionPolicy
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.parametrize("mpc_available", [False, True])
def test_autoland_status_requires_a_per_attempt_mpc_selection(
    app_config: AppConfig,
    mpc_available: bool,
) -> None:
    config = replace(
        app_config,
        landing=replace(app_config.landing, mpc_enabled=mpc_available),
    )
    world = SimulationWorld(config)

    status = world.manager.query("autoland status")

    assert status["mpc_available"] is mpc_available
    assert status["mpc_selected_for_session"] is False
    assert status["landing_mode"] == "legacy_safe_descent"
    assert status["impact_recovery_required"] is False
    assert status["hardware_output_available"] is False


def test_mpc_prepare_command_requires_exact_confirmation_in_flight() -> None:
    spec = build_registry().get("autoland prepare mpc")

    assert spec.confirmation.level is ConfirmationLevel.EXACT_PHRASE
    assert spec.confirmation.exact_phrase == "CONFIRM_MPC_AUTOLAND"
    assert spec.permission.allowed_modes == frozenset({RuntimeMode.DRY_RUN})
    assert spec.permission.allowed_states == frozenset({SystemState.FLIGHT_MANUAL})
    assert (
        PermissionPolicy()
        .decide(
            spec.action,
            SystemState.FLIGHT_MANUAL,
        )
        .allowed
    )
    assert (
        not PermissionPolicy()
        .decide(
            spec.action,
            SystemState.FLIGHT_READY,
        )
        .allowed
    )


@pytest.mark.asyncio
async def test_manager_requires_capability_and_explicit_per_attempt_confirmation(
    app_config: AppConfig,
) -> None:
    disabled = SimulationWorld(
        replace(
            app_config,
            landing=replace(app_config.landing, mpc_enabled=False),
        )
    )
    enabled = SimulationWorld(
        replace(
            app_config,
            landing=replace(app_config.landing, mpc_enabled=True),
        )
    )
    try:
        assert (await disabled.start()).ok
        assert (await enabled.start()).ok
        assert (await disabled._reach_flight_manual([])).ok
        unavailable = await disabled.manager.prepare_mpc_autoland(operator_confirmed=True)
        assert unavailable.code == "MPC_AUTOLAND_DISABLED"
        assert disabled.manager.state is SystemState.FLIGHT_MANUAL

        assert (await enabled._reach_flight_manual([])).ok
        unconfirmed = await enabled.manager.prepare_mpc_autoland()
        assert unconfirmed.code == "MPC_AUTOLAND_CONFIRMATION_REQUIRED"
        assert enabled.manager.state is SystemState.FLIGHT_MANUAL
        assert not enabled.manager.snapshot.autoland_mpc_selected

        enabled._set_landing_estimate(
            LandingEstimate(
                valid=True,
                ground_detected=True,
                height_m=1.0,
                vertical_velocity_mps=0.0,
                horizontal_velocity_mps=0.0,
                timestamp=enabled.clock.monotonic(),
                reason="simulated estimator valid",
            )
        )
        enabled._set_switches(autoland=1500)
        await enabled.manager.tick()
        legacy = await enabled.manager.prepare_autoland()
        assert legacy.ok
        assert enabled.manager.state is SystemState.AUTO_LANDING_READY
        assert not enabled.manager.snapshot.autoland_mpc_selected
        assert enabled.manager.query("autoland status")["landing_mode"] == "legacy_safe_descent"
        assert (await enabled.manager.abort_autoland()).ok
        assert not enabled.manager.snapshot.autoland_mpc_selected

        selected = await enabled.manager.prepare_mpc_autoland(operator_confirmed=True)
        assert selected.ok
        assert enabled.manager.state is SystemState.AUTO_LANDING_READY
        assert enabled.manager.snapshot.autoland_mpc_selected
        status = enabled.manager.query("autoland status")
        assert status["landing_mode"] == "impact_aware_mpc"
        assert status["mpc_selected_for_session"] is True

        assert (await enabled.manager.abort_autoland()).ok
        assert enabled.manager.state is SystemState.FLIGHT_MANUAL
        assert not enabled.manager.snapshot.autoland_mpc_selected
    finally:
        await disabled.shutdown()
        await enabled.shutdown()


@pytest.mark.asyncio
async def test_cli_exact_phrase_is_the_in_flight_activation_boundary(
    app_config: AppConfig,
) -> None:
    config = replace(
        app_config,
        landing=replace(app_config.landing, mpc_enabled=True),
    )
    world = SimulationWorld(config)
    try:
        assert (await world.start()).ok
        assert (await world._reach_flight_manual([])).ok
        world._set_landing_estimate(
            LandingEstimate(
                valid=True,
                ground_detected=True,
                height_m=1.0,
                vertical_velocity_mps=0.0,
                horizontal_velocity_mps=0.0,
                timestamp=world.clock.monotonic(),
                reason="simulated estimator valid",
            )
        )
        world._set_switches(autoland=1500)
        await world.manager.tick()

        rejected_reader = ScriptedConfirmationReader(("WRONG_PHRASE",))
        rejected = await CommandDispatcher(
            build_registry(),
            world,
            confirmation=ConfirmationService(rejected_reader),
        ).dispatch("autoland prepare mpc", render=False)

        assert rejected.result.status is CommandStatus.REJECTED
        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert not world.manager.snapshot.autoland_mpc_selected

        accepted_reader = ScriptedConfirmationReader(("CONFIRM_MPC_AUTOLAND",))
        accepted = await CommandDispatcher(
            build_registry(),
            world,
            confirmation=ConfirmationService(accepted_reader),
        ).dispatch("autoland prepare mpc", render=False)

        assert accepted.result.status is CommandStatus.SUCCESS
        assert world.manager.state is SystemState.AUTO_LANDING_READY
        assert world.manager.snapshot.autoland_mpc_selected
    finally:
        await world.shutdown()


@pytest.mark.parametrize("mpc_enabled", [False, True])
@pytest.mark.asyncio
async def test_both_optional_autoland_modes_complete_the_nominal_scenario(
    app_config: AppConfig,
    mpc_enabled: bool,
) -> None:
    config = replace(
        app_config,
        landing=replace(app_config.landing, mpc_enabled=mpc_enabled),
    )
    world = SimulationWorld(config)
    recovery_injections: list[bool] = []
    original_inject = world.inject_impact_recovery_completion

    def record_recovery_injection(completed: bool = True) -> None:
        recovery_injections.append(completed)
        original_inject(completed)

    world.inject_impact_recovery_completion = record_recovery_injection  # type: ignore[method-assign]

    try:
        result = await world.run_scenario("nominal")

        assert result.ok
        assert world.manager.state is SystemState.WALK
        assert not world.manager.snapshot.autoland_mpc_selected
        touchdown = next(
            record
            for record in world.manager._state_machine.history
            if record.new_state is SystemState.TOUCHDOWN_VERIFY
        )
        if mpc_enabled:
            assert recovery_injections == [True]
            assert "post-touchdown recovery" in touchdown.reason
        else:
            assert recovery_injections == []
            assert "legacy automatic landing" in touchdown.reason
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_mpc_autoland_waits_for_recovery_and_aborts_on_timeout(
    app_config: AppConfig,
) -> None:
    config = replace(
        app_config,
        landing=replace(app_config.landing, mpc_enabled=True),
    )
    world = SimulationWorld(config)
    world.inject_impact_recovery_completion = lambda completed=True: None  # type: ignore[method-assign]

    try:
        result = await world.run_scenario("nominal")

        assert not result.ok
        assert world.manager.state is SystemState.AUTO_LANDING
        assert world.manager.query("autoland status")["impact_recovery_required"] is True
        assert all(
            record.new_state is not SystemState.TOUCHDOWN_VERIFY
            for record in world.manager._state_machine.history
        )

        world.clock.advance(config.safety.impact_recovery_completion_timeout_s + 0.01)
        await world.step(0.01)

        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert not world.manager.snapshot.autoland_active
        assert not world.manager.snapshot.autoland_mpc_selected
        assert world.manager._aborted_impact_touchdown_latched

        retry = await world.manager.prepare_autoland()
        assert retry.code == "IMPACT_RECOVERY_REENTRY_BLOCKED"
        assert world.manager.state is SystemState.FLIGHT_MANUAL
        assert world.manager._aborted_impact_touchdown_latched
    finally:
        await world.shutdown()
