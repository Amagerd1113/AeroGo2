from __future__ import annotations

import asyncio

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.simulation.fault_injection import SimulatedFault
from aerogo2.simulation.scenarios import SCENARIOS
from aerogo2.simulation.world import ScenarioResult, SimulationWorld


@pytest.mark.asyncio
async def test_background_step_cannot_interleave_with_running_scenario(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = SimulationWorld(app_config)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_nominal() -> ScenarioResult:
        entered.set()
        await release.wait()
        return ScenarioResult(
            name="nominal",
            ok=True,
            final_state=world.manager.state,
            states=(SystemState.BOOT_SAFE, world.manager.state),
        )

    monkeypatch.setattr(world, "_scenario_nominal", blocked_nominal)
    scenario_task = asyncio.create_task(world.run_scenario("nominal"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        before = world.clock.monotonic()

        background_result = await world.step(0.5)
        second_scenario = await world.run_scenario("landing")

        injection = await world.inject(SimulatedFault.RC_LOSS)
        selection = await world.select_scenario("landing")
        pause = await world.pause()
        resume = await world.resume()
        assert world.scenario_running
        assert not background_result.ok
        assert background_result.code == "SCENARIO_RUNNING"
        assert world.clock.monotonic() == before
        assert not second_scenario.ok
        assert second_scenario.details["code"] == "SCENARIO_RUNNING"
    finally:
        release.set()
        assert injection.code == "SCENARIO_RUNNING"
        assert selection.code == "SCENARIO_RUNNING"
        assert pause.code == "SCENARIO_RUNNING"
        assert resume.code == "SCENARIO_RUNNING"
        assert world.selected_scenario == "nominal"
        await scenario_task
        await world.shutdown()

    assert not world.scenario_running


def test_scenario_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        SCENARIOS["unexpected"] = SCENARIOS["nominal"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_reset_and_queued_step_are_one_atomic_world_transaction(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = SimulationWorld(app_config)
    await world.start()
    entered_shutdown = asyncio.Event()
    release_shutdown = asyncio.Event()
    original_shutdown = world.manager.shutdown

    async def blocked_shutdown():
        entered_shutdown.set()
        await release_shutdown.wait()
        return await original_shutdown()

    monkeypatch.setattr(world.manager, "shutdown", blocked_shutdown)
    before = world.clock.monotonic()
    reset_task = asyncio.create_task(world.reset(start=False))
    await asyncio.wait_for(entered_shutdown.wait(), timeout=1.0)
    step_task = asyncio.create_task(world.step(0.5))
    await asyncio.sleep(0)
    assert not step_task.done()

    release_shutdown.set()
    reset_result, step_result = await asyncio.gather(reset_task, step_task)

    assert reset_result.ok
    assert not step_result.ok
    assert step_result.code == "SIM_NOT_RUNNING"
    assert world.clock.monotonic() == before
    assert not world.manager.started


def test_scenario_result_details_are_copied_and_read_only() -> None:
    source = {"code": "ORIGINAL", "nested": {"values": [1, 2]}}
    result = ScenarioResult(
        name="test",
        ok=False,
        final_state=SystemState.BOOT_SAFE,
        states=(SystemState.BOOT_SAFE,),
        details=source,
    )

    source["code"] = "MUTATED"
    source["extra"] = "LATE_ALIAS"
    source["nested"]["values"].append(3)

    assert result.details["code"] == "ORIGINAL"
    assert result.details["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        result.details["code"] = "FORBIDDEN"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.details["nested"]["values"] = ()  # type: ignore[index]


@pytest.mark.asyncio
async def test_fault_injection_requires_a_started_world(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)

    result = await world.inject(SimulatedFault.RC_LOSS)

    assert not result.ok
    assert result.code == "SIM_NOT_RUNNING"
