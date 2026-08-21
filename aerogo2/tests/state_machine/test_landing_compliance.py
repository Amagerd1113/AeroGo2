from __future__ import annotations

from dataclasses import replace
from typing import List

import pytest

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.common.models import LandingEstimate
from aerogo2.simulation.world import SimulationWorld


def _enabled_config(app_config: AppConfig) -> AppConfig:
    return replace(
        app_config,
        go2=replace(
            app_config.go2,
            landing_compliance_enabled=True,
            foot_force_contact_thresholds=(10, 20, 30, 40),
            landing_contact_min_feet=3,
            landing_contact_confirm_s=0.15,
            landing_compliance_settle_s=0.20,
        ),
    )


async def _reach_touchdown_verify(world: SimulationWorld) -> None:
    states: List[SystemState] = []
    assert (await world.start()).ok
    assert (await world._reach_flight_manual(states)).ok
    world._set_landing_estimate(
        LandingEstimate(
            valid=False,
            ground_detected=False,
            height_m=None,
            timestamp=world.clock.monotonic(),
            reason="manual landing",
        )
    )
    world.pixhawk.inject_landed_state(
        True,
        vertical_velocity_mps=0.0,
        relative_altitude_m=0.0,
    )
    world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
    for slot in world.config.esc.slots:
        world.pixhawk.inject_esc_rpm(slot, 0.0)
    steps = int(world.config.safety.touchdown_confirm_s / 0.05) + 3
    for _ in range(steps):
        await world.step(0.05)
        if world.manager.state is SystemState.TOUCHDOWN_VERIFY:
            break
    assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
    assert world.manager.snapshot.go2.joints_locked
    assert world.manager.snapshot.pixhawk.armed


async def _enter_landing_compliant(world: SimulationWorld) -> None:
    await _reach_touchdown_verify(world)
    world.pixhawk.inject_armed_state(False)
    world._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
    world.go2.inject_status(
        foot_force=(100, 100, 100, 0),
        foot_force_valid=True,
    )
    steps = int(world.config.go2.landing_contact_confirm_s / 0.05) + 4
    for _ in range(steps):
        await world.step(0.05)
        if world.manager.state is SystemState.LANDING_COMPLIANT:
            break
    assert world.manager.state is SystemState.LANDING_COMPLIANT


@pytest.mark.asyncio
async def test_enabled_compliance_cannot_be_bypassed_without_foot_contact(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(_enabled_config(app_config))
    try:
        await _reach_touchdown_verify(world)
        world.pixhawk.inject_armed_state(False)
        world._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
        world.go2.inject_status(
            foot_force=(0, 0, 0, 0),
            foot_force_valid=True,
        )
        await world.step(0.25)

        result = await world.manager.request_transform_walk(operator_confirmed=True)

        assert not result.ok
        assert result.code == "LANDING_COMPLIANCE_REQUIRED"
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
        assert world.manager.snapshot.go2.joints_locked
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_three_calibrated_feet_enter_balance_then_relock_before_transform(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(_enabled_config(app_config))
    try:
        await _enter_landing_compliant(world)

        assert world.manager.snapshot.go2.locomotion_mode == "BALANCE_STAND"
        assert not world.manager.snapshot.go2.joints_locked
        report = world.manager.query("landing compliance")
        assert report["contact_count"] == 3
        assert report["contact_safe"] is True

        early = await world.manager.request_transform_walk(operator_confirmed=True)
        assert not early.ok
        assert early.code == "LANDING_COMPLIANCE_SETTLE_REQUIRED"

        await world.step(world.config.go2.landing_compliance_settle_s + 0.05)
        result = await world.manager.request_transform_walk(operator_confirmed=True)

        assert result.ok
        assert world.manager.state is SystemState.WALK
        assert world.manager.snapshot.go2.joints_locked
        assert (await world.manager.walk_stand()).ok
        assert not world.manager.snapshot.go2.joints_locked
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_contact_loss_in_compliant_mode_relocks_then_faults(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(_enabled_config(app_config))
    try:
        await _enter_landing_compliant(world)
        world.go2.inject_status(
            foot_force=(0, 0, 0, 0),
            foot_force_valid=True,
        )

        await world.step(0.05)

        assert world.manager.state is SystemState.FAULT
        assert world.manager.snapshot.go2.joints_locked
        assert any(item.code == "GO2_FOOT_CONTACT_LOST" for item in world.manager.violations)
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_disarm_and_exact_zero_rpm_are_required_before_unlock(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(_enabled_config(app_config))
    try:
        await _reach_touchdown_verify(world)
        world.go2.inject_status(
            foot_force=(100, 100, 100, 100),
            foot_force_valid=True,
        )
        await world.step(0.25)
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
        assert world.manager.snapshot.go2.joints_locked

        world.pixhawk.inject_armed_state(False)
        world.pixhawk.inject_esc_rpm(1, 1.0)
        world._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
        await world.step(0.25)
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
        assert world.manager.snapshot.go2.joints_locked
    finally:
        await world.shutdown()
