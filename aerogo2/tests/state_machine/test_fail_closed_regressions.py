from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple
from unittest.mock import AsyncMock

import pytest

from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State, SystemState
from aerogo2.common.models import LandingEstimate, SystemSnapshot, TransitionRecord
from aerogo2.common.results import OperationResult
from aerogo2.manager.state_machine import StateMachine
from aerogo2.manager.transition_guards import TransitionGuards
from aerogo2.safety.safety_monitor import SafetyMonitor
from aerogo2.simulation.world import SimulationWorld


def evaluate_flight_precheck(
    app_config: AppConfig,
    snapshot: SystemSnapshot,
):
    return TransitionGuards(app_config).evaluate(
        SystemState.WALK,
        SystemState.WALK_TO_FLIGHT_PRECHECK,
        snapshot,
    )


@pytest.mark.parametrize(
    ("device", "expected_code"),
    [
        ("pixhawk", "PIXHAWK_TIMEOUT"),
        ("f446", "F446_TIMEOUT"),
        ("go2", "GO2_TIMEOUT"),
        ("rc", "RC_TIMEOUT"),
    ],
)
@pytest.mark.parametrize("bad_timestamp", [float("nan"), 10.001])
def test_transform_guard_rejects_nonfinite_or_future_device_timestamp(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    device: str,
    expected_code: str,
    bad_timestamp: float,
) -> None:
    if device == "pixhawk":
        snapshot = replace(
            safe_walk_snapshot,
            pixhawk=replace(
                safe_walk_snapshot.pixhawk,
                heartbeat_timestamp=bad_timestamp,
            ),
        )
    elif device == "f446":
        snapshot = replace(
            safe_walk_snapshot,
            f446=replace(safe_walk_snapshot.f446, timestamp=bad_timestamp),
        )
    elif device == "go2":
        snapshot = replace(
            safe_walk_snapshot,
            go2=replace(safe_walk_snapshot.go2, timestamp=bad_timestamp),
        )
    else:
        snapshot = replace(
            safe_walk_snapshot,
            rc=replace(safe_walk_snapshot.rc, timestamp=bad_timestamp),
        )

    result = evaluate_flight_precheck(app_config, snapshot)

    assert not result.permitted
    assert expected_code in result.codes


def test_transform_guard_rejects_missing_unhealthy_or_nonfinite_esc_telemetry(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    baseline = safe_walk_snapshot.pixhawk.esc
    variants = (
        (),
        (replace(baseline[0], healthy=False),) + baseline[1:],
        (replace(baseline[0], rpm=float("nan")),) + baseline[1:],
    )

    for esc in variants:
        snapshot = replace(
            safe_walk_snapshot,
            pixhawk=replace(
                safe_walk_snapshot.pixhawk,
                esc=esc,
                esc_rpm={},
                esc_online={},
            ),
        )
        result = evaluate_flight_precheck(app_config, snapshot)
        assert not result.permitted
        assert "ESC_RPM_NONZERO_DURING_TRANSFORM" in result.codes


def test_transform_guard_rejects_nonfinite_go2_velocity(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        go2=replace(safe_walk_snapshot.go2, velocity_mps=float("nan")),
    )

    result = evaluate_flight_precheck(app_config, snapshot)

    assert not result.permitted
    assert "GO2_MOVING_DURING_TRANSFORM" in result.codes


@pytest.mark.parametrize("field", ["failsafe", "rc_failsafe"])
def test_transform_guard_rejects_pixhawk_failsafe(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    field: str,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        pixhawk=replace(safe_walk_snapshot.pixhawk, **{field: True}),
    )

    result = evaluate_flight_precheck(app_config, snapshot)

    assert not result.permitted
    assert "PIXHAWK_FAILSAFE" in result.codes


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"used_current_adc": 1600}, "F446_OVERCURRENT"),
        ({"duty": 1}, "F446_DUTY_NONZERO"),
    ],
)
def test_transform_guard_rejects_f446_overcurrent_or_nonzero_duty(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    changes: dict,
    expected_code: str,
) -> None:
    if "used_current_adc" in changes:
        threshold = safe_walk_snapshot.f446.threshold_adc
        assert threshold is not None
        changes = {
            "used_current_adc": (threshold - app_config.f446.current_safe_margin_adc + 1),
        }
    snapshot = replace(
        safe_walk_snapshot,
        f446=replace(safe_walk_snapshot.f446, **changes),
    )

    result = evaluate_flight_precheck(app_config, snapshot)

    assert not result.permitted
    assert expected_code in result.codes


@pytest.mark.parametrize(
    ("current", "target", "configuration", "expected_code"),
    [
        (
            SystemState.WALK_TO_FLIGHT_PRECHECK,
            SystemState.TRANSFORM_TO_FLIGHT,
            Configuration.FLIGHT,
            "WALK_CONFIGURATION_NOT_CONFIRMED",
        ),
        (
            SystemState.FLIGHT_TO_WALK_PRECHECK,
            SystemState.TRANSFORM_TO_WALK,
            Configuration.WALK,
            "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
        ),
    ],
)
def test_second_stage_transform_guard_rechecks_source_configuration(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    current: SystemState,
    target: SystemState,
    configuration: Configuration,
    expected_code: str,
) -> None:
    f446_state = (
        app_config.f446.expected_walk_state
        if current is SystemState.WALK_TO_FLIGHT_PRECHECK
        else app_config.f446.expected_flight_state
    )
    snapshot = replace(
        safe_walk_snapshot,
        state=current,
        configuration=configuration,
        f446=replace(safe_walk_snapshot.f446, state=f446_state),
    )

    result = TransitionGuards(app_config).evaluate(current, target, snapshot)

    assert not result.permitted
    assert expected_code in result.codes


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SystemState.BOOT_SAFE, SystemState.WALK),
        (SystemState.WALK, SystemState.WALK_TO_FLIGHT_PRECHECK),
    ],
)
def test_active_fault_codes_block_safe_state_or_transform_entry(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    current: SystemState,
    target: SystemState,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        state=current,
        active_fault_codes=("SIMULATED_STICKY_FAULT",),
    )

    result = TransitionGuards(app_config).evaluate(current, target, snapshot)

    assert not result.permitted
    assert "ACTIVE_FAULTS_PRESENT" in result.codes


@pytest.mark.asyncio
async def test_stop_supervised_propagates_f446_operation_failure(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        world.f446.inject_status(state=F446State.LIMIT_FWD, duty=120)
        await world.manager.refresh_snapshot()
        failure = OperationResult.failure("F446_STOP_REJECTED", "simulated F446 stop rejection")
        failed_stop = AsyncMock(return_value=failure)

        with monkeypatch.context() as patch:
            patch.setattr(world.f446, "stop", failed_stop)
            result = await world.manager.stop_supervised()

        assert not result.ok
        assert "simulated F446 stop rejection" in result.message
        assert world.manager.snapshot.f446.duty == 120
        failed_stop.assert_awaited_once()
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_stop_supervised_preserves_active_flag_when_setpoint_stop_fails(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        assert (await world.pixhawk.send_velocity_setpoint(0.0, 0.0, 0.1, 0.0)).ok
        world.manager._setpoint_active = True
        await world.manager.refresh_snapshot()
        assert world.manager.snapshot.external_setpoint_active
        failure = OperationResult.failure(
            "SETPOINT_STOP_REJECTED",
            "simulated Pixhawk setpoint-stop rejection",
        )
        failed_stop = AsyncMock(return_value=failure)

        with monkeypatch.context() as patch:
            patch.setattr(world.pixhawk, "stop_external_setpoints", failed_stop)
            result = await world.manager.stop_supervised()

        assert not result.ok
        assert "simulated Pixhawk setpoint-stop rejection" in result.message
        assert world.pixhawk.external_setpoints_active
        assert world.manager.snapshot.external_setpoint_active
        failed_stop.assert_awaited_once()
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_state_transition_orders_log_before_mutation_and_publish_before_entry(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    events: List[Tuple[str, SystemState]] = []
    machine: StateMachine

    def logger(record: TransitionRecord) -> None:
        events.append(("log", machine.state))

    machine = StateMachine(TransitionGuards(app_config), clock, record_logger=logger)

    def subscriber(record: TransitionRecord) -> None:
        events.append(("publish", machine.state))

    async def entry_action(snapshot: SystemSnapshot) -> None:
        events.append(("entry", machine.state))

    machine.subscribe(subscriber)
    machine.set_entry_action(SystemState.WALK, entry_action)

    await machine.transition_to(
        SystemState.WALK,
        reason="ordering regression",
        snapshot=safe_walk_snapshot,
    )

    assert events == [
        ("log", SystemState.BOOT_SAFE),
        ("publish", SystemState.WALK),
        ("entry", SystemState.WALK),
    ]


@pytest.mark.asyncio
async def test_entry_action_failure_transitions_to_fault(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    machine = StateMachine(TransitionGuards(app_config), clock)

    async def fail_entry(snapshot: SystemSnapshot) -> None:
        raise RuntimeError("simulated WALK entry failure")

    machine.set_entry_action(SystemState.WALK, fail_entry)

    await machine.transition_to(
        SystemState.WALK,
        reason="entry failure regression",
        snapshot=safe_walk_snapshot,
    )

    assert machine.state is SystemState.FAULT
    assert [(item.previous_state, item.new_state) for item in machine.history] == [
        (SystemState.BOOT_SAFE, SystemState.WALK),
        (SystemState.WALK, SystemState.FAULT),
    ]
    assert "simulated WALK entry failure" in machine.history[-1].reason


@pytest.mark.asyncio
async def test_fault_entry_action_failure_does_not_recurse(
    app_config: AppConfig,
    clock: ManualClock,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    machine = StateMachine(TransitionGuards(app_config), clock)

    async def fail_fault_entry(snapshot: SystemSnapshot) -> None:
        raise RuntimeError("simulated FAULT entry failure")

    machine.set_entry_action(SystemState.FAULT, fail_fault_entry)

    await machine.transition_to(
        SystemState.FAULT,
        reason="direct fault regression",
        snapshot=safe_walk_snapshot,
    )

    assert machine.state is SystemState.FAULT
    assert len(machine.history) == 2
    assert machine.history[0].permitted
    assert machine.history[1].guard_codes == ("FAULT_ENTRY_ACTION_FAILED",)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (SystemState.WALK, False),
        (SystemState.FLIGHT_MANUAL, False),
        (SystemState.AUTO_LANDING_READY, True),
        (SystemState.AUTO_LANDING, True),
    ],
)
def test_manual_override_warning_is_scoped_to_autoland(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
    state: SystemState,
    expected: bool,
) -> None:
    snapshot = replace(
        safe_walk_snapshot,
        state=state,
        rc=replace(safe_walk_snapshot.rc, manual_override=True),
    )

    codes = {item.code for item in SafetyMonitor(app_config).evaluate(snapshot)}

    assert ("MANUAL_OVERRIDE_REQUESTED" in codes) is expected


@pytest.mark.asyncio
async def test_touchdown_height_instability_restarts_confirmation_hold(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        states: List[SystemState] = []
        assert (await world._reach_flight_manual(states)).ok
        assert (await world._reach_autoland(states)).ok
        assert world.manager.state is SystemState.AUTO_LANDING

        world.pixhawk.inject_landed_state(
            True,
            vertical_velocity_mps=0.0,
            relative_altitude_m=0.0,
        )
        world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
        for slot in app_config.esc.slots:
            world.pixhawk.inject_esc_rpm(slot, 0.0)

        def set_height(height_m: float) -> None:
            world._set_landing_estimate(
                LandingEstimate(
                    valid=True,
                    ground_detected=True,
                    height_m=height_m,
                    vertical_velocity_mps=0.0,
                    horizontal_velocity_mps=0.0,
                    timestamp=world.clock.monotonic(),
                    reason="simulated ground contact",
                )
            )

        step_s = 0.05
        half_hold_steps = int(app_config.safety.touchdown_confirm_s / step_s / 2)
        set_height(0.0)
        for _ in range(half_hold_steps):
            await world.step(step_s)
        assert world.manager.state is SystemState.AUTO_LANDING

        set_height(0.25)
        await world.step(step_s)
        for _ in range(half_hold_steps + 1):
            await world.step(step_s)

        assert world.manager.state is SystemState.AUTO_LANDING

        full_hold_steps = int(app_config.safety.touchdown_confirm_s / step_s) + 2
        for _ in range(full_hold_steps):
            await world.step(step_s)
            if world.manager.state is SystemState.TOUCHDOWN_VERIFY:
                break
        assert world.manager.state is SystemState.TOUCHDOWN_VERIFY
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_touchdown_rejects_unhealthy_zero_rpm_esc_telemetry(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        assert (await world.start()).ok
        states: List[SystemState] = []
        assert (await world._reach_flight_manual(states)).ok
        assert (await world._reach_autoland(states)).ok
        world.pixhawk.inject_landed_state(
            True,
            vertical_velocity_mps=0.0,
            relative_altitude_m=0.0,
        )
        world.pixhawk.inject_attitude(0.0, 0.0, 0.0)
        for slot in app_config.esc.slots:
            world.pixhawk.inject_esc_rpm(slot, 0.0, healthy=slot != 1)
        world._set_landing_estimate(
            LandingEstimate(
                valid=True,
                ground_detected=True,
                height_m=0.0,
                vertical_velocity_mps=0.0,
                horizontal_velocity_mps=0.0,
                timestamp=world.clock.monotonic(),
                reason="simulated ground contact",
            )
        )

        steps = int(app_config.safety.touchdown_confirm_s / 0.05) + 3
        for _ in range(steps):
            await world.step(0.05)

        assert world.manager.state is SystemState.AUTO_LANDING
    finally:
        await world.shutdown()
