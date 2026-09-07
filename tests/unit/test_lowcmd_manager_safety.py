from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest

from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.bridges.go2_lowlevel_interface import Go2OwnershipPermit
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import (
    AppConfig,
    Go2LowLevelConfig,
    compute_go2_joint_mapping_hash,
)
from aerogo2.common.enums import (
    Configuration,
    Go2ControlAuthorityState,
    MorphologyRequest,
    SystemState,
)
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import (
    Go2ControlAuthorityStatus,
    Go2LowLevelStatus,
    Go2MotorFeedback,
    LowCmdOwnershipState,
    RCStatus,
    SystemSnapshot,
)
from aerogo2.common.results import OperationResult
from aerogo2.hardware.runtime import HardwareWorld
from aerogo2.landing.impact_aware.integration import Go2JointPositionCommand
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.manager.system_manager import SystemManager
from aerogo2.manager.transition_guards import TransitionGuards
from aerogo2.safety.interlocks import SafetyInterlocks
from aerogo2.safety.safety_monitor import SafetyMonitor


def _enabled_lowcmd_config(config: AppConfig) -> AppConfig:
    joint_names = (
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
    )
    motor_ids = tuple(range(12))
    directions = (1,) * 12
    offsets = (0.0,) * 12
    mapping_version = "test-fixture-v1"
    mapping_hash = compute_go2_joint_mapping_hash(
        mapping_version,
        joint_names,
        motor_ids,
        directions,
        offsets,
    )
    low_level = Go2LowLevelConfig(
        enabled=True,
        low_state_topic="rt/lowstate",
        low_command_topic="rt/lowcmd",
        send_period_s=0.002,
        maximum_jitter_s=0.001,
        low_state_max_age_s=0.02,
        target_ttl_s=0.05,
        acquire_timeout_s=0.5,
        release_timeout_s=0.5,
        safe_hold_policy="capture_current",
        safe_hold_pose_rad=(0.0,) * 12,
        safe_hold_position_tolerance_rad=(0.02,) * 12,
        safe_hold_velocity_tolerance_rad_s=(0.05,) * 12,
        tracking_position_error_limit_rad=(0.2,) * 12,
        safe_hold_ack_timeout_s=0.5,
        restore_mode_form="0",
        restore_mode_name="normal",
        mapping_version=mapping_version,
        mapping_hash=mapping_hash,
        joint_names=joint_names,
        motor_ids=motor_ids,
        directions=directions,
        zero_offsets_rad=offsets,
        q_min_rad=(-2.0,) * 12,
        q_max_rad=(2.0,) * 12,
        dq_max_rad_s=(1.0,) * 12,
        maximum_delta_q_rad=(0.01,) * 12,
        kp=(5.0,) * 12,
        kd=(0.5,) * 12,
        tau_ff_nm=(0.0,) * 12,
        tau_limit_nm=(10.0,) * 12,
        temperature_limit_c=(70.0,) * 12,
    )
    return replace(
        config,
        safety=replace(config.safety, stationary_confirm_s=0.01),
        go2=replace(config.go2, low_level=low_level),
    )


class _FakeLowCmdOwner:
    def __init__(self, clock: ManualClock, mapping_hash: str) -> None:
        self.clock = clock
        self.mapping_hash = mapping_hash
        self.epoch = 41
        self.revoke_reasons: List[str] = []
        self.release_permits: List[Go2OwnershipPermit] = []
        self.submissions: List[Tuple[int, int, str]] = []
        self.connect_calls = 0
        self._status = Go2LowLevelStatus(
            timestamp=clock.monotonic(),
            ownership_state=LowCmdOwnershipState.DISCONNECTED,
        )

    async def connect(self) -> OperationResult:
        self.connect_calls += 1
        now = self.clock.monotonic()
        motors = tuple(
            Go2MotorFeedback(
                motor_id=index,
                joint_name=f"joint_{index}",
                q_rad=0.0,
                dq_rad_s=0.0,
                tau_est_nm=0.0,
                temperature_c=25.0,
                lost=False,
                timestamp=now,
            )
            for index in range(12)
        )
        self._status = Go2LowLevelStatus(
            timestamp=now,
            connected=True,
            ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
            healthy=True,
            low_state_timestamp=now,
            low_state_age_s=0.0,
            mapping_hash_verified=True,
            active_mapping_hash=self.mapping_hash,
            fault_reason=None,
            motors=motors,
        )
        return OperationResult.success("observing")

    async def acquire(self, permit: Go2OwnershipPermit) -> OperationResult:
        assert permit.mapping_hash == self.mapping_hash
        now = self.clock.monotonic()
        self._status = replace(
            self._status,
            timestamp=now,
            connected=True,
            ownership_state=LowCmdOwnershipState.HOLDING,
            owner_epoch=self.epoch,
            healthy=True,
            low_state_timestamp=now,
            publisher_active=True,
            writer_alive=True,
            watchdog_healthy=True,
            safe_hold_active=True,
            safe_hold_settled=True,
            high_level_released=True,
            network_exclusivity_verified=True,
            mapping_hash_verified=True,
            active_mapping_hash=self.mapping_hash,
            fault_reason=None,
        )
        return OperationResult.success("acquired", {"ownership_epoch": self.epoch})

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        self.submissions.append((command.sequence, ownership_epoch, mapping_hash))
        self._status = replace(
            self._status,
            timestamp=self.clock.monotonic(),
            ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
            target_sequence=command.sequence,
            mailbox_staged_target_sequence=command.sequence,
            target_deadline=command.valid_until_s,
            safe_hold_active=False,
            safe_hold_settled=False,
        )
        return OperationResult.success("submitted")

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        assert ownership_epoch in (None, self.epoch)
        self.revoke_reasons.append(reason)
        self._status = replace(
            self._status,
            timestamp=self.clock.monotonic(),
            ownership_state=LowCmdOwnershipState.SAFE_HOLD,
            healthy=True,
            target_sequence=None,
            mailbox_staged_target_sequence=None,
            target_age_s=None,
            target_deadline=None,
            safe_hold_active=True,
            safe_hold_settled=True,
            fault_reason=None,
        )
        return OperationResult.success("safe hold")

    async def release(
        self,
        permit: Go2OwnershipPermit,
        reason: str,
        *,
        ownership_epoch: int,
    ) -> OperationResult:
        del reason
        assert ownership_epoch == self.epoch
        self.release_permits.append(permit)
        self._status = replace(
            self._status,
            timestamp=self.clock.monotonic(),
            ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
            owner_epoch=0,
            healthy=True,
            publisher_active=False,
            writer_alive=False,
            watchdog_healthy=False,
            safe_hold_active=False,
            safe_hold_settled=False,
            high_level_released=False,
            network_exclusivity_verified=False,
            fault_reason=None,
        )
        return OperationResult.success("released")

    def status(self) -> Go2LowLevelStatus:
        return replace(self._status, timestamp=self.clock.monotonic())


@pytest.mark.asyncio
async def test_manager_connects_explicit_observe_only_lowstate_tier(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    commissioned = _enabled_lowcmd_config(app_config)
    low_level = replace(
        commissioned.go2.low_level,
        enabled=False,
        observe_only_enabled=True,
        q_min_rad=None,
        q_max_rad=None,
        kp=None,
        kd=None,
    )
    config = replace(commissioned, go2=replace(commissioned.go2, low_level=low_level))
    assert low_level.mapping_hash is not None
    owner = _FakeLowCmdOwner(clock, low_level.mapping_hash)
    manager = SystemManager(
        config=config,
        pixhawk=FakePixhawk(clock=clock, esc_mapping=config.esc.slots),
        f446=FakeF446(config=config.f446, clock=clock),
        go2=FakeGo2(clock=clock),
        landing_controller=SafeDescentController(config),
        clock=clock,
        go2_low_level=owner,
    )

    await manager._connect_go2_low_level_if_enabled()
    assert owner.connect_calls == 1
    await manager.refresh_snapshot()
    assert manager.snapshot.go2.low_level_status.ownership_state is (
        LowCmdOwnershipState.OBSERVE_ONLY
    )


async def _manager_at_flight_ready(
    config: AppConfig,
    clock: ManualClock,
) -> Tuple[SystemManager, FakePixhawk, FakeGo2, _FakeLowCmdOwner]:
    pixhawk = FakePixhawk(clock=clock, esc_mapping=config.esc.slots)
    f446 = FakeF446(
        config=config.f446,
        clock=clock,
        initial_configuration=Configuration.FLIGHT,
    )
    go2 = FakeGo2(clock=clock)
    mapping_hash = config.go2.low_level.mapping_hash
    assert mapping_hash is not None
    owner = _FakeLowCmdOwner(clock, mapping_hash)
    manager = SystemManager(
        config=config,
        pixhawk=pixhawk,
        f446=f446,
        go2=go2,
        landing_controller=SafeDescentController(config),
        clock=clock,
        go2_low_level=owner,
    )
    await manager.start()
    await pixhawk.connect()
    pixhawk.inject_telemetry_cycle()
    await f446.connect()
    await go2.connect()
    await owner.connect()
    manager._state_machine._state = SystemState.FLIGHT_READY
    go2.inject_status(
        joints_locked=True,
        locomotion_mode="JOINT_LOCK",
        standing=True,
        stable=True,
        moving=False,
        controller_active=False,
    )
    manager.accept_rc_status(
        RCStatus(
            connected=True,
            failsafe=False,
            channels={config.rc.flight_enable_channel: 1000},
            flight_enable=False,
            morphology_request=MorphologyRequest.FLIGHT_REQUEST,
            timestamp=clock.monotonic(),
        )
    )
    await manager.refresh_snapshot()
    return manager, pixhawk, go2, owner


@pytest.mark.asyncio
async def test_manager_lowcmd_lifecycle_is_ground_gated_and_epoch_bound(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, go2, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()

    acquired = await manager.acquire_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )
    assert acquired.ok
    assert manager.snapshot.go2.low_level_status.ownership_state is LowCmdOwnershipState.HOLDING

    go2.inject_status(
        joints_locked=False,
        locomotion_mode="STOPPED",
        controller_active=False,
    )
    manager._state_machine._state = SystemState.AUTO_LANDING
    manager._autoland_active = True
    pixhawk.inject_status(armed=True, landed=False)
    now = clock.monotonic()
    command = Go2JointPositionCommand(
        sequence=7,
        timestamp_s=now,
        valid_until_s=now + 0.04,
        joint_positions_rad=(0.1,) * 12,
        desired_contact_forces_world_n=(0.0,) * 12,
    )
    activated = await manager.activate_go2_low_level_control(command)
    assert activated.ok
    assert owner.submissions == [(7, owner.epoch, owner.mapping_hash)]

    unsafe_release = await manager.release_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )
    assert not unsafe_release.ok
    assert unsafe_release.code == "GO2_LOWCMD_RELEASE_STATE_INVALID"

    revoked = await manager.revoke_go2_low_level_control("test abort")
    assert revoked.ok
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.status().writer_alive

    manager._state_machine._state = SystemState.TOUCHDOWN_VERIFY
    pixhawk.inject_status(armed=False, landed=True)
    await manager.refresh_snapshot()
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    released = await manager.release_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )
    assert released.ok
    assert manager.state is SystemState.GO2_GROUND_HANDOVER
    assert not owner.status().owns_lowcmd
    assert owner.release_permits[-1].pixhawk_disarmed
    assert owner.release_permits[-1].rotors_stopped

    # The pre-handover cached mode must never complete the handover, even if it
    # already says JOINT_LOCK.
    go2.inject_status(
        joints_locked=True,
        locomotion_mode="JOINT_LOCK",
        standing=True,
        stable=True,
        moving=False,
        controller_active=False,
    )
    await manager.tick()
    assert manager.state is SystemState.GO2_GROUND_HANDOVER

    clock.advance(0.001)
    go2.inject_status(
        joints_locked=True,
        locomotion_mode="JOINT_LOCK",
        standing=True,
        stable=True,
        moving=False,
        controller_active=False,
    )
    await manager.tick()
    assert manager.state is SystemState.FLIGHT_READY, [
        violation.code for violation in manager.violations
    ]
    assert "GO2_LOWCMD_OWNER_MISSING" not in {violation.code for violation in manager.violations}


@pytest.mark.asyncio
async def test_ground_arm_authorization_rechecks_owner_health_after_gate_ack(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (
        await manager.acquire_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
        )
    ).ok
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()

    enable_started = asyncio.Event()
    finish_enable = asyncio.Event()
    calls: list[bool] = []
    original_gate = pixhawk.set_ground_arm_authorization

    async def delayed_gate(enabled: bool, ttl_s: float) -> OperationResult:
        calls.append(enabled)
        if enabled:
            enable_started.set()
            await finish_enable.wait()
        return await original_gate(enabled, ttl_s)

    monkeypatch.setattr(pixhawk, "set_ground_arm_authorization", delayed_gate)
    authorize_task = asyncio.create_task(manager.authorize_ground_arm())
    await asyncio.wait_for(enable_started.wait(), timeout=0.5)
    owner._status = replace(
        owner._status,
        healthy=False,
        watchdog_healthy=False,
        fault_reason="owner failed while Pixhawk ACK was pending",
    )
    finish_enable.set()

    result = await authorize_task

    assert not result.ok
    assert result.code == "GO2_LOWCMD_NOT_READY_FOR_ARM"
    assert calls == [True, False]
    assert not pixhawk.ground_arm_authorization_active()
    assert not manager.snapshot.ground_arm_authorized


@pytest.mark.asyncio
async def test_lowcmd_release_waits_for_authorize_and_revokes_gate_before_owner(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (
        await manager.acquire_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
        )
    ).ok
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()

    enable_started = asyncio.Event()
    finish_enable = asyncio.Event()
    events: list[str] = []
    original_gate = pixhawk.set_ground_arm_authorization
    original_release = owner.release

    async def delayed_gate(enabled: bool, ttl_s: float) -> OperationResult:
        if enabled:
            events.append("gate-enable-start")
            enable_started.set()
            await finish_enable.wait()
            events.append("gate-enable-ack")
        else:
            events.append("gate-disable")
        return await original_gate(enabled, ttl_s)

    async def recorded_release(
        permit: Go2OwnershipPermit,
        reason: str,
        *,
        ownership_epoch: int,
    ) -> OperationResult:
        events.append("owner-release")
        return await original_release(
            permit,
            reason,
            ownership_epoch=ownership_epoch,
        )

    monkeypatch.setattr(pixhawk, "set_ground_arm_authorization", delayed_gate)
    monkeypatch.setattr(owner, "release", recorded_release)
    authorize_task = asyncio.create_task(manager.authorize_ground_arm())
    await asyncio.wait_for(enable_started.wait(), timeout=0.5)
    release_task = asyncio.create_task(
        manager.release_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
            reason="serialized release regression",
        )
    )
    await asyncio.sleep(0)

    assert not release_task.done()
    assert not owner.release_permits
    finish_enable.set()
    authorized = await authorize_task
    released = await release_task

    assert authorized.ok
    assert released.ok
    assert events.index("gate-disable") < events.index("owner-release")
    assert not pixhawk.ground_arm_authorization_active()
    assert not manager.snapshot.ground_arm_authorized


@pytest.mark.asyncio
async def test_lowcmd_release_refuses_when_arm_gate_revoke_is_not_acknowledged(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (
        await manager.acquire_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
        )
    ).ok
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (await manager.authorize_ground_arm()).ok

    async def failed_gate_revoke(enabled: bool, ttl_s: float) -> OperationResult:
        assert not enabled
        assert ttl_s == 0.0
        return OperationResult.failure(
            "PIXHAWK_ARM_AUTH_GATE_TIMEOUT",
            "injected missing disable ACK",
        )

    monkeypatch.setattr(
        pixhawk,
        "set_ground_arm_authorization",
        failed_gate_revoke,
    )

    result = await manager.release_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )

    assert not result.ok
    assert result.code == "GO2_LOWCMD_RELEASE_ARM_GATE_REVOKE_FAILED"
    assert not owner.release_permits
    assert pixhawk.ground_arm_authorization_active()


@pytest.mark.asyncio
async def test_supervised_stop_revokes_arm_gate_before_other_failed_stops(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, _ = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (
        await manager.acquire_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
        )
    ).ok
    assert (await manager.authorize_ground_arm()).ok

    events: list[str] = []
    original_gate = pixhawk.set_ground_arm_authorization

    async def recorded_gate(enabled: bool, ttl_s: float) -> OperationResult:
        if not enabled:
            events.append("gate-disable")
        return await original_gate(enabled, ttl_s)

    async def failed_transform_stop() -> OperationResult:
        events.append("f446-stop-failed")
        return OperationResult.failure("F446_STOP_FAILED", "injected F446 stop failure")

    async def failed_lowcmd_revoke(reason: str) -> OperationResult:
        del reason
        events.append("lowcmd-revoke-failed")
        return OperationResult.failure("GO2_LOWCMD_REVOKE_FAILED", "injected owner failure")

    monkeypatch.setattr(pixhawk, "set_ground_arm_authorization", recorded_gate)
    monkeypatch.setattr(manager, "_stop_transform_outputs", failed_transform_stop)
    monkeypatch.setattr(manager, "_revoke_go2_low_level_internal", failed_lowcmd_revoke)

    result = await manager.stop_supervised()

    assert not result.ok
    assert events[:3] == [
        "gate-disable",
        "f446-stop-failed",
        "lowcmd-revoke-failed",
    ]
    assert not pixhawk.ground_arm_authorization_active()
    assert not manager.snapshot.ground_arm_authorized


@pytest.mark.asyncio
async def test_lowcmd_autoland_start_is_blocked_until_first_policy_transaction_exists(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, _, _, _ = await _manager_at_flight_ready(config, clock)
    manager._state_machine._state = SystemState.AUTO_LANDING_READY

    result = await manager.start_autoland()

    assert not result.ok
    assert result.code == "COORDINATED_ACTUATION_NOT_CONFIGURED"
    assert manager.state is SystemState.AUTO_LANDING_READY
    assert not manager.snapshot.autoland_active


@pytest.mark.asyncio
async def test_manager_allows_ground_handback_recovery_after_publisher_closed(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    assert (
        await manager.acquire_go2_low_level_control(
            operator_confirmed=True,
            robot_supported=True,
        )
    ).ok

    # Model a previous release whose local publisher Close succeeded but whose
    # exact MotionSwitcher Select/Check transaction did not. LowState then
    # becomes unusable. The only safe recovery is still allowed on verified
    # disarmed/landed/zero-rotor ground evidence.
    owner._status = replace(
        owner._status,
        ownership_state=LowCmdOwnershipState.FAULT,
        healthy=False,
        low_state_timestamp=0.0,
        publisher_active=False,
        writer_alive=False,
        watchdog_healthy=False,
        safe_hold_active=False,
        safe_hold_settled=False,
        high_level_released=True,
        motors=tuple(replace(motor, lost=True) for motor in owner._status.motors),
        fault_reason="publisher closed; high-level handback pending",
    )
    manager._state_machine._state = SystemState.FAULT
    pixhawk.inject_status(armed=False, landed=True)
    await manager.refresh_snapshot()

    released = await manager.release_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
        reason="retry closed-endpoint handback",
    )

    assert released.ok
    assert owner.release_permits
    assert not owner.status().ownership_pending


@pytest.mark.asyncio
async def test_idempotent_acquire_rebinds_executor_only_for_settled_holding(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, owner = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()

    acquired = await manager.acquire_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )
    assert acquired.ok
    assert manager._impact_lowcmd_executor is not None

    # Model cancellation/restart of the manager-side call after the owner has
    # already completed its critical handoff.
    manager._impact_lowcmd_executor = None
    rebound = await manager.acquire_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )

    assert rebound.ok
    assert manager._impact_lowcmd_executor is not None
    assert owner.status().owner_epoch == owner.epoch


def test_lowcmd_owner_cannot_transition_into_manual_morphology_state(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    now = safe_walk_snapshot.timestamp
    low_level = Go2LowLevelStatus(
        timestamp=now,
        connected=True,
        ownership_state=LowCmdOwnershipState.HOLDING,
        owner_epoch=5,
        healthy=True,
        low_state_timestamp=now,
        writer_alive=True,
        watchdog_healthy=True,
        safe_hold_active=True,
        safe_hold_settled=True,
        high_level_released=True,
        network_exclusivity_verified=True,
        mapping_hash_verified=True,
        active_mapping_hash=config.go2.low_level.mapping_hash,
        fault_reason=None,
    )
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.FLIGHT_READY,
        configuration=Configuration.FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=config.f446.expected_flight_state,
        ),
        go2=replace(safe_walk_snapshot.go2, low_level_status=low_level),
    )

    guard = TransitionGuards(config).evaluate(
        SystemState.FLIGHT_READY,
        SystemState.MANUAL_POSITIONING,
        snapshot,
    )

    assert not guard.permitted
    assert "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED" in guard.codes


def test_arm_readiness_uses_fresh_lowstate_and_settled_hold_not_stale_sport_state(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    now = safe_walk_snapshot.timestamp
    low_level = Go2LowLevelStatus(
        timestamp=now,
        connected=True,
        ownership_state=LowCmdOwnershipState.HOLDING,
        owner_epoch=8,
        healthy=True,
        low_state_timestamp=now,
        writer_alive=True,
        watchdog_healthy=True,
        safe_hold_active=True,
        safe_hold_settled=True,
        high_level_released=True,
        network_exclusivity_verified=True,
        mapping_hash_verified=True,
        active_mapping_hash=config.go2.low_level.mapping_hash,
        fault_reason=None,
    )
    snapshot = replace(
        safe_walk_snapshot,
        state=SystemState.FLIGHT_READY,
        configuration=Configuration.FLIGHT,
        f446=replace(
            safe_walk_snapshot.f446,
            state=config.f446.expected_flight_state,
        ),
        go2=replace(
            safe_walk_snapshot.go2,
            timestamp=0.0,
            joints_locked=False,
            controller_active=True,
            low_level_status=low_level,
        ),
    )

    ready = SafetyInterlocks(config).can_enter_flight_ready(snapshot)
    unsettled = SafetyInterlocks(config).can_enter_flight_ready(
        replace(
            snapshot,
            go2=replace(
                snapshot.go2,
                low_level_status=replace(low_level, safe_hold_settled=False),
            ),
        )
    )

    assert ready.permitted, ready.messages
    assert not unsettled.permitted
    assert "GO2_LOWCMD_NOT_READY_FOR_ARM" in unsettled.codes


@pytest.mark.asyncio
async def test_fault_entry_attempts_all_independent_stops_after_revoke_failure(
    app_config: AppConfig,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, _, _, _ = await _manager_at_flight_ready(config, clock)
    revoke = AsyncMock(return_value=OperationResult.failure("REVOKE_FAILED", "failed"))
    stop_setpoints = AsyncMock(side_effect=BridgeError("setpoint failure"))
    stop_f446 = AsyncMock(return_value=OperationResult.failure("F446_FAILED", "failed"))
    monkeypatch.setattr(manager, "_revoke_go2_low_level_internal", revoke)
    monkeypatch.setattr(manager, "_stop_setpoints", stop_setpoints)
    monkeypatch.setattr(manager, "_safe_f446_stop", stop_f446)
    manager._suppress_fault_entry_stop = True

    with pytest.raises(BridgeError):
        await manager._enter_fault_state(manager.snapshot)

    revoke.assert_awaited_once()
    stop_setpoints.assert_awaited_once()
    stop_f446.assert_awaited_once()


class _RuntimeManagerWithTransientTickFailure:
    def __init__(self) -> None:
        self.calls = 0

    async def tick(self) -> Tuple[()]:
        self.calls += 1
        raise RuntimeError("transient monitor failure")


class _RuntimeOwnerStatusSequence:
    def __init__(self) -> None:
        self.calls = 0

    def status(self) -> Go2LowLevelStatus:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient owner status failure")
        if self.calls == 2:
            return Go2LowLevelStatus(
                ownership_state=LowCmdOwnershipState.SAFE_HOLD,
                owner_epoch=1,
                writer_alive=True,
                safe_hold_active=True,
            )
        return Go2LowLevelStatus(ownership_state=LowCmdOwnershipState.OBSERVE_ONLY)


class _HardwareWorldKeepaliveHarness(HardwareWorld):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.manager = _RuntimeManagerWithTransientTickFailure()  # type: ignore[assignment]
        self.go2_low_level = _RuntimeOwnerStatusSequence()  # type: ignore[assignment]
        self.shutdown_calls = 0

    async def start(self) -> OperationResult:
        return OperationResult.success("started")

    async def shutdown(self) -> OperationResult:
        self.shutdown_calls += 1
        if self.shutdown_calls == 1:
            return OperationResult.failure("OWNER_PENDING", "pending")
        return OperationResult.success("stopped")


@pytest.mark.asyncio
async def test_runtime_keepalive_survives_status_and_tick_exceptions_until_release(
    app_config: AppConfig,
) -> None:
    world = _HardwareWorldKeepaliveHarness(app_config)
    stop_event = asyncio.Event()
    stop_event.set()

    result = await world.monitor_until_stopped(stop_event)

    assert result.ok
    assert world.shutdown_calls == 2
    assert world.manager.calls == 1
    assert world.go2_low_level.calls == 3


@pytest.mark.asyncio
async def test_fault_cannot_clear_to_boot_safe_while_lowcmd_handoff_is_pending(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    manager, pixhawk, _, _ = await _manager_at_flight_ready(config, clock)
    clock.advance(config.safety.stationary_confirm_s + 0.001)
    pixhawk.inject_telemetry_cycle()
    await manager.refresh_snapshot()
    acquired = await manager.acquire_go2_low_level_control(
        operator_confirmed=True,
        robot_supported=True,
    )
    assert acquired.ok
    manager._state_machine._state = SystemState.FAULT

    result = await manager.clear_fault()

    assert not result.ok
    assert result.code == "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED"
    assert manager.state is SystemState.FAULT


def test_safety_monitor_uses_lowcmd_health_instead_of_mode_six(
    app_config: AppConfig,
    safe_walk_snapshot: SystemSnapshot,
) -> None:
    config = _enabled_lowcmd_config(app_config)
    base = safe_walk_snapshot
    snapshot = replace(
        base,
        state=SystemState.AUTO_LANDING,
        pixhawk=replace(base.pixhawk, armed=True, landed=False),
        go2=replace(
            base.go2,
            joints_locked=False,
            controller_active=False,
            control_authority=Go2ControlAuthorityStatus(
                state=Go2ControlAuthorityState.LOWCMD_ACTIVE,
                timestamp=base.timestamp,
                generation=1,
                ownership_epoch=99,
                reason="test active owner",
            ),
            low_level_status=Go2LowLevelStatus(
                timestamp=base.timestamp,
                connected=True,
                ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
                owner_epoch=99,
                healthy=True,
                low_state_timestamp=base.timestamp,
                low_state_age_s=0.0,
                target_sequence=4,
                target_deadline=base.timestamp + 0.02,
                publisher_active=True,
                writer_alive=True,
                watchdog_healthy=True,
                safe_hold_active=False,
                high_level_released=True,
                network_exclusivity_verified=True,
                mapping_hash_verified=True,
                active_mapping_hash=config.go2.low_level.mapping_hash,
                fault_reason=None,
                motors=tuple(
                    Go2MotorFeedback(
                        motor_id=index,
                        joint_name=f"joint_{index}",
                        q_rad=0.0,
                        dq_rad_s=0.0,
                        tau_est_nm=0.0,
                        temperature_c=25.0,
                        lost=False,
                        timestamp=base.timestamp,
                    )
                    for index in range(12)
                ),
                tracking_error_timestamp=base.timestamp,
                tracking_reference_write_timestamp=base.timestamp - 0.001,
                tracking_reference_write_generation=1,
                tracking_reference_q_rad=(0.0,) * 12,
                position_error_rad=(0.0,) * 12,
            ),
        ),
    )

    codes = {item.code for item in SafetyMonitor(config).evaluate(snapshot)}

    assert "GO2_JOINT_LOCK_LOST" not in codes
    assert "GO2_LOWCMD_OWNER_UNHEALTHY" not in codes
    assert "GO2_LOWCMD_OWNER_MISSING" not in codes
    assert "GO2_LOWCMD_MPC_INACTIVE" not in codes


def test_fault_state_with_epoch_still_owns_lowcmd() -> None:
    status = Go2LowLevelStatus(
        ownership_state=LowCmdOwnershipState.FAULT,
        owner_epoch=12,
    )

    assert status.owns_lowcmd
    assert status.ownership_pending


@pytest.mark.parametrize(
    "status",
    (
        Go2LowLevelStatus(owner_epoch=3),
        Go2LowLevelStatus(writer_alive=True),
        Go2LowLevelStatus(high_level_released=True),
        Go2LowLevelStatus(safe_hold_active=True),
        Go2LowLevelStatus(safe_hold_settled=True),
        Go2LowLevelStatus(watchdog_healthy=True),
        Go2LowLevelStatus(network_exclusivity_verified=True),
        Go2LowLevelStatus(target_sequence=1),
        Go2LowLevelStatus(target_deadline=2.0),
        Go2LowLevelStatus(ownership_state=LowCmdOwnershipState.RELEASING),
    ),
)
def test_inconsistent_lowcmd_status_fails_closed_as_ownership_pending(
    status: Go2LowLevelStatus,
) -> None:
    assert status.ownership_pending
