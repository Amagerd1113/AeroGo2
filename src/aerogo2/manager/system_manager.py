"""Safety-owned orchestration for every AeroGo2 command path."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from aerogo2.bridges.f446_interface import F446Interface
from aerogo2.bridges.go2_interface import Go2Interface
from aerogo2.bridges.go2_lowlevel_interface import (
    Go2LowLevelInterface,
    Go2OwnershipPermit,
)
from aerogo2.bridges.pixhawk_interface import PixhawkInterface
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446State,
    Go2ControlAuthorityState,
    RuntimeMode,
    SafetySeverity,
    SystemState,
)
from aerogo2.common.exceptions import AeroGo2Error, BridgeError, TransitionRejected
from aerogo2.common.immutable import deep_thaw
from aerogo2.common.models import (
    F446Status,
    Go2ControlAuthorityStatus,
    Go2LowLevelStatus,
    Go2Status,
    ImpactLandingRecoveryEvidence,
    LandingCommand,
    LandingEstimate,
    LowCmdOwnershipState,
    OperatorRequest,
    RCStatus,
    SafetyViolation,
    SystemSnapshot,
    TransitionRecord,
    snapshot_to_dict,
)
from aerogo2.common.results import GuardResult, OperationResult
from aerogo2.landing.controller_base import LandingControllerBase
from aerogo2.landing.impact_aware.executor import ImpactAwareLowCmdExecutor
from aerogo2.landing.impact_aware.integration import Go2JointPositionCommand
from aerogo2.landing.safety_filter import LandingSafetyFilter
from aerogo2.manager.state_machine import StateMachine
from aerogo2.manager.transition_guards import TRANSFORM_STATES, TransitionGuards
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.go2_contact import assess_foot_contact
from aerogo2.safety.interlocks import SafetyInterlocks
from aerogo2.safety.pixhawk_freshness import (
    pixhawk_ground_state_is_current,
    pixhawk_touchdown_sources_are_current,
    timestamps_are_coherent,
)
from aerogo2.safety.safety_monitor import SafetyMonitor
from aerogo2.safety.watchdog import timestamp_age, timestamp_is_fresh

_GROUND_ARM_AUTHORIZATION_TTL_S = 30.0


class SystemManager:
    """The sole owner of bridge calls.

    The shell and simulator submit requests here. Neither is given a bridge
    reference, which makes the `Shell -> SystemManager -> Guard -> Bridge`
    boundary enforceable and testable.
    """

    def __init__(
        self,
        config: AppConfig,
        pixhawk: PixhawkInterface,
        f446: F446Interface,
        go2: Go2Interface,
        landing_controller: LandingControllerBase,
        safety_monitor: Optional[SafetyMonitor] = None,
        clock: Optional[Clock] = None,
        runtime_mode: RuntimeMode = RuntimeMode.DRY_RUN,
        event_logger: Optional[Any] = None,
        rc_monitor: Optional[RCMonitor] = None,
        go2_low_level: Optional[Go2LowLevelInterface] = None,
        impact_recovery_source: Optional[Callable[[], ImpactLandingRecoveryEvidence]] = None,
    ) -> None:
        self.config = config
        self._pixhawk = pixhawk
        self._f446 = f446
        self._go2 = go2
        self._go2_low_level = go2_low_level
        self._impact_recovery_source = impact_recovery_source
        self._impact_lowcmd_executor: Optional[ImpactAwareLowCmdExecutor] = None
        # Serialize the background tick against LowCmd ownership transfers.
        # Shell commands are already dispatched serially; this closes the
        # remaining snapshot-check-await-act race with the monitor task.
        self._operation_lock = asyncio.Lock()
        self._landing_controller = landing_controller
        self._landing_safety_filter = LandingSafetyFilter(config)
        self._landing_interlocks = SafetyInterlocks(config)
        self._clock = RealClock() if clock is None else clock
        self._runtime_mode = runtime_mode
        self._event_logger = event_logger
        self._rc_monitor = rc_monitor
        self._safety_monitor = SafetyMonitor(config) if safety_monitor is None else safety_monitor
        self._state_machine = StateMachine(
            TransitionGuards(config), self._clock, record_logger=self._log_transition
        )
        self._state_machine.set_entry_action(SystemState.BOOT_SAFE, self._enter_boot_safe)
        self._state_machine.set_entry_action(
            SystemState.MANUAL_POSITIONING,
            self._enter_manual_positioning,
        )
        self._state_machine.set_entry_action(SystemState.WALK, self._leave_manual_positioning)
        self._state_machine.set_entry_action(
            SystemState.GO2_JOINT_LOCK_WAIT,
            self._enter_go2_joint_lock_wait,
        )
        self._state_machine.set_entry_action(
            SystemState.GO2_GROUND_HANDOVER,
            self._enter_go2_joint_lock_wait,
        )
        self._state_machine.set_entry_action(
            SystemState.FLIGHT_READY,
            self._leave_manual_positioning,
        )
        self._state_machine.set_entry_action(
            SystemState.LANDING_COMPLIANT,
            self._enter_landing_compliant,
        )
        self._state_machine.set_entry_action(SystemState.FAULT, self._enter_fault_state)
        self._state_machine.set_entry_action(SystemState.EMERGENCY_STOP, self._enter_emergency_stop)
        self._state_machine.subscribe(self._on_state_transition)

        now = self._clock.monotonic()
        self._rc = RCStatus(timestamp=now)
        self._estimate = LandingEstimate(timestamp=now)
        self._snapshot = SystemSnapshot(timestamp=now, state=SystemState.BOOT_SAFE)
        self._go2_control_authority = Go2ControlAuthorityStatus(timestamp=now)
        self._impact_landing_session_id = 0
        self._impact_recovery = ImpactLandingRecoveryEvidence()
        self._last_confirmed_impact_recovery: Optional[ImpactLandingRecoveryEvidence] = None
        self._impact_recovery_wait_started_at: Optional[float] = None
        self._impact_recovery_finalization_started_at: Optional[float] = None
        self._impact_recovery_finalization_completed_at: Optional[float] = None
        self._post_touchdown_stable_since: Optional[float] = None
        self._post_touchdown_last_stability_check_at: Optional[float] = None
        self._impact_landing_exit_ready = False
        self._impact_recovery_safe_hold_confirmed = False
        self._impact_recovery_setpoints_stopped = False
        self._active_violations: Dict[str, SafetyViolation] = {}
        self._violation_history: List[SafetyViolation] = []
        self._last_landing_command = LandingCommand(timestamp=now)
        self._last_landing_update: Optional[float] = None
        self._next_landing_update_at: Optional[float] = None
        self._airborne_since: Optional[float] = None
        self._airborne_confirmed = False
        self._touchdown_since: Optional[float] = None
        self._touchdown_height_reference: Optional[float] = None
        # A failed post-impact recovery may not be reclassified by the simpler
        # manual-touchdown path while the same continuous landed episode holds.
        self._aborted_impact_touchdown_latched = False
        self._aborted_impact_airborne_since: Optional[float] = None
        self._aborted_impact_airborne_last_check_at: Optional[float] = None
        self._landing_contact_since: Optional[float] = None
        self._landing_compliant_since: Optional[float] = None
        self._go2_stationary_since: Optional[float] = None
        # Ownership transfer has a stronger dwell requirement than morphology
        # changes: the aircraft must also be positively disarmed and landed.
        self._lowcmd_ground_stationary_since: Optional[float] = None
        self._lowcmd_ground_stationary_source_start: Optional[Tuple[float, float]] = None
        # A pre-LowCmd SportModeState sample must never be reused to complete
        # the post-landing LowCmd-to-high-level handover.
        self._go2_ground_handover_started_at: Optional[float] = None
        self._f446_current_clear_since: Optional[float] = None
        self._f446_current_clear_key: Optional[Tuple[Any, ...]] = None
        self._touchdown_confirmed = False
        self._autoland_active = False
        self._setpoint_active = False
        self._ground_arm_authorized = False
        self._ground_arm_authorization_expires_at: Optional[float] = None
        self._maintenance_mode = False
        self._suppress_fault_entry_stop = False
        self._operator_confirmed_configuration: Optional[Configuration] = None
        self._manual_marked_configuration: Optional[Configuration] = None
        self._manual_motion_deadline: Optional[float] = None
        self._manual_last_direction: Optional[str] = None
        self._manual_motion_started = False
        self._go2_joint_lock_deadline: Optional[float] = None
        # Distinguish harmless cleanup after a partial startup from loss of
        # authoritative telemetry after the system has entered an operational
        # state.  FAULT is not itself proof that the vehicle is on the ground.
        self._ever_entered_operational_state = False
        self._go2_joint_lock_entered_at: Optional[float] = None
        self._go2_joint_lock_unsafe_since: Optional[float] = None
        self._operator_joint_lock_confirmed = False
        self._started = False

    @property
    def state(self) -> SystemState:
        return self._state_machine.state

    @property
    def started(self) -> bool:
        return self._started

    @property
    def snapshot(self) -> SystemSnapshot:
        return self._snapshot

    @property
    def runtime_mode(self) -> RuntimeMode:
        return self._runtime_mode

    @property
    def transitions(self) -> Tuple[TransitionRecord, ...]:
        return self._state_machine.history

    @property
    def violations(self) -> Tuple[SafetyViolation, ...]:
        return tuple(self._active_violations.values())

    @property
    def last_landing_command(self) -> LandingCommand:
        return self._last_landing_command

    def _control_writes_allowed(self) -> bool:
        return self._runtime_mode is RuntimeMode.DRY_RUN or (
            self._runtime_mode is RuntimeMode.HARDWARE and self.config.system.hardware_write_enabled
        )

    async def _on_state_transition(self, record: TransitionRecord) -> None:
        if record.new_state in (
            SystemState.BOOT_SAFE,
            SystemState.MANUAL_POSITIONING,
            SystemState.WALK,
            SystemState.GO2_JOINT_LOCK_WAIT,
            SystemState.LANDING_COMPLIANT,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            self._operator_joint_lock_confirmed = False

        if (
            record.previous_state is SystemState.FLIGHT_READY
            and record.new_state is not SystemState.FLIGHT_READY
        ):
            revoked = await self._revoke_ground_arm_authorization_unlocked(
                f"state changed to {record.new_state.name}"
            )
            if not revoked.ok and record.new_state is not SystemState.FAULT:
                # A normal-state transition must not silently commit while the
                # independent Pixhawk arm authority remains ambiguous.  Raising
                # makes StateMachine enter FAULT, whose entry action retries the
                # revocation independently.
                raise BridgeError(
                    f"{revoked.code}: ground-arm gate revoke failed during "
                    f"{record.previous_state.name}->{record.new_state.name}: "
                    f"{revoked.message}"
                )
            # For a direct FLIGHT_READY->FAULT transition, allow the FAULT
            # entry action to run and perform its own mandatory retry.  Raising
            # here would skip that entry action in StateMachine's publish path.

        if (
            record.previous_state is SystemState.FLIGHT_READY
            and record.new_state is SystemState.FLIGHT_MANUAL
        ):
            self._reset_touchdown_cycle()
        elif record.new_state in (
            SystemState.BOOT_SAFE,
            SystemState.WALK,
            SystemState.FLIGHT_READY,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            self._reset_touchdown_cycle()

    async def start(self) -> OperationResult:
        """Start services but leave every device disconnected in BOOT_SAFE."""

        if self._started:
            return OperationResult.success("System manager already started")
        self._started = True
        if self._event_logger is not None:
            self._event_logger.start()
        self._emit("SYSTEM_STARTED")
        await self.refresh_snapshot()
        return OperationResult.success("System started in BOOT_SAFE")

    async def _connect_go2_low_level_if_enabled(self) -> None:
        """Connect the read-only LowState endpoint when either tier requests it."""

        if not self.config.go2.low_level.observation_enabled:
            return
        if self._go2_low_level is None:
            raise BridgeError(
                "Go2 LowState observation is enabled but no low-level bridge was injected"
            )
        result = await self._go2_low_level.connect()
        if not result.ok:
            raise BridgeError(f"{result.code}: {result.message}")

    async def connect_all(self) -> OperationResult:
        """Connect injected bridges and adopt only a verified idle configuration."""

        try:
            await self._pixhawk.connect()
            await self._f446.connect()
            await self._go2.connect()
            await self._connect_go2_low_level_if_enabled()
            await self.refresh_snapshot()
            if self.state is SystemState.BOOT_SAFE:
                await self._adopt_boot_configuration()
                await self.refresh_snapshot()
            self._emit("DEVICE_CONNECTED", device="all")
            return OperationResult.success(f"All devices connected; state={self.state.name}")
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            await self._fault("DEVICE_CONNECT_FAILED", str(exc))
            return OperationResult.failure("DEVICE_CONNECT_FAILED", str(exc))

    async def connect_device(self, name: str) -> OperationResult:
        normalized = name.strip().lower()
        try:
            if normalized == "pixhawk":
                await self._pixhawk.connect()
            elif normalized == "f446":
                await self._f446.connect()
            elif normalized == "go2":
                await self._go2.connect()
                await self._connect_go2_low_level_if_enabled()
            else:
                return OperationResult.failure("UNKNOWN_DEVICE", f"Unknown device '{name}'")
            await self.refresh_snapshot()
            all_connected = (
                self._snapshot.pixhawk.connected
                and self._snapshot.f446.connected
                and (self._snapshot.go2.connected or not self.config.go2.enabled)
            )
            if all_connected and self.state is SystemState.BOOT_SAFE:
                await self._adopt_boot_configuration()
                await self.refresh_snapshot()
            self._emit("DEVICE_CONNECTED", device=normalized)
            return OperationResult.success(f"{normalized} connected")
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            await self._fault("DEVICE_CONNECT_FAILED", f"{normalized}: {exc}")
            return OperationResult.failure("DEVICE_CONNECT_FAILED", str(exc))

    async def disconnect_all(self) -> OperationResult:
        stop_result = await self.stop_supervised()
        if not stop_result.ok:
            return OperationResult.failure(
                "DISCONNECT_INHIBITED",
                f"Supervised outputs did not stop: {stop_result.message}",
            )
        ready = await self._prepare_disconnect()
        if not ready.ok:
            return ready
        release = await self._release_go2_low_level_for_shutdown("disconnect all")
        if not release.ok:
            return OperationResult.failure(
                "DISCONNECT_INHIBITED",
                f"LowCmd ownership was not safely released: {release.message}",
            )
        failures: List[str] = []
        for name, bridge in (
            ("go2", self._go2),
            ("f446", self._f446),
            ("pixhawk", self._pixhawk),
        ):
            try:
                await bridge.disconnect()
            except (BridgeError, OSError, RuntimeError) as exc:
                failures.append(f"{name}: {exc}")
        await self.refresh_snapshot()
        if failures:
            await self._fault("DISCONNECT_FAILED", "; ".join(failures))
            return OperationResult.failure("DISCONNECT_FAILED", "; ".join(failures))
        self._emit("DEVICE_DISCONNECTED", device="all")
        return OperationResult.success(
            f"All simulated devices disconnected; manager state is {self.state.name}"
        )

    async def disconnect_device(self, name: str) -> OperationResult:
        normalized = name.strip().lower()
        stop_result = await self.stop_supervised()
        if not stop_result.ok:
            return OperationResult.failure(
                "DISCONNECT_INHIBITED",
                f"Supervised outputs did not stop: {stop_result.message}",
            )
        ready = await self._prepare_disconnect()
        if not ready.ok:
            return ready
        release = await self._release_go2_low_level_for_shutdown(f"disconnect device {normalized}")
        if not release.ok:
            return OperationResult.failure(
                "DISCONNECT_INHIBITED",
                f"LowCmd ownership was not safely released: {release.message}",
            )
        try:
            if normalized == "pixhawk":
                await self._pixhawk.disconnect()
            elif normalized == "f446":
                await self._f446.disconnect()
            elif normalized == "go2":
                await self._go2.disconnect()
            else:
                return OperationResult.failure("UNKNOWN_DEVICE", f"Unknown device '{name}'")
        except (BridgeError, OSError, RuntimeError) as exc:
            await self._fault("DISCONNECT_FAILED", f"{normalized}: {exc}")
            return OperationResult.failure("DISCONNECT_FAILED", str(exc))
        await self.refresh_snapshot()
        self._emit("DEVICE_DISCONNECTED", device=normalized)
        return OperationResult.success(
            f"Simulated {normalized} disconnected; manager state is {self.state.name}"
        )

    async def refresh_snapshot(self) -> SystemSnapshot:
        if self.state not in {
            SystemState.BOOT_SAFE,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        }:
            self._ever_entered_operational_state = True
        pixhawk = self._pixhawk.get_status()
        f446 = self._f446.get_status()
        go2 = self._go2.get_status()
        now = self._clock.monotonic()
        low_level = go2.low_level_status
        if self._go2_low_level is not None and self.config.go2.low_level.observation_enabled:
            try:
                low_level = self._go2_low_level.status()
            except Exception as exc:
                low_level = Go2LowLevelStatus(
                    timestamp=now,
                    connected=False,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    healthy=False,
                    fault_reason=f"low-level status failed: {exc}",
                )
        elif self.config.go2.low_level.observation_enabled:
            low_level = Go2LowLevelStatus(
                timestamp=now,
                connected=False,
                ownership_state=LowCmdOwnershipState.DISCONNECTED,
                healthy=False,
                fault_reason="enabled LowState observer was not injected into SystemManager",
            )
        if low_level.low_state_timestamp > 0.0:
            low_level = replace(
                low_level,
                low_state_age_s=timestamp_age(now, low_level.low_state_timestamp),
            )
        if self._rc_monitor is not None:
            self._rc = self._rc_monitor.update_from_channels(
                pixhawk.rc_channels,
                failsafe=pixhawk.rc_failsafe,
                connected=pixhawk.connected,
                timestamp=now,
            )
        go2_velocity = cast(Tuple[float, float, float], tuple(go2.body_velocity))
        if go2_velocity == (0.0, 0.0, 0.0) and go2.velocity_mps != 0.0:
            go2_velocity = (go2.velocity_mps, 0.0, 0.0)
        go2_motion_invalid = len(go2_velocity) != 3 or any(
            not math.isfinite(item) for item in go2_velocity
        )
        pixhawk = replace(
            pixhawk,
            timestamp=pixhawk.heartbeat_timestamp,
            message_age_s=timestamp_age(now, pixhawk.heartbeat_timestamp),
            rc_failsafe=pixhawk.rc_failsafe,
            attitude_rpy=(pixhawk.roll_rad, pixhawk.pitch_rad, pixhawk.yaw_rad),
            local_position=(0.0, 0.0, -pixhawk.relative_altitude_m),
            local_velocity=(0.0, 0.0, pixhawk.vertical_velocity_mps),
            rc_channels=dict(self._rc.channels),
            esc_rpm={item.slot: item.rpm for item in pixhawk.esc},
            esc_online={item.slot: item.healthy for item in pixhawk.esc},
        )
        firmware_configuration = self._configuration_from_f446(f446)
        configuration: Configuration
        if not f446.connected:
            self._operator_confirmed_configuration = None
            self._manual_marked_configuration = None
        if firmware_configuration is not Configuration.UNKNOWN:
            configuration = firmware_configuration
            configuration_source = "f446_limit"
            self._operator_confirmed_configuration = None
        elif (
            self._operator_confirmed_configuration is not None
            and f446.connected
            and not f446.faulted
            and f446.state is F446State.IDLE
            and f446.duty == 0
        ):
            configuration = self._operator_confirmed_configuration
            configuration_source = "operator"
        else:
            configuration = Configuration.UNKNOWN
            configuration_source = "unconfirmed"
        current_clear_key = (
            f446.connected,
            configuration,
            f446.threshold_adc,
            self.config.f446.current_safe_margin_adc,
            self.config.f446.current_clear_hold_s,
        )
        if current_clear_key != self._f446_current_clear_key:
            self._f446_current_clear_since = None
            self._f446_current_clear_key = current_clear_key
        if self._f446_current_margin_is_clear(f446, now):
            if self._f446_current_clear_since is None:
                self._f446_current_clear_since = now
        else:
            self._f446_current_clear_since = None
        f446 = replace(
            f446,
            message_age_s=timestamp_age(now, f446.timestamp),
            raw_state=f446.state.value,
            configuration=configuration.value,
            r_is_raw=0 if f446.r_is_adc is None else f446.r_is_adc,
            l_is_raw=0 if f446.l_is_adc is None else f446.l_is_adc,
            used_raw=0 if f446.used_current_adc is None else f446.used_current_adc,
            threshold_raw=0 if f446.threshold_adc is None else f446.threshold_adc,
        )
        authority = self._reconcile_go2_control_authority(go2, low_level, now)
        go2 = replace(
            go2,
            message_age_s=timestamp_age(now, go2.timestamp),
            body_velocity=go2_velocity,
            moving=(
                go2_motion_invalid
                or any(
                    abs(item) >= self.config.safety.stationary_velocity_mps for item in go2_velocity
                )
                or go2.controller_active
            ),
            low_level_status=low_level,
            control_authority=authority,
        )
        stationary = (
            go2.connected
            and math.isfinite(go2.velocity_mps)
            and abs(go2.velocity_mps) < self.config.safety.stationary_velocity_mps
            and go2.stable
            and not go2.controller_active
            and not go2.moving
        )
        if stationary:
            if self._go2_stationary_since is None:
                self._go2_stationary_since = now
        else:
            self._go2_stationary_since = None
        lowcmd_owner_stationary = (
            low_level.ownership_pending and self._lowcmd_motor_feedback_is_stationary(low_level)
        )
        lowcmd_ground_stationary = (
            (lowcmd_owner_stationary if low_level.ownership_pending else stationary)
            and pixhawk_ground_state_is_current(
                pixhawk,
                now,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
            )
            and not pixhawk.armed
            and pixhawk.landed
        )
        if lowcmd_ground_stationary:
            if self._lowcmd_ground_stationary_since is None:
                self._lowcmd_ground_stationary_since = now
                self._lowcmd_ground_stationary_source_start = (
                    pixhawk.heartbeat_timestamp,
                    pixhawk.landed_state_timestamp,
                )
        else:
            self._lowcmd_ground_stationary_since = None
            self._lowcmd_ground_stationary_source_start = None
        operator = OperatorRequest(
            timestamp=self._rc.timestamp,
            flight_enable=self._rc.flight_enable,
            morphology_request=self._rc.morphology_request.value,
            auto_landing_request=self._rc.auto_landing_request.value,
            manual_override=self._rc.manual_override,
        )
        try:
            bridge_authorized = self._pixhawk.ground_arm_authorization_active()
        except Exception:
            # The snapshot must never claim a gate is closed merely because
            # its status callback failed.  ``tick`` treats this unknown state
            # as possibly active and executes the acknowledged revoke path.
            bridge_authorized = True
        if self._ground_arm_authorized and not bridge_authorized:
            self._ground_arm_authorized = False
            self._ground_arm_authorization_expires_at = None
        ground_arm_authorized = self._ground_arm_authorized and bridge_authorized
        ground_arm_authorization_expires_at = (
            self._ground_arm_authorization_expires_at if ground_arm_authorized else None
        )
        if self._impact_recovery_source is not None:
            try:
                recovery = self._impact_recovery_source()
                if not isinstance(recovery, ImpactLandingRecoveryEvidence):
                    raise TypeError("impact recovery source did not return typed evidence")
                self._impact_recovery = self._validate_impact_recovery_source_sample(
                    recovery,
                    now,
                )
            except Exception as exc:
                self._impact_recovery = ImpactLandingRecoveryEvidence(
                    timestamp=now,
                    valid_until=now,
                    landing_session_id=self._impact_landing_session_id,
                    reason=f"impact recovery source failed: {type(exc).__name__}: {exc}",
                )
        if not go2.connected:
            self._operator_joint_lock_confirmed = False
        joint_lock_confirmed = go2.connected and (
            go2.joints_locked or self._operator_joint_lock_confirmed
        )
        joint_lock_source = (
            "telemetry"
            if go2.connected and go2.joints_locked
            else "operator"
            if go2.connected and self._operator_joint_lock_confirmed
            else "none"
        )
        self._snapshot = SystemSnapshot(
            timestamp=now,
            state=self.state,
            pixhawk=pixhawk,
            f446=f446,
            go2=go2,
            operator=operator,
            rc=self._rc,
            configuration=configuration,
            configuration_source=configuration_source,
            landing_estimate=self._estimate,
            autoland_active=self._autoland_active,
            external_setpoint_active=self._setpoint_active,
            maintenance_mode=self._maintenance_mode,
            joint_lock_confirmed=joint_lock_confirmed,
            joint_lock_source=joint_lock_source,
            ground_arm_authorized=ground_arm_authorized,
            ground_arm_authorization_expires_at=ground_arm_authorization_expires_at,
            active_fault_codes=tuple(sorted(self._active_violations)),
            impact_landing_session_id=self._impact_landing_session_id,
            impact_recovery=self._impact_recovery,
            impact_recovery_wait_started_at=self._impact_recovery_wait_started_at,
            impact_recovery_finalization_started_at=(self._impact_recovery_finalization_started_at),
            post_touchdown_stable_since=self._post_touchdown_stable_since,
            post_touchdown_last_stability_check_at=(self._post_touchdown_last_stability_check_at),
            post_touchdown_stable_dwell_complete=(
                self._post_touchdown_stable_since is not None
                and self._post_touchdown_last_stability_check_at is not None
                and now - self._post_touchdown_stable_since
                >= self.config.safety.post_touchdown_stable_confirm_s
                and now - self._post_touchdown_last_stability_check_at
                <= self.config.safety.post_touchdown_stability_max_check_gap_s
            ),
            impact_landing_exit_ready=self._impact_landing_exit_ready,
        )
        return self._snapshot

    def accept_rc_status(self, status: RCStatus) -> None:
        """Accept telemetry produced by RCMonitor; this never sends RC data."""

        self._rc = status

    def accept_landing_estimate(self, estimate: LandingEstimate) -> None:
        """Inject estimator output only into the deterministic dry-run world.

        A future hardware estimator must be constructor-bound and carry its
        own source identity/generation; a shell or UI must not manufacture a
        landing observation that can advance the safety state machine.
        """

        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            raise RuntimeError("hardware landing estimates must come from a bound source")
        if not isinstance(estimate, LandingEstimate):
            raise TypeError("estimate must be a LandingEstimate")
        self._estimate = estimate

    def accept_impact_landing_recovery_evidence(
        self,
        evidence: ImpactLandingRecoveryEvidence,
    ) -> None:
        """Inject recovery evidence only into the deterministic dry-run world.

        Hardware must use the constructor-injected read-only source so a shell
        or UI cannot manufacture completion booleans.
        """

        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            raise RuntimeError("hardware recovery evidence must come from the bound source")
        if not isinstance(evidence, ImpactLandingRecoveryEvidence):
            raise TypeError("evidence must be ImpactLandingRecoveryEvidence")
        self._impact_recovery = self._validate_impact_recovery_source_sample(
            evidence,
            self._clock.monotonic(),
        )

    def _reset_impact_landing_completion(self, *, new_session: bool) -> None:
        if new_session:
            self._impact_landing_session_id += 1
            self._aborted_impact_touchdown_latched = False
        self._clear_aborted_impact_airborne_dwell()
        self._impact_recovery = ImpactLandingRecoveryEvidence()
        self._last_confirmed_impact_recovery = None
        self._impact_recovery_wait_started_at = None
        self._impact_recovery_finalization_started_at = None
        self._impact_recovery_finalization_completed_at = None
        self._post_touchdown_stable_since = None
        self._post_touchdown_last_stability_check_at = None
        self._impact_landing_exit_ready = False
        self._impact_recovery_safe_hold_confirmed = False
        self._impact_recovery_setpoints_stopped = False

    def _clear_aborted_impact_airborne_dwell(self) -> bool:
        """Discard partial evidence for clearing a post-impact abort latch."""

        changed = bool(
            self._aborted_impact_airborne_since is not None
            or self._aborted_impact_airborne_last_check_at is not None
        )
        self._aborted_impact_airborne_since = None
        self._aborted_impact_airborne_last_check_at = None
        return changed

    def _validate_impact_recovery_source_sample(
        self,
        candidate: ImpactLandingRecoveryEvidence,
        now: float,
    ) -> ImpactLandingRecoveryEvidence:
        """Fence replay/re-stamping of normal recovery completion evidence."""

        if not candidate.confirmed:
            return candidate
        if candidate.landing_session_id != self._impact_landing_session_id:
            return ImpactLandingRecoveryEvidence(
                timestamp=now,
                valid_until=now,
                landing_session_id=self._impact_landing_session_id,
                reason="confirmed recovery evidence belongs to another landing session",
            )
        previous = self._last_confirmed_impact_recovery
        if previous is not None:
            if candidate.sequence == previous.sequence:
                if candidate == previous:
                    return previous
                return ImpactLandingRecoveryEvidence(
                    timestamp=now,
                    valid_until=now,
                    landing_session_id=self._impact_landing_session_id,
                    reason="recovery evidence sequence was replayed with changed content",
                )
            domain_changed = (
                candidate.sequence < previous.sequence
                or candidate.timestamp <= previous.timestamp
                or candidate.go2_ownership_epoch != previous.go2_ownership_epoch
                or candidate.contact_epoch != previous.contact_epoch
                or candidate.fc_session_id != previous.fc_session_id
                or candidate.fc_control_epoch != previous.fc_control_epoch
                or candidate.fc_transport_generation != previous.fc_transport_generation
                or candidate.residual_zero_ack_timestamp != previous.residual_zero_ack_timestamp
                or candidate.residual_zero_execution_timestamp
                != previous.residual_zero_execution_timestamp
                or candidate.residual_zero_status_timestamp
                <= previous.residual_zero_status_timestamp
                or (
                    candidate.clear_through_command_sequence is not None
                    and previous.clear_through_command_sequence is not None
                    and candidate.clear_through_command_sequence
                    < previous.clear_through_command_sequence
                )
            )
            if domain_changed:
                return ImpactLandingRecoveryEvidence(
                    timestamp=now,
                    valid_until=now,
                    landing_session_id=self._impact_landing_session_id,
                    reason=(
                        "recovery evidence sequence, time, identity, contact epoch, "
                        "or FC clear watermark regressed"
                    ),
                )
        self._last_confirmed_impact_recovery = candidate
        return candidate

    def _set_go2_control_authority(
        self,
        state: Go2ControlAuthorityState,
        *,
        now: float,
        ownership_epoch: int,
        reason: str,
        timeout_s: Optional[float] = None,
        restart_transition: bool = False,
    ) -> Go2ControlAuthorityStatus:
        previous = self._go2_control_authority
        same_identity = (
            not restart_transition
            and previous.state is state
            and previous.ownership_epoch == ownership_epoch
        )
        if timeout_s is not None and same_identity and previous.transition_pending:
            started_at = previous.transition_started_at
            deadline = previous.transition_deadline
        elif timeout_s is not None:
            started_at = now
            deadline = now + timeout_s
        else:
            started_at = None
            deadline = None
        self._go2_control_authority = Go2ControlAuthorityStatus(
            state=state,
            timestamp=now,
            transition_started_at=started_at,
            transition_deadline=deadline,
            generation=previous.generation + (0 if same_identity else 1),
            ownership_epoch=ownership_epoch,
            reason=reason,
        )
        return self._go2_control_authority

    def _reconcile_go2_control_authority(
        self,
        go2: Go2Status,
        low_level: Go2LowLevelStatus,
        now: float,
    ) -> Go2ControlAuthorityStatus:
        """Reconcile one explicit authority state without trusting SportMode in LowCmd."""

        owner_state = low_level.ownership_state
        if owner_state is LowCmdOwnershipState.ACQUIRING:
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.LOWCMD_ACQUIRING,
                now=now,
                ownership_epoch=low_level.owner_epoch,
                reason="MotionSwitcher release and first safe-hold write are pending",
                timeout_s=self.config.go2.low_level.acquire_timeout_s,
            )
        if owner_state is LowCmdOwnershipState.MPC_ACTIVE:
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.LOWCMD_ACTIVE,
                now=now,
                ownership_epoch=low_level.owner_epoch,
                reason="exclusive LowCmd owner has an active MPC target lease",
            )
        if owner_state in {
            LowCmdOwnershipState.HOLDING,
            LowCmdOwnershipState.SAFE_HOLD,
        }:
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.LOWCMD_SAFE_HOLD,
                now=now,
                ownership_epoch=low_level.owner_epoch,
                reason="exclusive LowCmd owner is retaining the verified safe hold",
            )
        if owner_state is LowCmdOwnershipState.RELEASING:
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
                now=now,
                ownership_epoch=low_level.owner_epoch,
                reason="LowCmd endpoint is closing and high-level authority is being restored",
                timeout_s=self.config.go2.low_level.release_timeout_s,
            )
        if owner_state is LowCmdOwnershipState.FAULT or low_level.ownership_pending:
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.FAULT,
                now=now,
                ownership_epoch=low_level.owner_epoch,
                reason=low_level.fault_reason or "LowCmd ownership facts are inconsistent",
            )

        previous = self._go2_control_authority
        if previous.state is Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING:
            fence = self._go2_ground_handover_started_at
            causal_joint_lock = bool(
                fence is not None
                and go2.connected
                and go2.joints_locked
                and go2.timestamp > fence
                and timestamp_is_fresh(now, go2.timestamp, self.config.safety.go2_timeout_s)
            )
            if causal_joint_lock:
                return self._set_go2_control_authority(
                    Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK,
                    now=now,
                    ownership_epoch=0,
                    reason="post-handover SportMode sample confirmed JOINT_LOCK",
                )
            deadline = previous.transition_deadline
            if deadline is not None and now >= deadline:
                return self._set_go2_control_authority(
                    Go2ControlAuthorityState.FAULT,
                    now=now,
                    ownership_epoch=previous.ownership_epoch,
                    reason="high-level authority reacquisition timed out",
                )
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
                now=now,
                ownership_epoch=previous.ownership_epoch,
                reason="waiting for a causally-new JOINT_LOCK sample",
                timeout_s=self.config.go2.low_level.release_timeout_s,
            )

        if (
            go2.connected
            and go2.joints_locked
            and timestamp_is_fresh(now, go2.timestamp, self.config.safety.go2_timeout_s)
        ):
            return self._set_go2_control_authority(
                Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK,
                now=now,
                ownership_epoch=0,
                reason="fresh SportMode sample confirms high-level JOINT_LOCK",
            )
        return self._set_go2_control_authority(
            Go2ControlAuthorityState.UNKNOWN,
            now=now,
            ownership_epoch=0,
            reason="neither high-level JOINT_LOCK nor exclusive LowCmd authority is proven",
        )

    async def acquire_go2_low_level_control(
        self,
        *,
        operator_confirmed: bool = False,
        robot_supported: bool = False,
    ) -> OperationResult:
        async with self._operation_lock:
            return await self._acquire_go2_low_level_control_unlocked(
                operator_confirmed=operator_confirmed,
                robot_supported=robot_supported,
            )

    async def _acquire_go2_low_level_control_unlocked(
        self,
        *,
        operator_confirmed: bool,
        robot_supported: bool,
    ) -> OperationResult:
        """Acquire the sole LowCmd writer while still safely on the ground.

        Acquisition is intentionally separate from MPC activation.  The
        expected flight sequence is JOINT_LOCK -> ground-only acquisition ->
        operator/Pixhawk arm -> low-rate target submission in AUTO_LANDING.
        """

        unavailable = self._lowcmd_unavailable_result()
        if unavailable is not None:
            return unavailable
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "HARDWARE_WRITE_DISABLED",
                "LowCmd ownership requires an explicitly unlocked hardware process",
            )
        if self.state is not SystemState.FLIGHT_READY:
            return OperationResult.failure(
                "INVALID_STATE",
                "LowCmd may be acquired only from FLIGHT_READY before rotor arming",
            )
        if type(operator_confirmed) is not bool or not operator_confirmed:
            return OperationResult.failure(
                "OPERATOR_CONFIRMATION_REQUIRED",
                "An explicit operator confirmation is required for LowCmd ownership transfer",
            )
        if type(robot_supported) is not bool or not robot_supported:
            return OperationResult.failure(
                "ROBOT_SUPPORT_CONFIRMATION_REQUIRED",
                "Confirm that the vehicle is mechanically supported before LowCmd acquisition",
            )

        await self.refresh_snapshot()
        status = self._snapshot.go2.low_level_status
        if status.ownership_pending:
            if (
                status.ownership_state is LowCmdOwnershipState.HOLDING
                and self._lowcmd_status_healthy(status)
            ):
                rebound = self._bind_impact_lowcmd_executor(status)
                if not rebound.ok:
                    return rebound
                self._emit(
                    "GO2_LOWCMD_EXECUTOR_REBOUND",
                    ownership_epoch=status.owner_epoch,
                )
                return OperationResult.success(
                    "Go2 LowCmd ownership is already held; executor binding was reconciled",
                    {"ownership_epoch": status.owner_epoch},
                )
            return OperationResult.failure(
                "GO2_LOWCMD_OWNER_UNHEALTHY",
                status.fault_reason or "The existing LowCmd owner is unhealthy",
            )
        ground = self._lowcmd_ground_transfer_result(require_joint_lock=True)
        if not ground.ok:
            return ground
        acquiring = self._set_go2_control_authority(
            Go2ControlAuthorityState.LOWCMD_ACQUIRING,
            now=self._clock.monotonic(),
            ownership_epoch=0,
            reason="operator-authorized MotionSwitcher/LowCmd acquisition started",
            timeout_s=self.config.go2.low_level.acquire_timeout_s,
        )
        self._snapshot = replace(
            self._snapshot,
            go2=replace(self._snapshot.go2, control_authority=acquiring),
        )
        try:
            permit = self._build_lowcmd_permit(
                operator_authorized=operator_confirmed,
                robot_supported=robot_supported,
                timeout_s=self.config.go2.low_level.acquire_timeout_s,
                reason="operator-confirmed ground acquisition before flight",
            )
            assert self._go2_low_level is not None
            result = await self._go2_low_level.acquire(permit)
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            await self.refresh_snapshot()
            return OperationResult.failure("GO2_LOWCMD_ACQUIRE_FAILED", str(exc))
        if not result.ok:
            await self.refresh_snapshot()
            return result
        await self.refresh_snapshot()
        status = self._snapshot.go2.low_level_status
        if (
            status.ownership_state is not LowCmdOwnershipState.HOLDING
            or status.owner_epoch <= 0
            or not self._lowcmd_status_healthy(status)
        ):
            await self._revoke_go2_low_level_internal("acquisition acknowledgement invalid")
            return OperationResult.failure(
                "GO2_LOWCMD_ACQUIRE_UNCONFIRMED",
                "Owner did not confirm a healthy HOLDING stream after acquisition",
            )
        bound = self._bind_impact_lowcmd_executor(status)
        if not bound.ok:
            await self._revoke_go2_low_level_internal(
                "executor could not bind incomplete mapping/TTL configuration"
            )
            return bound
        self._emit("GO2_LOWCMD_ACQUIRED", ownership_epoch=status.owner_epoch)
        return OperationResult.success(
            "Go2 LowCmd owner acquired and verified in safe-hold",
            {"ownership_epoch": status.owner_epoch},
        )

    async def activate_go2_low_level_control(
        self,
        command: Go2JointPositionCommand,
    ) -> OperationResult:
        async with self._operation_lock:
            return await self._activate_go2_low_level_control_unlocked(command)

    async def _activate_go2_low_level_control_unlocked(
        self,
        command: Go2JointPositionCommand,
    ) -> OperationResult:
        """Submit the leg half only after a future atomic committer authorizes it."""

        unavailable = self._lowcmd_unavailable_result()
        if unavailable is not None:
            return unavailable
        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            return OperationResult.failure(
                "COORDINATED_ACTUATION_NOT_CONFIGURED",
                "Hardware leg activation remains locked until matching flight-controller residual firmware/transport and the cross-device committer are implemented",
            )
        if self.state is not SystemState.AUTO_LANDING or not self._autoland_active:
            return OperationResult.failure(
                "GO2_LOWCMD_ACTIVATION_STATE_INVALID",
                "MPC LowCmd targets are authorized only during active AUTO_LANDING",
            )
        if not isinstance(command, Go2JointPositionCommand):
            return OperationResult.failure(
                "GO2_LOWCMD_TARGET_INVALID",
                "command must be the validated leg half supplied by the atomic committer",
            )
        await self.refresh_snapshot()
        before = self._snapshot.go2.low_level_status
        if not before.owns_lowcmd or not self._lowcmd_status_healthy(before):
            return OperationResult.failure(
                "GO2_LOWCMD_NOT_READY",
                before.fault_reason or "A healthy LowCmd ownership epoch is required",
            )
        executor = self._impact_lowcmd_executor
        if executor is None:
            return OperationResult.failure(
                "GO2_LOWCMD_EXECUTOR_NOT_BOUND",
                "Acquire a new ownership epoch before activating MPC targets",
            )
        try:
            result = await executor.submit(command)
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            await self._revoke_go2_low_level_internal(f"target submission exception: {exc}")
            return OperationResult.failure("GO2_LOWCMD_SUBMIT_FAILED", str(exc))
        if not result.ok:
            await self._revoke_go2_low_level_internal(f"target submission rejected: {result.code}")
            return result
        await self.refresh_snapshot()
        after = self._snapshot.go2.low_level_status
        if (
            after.owner_epoch != before.owner_epoch
            or after.ownership_state is not LowCmdOwnershipState.MPC_ACTIVE
            or after.target_sequence != command.sequence
            or not self._lowcmd_status_healthy(after)
        ):
            await self._revoke_go2_low_level_internal("target activation acknowledgement invalid")
            return OperationResult.failure(
                "GO2_LOWCMD_ACTIVATION_UNCONFIRMED",
                "Owner did not confirm the submitted target in the same ownership epoch",
            )
        self._emit(
            "GO2_LOWCMD_TARGET_ACTIVE",
            ownership_epoch=after.owner_epoch,
            sequence=command.sequence,
        )
        return OperationResult.success(
            "Go2 LowCmd MPC target activated",
            {"ownership_epoch": after.owner_epoch, "sequence": command.sequence},
        )

    async def revoke_go2_low_level_control(
        self,
        reason: str = "operator request",
    ) -> OperationResult:
        """Revoke the MPC target while preserving the sole safe-hold stream."""

        async with self._operation_lock:
            return await self._revoke_go2_low_level_internal(reason)

    async def release_go2_low_level_control(
        self,
        *,
        operator_confirmed: bool = False,
        robot_supported: bool = False,
        reason: str = "operator-confirmed ground release",
    ) -> OperationResult:
        async with self._operation_lock:
            return await self._release_go2_low_level_control_unlocked(
                operator_confirmed=operator_confirmed,
                robot_supported=robot_supported,
                reason=reason,
            )

    async def _release_go2_low_level_control_unlocked(
        self,
        *,
        operator_confirmed: bool,
        robot_supported: bool,
        reason: str,
    ) -> OperationResult:
        """Return Go2 authority only after a second ground-only permit."""

        unavailable = self._lowcmd_unavailable_result()
        if unavailable is not None:
            return unavailable
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "HARDWARE_WRITE_DISABLED",
                "LowCmd release requires the explicitly unlocked hardware process that owns the writer",
            )
        if type(operator_confirmed) is not bool or not operator_confirmed:
            return OperationResult.failure(
                "OPERATOR_CONFIRMATION_REQUIRED",
                "An explicit operator confirmation is required to release LowCmd ownership",
            )
        if type(robot_supported) is not bool or not robot_supported:
            return OperationResult.failure(
                "ROBOT_SUPPORT_CONFIRMATION_REQUIRED",
                "Confirm mechanical support before releasing the continuous LowCmd stream",
            )
        if self.state not in {
            SystemState.FLIGHT_READY,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.LANDING_COMPLIANT,
            SystemState.BOOT_SAFE,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        }:
            return OperationResult.failure(
                "GO2_LOWCMD_RELEASE_STATE_INVALID",
                "LowCmd release is allowed only in a verified ground/fault handover state",
            )
        touchdown_handover = self.state is SystemState.TOUCHDOWN_VERIFY

        # The Pixhawk arm gate and the LowCmd handover are one serialized
        # authority transaction.  Clear and acknowledge the former before a
        # release permit can stop the sole LowCmd writer.
        gate_result = await self._revoke_ground_arm_authorization_unlocked(
            f"before Go2 LowCmd release: {reason}"
        )
        if not gate_result.ok:
            return OperationResult.failure(
                "GO2_LOWCMD_RELEASE_ARM_GATE_REVOKE_FAILED",
                "LowCmd release is inhibited because the Pixhawk ground-arm gate "
                f"could not be revoked: {gate_result.code}: {gate_result.message}",
                {"gate_code": gate_result.code},
            )
        await self.refresh_snapshot()
        try:
            gate_still_active = self._pixhawk.ground_arm_authorization_active()
        except Exception as exc:
            return OperationResult.failure(
                "GO2_LOWCMD_RELEASE_ARM_GATE_STATUS_FAILED",
                f"Cannot prove the Pixhawk ground-arm gate inactive: {type(exc).__name__}: {exc}",
            )
        if self._ground_arm_authorized or self._snapshot.ground_arm_authorized or gate_still_active:
            return OperationResult.failure(
                "GO2_LOWCMD_RELEASE_ARM_GATE_ACTIVE",
                "LowCmd release is inhibited until both manager and Pixhawk prove the arm gate inactive",
            )
        status = self._snapshot.go2.low_level_status
        ground = self._lowcmd_ground_transfer_result(require_joint_lock=False)
        if not ground.ok:
            return ground
        already_released = not status.ownership_pending
        if already_released and not touchdown_handover:
            return OperationResult.success("No Go2 LowCmd ownership is held")
        if status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE:
            revoked = await self._revoke_go2_low_level_internal(
                f"safe-hold before release: {reason}"
            )
            if not revoked.ok:
                return revoked
            # Revocation is asynchronous with respect to the physical system;
            # never reuse the pre-revoke ground observation to stop the writer.
            await self.refresh_snapshot()
            ground = self._lowcmd_ground_transfer_result(require_joint_lock=False)
            if not ground.ok:
                return ground
        if not already_released:
            reacquiring = self._set_go2_control_authority(
                Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
                now=self._clock.monotonic(),
                ownership_epoch=status.owner_epoch,
                reason="ground-authorized LowCmd-to-high-level handover started",
                timeout_s=self.config.go2.low_level.release_timeout_s,
            )
            self._snapshot = replace(
                self._snapshot,
                go2=replace(self._snapshot.go2, control_authority=reacquiring),
            )
            try:
                permit = self._build_lowcmd_permit(
                    operator_authorized=operator_confirmed,
                    robot_supported=robot_supported,
                    timeout_s=self.config.go2.low_level.release_timeout_s,
                    reason=reason,
                )
                assert self._go2_low_level is not None
                result = await self._go2_low_level.release(
                    permit,
                    reason,
                    ownership_epoch=status.owner_epoch,
                )
            except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
                await self.refresh_snapshot()
                return OperationResult.failure("GO2_LOWCMD_RELEASE_FAILED", str(exc))
            if not result.ok:
                await self.refresh_snapshot()
                return result
        # Fence out any SportModeState cached before the low-level endpoint
        # closed.  The bridge's MotionSwitcher ACK proves service restoration;
        # a later JOINT_LOCK sample is still required before normal high-level
        # locomotion resumes.
        self._go2_ground_handover_started_at = self._clock.monotonic()
        self._set_go2_control_authority(
            Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
            now=self._go2_ground_handover_started_at,
            ownership_epoch=0,
            reason="high-level service restored; waiting for post-handover JOINT_LOCK",
            timeout_s=self.config.go2.joint_lock_operator_timeout_s,
            restart_transition=True,
        )
        await self.refresh_snapshot()
        after = self._snapshot.go2.low_level_status
        if (
            after.ownership_pending
            or after.ownership_state is not LowCmdOwnershipState.OBSERVE_ONLY
        ):
            return OperationResult.failure(
                "GO2_LOWCMD_RELEASE_UNCONFIRMED",
                "Owner did not confirm writer shutdown and high-level handback",
            )
        self._impact_lowcmd_executor = None
        self._emit("GO2_LOWCMD_RELEASED", reason=reason)
        if touchdown_handover:
            # Require a strictly later SportModeState sample before accepting
            # mode=6.  A fresh-by-age sample may still predate the entire
            # LowCmd ownership interval.
            try:
                await self._state_machine.transition_to(
                    SystemState.GO2_GROUND_HANDOVER,
                    reason="LowCmd released on verified ground; waiting for mode=6",
                    snapshot=self._snapshot,
                )
            except TransitionRejected as exc:
                await self._fault("GO2_GROUND_HANDOVER_REJECTED", str(exc))
                return OperationResult.failure("GO2_GROUND_HANDOVER_REJECTED", str(exc))
            await self.refresh_snapshot()
            self._emit("GO2_GROUND_HANDOVER_STARTED")
            return self._joint_lock_operator_required_result()
        return OperationResult.success(
            "Go2 LowCmd endpoint released; high-level JOINT_LOCK confirmation is pending"
        )

    def _lowcmd_unavailable_result(self) -> Optional[OperationResult]:
        if not self.config.go2.low_level.enabled:
            return OperationResult.failure(
                "GO2_LOWCMD_DISABLED",
                "Go2 LowCmd is disabled until all hardware-specific parameters are verified",
            )
        if self._go2_low_level is None:
            return OperationResult.failure(
                "GO2_LOWCMD_NOT_INJECTED",
                "No exclusive Go2 LowCmd owner was injected into SystemManager",
            )
        return None

    def _build_lowcmd_permit(
        self,
        *,
        operator_authorized: bool,
        robot_supported: bool,
        timeout_s: Optional[float],
        reason: str,
    ) -> Go2OwnershipPermit:
        config = self.config.go2.low_level
        if timeout_s is None or config.mapping_version is None or config.mapping_hash is None:
            raise ValueError("LowCmd permit timeout and mapping identity must be configured")
        now = self._clock.monotonic()
        return Go2OwnershipPermit(
            timestamp_s=now,
            valid_until_s=now + timeout_s,
            operator_authorized=operator_authorized,
            robot_supported=robot_supported,
            pixhawk_disarmed=not self._snapshot.pixhawk.armed,
            rotors_stopped=assess_esc_telemetry(
                self._snapshot,
                self.config.esc.slots,
                exact_zero=True,
            ).safe,
            mapping_version=config.mapping_version,
            mapping_hash=config.mapping_hash,
            reason=reason,
        )

    def _lowcmd_ground_transfer_result(self, *, require_joint_lock: bool) -> OperationResult:
        snapshot = self._snapshot
        now = snapshot.timestamp
        low_state = snapshot.go2.low_level_status
        lowcmd_owner_pending = low_state.ownership_pending
        closed_endpoint_recovery = (
            low_state.owner_epoch > 0
            and low_state.ownership_state is LowCmdOwnershipState.FAULT
            and not low_state.publisher_active
            and not low_state.writer_alive
            and not low_state.safe_hold_active
            and not low_state.watchdog_healthy
            and low_state.target_sequence is None
        )
        pixhawk_ground_current = pixhawk_ground_state_is_current(
            snapshot.pixhawk,
            now,
            self.config.safety.pixhawk_timeout_s,
            self.config.safety.touchdown_max_source_age_s,
        )
        if not pixhawk_ground_current:
            return OperationResult.failure(
                "GO2_LOWCMD_PIXHAWK_EVIDENCE_STALE",
                "Fresh Pixhawk heartbeat and landed-state samples are required for a ground ownership transfer",
            )
        if not snapshot.f446.connected or not timestamp_is_fresh(
            now,
            snapshot.f446.timestamp,
            self.config.safety.f446_timeout_s,
        ):
            return OperationResult.failure(
                "GO2_LOWCMD_F446_EVIDENCE_STALE",
                "Fresh F446 status is required for a ground ownership transfer",
            )
        if (require_joint_lock or not lowcmd_owner_pending) and (
            not snapshot.go2.connected
            or not timestamp_is_fresh(
                now,
                snapshot.go2.timestamp,
                self.config.safety.go2_timeout_s,
            )
        ):
            return OperationResult.failure(
                "GO2_LOWCMD_GO2_EVIDENCE_STALE",
                "Fresh Go2 status is required for a ground ownership transfer",
            )
        if not closed_endpoint_recovery:
            maximum_low_state_age = self.config.go2.low_level.low_state_max_age_s
            if (
                maximum_low_state_age is None
                or not low_state.connected
                or not timestamp_is_fresh(
                    now,
                    low_state.low_state_timestamp,
                    maximum_low_state_age,
                )
            ):
                return OperationResult.failure(
                    "GO2_LOWCMD_LOWSTATE_EVIDENCE_STALE",
                    "Fresh LowState is required while a LowCmd endpoint may still exist",
                )
        if snapshot.pixhawk.armed or not snapshot.pixhawk.landed:
            return OperationResult.failure(
                "GO2_LOWCMD_GROUND_TRANSFER_REQUIRED",
                "LowCmd ownership transfer requires Pixhawk disarmed and landed",
            )
        if not assess_esc_telemetry(snapshot, self.config.esc.slots, exact_zero=True).safe:
            return OperationResult.failure(
                "GO2_LOWCMD_ROTORS_NOT_STOPPED",
                "Every configured ESC must be fresh, healthy, and exactly zero RPM",
            )
        if snapshot.configuration is not Configuration.FLIGHT:
            return OperationResult.failure(
                "GO2_LOWCMD_FLIGHT_CONFIGURATION_REQUIRED",
                "The fixed deployed FLIGHT configuration must be verified",
            )
        if snapshot.f446.duty != 0 or snapshot.f446.faulted:
            return OperationResult.failure(
                "GO2_LOWCMD_F446_UNSAFE",
                "The folding mechanism must remain stopped and fault-free",
            )
        if require_joint_lock and not snapshot.go2.joints_locked:
            return OperationResult.failure(
                "GO2_JOINT_LOCK_REQUIRED",
                "Initial LowCmd acquisition requires an authoritative mode=6 JOINT_LOCK",
            )
        if lowcmd_owner_pending and not closed_endpoint_recovery:
            if not self._lowcmd_motor_feedback_is_stationary(low_state):
                return OperationResult.failure(
                    "GO2_LOWCMD_ROBOT_NOT_STATIONARY",
                    "All 12 fresh LowState joint velocities must satisfy the commissioned stationary tolerance",
                )
        elif not snapshot.go2.stable or snapshot.go2.moving:
            return OperationResult.failure(
                "GO2_LOWCMD_ROBOT_NOT_STATIONARY",
                "Go2 must be stable and stationary during ownership transfer",
            )
        if not closed_endpoint_recovery:
            dwell = self._lowcmd_ground_stationary_dwell_result()
            if not dwell.ok:
                return dwell
        return OperationResult.success("LowCmd ground transfer conditions are satisfied")

    def _lowcmd_motor_feedback_is_stationary(self, status: Go2LowLevelStatus) -> bool:
        tolerances = self.config.go2.low_level.safe_hold_velocity_tolerance_rad_s
        if tolerances is None or len(tolerances) != 12 or len(status.motors) != 12:
            return False
        return all(
            not motor.lost
            and motor.dq_rad_s is not None
            and math.isfinite(motor.dq_rad_s)
            and abs(motor.dq_rad_s) <= tolerances[index]
            for index, motor in enumerate(status.motors)
        )

    def _lowcmd_status_healthy(self, status: Go2LowLevelStatus) -> bool:
        config = self.config.go2.low_level
        now = self._clock.monotonic()
        if status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE:
            deadline = status.target_deadline
            phase_valid = (
                status.target_sequence is not None
                and deadline is not None
                and math.isfinite(deadline)
                and now < deadline
                and not status.safe_hold_active
                and not status.safe_hold_settled
            )
        elif status.ownership_state in {
            LowCmdOwnershipState.HOLDING,
            LowCmdOwnershipState.SAFE_HOLD,
        }:
            phase_valid = (
                status.safe_hold_active
                and status.safe_hold_settled
                and status.target_sequence is None
                and status.target_deadline is None
            )
        else:
            phase_valid = False
        return (
            status.owns_lowcmd
            and status.connected
            and status.healthy
            and status.publisher_active
            and status.writer_alive
            and status.watchdog_healthy
            and status.high_level_released
            and status.network_exclusivity_verified
            and status.mapping_hash_verified
            and config.mapping_hash is not None
            and status.active_mapping_hash == config.mapping_hash
            and config.low_state_max_age_s is not None
            and timestamp_is_fresh(
                now,
                status.low_state_timestamp,
                config.low_state_max_age_s,
            )
            and phase_valid
        )

    def _bind_impact_lowcmd_executor(self, status: Go2LowLevelStatus) -> OperationResult:
        """Bind/rebind the low-rate executor to one settled HOLDING epoch."""

        config = self.config.go2.low_level
        if (
            status.ownership_state is not LowCmdOwnershipState.HOLDING
            or not self._lowcmd_status_healthy(status)
        ):
            return OperationResult.failure(
                "GO2_LOWCMD_EXECUTOR_BIND_UNSAFE",
                "Executor binding requires a healthy, settled HOLDING ownership epoch",
            )
        mapping_hash = config.mapping_hash
        maximum_ttl = config.target_ttl_s
        maximum_low_state_age = config.low_state_max_age_s
        if mapping_hash is None or maximum_ttl is None or maximum_low_state_age is None:
            return OperationResult.failure(
                "GO2_LOWCMD_EXECUTOR_CONFIG_INVALID",
                "Mapping hash, target TTL and LowState maximum age must be configured",
            )
        if self._go2_low_level is None:
            return OperationResult.failure(
                "GO2_LOWCMD_NOT_INJECTED",
                "No exclusive Go2 LowCmd owner was injected into SystemManager",
            )
        try:
            self._impact_lowcmd_executor = ImpactAwareLowCmdExecutor(
                self._go2_low_level,
                mapping_hash=mapping_hash,
                ownership_epoch=status.owner_epoch,
                maximum_command_ttl_s=maximum_ttl,
                maximum_low_state_age_s=maximum_low_state_age,
                monotonic_clock=self._clock.monotonic,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._impact_lowcmd_executor = None
            return OperationResult.failure(
                "GO2_LOWCMD_EXECUTOR_CONFIG_INVALID",
                str(exc),
            )
        return OperationResult.success(
            "Impact-aware LowCmd executor bound",
            {"ownership_epoch": status.owner_epoch},
        )

    async def _revoke_go2_low_level_internal(self, reason: str) -> OperationResult:
        if not self.config.go2.low_level.enabled or self._go2_low_level is None:
            return OperationResult.success("No Go2 LowCmd owner was injected")
        try:
            status = self._go2_low_level.status()
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return OperationResult.failure("GO2_LOWCMD_STATUS_FAILED", str(exc))
        if not status.ownership_pending:
            self._impact_lowcmd_executor = None
            return OperationResult.success("No Go2 LowCmd MPC lease is active")
        if status.owner_epoch <= 0:
            return OperationResult.failure(
                "GO2_LOWCMD_STATUS_INCONSISTENT",
                "LowCmd handover is pending but no valid ownership epoch is available",
            )
        try:
            result = await self._go2_low_level.revoke(
                reason,
                ownership_epoch=status.owner_epoch,
            )
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return OperationResult.failure("GO2_LOWCMD_REVOKE_FAILED", str(exc))
        if not result.ok:
            return result
        self._impact_lowcmd_executor = None
        await self.refresh_snapshot()
        after = self._snapshot.go2.low_level_status
        if (
            after.owner_epoch != status.owner_epoch
            or after.ownership_state
            not in {LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.SAFE_HOLD}
            or not after.safe_hold_active
            or not self._lowcmd_status_healthy(after)
        ):
            return OperationResult.failure(
                "GO2_LOWCMD_REVOKE_UNCONFIRMED",
                "The sole writer did not confirm a healthy safe-hold after MPC revocation",
            )
        self._emit("GO2_LOWCMD_REVOKED_TO_SAFE_HOLD", reason=reason)
        return OperationResult.success(
            "MPC target revoked; the same LowCmd owner remains in safe-hold",
            {"ownership_epoch": after.owner_epoch},
        )

    async def _release_go2_low_level_for_shutdown(self, reason: str) -> OperationResult:
        if not self.config.go2.low_level.enabled or self._go2_low_level is None:
            return OperationResult.success("No Go2 LowCmd owner was injected")
        status = self._go2_low_level.status()
        if not status.ownership_pending:
            return OperationResult.success("No Go2 LowCmd ownership is held")
        return OperationResult.failure(
            "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
            "LowCmd remains active. Confirm landed/disarmed/zero-RPM/mechanical support "
            "and call release_go2_low_level_control explicitly before " + reason,
        )

    async def authorize_ground_arm(self) -> OperationResult:
        """Open a short, one-shot window; this method never arms Pixhawk."""

        async with self._operation_lock:
            return await self._authorize_ground_arm_unlocked()

    async def _authorize_ground_arm_unlocked(self) -> OperationResult:
        """Authorize only while one locked authority/readiness transaction remains valid."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "HARDWARE_WRITE_DISABLED",
                "Ground-arm authorization requires an explicitly unlocked hardware process",
            )
        if self.state is not SystemState.FLIGHT_READY:
            return OperationResult.failure(
                "NOT_IN_FLIGHT_READY",
                "Ground-arm authorization requires FLIGHT_READY",
            )
        await self.refresh_snapshot()
        if (
            not pixhawk_ground_state_is_current(
                self._snapshot.pixhawk,
                self._snapshot.timestamp,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
            )
            or self._snapshot.pixhawk.armed
            or not self._snapshot.pixhawk.landed
        ):
            return OperationResult.failure(
                "GROUND_ARM_GROUND_PROOF_INVALID",
                "Ground-arm authorization requires fresh disarmed and landed Pixhawk evidence",
            )
        if self.config.go2.low_level.enabled:
            low_level = self._snapshot.go2.low_level_status
            authority = self._snapshot.go2.control_authority
            if (
                low_level.ownership_state is not LowCmdOwnershipState.HOLDING
                or not self._lowcmd_status_healthy(low_level)
                or authority.state is not Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
                or authority.ownership_epoch != low_level.owner_epoch
            ):
                return OperationResult.failure(
                    "GO2_LOWCMD_NOT_READY_FOR_ARM",
                    "Acquire and verify the exact sole LowCmd HOLDING/SAFE_HOLD authority before arm authorization",
                )
        guard = self._flight_readiness_guard(require_flight_enable_low=True)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0] if guard.codes else "FLIGHT_NOT_READY",
                "; ".join(guard.messages) or "Flight readiness checks failed",
                data=self._flight_readiness_report(require_flight_enable_low=True),
            )
        requested_at = self._clock.monotonic()
        try:
            # The bridge owns the protocol ACK deadline.  An outer wait_for
            # would only cancel this coroutine; it cannot impose a real bound
            # on a blocking transport or prove that the remote gate is closed.
            result = await self._pixhawk.set_ground_arm_authorization(
                True,
                _GROUND_ARM_AUTHORIZATION_TTL_S,
            )
        except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_AUTHORIZATION_FAILED",
                f"Pixhawk arm-gate transaction raised: {type(exc).__name__}: {exc}",
            )
        if not isinstance(result, OperationResult):
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_AUTHORIZATION_PROTOCOL_ERROR",
                "Pixhawk arm gate returned an invalid authorization result",
            )
        try:
            bridge_authorized = self._pixhawk.ground_arm_authorization_active()
        except Exception as exc:
            bridge_authorized = False
            bridge_status_error = f"{type(exc).__name__}: {exc}"
        else:
            bridge_status_error = ""
        if not result.ok or not bridge_authorized:
            return await self._fail_ground_arm_after_ack(
                (
                    result.code
                    if not result.ok and result.code
                    else "GROUND_ARM_AUTHORIZATION_FAILED"
                ),
                (
                    result.message
                    if not result.ok and result.message
                    else "Pixhawk arm gate did not become active"
                    + (f": {bridge_status_error}" if bridge_status_error else "")
                ),
                data=result.data,
            )

        # Treat the bridge ACK as provisional.  Publish the local half, refresh
        # every authority/readiness input, and revoke the physical gate on any
        # post-ACK disagreement.
        self._ground_arm_authorized = True
        self._ground_arm_authorization_expires_at = requested_at + _GROUND_ARM_AUTHORIZATION_TTL_S
        try:
            await self.refresh_snapshot()
        except Exception as exc:
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_POST_ACK_REFRESH_FAILED",
                f"Post-ACK system snapshot failed: {type(exc).__name__}: {exc}",
            )

        if self._clock.monotonic() >= self._ground_arm_authorization_expires_at:
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_AUTHORIZATION_EXPIRED",
                "Ground-arm authorization expired before its ACK transaction could be committed",
            )
        if self.state is not SystemState.FLIGHT_READY:
            return await self._fail_ground_arm_after_ack(
                "NOT_IN_FLIGHT_READY",
                "System left FLIGHT_READY while the Pixhawk arm gate ACK was pending",
            )
        if self._snapshot.pixhawk.armed:
            return await self._fail_ground_arm_after_ack(
                "PIXHAWK_ALREADY_ARMED",
                "Pixhawk became armed before ground-arm authorization was committed",
            )
        if (
            not pixhawk_ground_state_is_current(
                self._snapshot.pixhawk,
                self._snapshot.timestamp,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
            )
            or not self._snapshot.pixhawk.landed
        ):
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_GROUND_PROOF_INVALID",
                "Fresh landed Pixhawk evidence was lost while the arm-gate ACK was pending",
            )
        if self.config.go2.low_level.enabled:
            low_level = self._snapshot.go2.low_level_status
            authority = self._snapshot.go2.control_authority
            if (
                low_level.ownership_state is not LowCmdOwnershipState.HOLDING
                or not self._lowcmd_status_healthy(low_level)
                or authority.state is not Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
                or authority.ownership_epoch != low_level.owner_epoch
            ):
                return await self._fail_ground_arm_after_ack(
                    "GO2_LOWCMD_NOT_READY_FOR_ARM",
                    "LowCmd HOLDING/SAFE_HOLD authority changed while the Pixhawk arm gate ACK was pending",
                )
        guard = self._flight_readiness_guard(require_flight_enable_low=True)
        if not guard.permitted:
            return await self._fail_ground_arm_after_ack(
                guard.codes[0] if guard.codes else "FLIGHT_NOT_READY",
                "; ".join(guard.messages) or "Post-ACK flight readiness checks failed",
                data=self._flight_readiness_report(require_flight_enable_low=True),
            )
        if not self._snapshot.ground_arm_authorized:
            return await self._fail_ground_arm_after_ack(
                "GROUND_ARM_AUTHORIZATION_FAILED",
                "Manager/Pixhawk authorization state was not jointly active after refresh",
            )
        self._emit(
            "GROUND_ARM_AUTHORIZED",
            ttl_s=_GROUND_ARM_AUTHORIZATION_TTL_S,
            rc_channel=self.config.rc.flight_enable_channel,
        )
        return OperationResult.success(
            "Ground authorization active; move RadioMaster CH5 from LOW to HIGH within 30s",
            data={
                "authorized": True,
                "ttl_s": _GROUND_ARM_AUTHORIZATION_TTL_S,
                "rc_channel": self.config.rc.flight_enable_channel,
                "arms_pixhawk": False,
            },
        )

    async def revoke_ground_arm(self) -> OperationResult:
        async with self._operation_lock:
            result = await self._revoke_ground_arm_authorization_unlocked("operator request")
            try:
                await self.refresh_snapshot()
            except Exception as exc:
                return OperationResult.failure(
                    "GROUND_ARM_AUTH_REVOKE_REFRESH_FAILED",
                    f"Ground-arm gate was addressed but the inactive snapshot could not be refreshed: {type(exc).__name__}: {exc}",
                    {"revoke_code": result.code},
                )
            return result

    async def _fail_ground_arm_after_ack(
        self,
        code: str,
        message: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult:
        """Roll back a provisional ACK before exposing its validation failure."""

        rollback = await self._revoke_ground_arm_authorization_unlocked(
            f"post-ACK validation failed: {code}"
        )
        refresh_error = ""
        try:
            await self.refresh_snapshot()
        except Exception as exc:
            refresh_error = f"{type(exc).__name__}: {exc}"
        try:
            bridge_active = self._pixhawk.ground_arm_authorization_active()
        except Exception as exc:
            bridge_active = True
            if not refresh_error:
                refresh_error = f"{type(exc).__name__}: {exc}"
        rollback_confirmed = bool(
            rollback.ok
            and not bridge_active
            and not self._ground_arm_authorized
            and not self._snapshot.ground_arm_authorized
            and not refresh_error
        )
        detail = dict(data or {})
        detail.update(
            {
                "post_ack_failure_code": code,
                "gate_revoke_code": rollback.code,
                "gate_revoke_confirmed": rollback_confirmed,
            }
        )
        if not rollback_confirmed:
            suffix = f"; refresh/status error: {refresh_error}" if refresh_error else ""
            return OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_FAILED",
                f"{message}; provisional Pixhawk gate revocation was not confirmed: "
                f"{rollback.code}: {rollback.message}{suffix}",
                detail,
            )
        return OperationResult.failure(code, message, detail)

    async def _revoke_ground_arm_authorization_unlocked(self, reason: str) -> OperationResult:
        was_active = self._ground_arm_authorized
        self._ground_arm_authorized = False
        self._ground_arm_authorization_expires_at = None
        try:
            # The concrete bridge must implement its own bounded ACK exchange.
            # Cancelling an arbitrary bridge await here cannot prove remote
            # revocation and would create a misleading safety guarantee.
            result = await self._pixhawk.set_ground_arm_authorization(False, 0.0)
        except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._emit("GROUND_ARM_AUTH_REVOKE_FAILED", reason=reason, error=str(exc))
            return OperationResult.failure("GROUND_ARM_AUTH_REVOKE_FAILED", str(exc))
        if not isinstance(result, OperationResult):
            self._emit(
                "GROUND_ARM_AUTH_REVOKE_FAILED",
                reason=reason,
                error="invalid Pixhawk gate result",
            )
            return OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_PROTOCOL_ERROR",
                "Pixhawk arm gate returned an invalid revoke result",
            )
        if not result.ok:
            return result
        try:
            bridge_active = self._pixhawk.ground_arm_authorization_active()
        except Exception as exc:
            self._emit("GROUND_ARM_AUTH_REVOKE_FAILED", reason=reason, error=str(exc))
            return OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_STATUS_FAILED",
                f"Cannot confirm the Pixhawk arm gate inactive: {type(exc).__name__}: {exc}",
            )
        if bridge_active:
            self._emit(
                "GROUND_ARM_AUTH_REVOKE_FAILED",
                reason=reason,
                error="Pixhawk gate remained active after revoke ACK",
            )
            return OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_UNCONFIRMED",
                "Pixhawk arm gate remained active after its revoke acknowledgement",
            )
        self._emit("GROUND_ARM_AUTH_REVOKED", reason=reason, was_active=was_active)
        return OperationResult.success(
            "Ground authorization revoked; Pixhawk arm/disarm state was not changed",
            data={
                "authorized": False,
                "was_active": was_active,
                "pixhawk_armed": self._snapshot.pixhawk.armed,
            },
        )

    async def walk_stop(self) -> OperationResult:
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Go2 control writes are locked in this process",
            )
        if self.state is not SystemState.WALK:
            return OperationResult.failure("INVALID_STATE", "walk stop requires WALK")
        try:
            if not await self._go2.request_stop():
                raise BridgeError("Go2 rejected the stop request")
        except (BridgeError, OSError, RuntimeError) as exc:
            await self._fault("GO2_STOP_FAILED", str(exc))
            return OperationResult.failure("GO2_STOP_FAILED", str(exc))
        await self.refresh_snapshot()
        return OperationResult.success("Go2 high-level locomotion stop accepted")

    async def walk_stand(self) -> OperationResult:
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Go2 control writes are locked in this process",
            )
        if self.state is not SystemState.WALK:
            return OperationResult.failure("INVALID_STATE", "walk stand requires WALK")
        await self.refresh_snapshot()
        if self._snapshot.configuration is not Configuration.WALK:
            return OperationResult.failure(
                "WALK_CONFIGURATION_NOT_CONFIRMED",
                "Go2 stand is inhibited until WALK configuration is confirmed",
            )
        try:
            if not await self._go2.request_stand():
                raise BridgeError("Go2 rejected the stand request")
        except (BridgeError, OSError, RuntimeError) as exc:
            await self._fault("GO2_STAND_FAILED", str(exc))
            return OperationResult.failure("GO2_STAND_FAILED", str(exc))
        await self.refresh_snapshot()
        return OperationResult.success("Go2 high-level stand request accepted")

    async def reset_controller(self) -> OperationResult:
        if self.state in (SystemState.AUTO_LANDING_READY, SystemState.AUTO_LANDING):
            return OperationResult.failure(
                "CONTROLLER_ACTIVE",
                "Abort automatic landing before resetting the controller",
            )
        self._landing_controller.reset()
        self._last_landing_update = None
        self._next_landing_update_at = None
        self._last_landing_command = LandingCommand(
            timestamp=self._clock.monotonic(),
            valid=False,
            reason="controller reset",
        )
        self._emit("CONTROLLER_RESET")
        await self.refresh_snapshot()
        return OperationResult.success("Landing controller reset while inactive")

    async def config_diff(self) -> OperationResult:
        try:
            disk = load_config(self.config.source_path)
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("CONFIG_INVALID", str(exc))
        current = self._flatten_mapping(self.config.raw)
        candidate = self._flatten_mapping(disk.raw)
        differences = [
            {
                "key": key,
                "runtime": current.get(key),
                "disk": candidate.get(key),
            }
            for key in sorted(set(current) | set(candidate))
            if current.get(key) != candidate.get(key)
        ]
        return OperationResult.success(
            f"{len(differences)} configuration difference(s)",
            data={"differences": differences},
        )

    async def reload_config(self) -> OperationResult:
        """Validate disk configuration without creating mixed-runtime ownership.

        SimulationWorld owns the fake bridges and RC monitor, so this manager
        cannot atomically rebuild every config consumer. Any actual change is
        therefore rejected and must be applied by restarting the process.
        """

        await self.refresh_snapshot()
        connected = (
            self._snapshot.pixhawk.connected
            or self._snapshot.f446.connected
            or self._snapshot.go2.connected
        )
        if self.state is not SystemState.BOOT_SAFE or connected:
            return OperationResult.failure(
                "CONFIG_RELOAD_UNSAFE",
                "Config reload requires disconnected BOOT_SAFE; disconnect all first",
            )
        try:
            candidate = load_config(self.config.source_path)
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return OperationResult.failure("CONFIG_RELOAD_FAILED", str(exc))
        if dict(candidate.raw) != dict(self.config.raw):
            diff = await self.config_diff()
            self._emit("CONFIG_RELOAD_REJECTED", reason="restart required")
            return OperationResult(
                False,
                "CONFIG_RESTART_REQUIRED",
                "Configuration is valid but differs from the running component graph. "
                "Restart AeroGo2 to apply it atomically; runtime config was not changed.",
                diff.data,
            )
        self._emit("CONFIG_RELOAD_NO_CHANGE", source=str(candidate.source_path))
        return OperationResult.success(
            "Configuration is valid and unchanged; no runtime components were replaced"
        )

    async def preflight(self, profile: str = "all") -> OperationResult:
        """Evaluate and report the same guards used by the requested transition."""

        await self.refresh_snapshot()
        normalized = profile.strip().lower() or "all"
        target_by_profile = {
            "transform-flight": SystemState.WALK_TO_FLIGHT_PRECHECK,
            "home-walk": SystemState.HOMING_TO_WALK,
            "manual-position": SystemState.MANUAL_POSITIONING,
            "transform-walk": SystemState.FLIGHT_TO_WALK_PRECHECK,
            "autoland": SystemState.AUTO_LANDING_READY,
        }
        codes: List[str] = []
        messages: List[str] = []
        target = target_by_profile.get(normalized)
        if target is not None:
            if normalized == "transform-walk" and self.state is SystemState.LANDING_COMPLIANT:
                for result in (
                    self._landing_compliance_hold_result(),
                    self._landing_compliance_settle_result(),
                ):
                    if not result.ok and result.code not in codes:
                        codes.append(result.code)
                        messages.append(result.message)
            else:
                guard = TransitionGuards(self.config).evaluate(
                    self.state,
                    target,
                    self._snapshot,
                )
                codes.extend(guard.codes)
                messages.extend(guard.messages)
        elif normalized == "flight":
            guard = self._flight_readiness_guard(require_flight_enable_low=True)
            for code, message in zip(guard.codes, guard.messages):
                if code not in codes:
                    codes.append(code)
                    messages.append(message)
        elif normalized != "all":
            return OperationResult.failure(
                "UNKNOWN_PREFLIGHT_PROFILE",
                f"Unknown preflight profile '{profile}'",
            )

        if normalized in {"transform-flight", "home-walk", "manual-position"}:
            dwell = self._stationary_dwell_result()
            if not dwell.ok and dwell.code not in codes:
                codes.append(dwell.code)
                messages.append(dwell.message)
        if normalized in {"transform-flight", "transform-walk", "home-walk", "manual-position"}:
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok and current_clear.code not in codes:
                codes.append(current_clear.code)
                messages.append(current_clear.message)
        if (
            normalized == "transform-walk"
            and self.state is SystemState.FLIGHT_MANUAL
            and not self._touchdown_confirmed
        ):
            codes.append("TOUCHDOWN_NOT_CONFIRMED")
            messages.append("Touchdown must be continuously verified before transformation")

        if (
            normalized == "transform-walk"
            and self.config.go2.landing_compliance_enabled
            and self.state is SystemState.TOUCHDOWN_VERIFY
        ):
            codes.append("LANDING_COMPLIANCE_REQUIRED")
            messages.append(
                "Wait for Disarm, exact zero X8 RPM, calibrated foot contact, and "
                "LANDING_COMPLIANT entry"
            )

        monitor_violations = tuple(self._safety_monitor.evaluate(self._snapshot))
        for violation in monitor_violations:
            if violation.severity not in (SafetySeverity.FAULT, SafetySeverity.EMERGENCY):
                continue
            if violation.code not in codes:
                codes.append(violation.code)
                messages.append(violation.message)

        checks = [
            {"passed": False, "code": code, "message": message}
            for code, message in zip(codes, messages)
        ]
        if not checks:
            checks.append(
                {
                    "passed": True,
                    "code": "OK",
                    "message": "All authoritative preflight checks passed",
                }
            )
        data = {
            "profile": normalized,
            "state": self.state.name,
            "target_state": None if target is None else target.name,
            "permitted": not codes,
            "checks": checks,
            "go2_original_remote_confirmation_required": normalized == "transform-flight",
            "operator_warning": (
                "Confirm the Go2 original remote is no longer being used before FLIGHT morphology."
                if normalized == "transform-flight"
                else None
            ),
        }
        if codes:
            return OperationResult(
                False,
                "PREFLIGHT_FAILED",
                "; ".join(f"{code}: {message}" for code, message in zip(codes, messages)),
                data,
            )
        return OperationResult.success(f"Preflight {normalized} passed", data)

    async def enter_manual_positioning(
        self,
        operator_confirmed: bool = False,
    ) -> OperationResult:
        """Enter a guarded, volatile F446 positioning session."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "Manual positioning requires --hardware --enable-hardware-write",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                "Two-stage confirmation ENTER_F446_MANUAL is required",
            )
        if self.state is SystemState.MANUAL_POSITIONING:
            return OperationResult.success("F446 manual positioning is already active")
        if self.state not in {
            SystemState.BOOT_SAFE,
            SystemState.WALK,
            SystemState.FLIGHT_READY,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.LANDING_COMPLIANT,
        }:
            return OperationResult.failure(
                "INVALID_STATE",
                f"Manual positioning cannot start from {self.state.name}",
            )
        try:
            await self.refresh_snapshot()
            entry_state = self.state
            if entry_state is SystemState.LANDING_COMPLIANT:
                settled = self._landing_compliance_settle_result()
                if not settled.ok:
                    return settled
                hold = self._landing_compliance_hold_result()
                if not hold.ok:
                    return hold
                relock = await self._finish_landing_compliance(
                    "operator requested post-touchdown manual WALK positioning"
                )
                if not relock.ok or self.state is not SystemState.FLIGHT_READY:
                    return relock
                await self.refresh_snapshot()
            if self._snapshot.go2.connected and not await self._go2.request_stop():
                return OperationResult.failure(
                    "GO2_STOP_REJECTED",
                    "Go2 rejected the stop request",
                )
            await self.refresh_snapshot()
            dwell = self._stationary_dwell_result()
            if not dwell.ok:
                return dwell
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                return current_clear
            await self._state_machine.transition_to(
                SystemState.MANUAL_POSITIONING,
                reason="operator entered guarded F446 manual positioning",
                snapshot=self._snapshot,
            )
            await self.refresh_snapshot()
            self._emit("F446_MANUAL_POSITIONING_ENTERED")
            return OperationResult.success(
                (
                    "Manual positioning active; use mf/mr or limf/limr, ms, then "
                    "mark and confirm the observed walk/flight endpoint"
                ),
                code="F446_MANUAL_POSITIONING_ACTIVE",
                data={
                    "entry_state": entry_state.name,
                    "post_touchdown_recovery": entry_state
                    in {
                        SystemState.TOUCHDOWN_VERIFY,
                        SystemState.LANDING_COMPLIANT,
                    },
                    "manual_timeout_s": self.config.f446.transform_timeout_s,
                    "current_reporting": "automatic while duty is nonzero",
                },
            )
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("F446_MANUAL_ENTER_FAILED", str(exc))

    async def exit_manual_positioning(self) -> OperationResult:
        """Stop F446 and leave without accepting any physical configuration."""

        if self.state is not SystemState.MANUAL_POSITIONING:
            return OperationResult.failure(
                "INVALID_STATE",
                "F446 manual positioning is not active",
            )
        stop_result = await self._stop_transform_outputs()
        if not stop_result.ok:
            return stop_result
        self._operator_confirmed_configuration = None
        self._manual_marked_configuration = None
        self._manual_motion_started = False
        self._manual_last_direction = None
        self._manual_motion_deadline = None
        await self.refresh_snapshot()
        try:
            await self._state_machine.transition_to(
                SystemState.BOOT_SAFE,
                reason="operator exited manual positioning without configuration confirmation",
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            return OperationResult.failure("F446_MANUAL_EXIT_REJECTED", str(exc))
        await self.refresh_snapshot()
        self._emit("F446_MANUAL_POSITIONING_EXITED")
        return OperationResult.success(
            "Manual positioning exited; morphology remains unconfirmed in BOOT_SAFE"
        )

    async def start_f446_maintenance_motion(
        self,
        operation: str,
        duty: int,
    ) -> OperationResult:
        """Start mf/mr/limf/limr after a fresh recheck of every interlock."""

        normalized = operation.strip().lower()
        if normalized not in {"mf", "mr", "limf", "limr"}:
            return OperationResult.failure(
                "F446_INVALID_MAINTENANCE_OPERATION",
                f"Unsupported operation '{operation}'",
            )
        if isinstance(duty, bool) or not 1 <= duty <= 900:
            return OperationResult.failure(
                "F446_INVALID_DUTY",
                "F446 duty must be 1..900",
            )
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 motion writes are locked in this process",
            )

        await self.refresh_snapshot()
        guard = TransitionGuards(self.config).manual_motion_guard(self._snapshot)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0],
                "; ".join(
                    f"{code}: {message}" for code, message in zip(guard.codes, guard.messages)
                ),
            )
        dwell = self._stationary_dwell_result()
        if not dwell.ok:
            return dwell
        current_clear = self._current_clear_dwell_result()
        if not current_clear.ok:
            return current_clear

        if normalized.startswith("lim"):
            threshold = self._snapshot.f446.threshold_adc
            safe_ceiling = (
                self.config.safety.maximum_transform_current_adc
                - self.config.f446.current_safe_margin_adc
            )
            if (
                threshold is None
                or isinstance(threshold, bool)
                or threshold <= self.config.f446.current_safe_margin_adc
                or threshold > safe_ceiling
            ):
                return OperationResult.failure(
                    "F446_LIMIT_THRESHOLD_UNSAFE",
                    (
                        "Automatic limit motion requires "
                        f"{self.config.f446.current_safe_margin_adc + 1} <= "
                        f"threshold_adc <= {safe_ceiling}; run thr VALUE first"
                    ),
                )

        try:
            result = await self._f446.start_maintenance_motion(normalized, duty)
        except (BridgeError, asyncio.TimeoutError, OSError, RuntimeError, ValueError) as exc:
            stop_result = await self._safe_f446_stop()
            message = str(exc)
            if not stop_result.ok:
                message += f"; stop failed: {stop_result.message}"
            await self._fault("F446_MANUAL_COMMAND_FAILED", message, stop_attempted=True)
            return OperationResult.failure("F446_MANUAL_COMMAND_FAILED", message)
        if not result.ok:
            return result

        self._operator_confirmed_configuration = None
        self._manual_marked_configuration = None
        self._manual_motion_started = True
        self._manual_last_direction = "forward" if normalized.endswith("f") else "reverse"
        self._manual_motion_deadline = (
            self._clock.monotonic() + self.config.f446.transform_timeout_s
        )
        await self.refresh_snapshot()
        self._emit(
            "F446_MANUAL_MOTION_STARTED",
            operation=normalized,
            duty=duty,
            deadline=self._manual_motion_deadline,
        )
        data = dict(result.data)
        data.update(
            {
                "host_timeout_s": self.config.f446.transform_timeout_s,
                "current_reporting": "R_IS/L_IS/used raw ADC and mV",
                "next": "ms, then confirm walk or confirm flight",
            }
        )
        return OperationResult.success(result.message, data=data, code=result.code)

    async def set_f446_current_threshold(
        self,
        value: int,
        *,
        millivolts: bool = False,
    ) -> OperationResult:
        """Update and read back the local HW-039 stall threshold."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 parameter writes are locked in this process",
            )
        await self.refresh_snapshot()
        guard = TransitionGuards(self.config).manual_motion_guard(self._snapshot)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0],
                "; ".join(
                    f"{code}: {message}" for code, message in zip(guard.codes, guard.messages)
                ),
            )
        if self._snapshot.f446.duty != 0:
            return OperationResult.failure(
                "F446_PARAMETER_WRITE_WHILE_MOVING",
                "Run stop before changing the threshold",
            )

        threshold_adc = value * 4095 // 3300 if millivolts else value
        minimum = self.config.f446.current_safe_margin_adc + 1
        maximum = (
            self.config.safety.maximum_transform_current_adc
            - self.config.f446.current_safe_margin_adc
        )
        if isinstance(value, bool) or not minimum <= threshold_adc <= maximum:
            unit = "mV" if millivolts else "ADC"
            return OperationResult.failure(
                "F446_THRESHOLD_OUTSIDE_HOST_SAFETY_ENVELOPE",
                (
                    f"Requested {value}{unit} maps to ADC {threshold_adc}; "
                    f"allowed ADC range is {minimum}..{maximum}"
                ),
            )
        try:
            result = (
                await self._f446.set_current_threshold_mv(value)
                if millivolts
                else await self._f446.set_current_threshold_adc(value)
            )
        except (BridgeError, asyncio.TimeoutError, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("F446_THRESHOLD_WRITE_FAILED", str(exc))
        await self.refresh_snapshot()
        if result.ok:
            self._emit(
                "F446_THRESHOLD_UPDATED",
                threshold_adc=self._snapshot.f446.threshold_adc,
                threshold_mv=self._snapshot.f446.threshold_mv,
            )
        return result

    async def set_f446_timing_parameter(
        self,
        parameter: str,
        value: int,
    ) -> OperationResult:
        """Update one stopped F446 timing parameter and verify its readback."""

        normalized = parameter.strip().lower()
        if normalized not in {"timeout", "blank", "overms"}:
            return OperationResult.failure(
                "F446_INVALID_TIMING_PARAMETER",
                f"Unsupported F446 timing parameter '{parameter}'",
            )
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "F446_HARDWARE_WRITE_DISABLED",
                "F446 parameter writes are locked in this process",
            )
        await self.refresh_snapshot()
        guard = TransitionGuards(self.config).manual_motion_guard(self._snapshot)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0],
                "; ".join(
                    f"{code}: {message}" for code, message in zip(guard.codes, guard.messages)
                ),
            )
        if self._snapshot.f446.duty != 0:
            return OperationResult.failure(
                "F446_PARAMETER_WRITE_WHILE_MOVING",
                f"Run ms before changing {normalized}",
            )

        ranges = {
            "timeout": (100, min(60000, int(round(self.config.f446.transform_timeout_s * 1000.0)))),
            "blank": (0, 5000),
            "overms": (10, 3000),
        }
        minimum, maximum = ranges[normalized]
        if isinstance(value, bool) or not minimum <= value <= maximum:
            return OperationResult.failure(
                "F446_TIMING_OUTSIDE_HOST_SAFETY_ENVELOPE",
                f"{normalized} must be {minimum}..{maximum}ms",
            )

        current_timeout = self._snapshot.f446.timeout_ms or self.config.f446.firmware_timeout_ms
        current_blank = self._snapshot.f446.blanking_ms
        current_overms = self._snapshot.f446.overcurrent_ms
        candidate_timeout = value if normalized == "timeout" else current_timeout
        candidate_blank = value if normalized == "blank" else current_blank
        candidate_overms = value if normalized == "overms" else current_overms
        if candidate_blank + candidate_overms >= candidate_timeout:
            return OperationResult.failure(
                "F446_TIMING_COMBINATION_UNSAFE",
                "blank + overms must remain less than timeout",
            )

        setter_by_parameter = {
            "timeout": self._f446.set_motion_timeout_ms,
            "blank": self._f446.set_stall_blanking_ms,
            "overms": self._f446.set_overcurrent_duration_ms,
        }
        try:
            result = await setter_by_parameter[normalized](value)
        except (BridgeError, asyncio.TimeoutError, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("F446_TIMING_WRITE_FAILED", str(exc))
        await self.refresh_snapshot()
        if result.ok:
            self._emit(
                "F446_TIMING_UPDATED",
                parameter=normalized,
                value_ms=value,
                timeout_ms=self._snapshot.f446.timeout_ms,
                blanking_ms=self._snapshot.f446.blanking_ms,
                overcurrent_ms=self._snapshot.f446.overcurrent_ms,
            )
        return result

    async def mark_manual_configuration(
        self,
        target: Configuration,
        operator_confirmed: bool = False,
    ) -> OperationResult:
        """Record the operator's endpoint choice without leaving manual positioning."""

        if target not in {Configuration.WALK, Configuration.FLIGHT}:
            return OperationResult.failure(
                "INVALID_CONFIGURATION",
                "Manual endpoint target must be WALK or FLIGHT",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                f"Exact confirmation MARK_CURRENT_ENDPOINT_{target.value} is required",
            )
        if self.state is not SystemState.MANUAL_POSITIONING:
            return OperationResult.failure(
                "INVALID_STATE",
                "Manual endpoint marking requires MANUAL_POSITIONING",
            )

        validation = await self._validate_manual_configuration_target(target)
        if not validation.ok:
            return validation

        self._manual_marked_configuration = target
        self._operator_confirmed_configuration = target
        await self.refresh_snapshot()
        self._emit(
            "F446_MANUAL_ENDPOINT_MARKED",
            configuration=target.value,
            source=self._snapshot.configuration_source,
        )
        return OperationResult.success(
            (
                f"Current stopped position marked as {target.value}; "
                "system remains MANUAL_POSITIONING"
            ),
            code="F446_MANUAL_ENDPOINT_MARKED",
            data={
                "marked_endpoint": target.value,
                "configuration": self._snapshot.configuration.value,
                "configuration_source": self._snapshot.configuration_source,
                "state": self.state.name,
                "next": f"motor confirm {target.value.lower()}",
            },
        )

    async def confirm_manual_configuration(
        self,
        target: Configuration,
        operator_confirmed: bool = False,
    ) -> OperationResult:
        """Enter the state matching an independently operator-marked endpoint."""

        if target not in {Configuration.WALK, Configuration.FLIGHT}:
            return OperationResult.failure(
                "INVALID_CONFIGURATION",
                "Manual confirmation target must be WALK or FLIGHT",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                f"Exact confirmation CONFIRM_MANUAL_{target.value} is required",
            )
        if self.state is not SystemState.MANUAL_POSITIONING:
            return OperationResult.failure(
                "INVALID_STATE",
                "Manual configuration confirmation requires MANUAL_POSITIONING",
            )
        if self._manual_marked_configuration is not target:
            marked = (
                "NONE"
                if self._manual_marked_configuration is None
                else self._manual_marked_configuration.value
            )
            return OperationResult.failure(
                "F446_ENDPOINT_NOT_MARKED",
                (
                    f"Run motor endpoint {target.value.lower()} first; "
                    f"current operator mark is {marked}"
                ),
                data={"marked_endpoint": marked, "requested_endpoint": target.value},
            )

        validation = await self._validate_manual_configuration_target(target)
        if not validation.ok:
            return validation

        self._operator_confirmed_configuration = target
        await self.refresh_snapshot()
        target_state = (
            SystemState.WALK if target is Configuration.WALK else SystemState.GO2_JOINT_LOCK_WAIT
        )
        try:
            await self._state_machine.transition_to(
                target_state,
                reason=f"operator confirmed marked endpoint as {target.value}",
                snapshot=self._snapshot,
            )
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            self._operator_confirmed_configuration = None
            await self.refresh_snapshot()
            return OperationResult.failure("F446_MANUAL_CONFIRM_REJECTED", str(exc))

        self._emit(
            "F446_MANUAL_CONFIGURATION_CONFIRMED",
            configuration=target.value,
            source=self._snapshot.configuration_source,
        )
        if target is Configuration.FLIGHT:
            if self._runtime_mode is RuntimeMode.DRY_RUN:
                return await self._advance_go2_joint_lock_wait(simulate_operator=True)
            await self.refresh_snapshot()
            return self._joint_lock_operator_required_result()

        await self.refresh_snapshot()
        return OperationResult.success(
            f"Operator-confirmed {target.value}; system entered {target_state.name}",
            code="F446_MANUAL_CONFIGURATION_CONFIRMED",
            data={
                "configuration": target.value,
                "configuration_source": self._snapshot.configuration_source,
                "state": target_state.name,
            },
        )

    async def _validate_manual_configuration_target(
        self,
        target: Configuration,
    ) -> OperationResult:
        """Validate a stopped endpoint before marking it or entering its state."""

        await self.refresh_snapshot()
        if self._snapshot.f446.duty != 0:
            return OperationResult.failure(
                "F446_STILL_MOVING",
                "Run ms and verify duty=0 before marking the physical position",
            )
        required_direction = self.config.f446.direction_for(target.value)
        expected_state = self.config.f446.expected_state_for(target.value)
        opposite_target = (
            Configuration.FLIGHT.value if target is Configuration.WALK else Configuration.WALK.value
        )
        opposite = self.config.f446.expected_state_for(opposite_target)
        if self._snapshot.f446.state is opposite:
            return OperationResult.failure(
                "F446_LIMIT_TARGET_MISMATCH",
                f"F446 reports {opposite.value}; it cannot be marked as {target.value}",
            )
        if self._manual_motion_started and self._manual_last_direction != required_direction:
            return OperationResult.failure(
                "F446_DIRECTION_TARGET_MISMATCH",
                (
                    f"{target.value} requires last motion direction {required_direction}, "
                    f"observed {self._manual_last_direction}"
                ),
            )
        if self._snapshot.f446.state not in {F446State.IDLE, expected_state}:
            return OperationResult.failure(
                "F446_FINAL_STATE_INVALID",
                (
                    "Manual endpoint marking requires IDLE or the matching limit state; "
                    f"received {self._snapshot.f446.state.value}"
                ),
            )

        guard = TransitionGuards(self.config).manual_motion_guard(self._snapshot)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0],
                "; ".join(
                    f"{code}: {message}" for code, message in zip(guard.codes, guard.messages)
                ),
            )
        dwell = self._stationary_dwell_result()
        if not dwell.ok:
            return dwell
        current_clear = self._current_clear_dwell_result()
        if not current_clear.ok:
            return current_clear
        return OperationResult.success(f"Manual endpoint {target.value} is safe to accept")

    async def request_home_walk(self, operator_confirmed: bool = False) -> OperationResult:
        """Home an unverified F446 mechanism to the configured WALK limit."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Morphology control writes are locked in this process",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                "Two-stage confirmation HOME_F446_TO_WALK is required",
            )
        if self.state is not SystemState.BOOT_SAFE:
            return OperationResult.failure(
                "INVALID_STATE",
                f"F446 homing requires BOOT_SAFE, current state is {self.state.name}",
            )
        try:
            await self.refresh_snapshot()
            if self._snapshot.configuration is not Configuration.UNKNOWN:
                return OperationResult.failure(
                    "F446_HOME_NOT_REQUIRED",
                    "F446 homing is allowed only when physical configuration is UNKNOWN",
                )
            if not await self._go2.request_stop():
                raise BridgeError("Go2 rejected the stop request")
            await self.refresh_snapshot()
            dwell = self._stationary_dwell_result()
            if not dwell.ok:
                return dwell
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                return current_clear
            await self._state_machine.transition_to(
                SystemState.HOMING_TO_WALK,
                reason="operator requested guarded F446 homing to WALK",
                snapshot=self._snapshot,
            )
            self._emit("F446_HOME_WALK_STARTED")
            move_result = await self._f446.move_to_configuration(Configuration.WALK)
            if not move_result.ok:
                raise BridgeError(f"{move_result.code}: {move_result.message}")
            final_snapshot = await self.refresh_snapshot()
            if final_snapshot.configuration is not Configuration.WALK:
                raise BridgeError(
                    "F446_FINAL_STATE_MISMATCH: WALK limit was not authoritatively verified"
                )
            await self._state_machine.transition_to(
                SystemState.WALK,
                reason="F446 WALK home limit and zero duty verified",
                snapshot=final_snapshot,
            )
            self._emit("F446_HOME_WALK_VERIFIED")
            await self.refresh_snapshot()
            return OperationResult.success(
                "F446 homed to WALK; physical configuration is verified",
                code="F446_HOME_WALK_VERIFIED",
            )
        except (
            AeroGo2Error,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            stop_result = await self._safe_f446_stop()
            message = str(exc)
            if not stop_result.ok:
                message += f"; F446 stop failed: {stop_result.code}: {stop_result.message}"
            code = self._specific_f446_failure_code(
                message,
                fallback="F446_HOME_WALK_FAILED",
            )
            await self._fault(code, message, stop_attempted=True)
            return OperationResult.failure(code, message)

    async def request_transform_flight(self, operator_confirmed: bool = False) -> OperationResult:
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Morphology control writes are locked in this process",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                "Exact confirmation TRANSFORM_TO_FLIGHT is required",
            )
        if self.state is not SystemState.WALK:
            return OperationResult.failure(
                "INVALID_STATE",
                f"transform flight requires WALK, current state is {self.state.name}",
            )
        try:
            await self.refresh_snapshot()
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                return current_clear
            await self._state_machine.transition_to(
                SystemState.WALK_TO_FLIGHT_PRECHECK,
                reason="operator requested FLIGHT morphology",
                snapshot=self._snapshot,
            )
            self._emit("TRANSFORM_FLIGHT_REQUESTED")
            if not await self._go2.request_stop():
                raise BridgeError("Go2 rejected the stop request")
            await self.refresh_snapshot()
            dwell = self._stationary_dwell_result()
            if not dwell.ok:
                await self._state_machine.transition_to(
                    SystemState.WALK,
                    reason="flight precheck waiting for stationary dwell",
                    snapshot=self._snapshot,
                )
                await self.refresh_snapshot()
                return dwell
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                await self._state_machine.transition_to(
                    SystemState.WALK,
                    reason="flight precheck waiting for F446 current-clear hold",
                    snapshot=self._snapshot,
                )
                await self.refresh_snapshot()
                return current_clear
            await self._state_machine.transition_to(
                SystemState.TRANSFORM_TO_FLIGHT,
                reason="Go2 stopped; transform interlocks rechecked",
                snapshot=self._snapshot,
            )
            self._emit("TRANSFORM_FLIGHT_STARTED")
            move_result = await self._f446.move_to_configuration(Configuration.FLIGHT)
            if not move_result.ok:
                raise BridgeError(f"{move_result.code}: {move_result.message}")
            await self.refresh_snapshot()
            if not await self._go2.request_stop():
                raise BridgeError("GO2_STOP_FAILED: Go2 rejected the stop request")
            await self.refresh_snapshot()
            self._emit("FLIGHT_LIMIT_REACHED")
            await self._state_machine.transition_to(
                SystemState.GO2_JOINT_LOCK_WAIT,
                reason="F446 FLIGHT endpoint verified; waiting for operator-selected mode=6",
                snapshot=self._snapshot,
            )
            await self.refresh_snapshot()
            self._emit(
                "GO2_JOINT_LOCK_OPERATOR_REQUIRED",
                timeout_s=self.config.go2.joint_lock_operator_timeout_s,
                current_mode=self._snapshot.go2.locomotion_mode,
            )
            if self._runtime_mode is RuntimeMode.DRY_RUN:
                return await self._advance_go2_joint_lock_wait(simulate_operator=True)
            return self._joint_lock_operator_required_result()
        except (
            AeroGo2Error,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            stop_result = await self._safe_f446_stop()
            message = str(exc)
            if not stop_result.ok:
                message += f"; F446 stop failed: {stop_result.code}: {stop_result.message}"
            code = self._specific_f446_failure_code(
                message,
                fallback="TRANSFORM_FLIGHT_FAILED",
            )
            await self._fault(code, message, stop_attempted=True)
            return OperationResult.failure(code, message)

    async def request_transform_walk(self, operator_confirmed: bool = False) -> OperationResult:
        if not self._control_writes_allowed():
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Morphology control writes are locked in this process",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                "Exact confirmation TRANSFORM_TO_WALK is required",
            )
        if self.state not in (
            SystemState.FLIGHT_READY,
            SystemState.FLIGHT_MANUAL,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.LANDING_COMPLIANT,
        ):
            return OperationResult.failure(
                "INVALID_STATE", "transform walk requires a landed FLIGHT state"
            )
        try:
            await self.refresh_snapshot()
            if (
                self.config.go2.landing_compliance_enabled
                and self.state is SystemState.TOUCHDOWN_VERIFY
            ):
                return OperationResult.failure(
                    "LANDING_COMPLIANCE_REQUIRED",
                    "Wait for calibrated foot contact, Disarm, zero X8 RPM, and "
                    "automatic LANDING_COMPLIANT entry before transforming",
                    data=dict(self._landing_compliance_report()),
                )
            if self.state is SystemState.LANDING_COMPLIANT:
                settled = self._landing_compliance_settle_result()
                if not settled.ok:
                    return settled
                hold = self._landing_compliance_hold_result()
                if not hold.ok:
                    return hold
                relock = await self._finish_landing_compliance(
                    "operator confirmed transform to WALK"
                )
                if not relock.ok:
                    raise BridgeError(f"{relock.code}: {relock.message}")
                if relock.code == "GO2_JOINT_LOCK_OPERATOR_REQUIRED":
                    return relock

            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                return current_clear
            if self.state is SystemState.FLIGHT_MANUAL and not self._touchdown_confirmed:
                return OperationResult.failure(
                    "TOUCHDOWN_NOT_CONFIRMED",
                    "Touchdown must be continuously verified before transformation",
                )
            await self._state_machine.transition_to(
                SystemState.FLIGHT_TO_WALK_PRECHECK,
                reason="operator requested WALK after touchdown/disarm",
                snapshot=self._snapshot,
            )
            if not await self._go2.request_stop():
                raise BridgeError("Go2 rejected the stop request")
            await self.refresh_snapshot()
            dwell = self._stationary_dwell_result()
            if not dwell.ok:
                await self._state_machine.transition_to(
                    SystemState.FLIGHT_READY,
                    reason="walk precheck waiting for stationary dwell",
                    snapshot=self._snapshot,
                )
                await self.refresh_snapshot()
                return dwell
            current_clear = self._current_clear_dwell_result()
            if not current_clear.ok:
                await self._state_machine.transition_to(
                    SystemState.FLIGHT_READY,
                    reason="walk precheck waiting for F446 current-clear hold",
                    snapshot=self._snapshot,
                )
                await self.refresh_snapshot()
                return current_clear
            await self._state_machine.transition_to(
                SystemState.TRANSFORM_TO_WALK,
                reason="flight-to-walk interlocks rechecked",
                snapshot=self._snapshot,
            )
            self._emit("TRANSFORM_WALK_STARTED")
            move_result = await self._f446.move_to_configuration(Configuration.WALK)
            if not move_result.ok:
                raise BridgeError(f"{move_result.code}: {move_result.message}")
            await self.refresh_snapshot()
            await self._state_machine.transition_to(
                SystemState.WALK,
                reason="F446 WALK limit and zero duty verified",
                snapshot=self._snapshot,
            )
            self._touchdown_confirmed = False
            self._landing_contact_since = None
            self._landing_compliant_since = None
            self._emit("WALK_CONFIGURATION_VERIFIED")
            await self.refresh_snapshot()
            return OperationResult.success("WALK verified; walking may resume")
        except (
            AeroGo2Error,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            stop_result = await self._safe_f446_stop()
            message = str(exc)
            if not stop_result.ok:
                message += f"; F446 stop failed: {stop_result.code}: {stop_result.message}"
            code = self._specific_f446_failure_code(
                message,
                fallback="TRANSFORM_WALK_FAILED",
            )
            await self._fault(code, message, stop_attempted=True)
            return OperationResult.failure(code, message)

    async def prepare_autoland(self) -> OperationResult:
        async with self._operation_lock:
            return await self._prepare_autoland_unlocked()

    async def _prepare_autoland_unlocked(self) -> OperationResult:
        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Phase 1 automatic landing is available only in DRY-RUN",
            )
        if self.state is not SystemState.FLIGHT_MANUAL:
            return OperationResult.failure(
                "INVALID_STATE", "autoland prepare requires FLIGHT_MANUAL"
            )
        self._landing_controller.reset()
        self._last_landing_update = None
        self._next_landing_update_at = None
        await self.refresh_snapshot()
        try:
            await self._state_machine.transition_to(
                SystemState.AUTO_LANDING_READY,
                reason="landing estimator/controller initialized; no setpoint sent",
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            return OperationResult.failure("AUTOLAND_PRECHECK_FAILED", str(exc))
        self._reset_impact_landing_completion(new_session=True)
        self._emit("AUTOLAND_READY")
        await self.refresh_snapshot()
        return OperationResult.success("Automatic landing ready; setpoints remain stopped")

    async def start_autoland(self) -> OperationResult:
        async with self._operation_lock:
            return await self._start_autoland_unlocked()

    async def _start_autoland_unlocked(self) -> OperationResult:
        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Phase 1 automatic landing is available only in DRY-RUN",
            )
        if self.state is not SystemState.AUTO_LANDING_READY:
            return OperationResult.failure(
                "INVALID_STATE", "autoland start requires AUTO_LANDING_READY"
            )
        if self.config.go2.low_level.enabled:
            return OperationResult.failure(
                "COORDINATED_ACTUATION_NOT_CONFIGURED",
                "LowCmd-enabled automatic landing requires an injected, bounded first-policy activation transaction; the legacy velocity-setpoint starter is intentionally blocked",
            )
        await self.refresh_snapshot()
        try:
            await self._state_machine.transition_to(
                SystemState.AUTO_LANDING,
                reason="operator selected AUTO_EXECUTE",
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            return OperationResult.failure("AUTOLAND_START_REJECTED", str(exc))
        self._autoland_active = True
        self._last_landing_update = None
        self._next_landing_update_at = None
        await self.refresh_snapshot()
        first = await self._update_autoland_unlocked()
        resulting_state = self._state_machine.state
        if (
            first.ok
            and resulting_state is SystemState.AUTO_LANDING
            and self._autoland_active
            and self._setpoint_active
        ):
            self._emit("AUTOLAND_STARTED")
            return first
        return OperationResult.failure(
            "AUTOLAND_START_FAILED",
            "The first automatic-landing command was not activated; the manager returned to its fail-safe state",
            {
                "first_result_code": first.code,
                "first_result_message": first.message,
                "state": resulting_state.name,
            },
        )

    async def update_autoland(self) -> OperationResult:
        async with self._operation_lock:
            return await self._update_autoland_unlocked()

    async def _update_autoland_unlocked(self) -> OperationResult:
        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Phase 1 automatic landing is available only in DRY-RUN",
            )
        if self.state is not SystemState.AUTO_LANDING or not self._autoland_active:
            return OperationResult.failure("AUTOLAND_INACTIVE", "Automatic landing is inactive")
        await self.refresh_snapshot()
        if (
            self._snapshot.rc.failsafe
            or self._snapshot.rc.manual_override
            or self._snapshot.rc.auto_landing_request is not AutoLandingRequest.AUTO_EXECUTE
            or self._snapshot.pixhawk.failsafe
        ):
            return await self._abort_autoland_unlocked("manual override or flight failsafe")

        # Revalidate every independently timed landing input before honoring a
        # not-yet-due controller period.  A previous descent setpoint must not
        # remain supervised merely because this invocation would send no new
        # packet.
        interlock = self._landing_interlocks.can_send_landing_setpoint(self._snapshot)
        if not interlock.permitted:
            reason = "; ".join(interlock.messages) or "landing setpoint interlock rejected"
            self._last_landing_command = self._landing_safety_filter.invalid(
                self._snapshot.timestamp,
                reason,
            )
            return await self._abort_autoland_unlocked(reason)

        if self._impact_recovery_setpoints_stopped or self._impact_recovery.confirmed:
            if self._impact_recovery_wait_started_at is None:
                self._impact_recovery_wait_started_at = self._clock.monotonic()
                self._emit(
                    "IMPACT_RECOVERY_COMPLETION_WAIT_STARTED",
                    landing_session_id=self._impact_landing_session_id,
                )
            if self._setpoint_active or self._snapshot.external_setpoint_active:
                try:
                    await self._stop_setpoints()
                except (BridgeError, OSError, RuntimeError) as exc:
                    await self._fault("AUTOLAND_SETPOINT_STOP_FAILED", str(exc))
                    return OperationResult.failure(
                        "AUTOLAND_FINALIZATION_STOP_FAILED",
                        str(exc),
                    )
            self._impact_recovery_setpoints_stopped = True
            self._next_landing_update_at = None
            await self.refresh_snapshot()
            return OperationResult.failure(
                "AUTOLAND_FINALIZATION_FENCED",
                "Post-touchdown recovery has fenced this landing session; external setpoints cannot restart",
            )

        now = self._clock.monotonic()
        period_s = 1.0 / self.config.landing.controller_hz
        due_at = self._next_landing_update_at
        if due_at is not None and now + 1e-9 < due_at:
            return OperationResult.success(
                "Automatic landing controller update is not due yet",
                data={"next_update_in_s": max(0.0, due_at - now)},
            )

        previous = self._last_landing_update
        dt = period_s if previous is None else now - previous
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.config.landing.controller_timeout_s:
            violation = self._safety_monitor.controller_timeout_violation(self._snapshot)
            self._record_runtime_violation(violation)
            self._emit(
                "AUTOLAND_CONTROLLER_TIMEOUT",
                safety_violations=(violation,),
            )
            return await self._abort_autoland_unlocked("automatic landing controller timeout")

        self._last_landing_update = now
        if due_at is None:
            self._next_landing_update_at = now + period_s
        else:
            elapsed_from_due = max(0.0, now - due_at)
            periods_to_advance = int(math.floor(elapsed_from_due / period_s)) + 1
            self._next_landing_update_at = due_at + periods_to_advance * period_s
        candidate = self._landing_controller.update(self._snapshot, dt)
        command = self._landing_safety_filter.apply(candidate, self._snapshot, dt)
        self._last_landing_command = command
        if not command.valid:
            if "timeout" in command.reason.lower():
                violation = self._safety_monitor.controller_timeout_violation(self._snapshot)
                self._record_runtime_violation(violation)
                self._emit(
                    "AUTOLAND_CONTROLLER_TIMEOUT",
                    safety_violations=(violation,),
                )
            return await self._abort_autoland_unlocked(command.reason)
        try:
            if self._runtime_mode is not RuntimeMode.DRY_RUN:
                return OperationResult.failure(
                    "PHASE_NOT_AVAILABLE",
                    "Phase 1 automatic landing is available only in DRY-RUN",
                )
            setpoint_result = await self._pixhawk.send_velocity_setpoint(
                command.vx_des,
                command.vy_des,
                command.vz_des,
                command.yaw_rate_des,
            )
            if not setpoint_result.ok:
                raise BridgeError(setpoint_result.message)
        except (BridgeError, OSError, RuntimeError) as exc:
            return await self._abort_autoland_unlocked(f"setpoint rejected: {exc}")
        self._setpoint_active = True
        self._emit("LANDING_COMMAND", landing_command=command)
        await self.refresh_snapshot()
        return OperationResult.success("Simulated landing setpoint recorded")

    async def abort_autoland(self, reason: str = "operator request") -> OperationResult:
        async with self._operation_lock:
            return await self._abort_autoland_unlocked(reason)

    def _prepare_autoland_abort(self, reason: str) -> None:
        """Invalidate touchdown timing before any abort-side await can yield.

        Once the impact-recovery wait has begun, the same continuously landed
        episode must not be reclassified by FLIGHT_MANUAL's simpler touchdown
        path.  Keep that fail-closed latch separate from recovery bookkeeping:
        the latter may be reset after a successful stop, while the latch is
        cleared only by an independently observed airborne cycle (or a fully
        guarded touchdown completion).
        """

        post_impact_abort = self._impact_recovery_wait_started_at is not None
        self._touchdown_since = None
        self._touchdown_height_reference = None
        self._clear_aborted_impact_airborne_dwell()
        if post_impact_abort and not self._aborted_impact_touchdown_latched:
            self._aborted_impact_touchdown_latched = True
            self._emit(
                "ABORTED_IMPACT_TOUCHDOWN_LATCHED",
                landing_session_id=self._impact_landing_session_id,
                reason=reason,
            )

    async def _abort_autoland_unlocked(
        self,
        reason: str = "operator request",
    ) -> OperationResult:
        self._prepare_autoland_abort(reason)
        revoke_result = await self._revoke_go2_low_level_internal(
            f"automatic landing aborted: {reason}"
        )
        try:
            await self._stop_setpoints()
        except (BridgeError, OSError, RuntimeError) as exc:
            self._autoland_active = False
            self._next_landing_update_at = None
            await self._fault("AUTOLAND_SETPOINT_STOP_FAILED", str(exc))
            return OperationResult.failure("AUTOLAND_ABORT_FAILED", str(exc))
        if not revoke_result.ok:
            self._autoland_active = False
            self._next_landing_update_at = None
            await self._fault(
                "GO2_LOWCMD_REVOKE_FAILED",
                revoke_result.message,
            )
            return OperationResult.failure("AUTOLAND_ABORT_FAILED", revoke_result.message)
        self._autoland_active = False
        self._next_landing_update_at = None
        self._reset_impact_landing_completion(new_session=False)
        self._last_landing_command = LandingCommand(
            valid=False, reason=reason, timestamp=self._clock.monotonic()
        )
        transition = await self._finish_autoland_abort(reason)
        if not transition.ok:
            return transition
        self._emit("MANUAL_OVERRIDE", reason=reason)
        await self.refresh_snapshot()
        return OperationResult.success(
            "External setpoints stopped; RadioMaster retains flight control"
        )

    async def tick(self) -> Tuple[SafetyViolation, ...]:
        async with self._operation_lock:
            return await self._tick_unlocked()

    async def _tick_unlocked(self) -> Tuple[SafetyViolation, ...]:
        """Refresh telemetry, evaluate safety, and advance passive state changes."""

        previously_armed = self._snapshot.pixhawk.armed
        await self.refresh_snapshot()
        if previously_armed and not self._snapshot.pixhawk.armed:
            self._emit("PIXHAWK_DISARMED")

        try:
            bridge_ground_arm_authorized = self._pixhawk.ground_arm_authorization_active()
            bridge_ground_arm_status_error = ""
        except Exception as exc:
            # Unknown is possibly active.  Continue through the same bounded
            # revoke/FAULT path instead of allowing an exception to escape and
            # terminate the safety loop.
            bridge_ground_arm_authorized = True
            bridge_ground_arm_status_error = f"{type(exc).__name__}: {exc}"
        if self._ground_arm_authorized and not bridge_ground_arm_authorized:
            self._ground_arm_authorized = False
            self._ground_arm_authorization_expires_at = None

        # The bridge keepalive can only maintain the remote lease; it cannot
        # validate the changing system snapshot.  Until a strict armed rising
        # edge consumes the one-shot gate, the manager therefore re-proves the
        # ground conditions on every serialized safety tick.  A bridge/local
        # disagreement is also unsafe: an independently live remote gate must
        # be closed even if the local half has already been lost.
        gate_may_be_active = bool(self._ground_arm_authorized or bridge_ground_arm_authorized)
        pixhawk_has_consumed_gate = self._snapshot.pixhawk.armed is True
        if gate_may_be_active and not pixhawk_has_consumed_gate:
            authorization_expires_at = self._ground_arm_authorization_expires_at
            ground_proof_valid = bool(
                self._ground_arm_authorized
                and bridge_ground_arm_authorized
                and not bridge_ground_arm_status_error
                and authorization_expires_at is not None
                and self._clock.monotonic() < authorization_expires_at
                and pixhawk_ground_state_is_current(
                    self._snapshot.pixhawk,
                    self._snapshot.timestamp,
                    self.config.safety.pixhawk_timeout_s,
                    self.config.safety.touchdown_max_source_age_s,
                )
                and self._snapshot.pixhawk.armed is False
                and self._snapshot.pixhawk.landed is True
            )
            if not ground_proof_valid:
                watchdog_reason = "ground proof lost before Pixhawk armed rising edge"
                if bridge_ground_arm_status_error:
                    watchdog_reason = (
                        "ground-arm gate status is unknown before Pixhawk armed rising "
                        f"edge: {bridge_ground_arm_status_error}"
                    )
                gate_result = await self._revoke_ground_arm_authorization_unlocked(watchdog_reason)
                if not gate_result.ok:
                    await self._fault(
                        "GROUND_ARM_AUTH_WATCHDOG_REVOKE_FAILED",
                        "Ground-arm proof was lost and the physical gate revoke was not "
                        f"confirmed: {gate_result.code}: {gate_result.message}",
                    )
                    return tuple(self._safety_monitor.evaluate(self._snapshot))
                self._emit(
                    "GROUND_ARM_AUTH_WATCHDOG_REVOKED",
                    reason=watchdog_reason,
                )
                await self.refresh_snapshot()

        deadline = self._manual_motion_deadline
        if (
            self.state is SystemState.MANUAL_POSITIONING
            and deadline is not None
            and self._snapshot.f446.duty != 0
            and self._clock.monotonic() >= deadline
        ):
            stop_result = await self._stop_transform_outputs()
            if not stop_result.ok:
                await self._fault(
                    "F446_MANUAL_TIMEOUT_STOP_FAILED",
                    stop_result.message,
                    stop_attempted=True,
                )
                return tuple(self._safety_monitor.evaluate(self._snapshot))
            self._emit(
                "F446_MANUAL_MOTION_TIMEOUT",
                timeout_s=self.config.f446.transform_timeout_s,
            )
            await self.refresh_snapshot()

        violations = self._filter_go2_joint_lock_wait_violations(
            tuple(self._safety_monitor.evaluate(self._snapshot))
        )
        self._update_violations(violations)
        blocking = [
            item
            for item in violations
            if item.severity in (SafetySeverity.FAULT, SafetySeverity.EMERGENCY)
        ]

        if self.state in (SystemState.AUTO_LANDING, SystemState.AUTO_LANDING_READY):
            takeover_codes = {
                "AUTOLAND_ESTIMATOR_INVALID",
                "PIXHAWK_TIMEOUT",
                "PIXHAWK_TOUCHDOWN_SOURCE_INCOHERENT",
                "PIXHAWK_TOUCHDOWN_SOURCE_STALE",
                "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID",
                "RC_FAILSAFE",
                "RC_TIMEOUT",
            }
            takeover = (
                self._snapshot.rc.failsafe
                or self._snapshot.rc.manual_override
                or self._snapshot.rc.auto_landing_request is AutoLandingRequest.MANUAL
                or self._snapshot.pixhawk.failsafe
                or any(item.code in takeover_codes for item in violations)
            )
            if takeover:
                await self._abort_autoland_unlocked(
                    "manual takeover or automatic-landing input failure"
                )
                return violations

        if self.state in TRANSFORM_STATES and blocking:
            stop_result = await self._stop_transform_outputs()
            message = blocking[0].message
            if not stop_result.ok:
                message += f"; F446/setpoint stop incomplete: {stop_result.message}"
            await self._fault(blocking[0].code, message, stop_attempted=True)
            return violations

        post_handover_go2_sample = self._post_handover_go2_sample_is_fresh()
        expected_handover_wait_codes = {"RC_TIMEOUT", "RC_FAILSAFE"}
        if self.state is SystemState.GO2_GROUND_HANDOVER and not post_handover_go2_sample:
            # SportModeState can be absent while the high-level service is
            # released.  The bounded joint-lock handover below owns this wait;
            # stale pre-handover Go2 data must neither pass nor fault it early.
            expected_handover_wait_codes.update(
                {"GO2_TIMEOUT", "GO2_UNSAFE_DURING_GROUND_HANDOVER"}
            )
        ground_handover_blocking = [
            item for item in blocking if item.code not in expected_handover_wait_codes
        ]
        if self.state is SystemState.GO2_GROUND_HANDOVER and ground_handover_blocking:
            stop_result = await self._stop_transform_outputs()
            message = ground_handover_blocking[0].message
            if not stop_result.ok:
                message += f"; F446/setpoint stop incomplete: {stop_result.message}"
            await self._fault(ground_handover_blocking[0].code, message, stop_attempted=True)
            return violations

        if self.state in {
            SystemState.GO2_JOINT_LOCK_WAIT,
            SystemState.GO2_GROUND_HANDOVER,
        }:
            await self._advance_go2_joint_lock_wait()
            return violations

        non_escalating_codes = {
            "AUTOLAND_CONTROLLER_TIMEOUT",
            "AUTOLAND_ESTIMATOR_INVALID",
            "MANUAL_OVERRIDE_REQUESTED",
            "PIXHAWK_TIMEOUT",
            "PIXHAWK_TOUCHDOWN_SOURCE_INCOHERENT",
            "PIXHAWK_TOUCHDOWN_SOURCE_STALE",
            "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID",
            "RC_FAILSAFE",
            "RC_TIMEOUT",
        }
        escalating = [item for item in blocking if item.code not in non_escalating_codes]
        if escalating and self.state not in (
            SystemState.BOOT_SAFE,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            stop_result = await self._stop_supervised_unlocked()
            message = escalating[0].message
            if not stop_result.ok:
                message += f"; supervised stop incomplete: {stop_result.message}"
            await self._fault(escalating[0].code, message, stop_attempted=True)
            return violations

        if (
            self.state is SystemState.FLIGHT_READY
            and self._snapshot.pixhawk.armed
            and self._snapshot.ground_arm_authorized
        ):
            await self._state_machine.transition_to(
                SystemState.FLIGHT_MANUAL,
                reason="Pixhawk armed after AeroGo2 authorization and RadioMaster request",
                snapshot=self._snapshot,
            )
            self._emit("PIXHAWK_ARMED")
            await self.refresh_snapshot()

        if self.state is SystemState.AUTO_LANDING:
            # Evaluate touchdown/recovery before producing another setpoint.
            # Once finalization has fenced the controller, this landing session
            # may never restart it merely because completion evidence expires.
            await self._check_touchdown()
            if (
                self.state is SystemState.AUTO_LANDING
                and not self._impact_recovery_setpoints_stopped
            ):
                await self._update_autoland_unlocked()
        elif self.state is SystemState.FLIGHT_MANUAL:
            await self._check_touchdown()
        if self.state is SystemState.TOUCHDOWN_VERIFY or (
            self.state is SystemState.FLIGHT_READY and self._touchdown_confirmed
        ):
            await self._check_landing_compliance()
        return violations

    async def _stop_transform_outputs(self) -> OperationResult:
        """Stop only manager-owned F446 motion and external setpoints."""

        if not self._control_writes_allowed():
            self._setpoint_active = False
            self._next_landing_update_at = None
            await self.refresh_snapshot()
            return OperationResult.success(
                "Control writes are locked; no output command was issued"
            )
        failures: List[str] = []
        try:
            f446_result = await self._f446.stop()
            if not f446_result.ok:
                failures.append(f"F446: {f446_result.code}: {f446_result.message}")
        except (BridgeError, OSError, RuntimeError) as exc:
            failures.append(f"F446: {exc}")
        try:
            await self._stop_setpoints()
        except (BridgeError, OSError, RuntimeError) as exc:
            failures.append(f"setpoint: {exc}")
        self._manual_motion_deadline = None
        await self.refresh_snapshot()
        if failures:
            return OperationResult.failure("TRANSFORM_STOP_PARTIAL", "; ".join(failures))
        return OperationResult.success("F446 motion and automatic setpoints stopped")

    async def stop_transform_motion(self) -> OperationResult:
        """Implement transform/motor stop without commanding Go2."""

        was_manual = self.state is SystemState.MANUAL_POSITIONING
        was_transforming = self.state in TRANSFORM_STATES
        result = await self._stop_transform_outputs()
        if was_manual:
            if result.ok:
                self._emit("F446_MANUAL_MOTION_STOPPED")
            elif self.state not in (SystemState.FAULT, SystemState.EMERGENCY_STOP):
                await self._fault("TRANSFORM_STOP_FAILED", result.message, stop_attempted=True)
        elif was_transforming:
            code = "TRANSFORM_STOPPED" if result.ok else "TRANSFORM_STOP_FAILED"
            await self._fault(code, result.message, stop_attempted=True)
        elif not result.ok and self.state not in (
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            await self._fault("TRANSFORM_STOP_FAILED", result.message, stop_attempted=True)
        return result

    async def stop_supervised(self) -> OperationResult:
        async with self._operation_lock:
            return await self._stop_supervised_unlocked()

    async def _stop_supervised_unlocked(self) -> OperationResult:
        """Stop F446, Go2 and external setpoints, but never arm/disarm rotors."""

        initial_state = self.state
        aborting_autoland = initial_state in {
            SystemState.AUTO_LANDING,
            SystemState.AUTO_LANDING_READY,
        }
        if aborting_autoland:
            # Do this before the first bridge await.  Even a partial/failed
            # supervised stop must not leave an old AUTO_LANDING touchdown
            # timer available to FLIGHT_MANUAL on the same landed episode.
            self._prepare_autoland_abort("supervised stop")
        failures: List[str] = []
        # This gate is an independent arm authority and therefore the first
        # stop action.  Every later actuator stop is attempted even when its
        # acknowledgement is missing.
        try:
            gate_result = await self._revoke_ground_arm_authorization_unlocked("supervised stop")
        except Exception as exc:
            gate_result = OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
        gate_revoke_failed = not gate_result.ok
        if gate_revoke_failed:
            failures.append(f"Pixhawk arm gate: {gate_result.code}: {gate_result.message}")
        transform_result = await self._stop_transform_outputs()
        if not transform_result.ok:
            failures.append(transform_result.message)
        lowcmd_result = await self._revoke_go2_low_level_internal("supervised stop")
        if not lowcmd_result.ok:
            failures.append(f"Go2 LowCmd: {lowcmd_result.message}")
        # Re-read the owner directly.  A revoke failure or an ambiguous
        # transitional/fault state must never be followed by a competing
        # SportClient RPC.
        await self.refresh_snapshot()
        low_level = self._snapshot.go2.low_level_status
        sport_handover_confirmed = (
            lowcmd_result.ok
            and not low_level.ownership_pending
            and low_level.ownership_state
            in {
                LowCmdOwnershipState.DISABLED,
                LowCmdOwnershipState.DISCONNECTED,
                LowCmdOwnershipState.OBSERVE_ONLY,
            }
        )
        if (
            self._control_writes_allowed()
            and self._snapshot.go2.connected
            and sport_handover_confirmed
        ):
            try:
                if initial_state is SystemState.LANDING_COMPLIANT:
                    if self._runtime_mode is RuntimeMode.DRY_RUN:
                        if not await self._go2.request_flight_pose():
                            raise BridgeError("simulated Go2 JOINT_LOCK restoration failed")
                    elif not await self._go2.request_stop():
                        raise BridgeError("Go2 rejected the emergency stop request")
                elif not await self._go2.request_stop():
                    raise BridgeError("Go2 rejected the stop request")
            except (BridgeError, OSError, RuntimeError) as exc:
                failures.append(f"Go2: {exc}")
        self._autoland_active = False
        self._next_landing_update_at = None
        await self.refresh_snapshot()

        if failures:
            message = "; ".join(failures)
            if self.state not in (SystemState.FAULT, SystemState.EMERGENCY_STOP):
                await self._fault("SUPERVISED_STOP_FAILED", message, stop_attempted=True)
            return OperationResult.failure(
                (
                    "GROUND_ARM_AUTH_REVOKE_FAILED"
                    if gate_revoke_failed
                    else "SUPERVISED_STOP_PARTIAL"
                ),
                message,
            )

        if aborting_autoland:
            self._reset_impact_landing_completion(new_session=False)
            self._last_landing_command = LandingCommand(
                timestamp=self._clock.monotonic(),
                valid=False,
                reason="supervised stop",
            )
            transition = await self._finish_autoland_abort("supervised stop")
            if not transition.ok:
                return transition
        elif initial_state is SystemState.MANUAL_POSITIONING:
            self._emit("F446_MANUAL_MOTION_STOPPED")
        elif initial_state in TRANSFORM_STATES:
            await self._fault(
                "TRANSFORM_STOPPED",
                "Supervised stop interrupted an active morphology transaction",
                stop_attempted=True,
            )

        await self.refresh_snapshot()
        if self._snapshot.pixhawk.armed:
            return OperationResult.success(
                "Pixhawk is armed. Rotor shutdown is not performed by this console. "
                "Automatic setpoints have been stopped. Use RadioMaster to control or "
                "land the vehicle."
            )
        return OperationResult.success(
            "F446 transformation, Go2 request, and automatic setpoints stopped"
        )

    async def interrupt_transform(self) -> OperationResult:
        """Stop an in-progress morphology change; manual positioning remains active."""

        if self.state not in TRANSFORM_STATES:
            return OperationResult.failure(
                "NO_TRANSFORM_IN_PROGRESS",
                "No morphology transformation is in progress",
            )
        manual_positioning = self.state is SystemState.MANUAL_POSITIONING
        stop_result = await self._stop_transform_outputs()
        if manual_positioning:
            if not stop_result.ok:
                await self._fault(
                    "TRANSFORM_INTERRUPT_STOP_PARTIAL",
                    stop_result.message,
                    stop_attempted=True,
                )
                return stop_result
            self._emit("F446_MANUAL_MOTION_STOPPED")
            return OperationResult.success(
                "Manual F446 motion stopped; positioning session remains active"
            )

        await self._fault(
            "TRANSFORM_INTERRUPTED",
            "Operator interrupted morphology transformation with Ctrl+C",
            stop_attempted=True,
        )
        if not stop_result.ok:
            return OperationResult.failure(
                "TRANSFORM_INTERRUPT_STOP_PARTIAL",
                "Transformation interrupted and FAULT latched; stop was partial: "
                + stop_result.message,
            )
        return OperationResult.success(
            "Transformation stopped and TRANSFORM_INTERRUPTED fault latched"
        )

    async def clear_fault(self) -> OperationResult:
        """Clear manager faults only; never sends the F446 `clear` command."""

        await self.refresh_snapshot()
        if self._snapshot.go2.low_level_status.ownership_pending:
            return OperationResult.failure(
                "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
                "LowCmd ownership/handoff is still pending; complete the explicit supported-ground release before clearing FAULT",
            )
        current = tuple(self._safety_monitor.evaluate(self._snapshot))
        blocking = [
            item
            for item in current
            if item.severity in (SafetySeverity.FAULT, SafetySeverity.EMERGENCY)
        ]
        if self._snapshot.f446.faulted:
            return OperationResult.failure(
                "F446_FAULT_ACTIVE",
                "F446 fault must be cleared by an explicit future maintenance workflow",
            )
        if blocking:
            return OperationResult.failure(
                "FAULT_CONDITION_ACTIVE",
                "; ".join(f"{item.code}: {item.message}" for item in blocking),
            )
        self._active_violations.clear()
        safe = replace(self._snapshot, active_fault_codes=())
        if self.state in (SystemState.FAULT, SystemState.EMERGENCY_STOP):
            try:
                await self._state_machine.transition_to(
                    SystemState.BOOT_SAFE,
                    reason="operator acknowledged faults after safe re-evaluation",
                    snapshot=safe,
                )
            except TransitionRejected as exc:
                return OperationResult.failure("FAULT_CLEAR_REJECTED", str(exc))
        self._emit("FAULT_CLEARED")
        await self.refresh_snapshot()
        return OperationResult.success("Manager faults cleared; system remains BOOT_SAFE")

    async def shutdown(self) -> OperationResult:
        stop_result = await self.stop_supervised()
        if stop_result.ok:
            result = await self.disconnect_all()
        else:
            result = OperationResult.failure(
                "SHUTDOWN_STOP_FAILED",
                "Devices remain connected because supervised stop was incomplete: "
                + stop_result.message,
            )
        if not result.ok:
            # Keep monitoring/logging alive.  In particular, a rejected
            # ground handover must not make the non-daemon LowCmd writer an
            # orphan with no manager watchdog.
            self._emit("SYSTEM_EXIT_INHIBITED", command_result=result)
            return result
        self._emit("SYSTEM_EXITED", command_result=result)
        if self._event_logger is not None:
            self._event_logger.stop()
        self._started = False
        return result

    def query(self, name: str) -> Mapping[str, Any]:
        """Return a command-specific read-only view through one public API."""

        normalized = " ".join(name.strip().lower().split())
        legacy_names = {"status", "transitions", "faults", "config", "controller"}
        if normalized in legacy_names:
            return self._legacy_query(normalized)
        if normalized == "state transitions":
            return self._legacy_query("transitions")
        if normalized == "config show":
            return self._legacy_query("config")
        if normalized in {"controller status", "autoland status"}:
            return self._legacy_query("controller")
        return self._semantic_query(normalized)

    def safety_report(self) -> Mapping[str, Any]:
        violations = tuple(self._safety_monitor.evaluate(self._snapshot))
        return {
            "healthy": not any(
                item.severity in (SafetySeverity.FAULT, SafetySeverity.EMERGENCY)
                for item in violations
            ),
            "violations": [self._violation_dict(item) for item in violations],
        }

    def _legacy_query(self, name: str) -> Mapping[str, Any]:
        if name in ("status", "state"):
            return cast(Mapping[str, Any], snapshot_to_dict(self._snapshot))
        if name == "transitions":
            return {
                "transitions": [
                    {
                        "timestamp": item.timestamp,
                        "previous_state": item.previous_state.name,
                        "new_state": item.new_state.name,
                        "reason": item.reason,
                        "permitted": item.permitted,
                        "guard_codes": list(item.guard_codes),
                        "entry_action_error": item.entry_action_error,
                    }
                    for item in self._state_machine.history
                ]
            }
        if name == "faults":
            return {
                "active": [self._violation_dict(item) for item in self.violations],
                "history": [self._violation_dict(item) for item in self._violation_history],
            }
        if name == "config":
            return cast(Dict[str, Any], deep_thaw(self.config.raw))
        if name == "controller":
            return {
                "active": self._autoland_active,
                "external_setpoint_active": self._setpoint_active,
                "last_command": {
                    "vx_des": self._last_landing_command.vx_des,
                    "vy_des": self._last_landing_command.vy_des,
                    "vz_des": self._last_landing_command.vz_des,
                    "yaw_rate_des": self._last_landing_command.yaw_rate_des,
                    "valid": self._last_landing_command.valid,
                    "reason": self._last_landing_command.reason,
                },
            }
        return {"error": f"unknown query '{name}'"}

    def _semantic_query(self, name: str) -> Mapping[str, Any]:
        snapshot = snapshot_to_dict(self._snapshot)
        pixhawk = snapshot["pixhawk"]
        f446 = snapshot["f446"]
        go2 = snapshot["go2"]
        rc = snapshot["rc"]

        if name == "state":
            return {
                "state": self.state.name,
                "configuration": self._snapshot.configuration.value,
                "maintenance_mode": self._snapshot.maintenance_mode,
                "active_fault_codes": list(self._snapshot.active_fault_codes),
            }
        if name == "devices":
            return {
                subsystem: {
                    "connected": snapshot[subsystem]["connected"],
                    "message_age_s": snapshot[subsystem]["message_age_s"],
                }
                for subsystem in ("pixhawk", "f446", "go2")
            }

        safety_views = {
            "health",
            "state guards",
            "audit",
            "audit pixhawk",
            "audit f446",
            "audit rc",
            "audit configuration",
            "check invariant",
            "check communication",
            "check sensors",
            "walk permit",
        }
        if name in {"flight enable-check", "flight ready"}:
            return self._flight_readiness_report(require_flight_enable_low=name == "flight ready")
        if name in safety_views:
            report = dict(self.safety_report())
            report["view"] = name
            target = name.split()[-1]
            if target in {"pixhawk", "f446", "rc"}:
                report[target] = snapshot[target]
            elif name == "audit configuration":
                report["configuration_source"] = str(self.config.source_path)
            elif name == "walk permit":
                report["permitted"] = (
                    self.state is SystemState.WALK
                    and self._snapshot.configuration is Configuration.WALK
                    and bool(report["healthy"])
                )
            return report

        if name == "rc":
            return {"parsed": rc, "operator_request": snapshot["operator"]}
        if name == "rc raw":
            return {
                "channels": dict(self._snapshot.rc.channels),
                "connected": self._snapshot.rc.connected,
                "failsafe": self._snapshot.rc.failsafe,
                "timestamp": self._snapshot.rc.timestamp,
            }
        if name == "rc mapping":
            return {"mapping": deep_thaw(self.config.raw.get("rc", {}))}
        if name == "rc check":
            timestamp = self._snapshot.rc.timestamp
            timestamp_valid = (
                math.isfinite(timestamp)
                and timestamp > 0.0
                and timestamp <= self._snapshot.timestamp
            )
            age = (
                timestamp_age(self._snapshot.timestamp, timestamp) if timestamp_valid else math.inf
            )
            return {
                "connected": self._snapshot.rc.connected,
                "failsafe": self._snapshot.rc.failsafe,
                "flight_enable": self._snapshot.rc.flight_enable,
                "morphology_request": self._snapshot.rc.morphology_request.value,
                "auto_landing_request": self._snapshot.rc.auto_landing_request.value,
                "valid": timestamp_valid,
                "fresh": timestamp_valid
                and timestamp_is_fresh(
                    self._snapshot.timestamp,
                    timestamp,
                    self.config.safety.rc_timeout_s,
                ),
                "message_age_s": age,
            }

        if name == "pixhawk status":
            return {"pixhawk": pixhawk}
        if name == "pixhawk messages":
            return {
                "connected": self._snapshot.pixhawk.connected,
                "message_age_s": self._snapshot.pixhawk.message_age_s,
                "supported_message_types": [
                    "HEARTBEAT",
                    "SYS_STATUS",
                    "RC_CHANNELS",
                    "ATTITUDE",
                    "LOCAL_POSITION_NED",
                    "EXTENDED_SYS_STATE",
                    "BATTERY_STATUS",
                    "ESC_TELEMETRY_1_TO_4",
                    "ESC_TELEMETRY_5_TO_8",
                    "STATUSTEXT",
                ],
                "raw_message_buffer_available": False,
            }
        if name == "pixhawk statustext":
            return {"statustext": list(self._snapshot.pixhawk.statustext)}
        if name == "pixhawk params":
            return {
                "configured": deep_thaw(self.config.raw.get("pixhawk", {})),
                "readback_available": False,
            }

        if name == "motor current":
            return {
                "r_is_raw": self._snapshot.f446.r_is_raw,
                "r_is_mv": self._snapshot.f446.r_is_mv,
                "l_is_raw": self._snapshot.f446.l_is_raw,
                "l_is_mv": self._snapshot.f446.l_is_mv,
                "used_raw": self._snapshot.f446.used_raw,
                "used_mv": self._snapshot.f446.used_mv,
                "threshold_raw": self._snapshot.f446.threshold_raw,
                "over_active": self._snapshot.f446.over_active,
            }
        if name == "motor parameters":
            return {
                "configured": deep_thaw(self.config.raw.get("f446", {})),
                "reported": {
                    "manual_limit": self._snapshot.f446.manual_limit,
                    "blanking_ms": self._snapshot.f446.blanking_ms,
                    "overcurrent_ms": self._snapshot.f446.overcurrent_ms,
                    "timeout_ms": self._snapshot.f446.timeout_ms,
                },
            }
        if name == "motor status":
            return {"f446": f446}

        if name == "go2 status":
            return {
                "go2": go2,
                "joint_lock_telemetry": self._snapshot.go2.joints_locked,
                "joint_lock_confirmed": self._snapshot.joint_lock_confirmed,
                "joint_lock_source": self._snapshot.joint_lock_source,
                "joint_lock_state_codes": list(self.config.go2.joint_lock_state_codes),
                "accepted_state_codes": list(self.config.go2.accepted_state_codes),
            }
        if name == "touchdown status":
            return self._touchdown_report()
        if name == "landing compliance":
            return self._landing_compliance_report()
        if name == "go2 motion":
            return {
                "body_velocity": list(self._snapshot.go2.body_velocity),
                "body_rpy": list(self._snapshot.go2.body_rpy),
                "standing": self._snapshot.go2.standing,
                "moving": self._snapshot.go2.moving,
                "stable": self._snapshot.go2.stable,
                "locomotion_mode": self._snapshot.go2.locomotion_mode,
            }
        if name == "go2 controller":
            return {
                "ubuntu_high_level_controller_active": self._snapshot.go2.controller_active,
                "original_remote_override_available": False,
                "warning": (
                    "Ubuntu cannot prove or suppress Go2 original-remote use; operator "
                    "confirmation is required before FLIGHT morphology."
                ),
            }

        esc_items = list(pixhawk["esc"])
        if name == "esc":
            return {"esc": esc_items}
        if name in {"esc 1", "esc 2", "esc 3", "esc 4"}:
            slot = int(name[-1])
            item = next((entry for entry in esc_items if entry["slot"] == slot), None)
            return {"slot": slot, "telemetry": item}
        if name == "esc mapping":
            display_shift = self.config.esc.mavlink_display_shift
            return {
                "mapping": dict(self.config.esc.slots),
                "mavlink_display_shift": display_shift,
                "expected_raw_slots": {
                    slot: slot + display_shift for slot in self.config.esc.slots
                },
            }
        if name == "esc health":
            tuple_items = self._snapshot.pixhawk.esc
            tuple_slots = [item.slot for item in tuple_items]
            assessment = assess_esc_telemetry(
                self._snapshot,
                self.config.esc.slots,
            )
            return {
                "healthy": assessment.safe,
                "complete": assessment.complete,
                "consistent": assessment.consistent,
                "expected_slots": sorted(self.config.esc.slots),
                "observed_slots": sorted(set(tuple_slots)),
                "esc": esc_items,
                "checks": [
                    {
                        "passed": assessment.complete,
                        "code": "ESC_TELEMETRY_COMPLETE",
                        "message": "All configured ESC views must contain exactly the expected slots.",
                    },
                    {
                        "passed": assessment.consistent,
                        "code": "ESC_TELEMETRY_CONSISTENT",
                        "message": "Tuple, RPM, online, and physical-position views must agree.",
                    },
                    {
                        "passed": assessment.healthy,
                        "code": "ESC_TELEMETRY_HEALTHY",
                        "message": "Every configured ESC must be online and healthy.",
                    },
                    {
                        "passed": assessment.rpm_safe,
                        "code": "ESC_RPM_FINITE",
                        "message": "Every configured ESC RPM must be a genuine finite number.",
                    },
                ],
            }

        if name == "transform status":
            joint_lock_deadline = self._go2_joint_lock_deadline
            now = self._clock.monotonic()
            joint_lock_remaining_s = (
                0.0 if joint_lock_deadline is None else max(0.0, joint_lock_deadline - now)
            )
            joint_lock_grace_remaining_s = (
                0.0
                if self._go2_joint_lock_entered_at is None
                else max(
                    0.0,
                    self.config.go2.joint_lock_transition_grace_s
                    - (now - self._go2_joint_lock_entered_at),
                )
            )
            joint_lock_unsafe_observed_s = (
                0.0
                if self._go2_joint_lock_unsafe_since is None
                else max(0.0, now - self._go2_joint_lock_unsafe_since)
            )
            return {
                "system_state": self.state.name,
                "configuration": self._snapshot.configuration.value,
                "go2_mode": self._snapshot.go2.locomotion_mode,
                "go2_joints_locked": self._snapshot.go2.joints_locked,
                "joint_lock_confirmed": self._snapshot.joint_lock_confirmed,
                "joint_lock_source": self._snapshot.joint_lock_source,
                "joint_lock_state_codes": list(self.config.go2.joint_lock_state_codes),
                "joint_lock_operator_remaining_s": joint_lock_remaining_s,
                "joint_lock_transition_grace_remaining_s": joint_lock_grace_remaining_s,
                "joint_lock_unsafe_observed_s": joint_lock_unsafe_observed_s,
                "operator_marked_endpoint": (
                    None
                    if self._manual_marked_configuration is None
                    else self._manual_marked_configuration.value
                ),
                "f446_state": self._snapshot.f446.state.value,
                "f446_duty": self._snapshot.f446.duty,
            }
        if name == "walk status":
            return {
                "system_state": self.state.name,
                "configuration": self._snapshot.configuration.value,
                "walking_permitted": (
                    self.state is SystemState.WALK
                    and self._snapshot.configuration is Configuration.WALK
                ),
                "go2": go2,
            }
        if name == "flight status":
            return {
                "system_state": self.state.name,
                "configuration": self._snapshot.configuration.value,
                "pixhawk": pixhawk,
                "ubuntu_direct_arm_disarm_available": False,
                "ground_arm_authorized": self._snapshot.ground_arm_authorized,
                "ground_arm_authorization_expires_at": (
                    self._snapshot.ground_arm_authorization_expires_at
                ),
                "second_key_rc_channel": self.config.rc.flight_enable_channel,
            }
        if name == "flight auth-status":
            expires_at = self._snapshot.ground_arm_authorization_expires_at
            return {
                "authorized": self._snapshot.ground_arm_authorized,
                "expires_at": expires_at,
                "remaining_s": max(0.0, expires_at - self._snapshot.timestamp)
                if expires_at is not None
                else 0.0,
                "arms_pixhawk": False,
                "second_key": f"RadioMaster RC{self.config.rc.flight_enable_channel} LOW-to-HIGH",
                "pixhawk_armed": self._snapshot.pixhawk.armed,
            }

        controller = self._legacy_query("controller")
        if name == "controller timing":
            return {
                "controller_hz": self.config.landing.controller_hz,
                "controller_timeout_s": self.config.landing.controller_timeout_s,
                "last_update_monotonic": self._last_landing_update,
            }
        if name == "controller inputs":
            return {
                "landing_estimate": snapshot["landing_estimate"],
                "operator_request": snapshot["operator"],
                "pixhawk": pixhawk,
            }
        if name == "controller output":
            return {"last_command": controller["last_command"]}
        return {"error": f"unknown query '{name}'"}

    def _flight_readiness_guard(self, *, require_flight_enable_low: bool) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []
        if self.state is not SystemState.FLIGHT_READY:
            codes.append("NOT_IN_FLIGHT_READY")
            messages.append("Flight enable is permitted only from FLIGHT_READY.")

        interlock = self._landing_interlocks.can_enter_flight_ready(
            self._snapshot,
            require_flight_enable_low=require_flight_enable_low,
        )
        for code, message in zip(interlock.codes, interlock.messages):
            if code not in codes:
                codes.append(code)
                messages.append(message)
        return GuardResult(not codes, tuple(codes), tuple(messages))

    def _flight_readiness_report(
        self,
        *,
        require_flight_enable_low: bool,
    ) -> Mapping[str, Any]:
        guard = self._flight_readiness_guard(require_flight_enable_low=require_flight_enable_low)
        checks = [
            {"passed": False, "code": code, "message": message}
            for code, message in zip(guard.codes, guard.messages)
        ]
        if not checks:
            checks.append(
                {
                    "passed": True,
                    "code": "OK",
                    "message": "All authoritative flight-enable checks passed.",
                }
            )
        return {
            "permitted": guard.permitted,
            "state": self.state.name,
            "configuration": self._snapshot.configuration.value,
            "flight_enable_requested": self._snapshot.rc.flight_enable,
            "flight_enable_must_be_low": require_flight_enable_low,
            "checks": checks,
        }

    def _reset_touchdown_candidate(self) -> None:
        self._touchdown_since = None
        self._touchdown_height_reference = None

    def _reset_touchdown_cycle(self) -> None:
        self._airborne_since = None
        self._airborne_confirmed = False
        self._touchdown_confirmed = False
        self._reset_touchdown_candidate()

    def _esc_telemetry_confirms_touchdown(self) -> bool:
        return bool(
            assess_esc_telemetry(
                self._snapshot,
                self.config.esc.slots,
                maximum_abs_rpm=self.config.safety.touchdown_max_esc_rpm,
            ).safe
        )

    async def _clear_post_touchdown_stability(self) -> None:
        """Clear the published dwell result when any stability premise is lost."""

        changed = bool(
            self._post_touchdown_stable_since is not None or self._impact_landing_exit_ready
        )
        self._post_touchdown_stable_since = None
        self._post_touchdown_last_stability_check_at = None
        self._impact_landing_exit_ready = False
        if changed:
            await self.refresh_snapshot()

    def _airborne_sample_is_valid(self) -> bool:
        status = self._snapshot.pixhawk
        now = self._clock.monotonic()
        return (
            pixhawk_touchdown_sources_are_current(
                status,
                now,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
                self.config.safety.touchdown_max_source_skew_s,
            )
            and status.armed
            and not status.landed
        )

    def _update_airborne_confirmation(self, now: float) -> None:
        if self._airborne_confirmed:
            return
        if not self._airborne_sample_is_valid():
            self._airborne_since = None
            self._reset_touchdown_candidate()
            return
        if self._airborne_since is None:
            self._airborne_since = now
            return
        if now - self._airborne_since < self.config.safety.airborne_confirm_s:
            return
        self._airborne_confirmed = True
        self._reset_touchdown_candidate()
        self._emit("AIRBORNE_CONFIRMED")

    def _touchdown_height(self) -> float:
        height = self._snapshot.pixhawk.relative_altitude_m
        if (
            self.state is SystemState.AUTO_LANDING
            and self._snapshot.landing_estimate.height_m is not None
        ):
            height = self._snapshot.landing_estimate.height_m
        return height

    def _touchdown_conditions(self, height: float) -> Mapping[str, bool]:
        status = self._snapshot.pixhawk
        estimate = self._snapshot.landing_estimate
        now = self._clock.monotonic()
        automatic_landing = self.state is SystemState.AUTO_LANDING
        source_timestamps = [
            status.attitude_timestamp,
            status.kinematics_timestamp,
            status.landed_state_timestamp,
        ]
        estimate_current = True
        if automatic_landing:
            source_timestamps.append(estimate.timestamp)
            estimate_current = bool(
                estimate.valid
                and estimate.ground_detected
                and estimate.height_m is not None
                and math.isfinite(estimate.height_m)
                and timestamp_is_fresh(
                    now,
                    estimate.timestamp,
                    self.config.safety.controller_timeout_s,
                )
            )
        return {
            "pixhawk_sources_current": pixhawk_touchdown_sources_are_current(
                status,
                now,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
                self.config.safety.touchdown_max_source_skew_s,
            ),
            "landing_estimate_current": estimate_current,
            "source_timestamps_coherent": timestamps_are_coherent(
                source_timestamps,
                self.config.safety.touchdown_max_source_skew_s,
            ),
            "pixhawk_landed": status.landed,
            "height_finite": math.isfinite(height),
            "vertical_velocity_finite": math.isfinite(status.vertical_velocity_mps),
            "roll_finite": math.isfinite(status.roll_rad),
            "pitch_finite": math.isfinite(status.pitch_rad),
            "vertical_speed_safe": (
                math.isfinite(status.vertical_velocity_mps)
                and abs(status.vertical_velocity_mps)
                <= self.config.safety.touchdown_max_vertical_speed_mps
            ),
            "roll_safe": (
                math.isfinite(status.roll_rad)
                and abs(status.roll_rad) <= self.config.safety.touchdown_max_tilt_rad
            ),
            "pitch_safe": (
                math.isfinite(status.pitch_rad)
                and abs(status.pitch_rad) <= self.config.safety.touchdown_max_tilt_rad
            ),
            "esc_rpm_safe": self._esc_telemetry_confirms_touchdown(),
        }

    def _touchdown_report(self) -> Mapping[str, Any]:
        now = self._clock.monotonic()
        height = self._touchdown_height()
        airborne_elapsed = (
            0.0 if self._airborne_since is None else max(0.0, now - self._airborne_since)
        )
        touchdown_elapsed = (
            0.0 if self._touchdown_since is None else max(0.0, now - self._touchdown_since)
        )
        height_delta = (
            None
            if self._touchdown_height_reference is None or not math.isfinite(height)
            else abs(height - self._touchdown_height_reference)
        )
        return {
            "state": self.state.name,
            "airborne_confirmed": self._airborne_confirmed,
            "airborne_candidate_elapsed_s": airborne_elapsed,
            "airborne_confirm_s": self.config.safety.airborne_confirm_s,
            "touchdown_detection_enabled": self._airborne_confirmed,
            "touchdown_candidate_elapsed_s": touchdown_elapsed,
            "touchdown_confirm_s": self.config.safety.touchdown_confirm_s,
            "touchdown_confirmed": self._touchdown_confirmed,
            "height_m": height,
            "height_reference_m": self._touchdown_height_reference,
            "height_delta_m": height_delta,
            "maximum_height_delta_m": self.config.safety.touchdown_max_height_delta_m,
            "pixhawk_armed": self._snapshot.pixhawk.armed,
            "pixhawk_landed": self._snapshot.pixhawk.landed,
            "conditions": dict(self._touchdown_conditions(height)),
        }

    async def _check_touchdown(self) -> None:
        now = self._clock.monotonic()
        self._update_airborne_confirmation(now)
        if not self._airborne_confirmed:
            return

        status = self._snapshot.pixhawk
        height = self._touchdown_height()
        source_timestamps = (
            status.attitude_timestamp,
            status.kinematics_timestamp,
            status.landed_state_timestamp,
        )
        if (
            self.config.go2.low_level.enabled
            and self.state is SystemState.FLIGHT_MANUAL
            and self._aborted_impact_touchdown_latched
        ):
            independent_airborne_evidence = bool(
                pixhawk_touchdown_sources_are_current(
                    status,
                    now,
                    self.config.safety.pixhawk_timeout_s,
                    self.config.safety.touchdown_max_source_age_s,
                    self.config.safety.touchdown_max_source_skew_s,
                )
                and timestamps_are_coherent(
                    source_timestamps,
                    self.config.safety.touchdown_max_source_skew_s,
                )
                and type(status.landed) is bool
                and not status.landed
                and math.isfinite(height)
                and math.isfinite(status.vertical_velocity_mps)
                and math.isfinite(status.maximum_esc_rpm)
                and (
                    height > self.config.safety.touchdown_max_height_delta_m
                    or abs(status.vertical_velocity_mps)
                    > self.config.safety.touchdown_max_vertical_speed_mps
                    or status.maximum_esc_rpm > self.config.safety.touchdown_max_esc_rpm
                )
            )
            self._reset_touchdown_candidate()
            if not independent_airborne_evidence:
                if self._clear_aborted_impact_airborne_dwell():
                    self._emit(
                        "ABORTED_IMPACT_AIRBORNE_DWELL_INTERRUPTED",
                        reason="airborne evidence became invalid, stale, incoherent, or absent",
                    )
                return
            last_check = self._aborted_impact_airborne_last_check_at
            if self._aborted_impact_airborne_since is not None and (
                last_check is None
                or now < last_check
                or now - last_check > self.config.safety.post_touchdown_stability_max_check_gap_s
            ):
                self._clear_aborted_impact_airborne_dwell()
                self._emit(
                    "ABORTED_IMPACT_AIRBORNE_OBSERVATION_GAP",
                    reason="airborne evidence was not checked continuously",
                )
            if self._aborted_impact_airborne_since is None:
                self._aborted_impact_airborne_since = now
                self._aborted_impact_airborne_last_check_at = now
                self._emit(
                    "ABORTED_IMPACT_AIRBORNE_DWELL_STARTED",
                    required_dwell_s=self.config.safety.aborted_impact_airborne_confirm_s,
                )
                return
            self._aborted_impact_airborne_last_check_at = now
            if (
                now - self._aborted_impact_airborne_since
                < self.config.safety.aborted_impact_airborne_confirm_s
            ):
                return
            self._aborted_impact_touchdown_latched = False
            self._clear_aborted_impact_airborne_dwell()
            self._emit(
                "ABORTED_IMPACT_TOUCHDOWN_LATCH_CLEARED",
                reason="independent airborne evidence held continuously",
                confirmed_dwell_s=self.config.safety.aborted_impact_airborne_confirm_s,
            )
            return
        if (
            self.config.go2.low_level.enabled
            and self.state is SystemState.AUTO_LANDING
            and self._impact_recovery_wait_started_at is not None
            and now - self._impact_recovery_wait_started_at
            >= self.config.safety.impact_recovery_completion_timeout_s
        ):
            self._emit(
                "IMPACT_RECOVERY_COMPLETION_TIMEOUT",
                landing_session_id=self._impact_landing_session_id,
                reason="normal recovery did not reach the guarded exit before its total deadline",
            )
            await self._abort_autoland_unlocked("post-touchdown recovery completion timed out")
            return
        touchdown = all(self._touchdown_conditions(height).values())
        if not touchdown:
            self._reset_touchdown_candidate()
            if self.state is SystemState.AUTO_LANDING and self.config.go2.low_level.enabled:
                await self._clear_post_touchdown_stability()
            return
        if self._touchdown_height_reference is None:
            self._touchdown_height_reference = height
            self._touchdown_since = now
            return
        if (
            abs(height - self._touchdown_height_reference)
            > self.config.safety.touchdown_max_height_delta_m
        ):
            self._touchdown_height_reference = height
            self._touchdown_since = now
            if self.state is SystemState.AUTO_LANDING and self.config.go2.low_level.enabled:
                await self._clear_post_touchdown_stability()
            return
        if self._touchdown_since is None:
            self._touchdown_since = now
            return
        if now - self._touchdown_since < self.config.safety.touchdown_confirm_s:
            return
        if self.state is SystemState.AUTO_LANDING and self.config.go2.low_level.enabled:
            await self._advance_impact_recovery_exit(now)
            return
        await self._complete_touchdown_transition(
            "manual landing touchdown conditions held for configured duration"
        )

    def _impact_recovery_evidence_failure(self, now: float) -> Optional[str]:
        evidence = self._impact_recovery
        if evidence.landing_session_id != self._impact_landing_session_id:
            return "recovery evidence belongs to another automatic-landing session"
        if not evidence.confirmed:
            return evidence.reason or "post-touchdown recovery evidence is incomplete"
        maximum_age = self.config.safety.impact_recovery_status_max_age_s
        if (
            not timestamp_is_fresh(now, evidence.timestamp, maximum_age)
            or not timestamp_is_fresh(
                now,
                evidence.residual_zero_status_timestamp,
                maximum_age,
            )
            or now >= evidence.valid_until
        ):
            return "post-touchdown recovery or persistent-zero FC status is stale or expired"
        low_level = self._snapshot.go2.low_level_status
        expected_epoch = low_level.owner_epoch if self.config.go2.low_level.enabled else 0
        if evidence.go2_ownership_epoch != expected_epoch:
            return "recovery evidence does not match the active Go2 ownership epoch"
        return None

    def _post_touchdown_stability_is_valid(self, now: float) -> bool:
        if self._impact_recovery_evidence_failure(now) is not None:
            return False
        pixhawk = self._snapshot.pixhawk
        estimate = self._snapshot.landing_estimate
        pixhawk_sources_current = pixhawk_touchdown_sources_are_current(
            pixhawk,
            now,
            self.config.safety.pixhawk_timeout_s,
            self.config.safety.touchdown_max_source_age_s,
            self.config.safety.touchdown_max_source_skew_s,
        )
        if (
            not pixhawk_sources_current
            or not timestamps_are_coherent(
                (
                    pixhawk.attitude_timestamp,
                    pixhawk.kinematics_timestamp,
                    pixhawk.landed_state_timestamp,
                    estimate.timestamp,
                ),
                self.config.safety.touchdown_max_source_skew_s,
            )
            or not pixhawk.landed
            or not math.isfinite(pixhawk.vertical_velocity_mps)
            or abs(pixhawk.vertical_velocity_mps)
            > self.config.safety.touchdown_max_vertical_speed_mps
            or not math.isfinite(pixhawk.roll_rad)
            or not math.isfinite(pixhawk.pitch_rad)
            or abs(pixhawk.roll_rad) > self.config.safety.touchdown_max_tilt_rad
            or abs(pixhawk.pitch_rad) > self.config.safety.touchdown_max_tilt_rad
            or not estimate.valid
            or not estimate.ground_detected
            or not timestamp_is_fresh(
                now,
                estimate.timestamp,
                self.config.safety.controller_timeout_s,
            )
            or estimate.horizontal_velocity_mps is None
            or not math.isfinite(estimate.horizontal_velocity_mps)
            or abs(estimate.horizontal_velocity_mps)
            > self.config.landing.maximum_horizontal_speed_mps
            or not self._esc_telemetry_confirms_touchdown()
            or not self._impact_recovery_setpoints_stopped
            or self._snapshot.external_setpoint_active
        ):
            return False
        if self.config.go2.low_level.enabled:
            low_level = self._snapshot.go2.low_level_status
            tracking_unsafe = any(
                violation.code == "GO2_JOINT_TRACKING_ERROR"
                for violation in self._safety_monitor.evaluate(self._snapshot)
            )
            return bool(
                self._impact_recovery_safe_hold_confirmed
                and low_level.ownership_state
                in {LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.SAFE_HOLD}
                and self._lowcmd_status_healthy(low_level)
                and self._lowcmd_motor_feedback_is_stationary(low_level)
                and not tracking_unsafe
            )
        return bool(
            self._snapshot.go2.control_authority.state
            is Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK
            and self._snapshot.go2.joints_locked
            and self._snapshot.go2.stable
            and not self._snapshot.go2.moving
        )

    async def _advance_impact_recovery_exit(self, now: float) -> None:
        if self._impact_recovery_wait_started_at is None:
            self._impact_recovery_wait_started_at = now
            self._emit(
                "IMPACT_RECOVERY_COMPLETION_WAIT_STARTED",
                landing_session_id=self._impact_landing_session_id,
            )
        failure = self._impact_recovery_evidence_failure(now)
        if failure is not None:
            await self._clear_post_touchdown_stability()
            if (
                self._impact_recovery_wait_started_at is not None
                and now - self._impact_recovery_wait_started_at
                >= self.config.safety.impact_recovery_completion_timeout_s
            ):
                self._emit(
                    "IMPACT_RECOVERY_COMPLETION_TIMEOUT",
                    landing_session_id=self._impact_landing_session_id,
                    reason=failure,
                )
                await self._abort_autoland_unlocked("post-touchdown recovery completion timed out")
                return
            self._emit(
                "IMPACT_RECOVERY_COMPLETION_REQUIRED",
                landing_session_id=self._impact_landing_session_id,
                reason=failure,
            )
            return

        if self._impact_recovery_finalization_started_at is None:
            self._impact_recovery_finalization_started_at = now
            self._emit(
                "IMPACT_RECOVERY_FINALIZATION_STARTED",
                landing_session_id=self._impact_landing_session_id,
            )

        if not self._impact_recovery_setpoints_stopped:
            try:
                await self._stop_setpoints()
            except (BridgeError, OSError, RuntimeError) as exc:
                await self._fault("AUTOLAND_SETPOINT_STOP_FAILED", str(exc))
                return
            self._impact_recovery_setpoints_stopped = True
            self._next_landing_update_at = None
            self._emit(
                "IMPACT_RECOVERY_SETPOINTS_QUIESCED",
                landing_session_id=self._impact_landing_session_id,
            )
            await self.refresh_snapshot()
            now = self._clock.monotonic()
            if (
                self._impact_recovery_finalization_started_at is not None
                and now - self._impact_recovery_finalization_started_at
                >= self.config.safety.impact_recovery_finalization_timeout_s
            ):
                self._emit(
                    "IMPACT_RECOVERY_FINALIZATION_TIMEOUT",
                    landing_session_id=self._impact_landing_session_id,
                    stage="setpoint_quiesce",
                )
                await self._abort_autoland_unlocked(
                    "post-touchdown setpoint finalization timed out"
                )
                return

        if not self._impact_recovery_safe_hold_confirmed:
            revoked = await self._revoke_go2_low_level_internal(
                "post-touchdown recovery complete and FC residual zero confirmed"
            )
            if not revoked.ok:
                await self._fault("GO2_LOWCMD_REVOKE_FAILED", revoked.message)
                return
            self._impact_recovery_safe_hold_confirmed = True
            self._emit(
                "IMPACT_RECOVERY_GO2_SAFE_HOLD_CONFIRMED",
                landing_session_id=self._impact_landing_session_id,
                ownership_epoch=self._impact_recovery.go2_ownership_epoch,
            )
            await self.refresh_snapshot()
            now = self._clock.monotonic()
            if (
                self._impact_recovery_finalization_started_at is not None
                and now - self._impact_recovery_finalization_started_at
                >= self.config.safety.impact_recovery_finalization_timeout_s
            ):
                self._emit(
                    "IMPACT_RECOVERY_FINALIZATION_TIMEOUT",
                    landing_session_id=self._impact_landing_session_id,
                    stage="go2_safe_hold",
                )
                await self._abort_autoland_unlocked(
                    "post-touchdown Go2 safe-hold finalization timed out"
                )
                return

        if self._impact_recovery_finalization_completed_at is None:
            # The finalization budget covers only the two quiescing actions
            # above (FC/setpoint stop and Go2 safe-hold), including their
            # acknowledgement refreshes.  Once both have completed inside the
            # budget, the independent stable-dwell observation must not keep
            # aging this action timer.
            completed_at = self._clock.monotonic()
            started_at = self._impact_recovery_finalization_started_at
            if (
                started_at is None
                or not self._impact_recovery_setpoints_stopped
                or not self._impact_recovery_safe_hold_confirmed
            ):
                await self._clear_post_touchdown_stability()
                return
            if (
                completed_at - started_at
                >= self.config.safety.impact_recovery_finalization_timeout_s
            ):
                self._emit(
                    "IMPACT_RECOVERY_FINALIZATION_TIMEOUT",
                    landing_session_id=self._impact_landing_session_id,
                    stage="action_completion",
                )
                await self._abort_autoland_unlocked(
                    "post-touchdown actuator finalization timed out"
                )
                return
            self._impact_recovery_finalization_completed_at = completed_at
            self._emit(
                "IMPACT_RECOVERY_FINALIZATION_COMPLETED",
                landing_session_id=self._impact_landing_session_id,
                elapsed_s=completed_at - started_at,
            )

        if not self._post_touchdown_stability_is_valid(now):
            await self._clear_post_touchdown_stability()
            return
        last_check = self._post_touchdown_last_stability_check_at
        if self._post_touchdown_stable_since is not None and (
            last_check is None
            or now < last_check
            or now - last_check > self.config.safety.post_touchdown_stability_max_check_gap_s
        ):
            await self._clear_post_touchdown_stability()
            self._emit(
                "POST_TOUCHDOWN_STABILITY_OBSERVATION_GAP",
                landing_session_id=self._impact_landing_session_id,
            )
            now = self._clock.monotonic()
            if not self._post_touchdown_stability_is_valid(now):
                return
        if self._post_touchdown_stable_since is None:
            self._post_touchdown_stable_since = now
            self._post_touchdown_last_stability_check_at = now
            self._emit(
                "POST_TOUCHDOWN_STABLE_DWELL_STARTED",
                landing_session_id=self._impact_landing_session_id,
            )
            await self.refresh_snapshot()
            return
        self._post_touchdown_last_stability_check_at = now
        if (
            now - self._post_touchdown_stable_since
            < self.config.safety.post_touchdown_stable_confirm_s
        ):
            return

        self._impact_landing_exit_ready = True
        await self.refresh_snapshot()
        final_now = self._clock.monotonic()
        if (
            not self._snapshot.post_touchdown_stable_dwell_complete
            or not self._post_touchdown_stability_is_valid(final_now)
        ):
            await self._clear_post_touchdown_stability()
            return
        await self._complete_touchdown_transition(
            "post-touchdown recovery, FC CLEAR/execution plus persistent-zero status, Go2 safe-hold, and stable dwell confirmed"
        )

    async def _complete_touchdown_transition(self, reason: str) -> None:
        automatic_landing = self.state is SystemState.AUTO_LANDING
        impact_aware_landing = automatic_landing and self.config.go2.low_level.enabled
        if not automatic_landing and self._aborted_impact_touchdown_latched:
            self._touchdown_since = None
            self._touchdown_height_reference = None
            self._emit(
                "ABORTED_IMPACT_TOUCHDOWN_RECLASSIFICATION_BLOCKED",
                reason="the same landed episode still requires recovery or a new airborne cycle",
            )
            return
        if not automatic_landing and self.config.go2.low_level.enabled:
            revoked = await self._revoke_go2_low_level_internal(
                "manual touchdown confirmed; retain safe-hold through handover"
            )
            if not revoked.ok:
                await self._fault("GO2_LOWCMD_REVOKE_FAILED", revoked.message)
                return
        try:
            await self._stop_setpoints()
        except (BridgeError, OSError, RuntimeError) as exc:
            await self._fault("AUTOLAND_SETPOINT_STOP_FAILED", str(exc))
            return
        self._next_landing_update_at = None
        await self.refresh_snapshot()
        if impact_aware_landing:
            final_now = self._clock.monotonic()
            completion_started = self._impact_recovery_wait_started_at
            finalization_started = self._impact_recovery_finalization_started_at
            completion_timed_out = bool(
                completion_started is not None
                and final_now - completion_started
                >= self.config.safety.impact_recovery_completion_timeout_s
            )
            finalization_timed_out = bool(
                self._impact_recovery_finalization_completed_at is None
                and finalization_started is not None
                and final_now - finalization_started
                >= self.config.safety.impact_recovery_finalization_timeout_s
            )
            if completion_timed_out or finalization_timed_out:
                if completion_timed_out:
                    self._emit(
                        "IMPACT_RECOVERY_COMPLETION_TIMEOUT",
                        landing_session_id=self._impact_landing_session_id,
                        reason="deadline crossed during final touchdown transaction",
                        stage="final_stop_recheck",
                    )
                if finalization_timed_out:
                    self._emit(
                        "IMPACT_RECOVERY_FINALIZATION_TIMEOUT",
                        landing_session_id=self._impact_landing_session_id,
                        stage="final_stop_recheck",
                    )
                await self._abort_autoland_unlocked(
                    "post-touchdown recovery deadline crossed during finalization"
                )
                return
            dwell_start = self._post_touchdown_stable_since
            last_stability_check = self._post_touchdown_last_stability_check_at
            if (
                self.state is not SystemState.AUTO_LANDING
                or not self._impact_landing_exit_ready
                or dwell_start is None
                or last_stability_check is None
                or final_now - dwell_start < self.config.safety.post_touchdown_stable_confirm_s
                or final_now < last_stability_check
                or final_now - last_stability_check
                > self.config.safety.post_touchdown_stability_max_check_gap_s
                or not self._post_touchdown_stability_is_valid(final_now)
            ):
                await self._clear_post_touchdown_stability()
                self._emit(
                    "POST_TOUCHDOWN_FINAL_RECHECK_FAILED",
                    landing_session_id=self._impact_landing_session_id,
                )
                return
        try:
            await self._state_machine.transition_to(
                SystemState.TOUCHDOWN_VERIFY,
                reason=reason,
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            await self._fault("TOUCHDOWN_EXIT_REJECTED", str(exc))
            return
        if self.state is not SystemState.TOUCHDOWN_VERIFY:
            # StateMachine converts subscriber/entry-action failures to FAULT
            # and returns the original permitted record.  Never publish local
            # touchdown completion flags unless the actual committed state is
            # still exactly TOUCHDOWN_VERIFY.
            await self._fault(
                "TOUCHDOWN_EXIT_ENTRY_FAILED",
                "TOUCHDOWN_VERIFY publish/entry failed; actual state is " + self.state.name,
            )
            return
        # Publish completion flags only after every preceding action and the
        # guarded state transition succeeded.  A partial transaction can no
        # longer masquerade as a confirmed touchdown.
        self._touchdown_confirmed = True
        self._aborted_impact_touchdown_latched = False
        self._clear_aborted_impact_airborne_dwell()
        self._autoland_active = False
        self._emit("TOUCHDOWN_CONFIRMED", landing_session_id=self._impact_landing_session_id)
        await self.refresh_snapshot()

    def _landing_compliance_entry_result(self) -> OperationResult:
        if not self.config.go2.landing_compliance_enabled:
            return OperationResult.failure(
                "LANDING_COMPLIANCE_DISABLED",
                "Calibrate all four foot-force thresholds before enabling landing compliance",
            )
        lowcmd_handover_complete = (
            self.config.go2.low_level.enabled
            and self.state is SystemState.FLIGHT_READY
            and self._touchdown_confirmed
        )
        if self.state is not SystemState.TOUCHDOWN_VERIFY and not lowcmd_handover_complete:
            return OperationResult.failure(
                "INVALID_STATE",
                "Landing compliance entry requires TOUCHDOWN_VERIFY or a completed LowCmd ground handover",
            )
        snapshot = self._snapshot
        contact = assess_foot_contact(snapshot.go2, self.config.go2)
        if not pixhawk_ground_state_is_current(
            snapshot.pixhawk,
            snapshot.timestamp,
            self.config.safety.pixhawk_timeout_s,
            self.config.safety.touchdown_max_source_age_s,
        ):
            return OperationResult.failure(
                "PIXHAWK_GROUND_STATE_STALE",
                "Fresh Pixhawk heartbeat and landed-state samples are required",
            )
        if not snapshot.pixhawk.landed:
            return OperationResult.failure("TOUCHDOWN_NOT_CONFIRMED", "Pixhawk is not landed")
        if snapshot.pixhawk.armed:
            return OperationResult.failure("PIXHAWK_ARMED", "Disarm with RadioMaster first")
        if not assess_esc_telemetry(
            snapshot,
            self.config.esc.slots,
            exact_zero=True,
        ).safe:
            return OperationResult.failure(
                "ESC_RPM_NOT_ZERO",
                "Every configured X8 must be online, healthy, finite, and exactly zero RPM",
            )
        if snapshot.configuration is not Configuration.FLIGHT:
            return OperationResult.failure(
                "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                "The mechanism must remain in verified FLIGHT configuration",
            )
        if snapshot.f446.duty != 0 or snapshot.f446.faulted:
            return OperationResult.failure("F446_NOT_SAFE", "F446 must be stopped and fault-free")
        low_level = snapshot.go2.low_level_status
        if low_level.ownership_pending:
            return OperationResult.failure(
                "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
                "Explicitly release LowCmd after verified ground/disarm/zero-RPM/support "
                "before entering BalanceStand landing compliance",
            )
        if not snapshot.joint_lock_confirmed:
            return OperationResult.failure(
                "GO2_JOINT_LOCK_REQUIRED",
                "Go2 joint lock must remain confirmed after any LowCmd-to-high-level handover",
            )
        if not snapshot.go2.stable or snapshot.go2.moving or snapshot.go2.controller_active:
            return OperationResult.failure(
                "GO2_NOT_STATIONARY",
                "Go2 must be stable and stationary before releasing joint lock",
            )
        if not contact.valid:
            return OperationResult.failure(
                "GO2_FOOT_FORCE_INVALID",
                "All four calibrated foot-force channels must be valid",
            )
        if not contact.safe:
            return OperationResult.failure(
                "GO2_FOOT_CONTACT_INSUFFICIENT",
                f"Only {contact.contact_count} feet report contact; "
                f"{contact.required_count} are required",
            )
        if snapshot.active_fault_codes:
            return OperationResult.failure(
                "ACTIVE_FAULTS_PRESENT",
                "Active safety faults prevent landing compliance",
            )
        return OperationResult.success("Landing compliance entry conditions are held")

    def _landing_compliance_hold_result(self) -> OperationResult:
        if self.state is not SystemState.LANDING_COMPLIANT:
            return OperationResult.failure(
                "INVALID_STATE",
                "Landing compliance hold requires LANDING_COMPLIANT",
            )
        snapshot = self._snapshot
        contact = assess_foot_contact(snapshot.go2, self.config.go2)
        if not pixhawk_ground_state_is_current(
            snapshot.pixhawk,
            snapshot.timestamp,
            self.config.safety.pixhawk_timeout_s,
            self.config.safety.touchdown_max_source_age_s,
        ):
            return OperationResult.failure(
                "PIXHAWK_GROUND_STATE_STALE",
                "Fresh Pixhawk heartbeat and landed-state samples must remain available",
            )
        if snapshot.pixhawk.armed:
            return OperationResult.failure(
                "PIXHAWK_ARMED_DURING_LANDING_COMPLIANCE",
                "Pixhawk must remain disarmed",
            )
        if not snapshot.pixhawk.landed:
            return OperationResult.failure("TOUCHDOWN_LOST", "Pixhawk landed state was lost")
        if not assess_esc_telemetry(
            snapshot,
            self.config.esc.slots,
            exact_zero=True,
        ).safe:
            return OperationResult.failure(
                "ESC_RPM_NONZERO_DURING_LANDING_COMPLIANCE",
                "Every configured X8 must remain exactly zero RPM",
            )
        if snapshot.configuration is not Configuration.FLIGHT:
            return OperationResult.failure(
                "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                "The mechanism must remain in FLIGHT configuration",
            )
        if snapshot.f446.duty != 0 or snapshot.f446.faulted:
            return OperationResult.failure("F446_NOT_SAFE", "F446 must remain stopped")
        if not contact.valid:
            return OperationResult.failure(
                "GO2_FOOT_FORCE_INVALID",
                "All four calibrated foot-force channels must remain valid",
            )
        if not contact.safe:
            return OperationResult.failure(
                "GO2_FOOT_CONTACT_LOST",
                f"Only {contact.contact_count} feet report contact; "
                f"{contact.required_count} are required",
            )
        if (
            snapshot.go2.locomotion_mode != "BALANCE_STAND"
            or snapshot.go2.joints_locked
            or not snapshot.go2.stable
            or snapshot.go2.moving
        ):
            return OperationResult.failure(
                "GO2_LANDING_COMPLIANCE_LOST",
                "Go2 must authoritatively report a stable BALANCE_STAND posture",
            )
        return OperationResult.success("Landing compliance remains safe")

    def _landing_compliance_settle_result(self) -> OperationResult:
        since = self._landing_compliant_since
        now = self._clock.monotonic()
        if since is None or not math.isfinite(since) or not math.isfinite(now):
            return OperationResult.failure(
                "LANDING_COMPLIANCE_SETTLE_REQUIRED",
                "Landing compliance settle timing has not started",
            )
        elapsed = now - since
        required = self.config.go2.landing_compliance_settle_s
        if elapsed < required:
            return OperationResult.failure(
                "LANDING_COMPLIANCE_SETTLE_REQUIRED",
                f"Keep BalanceStand stable for {required - elapsed:.2f}s more",
                data={"elapsed_s": elapsed, "required_s": required},
            )
        return OperationResult.success(
            "Landing compliance settle complete",
            data={"elapsed_s": elapsed, "required_s": required},
        )

    async def _check_landing_compliance(self) -> None:
        if not self.config.go2.landing_compliance_enabled:
            self._landing_contact_since = None
            return
        entry = self._landing_compliance_entry_result()
        now = self._clock.monotonic()
        if not entry.ok:
            self._landing_contact_since = None
            return
        if self._landing_contact_since is None:
            self._landing_contact_since = now
            return
        if now - self._landing_contact_since < self.config.go2.landing_contact_confirm_s:
            return
        try:
            await self._state_machine.transition_to(
                SystemState.LANDING_COMPLIANT,
                reason="calibrated foot contact, Disarm, and zero X8 RPM held",
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            self._landing_contact_since = None
            self._emit("LANDING_COMPLIANCE_REJECTED", message=str(exc))
            return
        self._landing_contact_since = None
        await self.refresh_snapshot()

    def _landing_compliance_report(self) -> Mapping[str, Any]:
        contact = assess_foot_contact(self._snapshot.go2, self.config.go2)
        now = self._clock.monotonic()
        contact_elapsed = (
            0.0
            if self._landing_contact_since is None
            else max(0.0, now - self._landing_contact_since)
        )
        settle_elapsed = (
            0.0
            if self._landing_compliant_since is None
            else max(0.0, now - self._landing_compliant_since)
        )
        return {
            "enabled": self.config.go2.landing_compliance_enabled,
            "state": self.state.name,
            "foot_force": list(self._snapshot.go2.foot_force),
            "foot_force_valid": self._snapshot.go2.foot_force_valid,
            "thresholds": list(self.config.go2.foot_force_contact_thresholds),
            "contacts": list(contact.contacts),
            "contact_count": contact.contact_count,
            "required_contact_count": contact.required_count,
            "contact_safe": contact.safe,
            "contact_confirm_elapsed_s": contact_elapsed,
            "contact_confirm_required_s": self.config.go2.landing_contact_confirm_s,
            "settle_elapsed_s": settle_elapsed,
            "settle_required_s": self.config.go2.landing_compliance_settle_s,
            "pixhawk_armed": self._snapshot.pixhawk.armed,
            "pixhawk_landed": self._snapshot.pixhawk.landed,
            "maximum_esc_rpm": self._snapshot.pixhawk.maximum_esc_rpm,
            "go2_mode": self._snapshot.go2.locomotion_mode,
            "joints_locked": self._snapshot.go2.joints_locked,
            "joint_lock_confirmed": self._snapshot.joint_lock_confirmed,
            "joint_lock_source": self._snapshot.joint_lock_source,
        }

    async def _fault(
        self,
        code: str,
        message: str,
        *,
        stop_attempted: bool = False,
    ) -> None:
        violation = SafetyViolation(
            code=code,
            severity=SafetySeverity.FAULT,
            message=message,
            recommended_action="Inspect and correct the cause before clear-fault",
            timestamp=self._clock.monotonic(),
        )
        self._active_violations[code] = violation
        self._violation_history.append(violation)
        if self.state is not SystemState.FAULT:
            fault_snapshot = replace(
                self._snapshot,
                active_fault_codes=tuple(sorted(self._active_violations)),
            )
            previous_suppression = self._suppress_fault_entry_stop
            self._suppress_fault_entry_stop = stop_attempted
            try:
                try:
                    await self._state_machine.transition_to(
                        SystemState.FAULT,
                        reason=f"{code}: {message}",
                        snapshot=fault_snapshot,
                    )
                except TransitionRejected:
                    # Preserve the active violation if a programming error removed a
                    # required fault edge from the transition graph.
                    pass
            finally:
                self._suppress_fault_entry_stop = previous_suppression
        self._emit("FAULT_ENTERED", fault_code=code, message=message)
        await self.refresh_snapshot()

    def _update_violations(self, violations: Sequence[SafetyViolation]) -> None:
        current = {item.code for item in violations}
        for item in violations:
            if item.code not in self._active_violations:
                self._violation_history.append(item)
            self._active_violations[item.code] = item
        sticky = {
            "TRANSFORM_FLIGHT_FAILED",
            "TRANSFORM_WALK_FAILED",
            "F446_TRANSFORM_TIMEOUT",
            "F446_FINAL_STATE_MISMATCH",
            "F446_FINAL_DUTY_NONZERO",
            "F446_DISCONNECTED",
        }
        for code in tuple(self._active_violations):
            if code not in current and code not in sticky:
                del self._active_violations[code]

    def _record_runtime_violation(self, violation: SafetyViolation) -> None:
        if violation.code not in self._active_violations:
            self._violation_history.append(violation)
        self._active_violations[violation.code] = violation

    async def _finish_autoland_abort(self, reason: str) -> OperationResult:
        if self.state not in (SystemState.AUTO_LANDING, SystemState.AUTO_LANDING_READY):
            return OperationResult.success("Automatic landing was already inactive")
        override = replace(
            self._snapshot,
            rc=replace(self._snapshot.rc, manual_override=True),
            autoland_active=False,
            external_setpoint_active=False,
        )
        try:
            await self._state_machine.transition_to(
                SystemState.FLIGHT_MANUAL,
                reason=f"automatic landing aborted: {reason}",
                snapshot=override,
            )
        except TransitionRejected as exc:
            await self._fault("AUTOLAND_ABORT_TRANSITION_FAILED", str(exc))
            return OperationResult.failure("AUTOLAND_ABORT_FAILED", str(exc))
        if self.state is not SystemState.FLIGHT_MANUAL:
            return OperationResult.failure(
                "AUTOLAND_ABORT_FAILED",
                f"Abort entry action failed; manager entered {self.state.name}",
            )
        return OperationResult.success("Returned to FLIGHT_MANUAL")

    async def _adopt_boot_configuration(self) -> None:
        configuration = self._snapshot.configuration
        if configuration is Configuration.WALK:
            await self._state_machine.transition_to(
                SystemState.WALK,
                reason="devices verified in safe WALK configuration",
                snapshot=self._snapshot,
            )
            return
        elif configuration is Configuration.FLIGHT:
            if not self._control_writes_allowed():
                self._emit(
                    "GO2_JOINT_LOCK_REQUIRED",
                    message=(
                        "FLIGHT configuration detected, but read-only mode cannot disable "
                        "the Go2 original remote and confirm JOINT_LOCK"
                    ),
                )
                return
            if self._snapshot.go2.joints_locked:
                if not await self._go2.request_flight_pose():
                    raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected the request")
                await self.refresh_snapshot()
                await self._state_machine.transition_to(
                    SystemState.FLIGHT_READY,
                    reason="devices verified in safe FLIGHT configuration with mode=6",
                    snapshot=self._snapshot,
                )
                return
            if not await self._go2.request_stop():
                raise BridgeError("GO2_STOP_FAILED: Go2 rejected the stop request")
            await self.refresh_snapshot()
            await self._state_machine.transition_to(
                SystemState.GO2_JOINT_LOCK_WAIT,
                reason="FLIGHT configuration verified; waiting for operator-selected mode=6",
                snapshot=self._snapshot,
            )
            await self.refresh_snapshot()
            self._emit(
                "GO2_JOINT_LOCK_OPERATOR_REQUIRED",
                timeout_s=self.config.go2.joint_lock_operator_timeout_s,
                current_mode=self._snapshot.go2.locomotion_mode,
            )
            if self._runtime_mode is RuntimeMode.DRY_RUN:
                result = await self._advance_go2_joint_lock_wait(simulate_operator=True)
                if not result.ok:
                    raise BridgeError(f"{result.code}: {result.message}")
            return
        else:
            self._emit("F446_CONFIGURATION_UNKNOWN", state=self._snapshot.f446.state.value)
            return

    def _joint_lock_operator_required_result(self) -> OperationResult:
        now = self._clock.monotonic()
        deadline = self._go2_joint_lock_deadline
        remaining = (
            self.config.go2.joint_lock_operator_timeout_s
            if deadline is None
            else max(0.0, deadline - now)
        )
        return OperationResult.success(
            (
                "FLIGHT endpoint is verified and F446 is stopped. In the Unitree phone app, "
                "select Joint Lock. AeroGo2 accepts mode 6 or the configured Lock On "
                "state code (1002 by default), then waits for the filtered motion to settle. "
                "If neither is reported, visually verify Lock On, then run "
                f"go2 confirm-lock; {remaining:.1f}s remain. Do not command Go2 locomotion."
            ),
            code="GO2_JOINT_LOCK_OPERATOR_REQUIRED",
            data={
                "state": self.state.name,
                "current_mode": self._snapshot.go2.locomotion_mode,
                "required_mode": "JOINT_LOCK",
                "required_mode_code": 6,
                "accepted_lock_state_codes": list(self.config.go2.joint_lock_state_codes),
                "manual_fallback_command": "go2 confirm-lock",
                "manual_fallback_phrase": "CONFIRM_GO2_JOINT_LOCK",
                "remaining_s": remaining,
                "f446_duty": self._snapshot.f446.duty,
                "automatic_transition": "FLIGHT_READY",
                "post_handover_sport_sample_received": (
                    self.state is not SystemState.GO2_GROUND_HANDOVER
                    or self._post_handover_go2_sample_is_fresh()
                ),
            },
        )

    async def confirm_operator_joint_lock(
        self,
        operator_confirmed: bool = False,
    ) -> OperationResult:
        """Accept a guarded operator assertion without falsifying Go2 telemetry."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "HARDWARE_WRITE_DISABLED",
                "Operator joint-lock confirmation requires an unlocked hardware process",
            )
        if not operator_confirmed:
            return OperationResult.failure(
                "CONFIRMATION_REQUIRED",
                "Exact confirmation CONFIRM_GO2_JOINT_LOCK is required",
            )
        if self.state is SystemState.FLIGHT_READY:
            await self.refresh_snapshot()
            if self._snapshot.joint_lock_confirmed:
                return OperationResult.success(
                    "Joint Lock was already confirmed and FLIGHT_READY is active.",
                    code="GO2_JOINT_LOCK_ALREADY_CONFIRMED",
                    data={
                        "state": self.state.name,
                        "joint_lock_telemetry": self._snapshot.go2.joints_locked,
                        "joint_lock_confirmed": self._snapshot.joint_lock_confirmed,
                        "joint_lock_source": self._snapshot.joint_lock_source,
                    },
                )
            return OperationResult.failure(
                "GO2_JOINT_LOCK_REQUIRED",
                "FLIGHT_READY lost its joint-lock confirmation; supervised recovery is required",
            )
        if self.state is not SystemState.GO2_JOINT_LOCK_WAIT:
            return OperationResult.failure(
                "INVALID_STATE",
                "Operator joint-lock confirmation requires GO2_JOINT_LOCK_WAIT",
            )

        await self.refresh_snapshot()
        if self._snapshot.go2.joints_locked:
            return await self._advance_go2_joint_lock_wait()
        if self._snapshot.go2.fault_code not in self.config.go2.accepted_state_codes:
            return OperationResult.failure(
                "GO2_STATE_CODE_NOT_ACCEPTED",
                (
                    f"Go2 error_code={self._snapshot.go2.fault_code} is not in "
                    f"go2.accepted_state_codes={list(self.config.go2.accepted_state_codes)}"
                ),
            )

        candidate = replace(
            self._snapshot,
            joint_lock_confirmed=True,
            joint_lock_source="operator",
        )
        interlock = self._landing_interlocks.can_enter_flight_ready(
            candidate,
            require_flight_enable_low=True,
        )
        if not interlock.permitted:
            return OperationResult.failure(
                interlock.codes[0] if interlock.codes else "GO2_OPERATOR_LOCK_REJECTED",
                "; ".join(interlock.messages) or "Operator lock confirmation checks failed",
                data={
                    "checks": [
                        {"code": code, "message": message, "passed": False}
                        for code, message in zip(interlock.codes, interlock.messages)
                    ]
                },
            )

        try:
            if not await self._go2.finalize_operator_joint_lock():
                raise BridgeError("Go2 rejected joystick disable after operator lock confirmation")
            self._operator_joint_lock_confirmed = True
            await self.refresh_snapshot()
            await self._state_machine.transition_to(
                SystemState.FLIGHT_READY,
                reason=(
                    "operator confirmed phone-app joint lock; joystick input disabled; "
                    "raw SportModeState retained"
                ),
                snapshot=self._snapshot,
            )
            self._go2_joint_lock_deadline = None
            self._emit(
                "GO2_JOINT_LOCK_OPERATOR_CONFIRMED",
                raw_mode=self._snapshot.go2.locomotion_mode,
                raw_joints_locked=self._snapshot.go2.joints_locked,
            )
            self._emit("FLIGHT_READY")
            await self.refresh_snapshot()
            return OperationResult.success(
                (
                    "FLIGHT_READY entered using explicit operator joint-lock confirmation. "
                    "Raw joints_locked remains unchanged and source=operator."
                ),
                code="FLIGHT_READY_OPERATOR_LOCK",
                data={
                    "state": self.state.name,
                    "joint_lock_telemetry": self._snapshot.go2.joints_locked,
                    "joint_lock_confirmed": self._snapshot.joint_lock_confirmed,
                    "joint_lock_source": self._snapshot.joint_lock_source,
                },
            )
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            self._operator_joint_lock_confirmed = False
            await self.refresh_snapshot()
            message = str(exc)
            await self._fault("GO2_OPERATOR_LOCK_FINALIZE_FAILED", message)
            return OperationResult.failure("GO2_OPERATOR_LOCK_FINALIZE_FAILED", message)

    async def _advance_go2_joint_lock_wait(
        self,
        *,
        simulate_operator: bool = False,
    ) -> OperationResult:
        if self.state not in {
            SystemState.GO2_JOINT_LOCK_WAIT,
            SystemState.GO2_GROUND_HANDOVER,
        }:
            return OperationResult.failure(
                "INVALID_STATE",
                "Go2 joint-lock completion requires a JOINT_LOCK_WAIT/GROUND_HANDOVER state",
            )
        if self.state is SystemState.GO2_GROUND_HANDOVER:
            if self._go2_ground_handover_started_at is None:
                message = "Ground handover has no causal SportModeState timestamp barrier"
                await self._fault("GO2_HANDOVER_BARRIER_MISSING", message)
                return OperationResult.failure("GO2_HANDOVER_BARRIER_MISSING", message)
            if not self._post_handover_go2_sample_is_fresh():
                deadline = self._go2_joint_lock_deadline
                if deadline is not None and self._clock.monotonic() >= deadline:
                    message = (
                        "No fresh SportModeState sample arrived after the LowCmd handover "
                        f"within {self.config.go2.joint_lock_operator_timeout_s:.1f}s"
                    )
                    await self._fault("GO2_POST_HANDOVER_STATE_TIMEOUT", message)
                    return OperationResult.failure("GO2_POST_HANDOVER_STATE_TIMEOUT", message)
                return self._joint_lock_operator_required_result()
        if simulate_operator and not self._snapshot.go2.joints_locked:
            try:
                if not await self._go2.request_flight_pose():
                    raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected the request")
            except (BridgeError, OSError, RuntimeError) as exc:
                message = str(exc)
                await self._fault("GO2_JOINT_LOCK_FAILED", message)
                return OperationResult.failure("GO2_JOINT_LOCK_FAILED", message)
            await self.refresh_snapshot()

        deadline = self._go2_joint_lock_deadline
        if deadline is not None and self._clock.monotonic() >= deadline:
            message = (
                "Go2 did not report and settle a valid joint-lock signal within "
                f"{self.config.go2.joint_lock_operator_timeout_s:.1f}s"
            )
            await self._fault("GO2_JOINT_LOCK_OPERATOR_TIMEOUT", message)
            return OperationResult.failure("GO2_JOINT_LOCK_OPERATOR_TIMEOUT", message)

        if not self._snapshot.go2.joints_locked:
            return self._joint_lock_operator_required_result()

        if not self._go2_joint_lock_is_settled():
            return OperationResult.success(
                (
                    "Joint Lock signal detected; waiting for Go2 motion telemetry to settle "
                    "before FLIGHT_READY."
                ),
                code="GO2_JOINT_LOCK_SETTLING",
                data={
                    "fault_code": self._snapshot.go2.fault_code,
                    "mode": self._snapshot.go2.locomotion_mode,
                    "velocity_mps": self._snapshot.go2.velocity_mps,
                    "body_velocity": list(self._snapshot.go2.body_velocity),
                },
            )

        try:
            # The operator/app already selected mode 6. This call only disables
            # joystick input and re-verifies the authoritative SportModeState.
            if not await self._go2.request_flight_pose():
                raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected finalization")
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                raise BridgeError("GO2_JOINT_LOCK_UNCONFIRMED: Go2 lost mode=6 during finalization")
            await self._state_machine.transition_to(
                SystemState.FLIGHT_READY,
                reason="Go2 joint-lock telemetry verified and joystick input disabled",
                snapshot=self._snapshot,
            )
            self._go2_joint_lock_deadline = None
            self._go2_ground_handover_started_at = None
            self._emit("FLIGHT_CONFIGURATION_VERIFIED")
            self._emit("FLIGHT_READY")
            await self.refresh_snapshot()
            return OperationResult.success(
                "FLIGHT verified with Go2 joint-lock telemetry; only RadioMaster may arm",
                code="FLIGHT_READY",
            )
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            message = str(exc)
            code = (
                "GO2_JOINT_LOCK_UNCONFIRMED"
                if message.startswith("GO2_JOINT_LOCK_UNCONFIRMED:")
                else "GO2_JOINT_LOCK_FAILED"
            )
            await self._fault(code, message)
            return OperationResult.failure(code, message)

    async def _prepare_disconnect(self) -> OperationResult:
        # State names alone are not proof of physical safety: FAULT and
        # EMERGENCY_STOP can be entered while airborne.  Refresh and enforce
        # the same observable ground facts before any transport disappears.
        await self.refresh_snapshot()
        low_level = self._snapshot.go2.low_level_status
        if low_level.ownership_pending:
            return OperationResult.failure(
                "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
                "Release the continuous LowCmd owner through the explicit ground handover before disconnecting",
            )
        all_required_disconnected = (
            not self._snapshot.pixhawk.connected
            and not self._snapshot.f446.connected
            and (not self.config.go2.enabled or not self._snapshot.go2.connected)
        )
        if self.state is SystemState.BOOT_SAFE and all_required_disconnected:
            return OperationResult.success("No connected hardware transport remains")
        missing_authority = []
        if not self._snapshot.pixhawk.connected:
            missing_authority.append("Pixhawk")
        if not self._snapshot.f446.connected:
            missing_authority.append("F446")
        if self.config.go2.enabled and not self._snapshot.go2.connected:
            missing_authority.append("Go2")
        if missing_authority and self._ever_entered_operational_state:
            return OperationResult.failure(
                "GROUND_PROOF_UNAVAILABLE",
                "Cannot prove a safe ground disconnect after operational telemetry was lost: "
                + ", ".join(missing_authority),
            )
        if self._snapshot.pixhawk.connected:
            if not pixhawk_ground_state_is_current(
                self._snapshot.pixhawk,
                self._snapshot.timestamp,
                self.config.safety.pixhawk_timeout_s,
                self.config.safety.touchdown_max_source_age_s,
            ):
                return OperationResult.failure(
                    "GROUND_PROOF_UNAVAILABLE",
                    "Fresh Pixhawk heartbeat and landed-state samples are required before disconnect",
                )
            if self._snapshot.pixhawk.armed or not self._snapshot.pixhawk.landed:
                return OperationResult.failure(
                    "UNSAFE_DISCONNECT",
                    "Pixhawk must be disarmed and report landed before disconnect",
                )
            if not assess_esc_telemetry(
                self._snapshot,
                self.config.esc.slots,
                exact_zero=True,
            ).safe:
                return OperationResult.failure(
                    "UNSAFE_DISCONNECT",
                    "Every configured rotor must have fresh, healthy, finite, exactly-zero RPM before disconnect",
                )
        if self._snapshot.f446.connected and (
            self._snapshot.f446.duty != 0 or self._snapshot.f446.faulted
        ):
            return OperationResult.failure(
                "UNSAFE_DISCONNECT",
                "F446 must be stopped and fault-free before disconnect",
            )
        if self.state in (
            SystemState.BOOT_SAFE,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            return OperationResult.success(f"Disconnect permitted in {self.state.name}")
        if self.state not in (SystemState.WALK, SystemState.FLIGHT_READY):
            return OperationResult.failure(
                "UNSAFE_DISCONNECT",
                "Disconnect requires BOOT_SAFE, FAULT, or a disarmed safe idle state",
            )
        try:
            await self._state_machine.transition_to(
                SystemState.BOOT_SAFE,
                reason="operator requested safe device disconnect",
                snapshot=self._snapshot,
            )
        except TransitionRejected as exc:
            return OperationResult.failure("UNSAFE_DISCONNECT", str(exc))
        await self.refresh_snapshot()
        if self.state is not SystemState.BOOT_SAFE:
            return OperationResult.failure(
                "UNSAFE_DISCONNECT",
                f"BOOT_SAFE entry failed; manager entered {self.state.name}",
            )
        return OperationResult.success("Manager entered BOOT_SAFE for disconnect")

    def _stationary_dwell_result(self) -> OperationResult:
        now = self._clock.monotonic()
        since = self._go2_stationary_since
        if since is None or not math.isfinite(now) or not math.isfinite(since):
            return OperationResult.failure(
                "GO2_STATIONARY_DWELL_REQUIRED",
                "Go2 stationary confirmation has not started",
            )
        elapsed = now - since
        required = self.config.safety.stationary_confirm_s
        if elapsed < required:
            return OperationResult.failure(
                "GO2_STATIONARY_DWELL_REQUIRED",
                f"Go2 must remain stationary for {required:.3f}s "
                f"({max(0.0, elapsed):.3f}s observed)",
            )
        return OperationResult.success("Go2 stationary dwell confirmed")

    def _lowcmd_ground_stationary_dwell_result(self) -> OperationResult:
        now = self._clock.monotonic()
        since = self._lowcmd_ground_stationary_since
        source_start = self._lowcmd_ground_stationary_source_start
        if (
            since is None
            or source_start is None
            or not math.isfinite(now)
            or not math.isfinite(since)
        ):
            return OperationResult.failure(
                "GO2_GROUND_STATIONARY_DWELL_REQUIRED",
                "Disarmed, landed, stationary Go2 confirmation has not started",
            )
        elapsed = now - since
        required = self.config.safety.stationary_confirm_s
        if elapsed < required:
            return OperationResult.failure(
                "GO2_GROUND_STATIONARY_DWELL_REQUIRED",
                "Go2 must remain disarmed, landed, stable, and stationary for "
                f"{required:.3f}s ({max(0.0, elapsed):.3f}s observed)",
            )
        heartbeat_start, landed_start = source_start
        if (
            self._snapshot.pixhawk.heartbeat_timestamp <= heartbeat_start
            or self._snapshot.pixhawk.landed_state_timestamp <= landed_start
        ):
            return OperationResult.failure(
                "GO2_GROUND_STATIONARY_DWELL_REQUIRED",
                "Ground dwell requires causally newer heartbeat and landed-state samples",
            )
        return OperationResult.success("LowCmd ground stationary dwell confirmed")

    def _post_handover_go2_sample_is_fresh(self) -> bool:
        barrier = self._go2_ground_handover_started_at
        status = self._snapshot.go2
        timestamp = status.timestamp
        return (
            barrier is not None
            and math.isfinite(barrier)
            and status.connected
            and math.isfinite(timestamp)
            and timestamp > barrier
            and timestamp_is_fresh(
                self._snapshot.timestamp,
                timestamp,
                self.config.safety.go2_timeout_s,
            )
        )

    def _go2_joint_lock_is_settled(self) -> bool:
        status = self._snapshot.go2
        components = status.body_velocity
        limit = self.config.safety.stationary_velocity_mps
        return (
            status.connected
            and status.joints_locked
            and math.isfinite(status.velocity_mps)
            and abs(status.velocity_mps) < limit
            and len(components) == 3
            and all(math.isfinite(value) and abs(value) < limit for value in components)
            and status.stable
            and not status.moving
            and not status.controller_active
        )

    def _filter_go2_joint_lock_wait_violations(
        self,
        violations: Tuple[SafetyViolation, ...],
    ) -> Tuple[SafetyViolation, ...]:
        """Debounce only the phone Lock On posture transient.

        Device freshness, RC, ESC, Pixhawk and F446 violations are never filtered.
        A locomotion/unsafe-speed violation becomes blocking after the configured
        initial grace and continuous confirmation windows.
        """

        if self.state is not SystemState.GO2_JOINT_LOCK_WAIT:
            self._go2_joint_lock_unsafe_since = None
            return violations

        unsafe_code = "GO2_UNSAFE_DURING_JOINT_LOCK"
        if not any(item.code == unsafe_code for item in violations):
            self._go2_joint_lock_unsafe_since = None
            return violations

        now = self._clock.monotonic()
        entered_at = self._go2_joint_lock_entered_at
        if entered_at is None:
            entered_at = now
            self._go2_joint_lock_entered_at = now

        if now - entered_at < self.config.go2.joint_lock_transition_grace_s:
            self._go2_joint_lock_unsafe_since = None
            return tuple(item for item in violations if item.code != unsafe_code)

        if self._go2_joint_lock_unsafe_since is None:
            self._go2_joint_lock_unsafe_since = now
            return tuple(item for item in violations if item.code != unsafe_code)

        if now - self._go2_joint_lock_unsafe_since < self.config.go2.joint_lock_unsafe_confirm_s:
            return tuple(item for item in violations if item.code != unsafe_code)
        return violations

    def _f446_current_margin_is_clear(
        self,
        status: F446Status,
        now: float,
    ) -> bool:
        current = status.used_current_adc
        threshold = status.threshold_adc
        if (
            not status.connected
            or timestamp_age(now, status.timestamp) > self.config.safety.f446_timeout_s
            or current is None
            or isinstance(current, bool)
            or threshold is None
            or isinstance(threshold, bool)
        ):
            return False
        try:
            current_value = float(current)
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            return False
        margin = self.config.f446.current_safe_margin_adc
        if (
            not math.isfinite(current_value)
            or current_value < 0.0
            or not math.isfinite(threshold_value)
            or threshold_value <= float(margin)
        ):
            return False
        return current_value <= threshold_value - float(margin)

    def _current_clear_dwell_result(self) -> OperationResult:
        now = self._clock.monotonic()
        if not self._f446_current_margin_is_clear(self._snapshot.f446, now):
            return OperationResult.failure(
                "F446_CURRENT_MARGIN_UNSAFE",
                "F446 current must be finite, non-negative, fresh, and no greater "
                "than threshold_raw minus current_safe_margin_adc",
            )
        since = self._f446_current_clear_since
        if since is None or not math.isfinite(now) or not math.isfinite(since):
            return OperationResult.failure(
                "F446_CURRENT_CLEAR_HOLD_REQUIRED",
                "F446 current-clear confirmation has not started",
            )
        elapsed = now - since
        required = self.config.f446.current_clear_hold_s
        if elapsed < required and not math.isclose(
            elapsed,
            required,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return OperationResult.failure(
                "F446_CURRENT_CLEAR_HOLD_REQUIRED",
                f"F446 current must remain below its safe margin for {required:.3f}s "
                f"({max(0.0, elapsed):.3f}s observed)",
            )
        return OperationResult.success("F446 current-clear hold confirmed")

    async def report_monitor_failure(self, message: str) -> OperationResult:
        detail = message.strip() or "unknown safety monitor failure"
        stop_result = await self.stop_supervised()
        if not stop_result.ok:
            detail += f"; supervised stop incomplete: {stop_result.message}"
        await self._fault("SAFETY_MONITOR_FAILURE", detail)
        return OperationResult.failure("SAFETY_MONITOR_FAILURE", detail)

    @staticmethod
    def _flatten_mapping(
        value: Mapping[Any, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(item, Mapping):
                flattened.update(SystemManager._flatten_mapping(item, dotted))
            else:
                flattened[dotted] = item
        return flattened

    async def _stop_setpoints(self) -> None:
        if not self._control_writes_allowed():
            self._setpoint_active = False
            return
        result = await self._pixhawk.stop_external_setpoints()
        if not result.ok:
            raise BridgeError(f"{result.code}: {result.message}")
        self._setpoint_active = False

    async def _safe_f446_stop(self) -> OperationResult:
        if not self._control_writes_allowed():
            return OperationResult.success("F446 writes are locked; no stop command was issued")
        try:
            result = await self._f446.stop()
        except (BridgeError, OSError, RuntimeError) as exc:
            return OperationResult.failure("F446_STOP_FAILED", str(exc))
        if not result.ok:
            return OperationResult.failure(result.code, result.message)
        return result

    async def _enter_boot_safe(self, snapshot: SystemSnapshot) -> None:
        await self._stop_setpoints()
        self._autoland_active = False
        self._maintenance_mode = False
        self._operator_confirmed_configuration = None
        self._manual_marked_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False
        self._go2_joint_lock_deadline = None
        self._go2_ground_handover_started_at = None
        self._go2_joint_lock_entered_at = None
        self._go2_joint_lock_unsafe_since = None

    async def _enter_manual_positioning(self, snapshot: SystemSnapshot) -> None:
        self._maintenance_mode = True
        self._manual_marked_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False

    async def _enter_go2_joint_lock_wait(self, snapshot: SystemSnapshot) -> None:
        self._maintenance_mode = False
        self._manual_marked_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False
        now = self._clock.monotonic()
        self._go2_joint_lock_deadline = now + self.config.go2.joint_lock_operator_timeout_s
        if snapshot.state is not SystemState.GO2_GROUND_HANDOVER:
            self._go2_ground_handover_started_at = None
        self._go2_joint_lock_entered_at = now
        self._go2_joint_lock_unsafe_since = None

    async def _leave_manual_positioning(self, snapshot: SystemSnapshot) -> None:
        self._maintenance_mode = False
        self._manual_marked_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False
        self._go2_joint_lock_deadline = None
        self._go2_ground_handover_started_at = None
        self._go2_joint_lock_entered_at = None
        self._go2_joint_lock_unsafe_since = None

    async def _enter_landing_compliant(self, snapshot: SystemSnapshot) -> None:
        if not self._control_writes_allowed():
            raise BridgeError("Go2 landing compliance requires unlocked hardware writes")
        failure: Optional[Exception] = None
        try:
            await self.refresh_snapshot()
            low_level = self._snapshot.go2.low_level_status
            if low_level.ownership_pending:
                raise BridgeError(
                    "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED: complete the explicit "
                    "ground/support handover before requesting BalanceStand"
                )
            if not await self._go2.request_landing_pose():
                raise BridgeError("Go2 rejected the BalanceStand landing posture")
            await self.refresh_snapshot()
            hold = self._landing_compliance_hold_result()
            if not hold.ok:
                raise BridgeError(f"{hold.code}: {hold.message}")
            self._landing_compliant_since = self._clock.monotonic()
            self._emit(
                "LANDING_COMPLIANCE_STARTED",
                settle_s=self.config.go2.landing_compliance_settle_s,
            )
            return
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            failure = exc
        self._landing_compliant_since = None
        relock_error: Optional[str] = None
        try:
            if not await self._go2.request_flight_pose():
                relock_error = "Go2 rejected emergency JOINT_LOCK restoration"
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                relock_error = "Go2 did not confirm emergency JOINT_LOCK restoration"
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            relock_error = str(exc)
        message = str(failure) if failure is not None else "unknown landing compliance failure"
        if relock_error is not None:
            message += f"; relock failed: {relock_error}"
        raise BridgeError(message)

    async def _finish_landing_compliance(self, reason: str) -> OperationResult:
        if self.state is not SystemState.LANDING_COMPLIANT:
            return OperationResult.failure(
                "INVALID_STATE",
                "Go2 landing compliance can only finish from LANDING_COMPLIANT",
            )
        try:
            if not await self._go2.request_stop():
                raise BridgeError("Go2 rejected the stop request before manual JOINT_LOCK")
            await self.refresh_snapshot()
            await self._state_machine.transition_to(
                SystemState.GO2_JOINT_LOCK_WAIT,
                reason=f"landing compliance finished; waiting for operator mode=6: {reason}",
                snapshot=self._snapshot,
            )
            self._landing_compliant_since = None
            await self.refresh_snapshot()
            self._emit(
                "GO2_JOINT_LOCK_OPERATOR_REQUIRED",
                timeout_s=self.config.go2.joint_lock_operator_timeout_s,
                current_mode=self._snapshot.go2.locomotion_mode,
                reason=reason,
            )
            if self._runtime_mode is RuntimeMode.DRY_RUN:
                return await self._advance_go2_joint_lock_wait(simulate_operator=True)
            return self._joint_lock_operator_required_result()
        except (
            AeroGo2Error,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return OperationResult.failure("GO2_LANDING_RELOCK_FAILED", str(exc))

    async def _enter_fault_state(self, snapshot: SystemSnapshot) -> None:
        self._autoland_active = False
        self._maintenance_mode = False
        self._operator_confirmed_configuration = None
        self._manual_marked_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False
        self._go2_joint_lock_deadline = None
        self._go2_ground_handover_started_at = None
        self._go2_joint_lock_entered_at = None
        self._go2_joint_lock_unsafe_since = None
        if self._suppress_fault_entry_stop and not self.config.go2.low_level.enabled:
            return
        # Every action is idempotent and independently safety-relevant.  A
        # failed LowCmd revoke must not prevent setpoint/F446 stop attempts,
        # and a caller's earlier partial stop must not suppress the revoke that
        # keeps the same sole writer in conservative safe-hold.
        failures: List[str] = []
        try:
            gate = await self._revoke_ground_arm_authorization_unlocked("FAULT entered")
            if not gate.ok:
                failures.append(f"{gate.code}: {gate.message}")
        except Exception as exc:
            failures.append(f"GROUND_ARM_AUTH_REVOKE_EXCEPTION: {exc}")
        try:
            revoke = await self._revoke_go2_low_level_internal(
                "FAULT entered; retain sole-writer safe-hold"
            )
            if not revoke.ok:
                failures.append(f"{revoke.code}: {revoke.message}")
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"GO2_LOWCMD_REVOKE_EXCEPTION: {exc}")
        try:
            await self._stop_setpoints()
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"SETPOINT_STOP_FAILED: {exc}")
        try:
            stop_result = await self._safe_f446_stop()
            if not stop_result.ok:
                failures.append(f"{stop_result.code}: {stop_result.message}")
        except (AeroGo2Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"F446_STOP_EXCEPTION: {exc}")
        if failures:
            raise BridgeError("; ".join(failures))

    async def _enter_emergency_stop(self, snapshot: SystemSnapshot) -> None:
        self._manual_marked_configuration = None
        self._go2_joint_lock_deadline = None
        self._go2_ground_handover_started_at = None
        self._go2_joint_lock_entered_at = None
        self._go2_joint_lock_unsafe_since = None
        try:
            gate = await self._revoke_ground_arm_authorization_unlocked("EMERGENCY_STOP entered")
        except Exception as exc:
            gate = OperationResult.failure(
                "GROUND_ARM_AUTH_REVOKE_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
        result = await self._stop_supervised_unlocked()
        if not gate.ok:
            raise BridgeError(f"{gate.code}: {gate.message}; stop={result.code}: {result.message}")
        if not result.ok:
            raise BridgeError(f"{result.code}: {result.message}")

    @staticmethod
    def _specific_f446_failure_code(message: str, fallback: str) -> str:
        for code in (
            "GO2_JOINT_LOCK_FAILED",
            "GO2_JOINT_LOCK_UNCONFIRMED",
            "F446_OVERCURRENT",
            "F446_FAULT",
            "F446_TRANSFORM_TIMEOUT",
            "F446_FINAL_STATE_MISMATCH",
            "F446_FINAL_DUTY_NONZERO",
            "F446_DISCONNECTED",
        ):
            if message.startswith(f"{code}:"):
                return code
        return fallback

    def _configuration_from_f446(self, status: F446Status) -> Configuration:
        if status.faulted or status.duty != 0:
            return Configuration.UNKNOWN
        if status.state is self.config.f446.expected_flight_state:
            return Configuration.FLIGHT
        if status.state is self.config.f446.expected_walk_state:
            return Configuration.WALK
        return Configuration.UNKNOWN

    def _log_transition(self, record: Any) -> None:
        transition_violations: Tuple[SafetyViolation, ...] = ()
        if not record.permitted:
            violation = self._safety_monitor.invalid_transition_violation(
                self._snapshot,
                record.new_state,
                "; ".join(record.guard_codes),
            )
            self._record_runtime_violation(violation)
            transition_violations = (violation,)
        self._emit(
            "STATE_TRANSITION",
            previous_state=record.previous_state.name,
            new_state=record.new_state.name,
            transition_reason=record.reason,
            command_result="PERMITTED" if record.permitted else "REJECTED",
            safety_violations=transition_violations,
            guard_codes=list(record.guard_codes),
            entry_action_error=record.entry_action_error,
        )

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self._event_logger is not None:
            context: Dict[str, Any] = {
                "pixhawk_status": self._snapshot.pixhawk,
                "f446_status": self._snapshot.f446,
                "go2_status": self._snapshot.go2,
                "operator_request": self._snapshot.operator,
                "safety_violations": tuple(self.violations),
                "landing_command": self._last_landing_command,
            }
            context.update(fields)
            self._event_logger.emit(
                event_type=event_type,
                system_state=self.state.name,
                monotonic_timestamp=self._clock.monotonic(),
                **context,
            )

    @staticmethod
    def _violation_dict(item: SafetyViolation) -> Mapping[str, Any]:
        return {
            "code": item.code,
            "severity": item.severity.value,
            "message": item.message,
            "recommended_action": item.recommended_action,
            "timestamp": item.timestamp,
        }
