"""Safety-owned orchestration for every AeroGo2 command path."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from aerogo2.bridges.f446_interface import F446Interface
from aerogo2.bridges.go2_interface import Go2Interface
from aerogo2.bridges.pixhawk_interface import PixhawkInterface
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446State,
    RuntimeMode,
    SafetySeverity,
    SystemState,
)
from aerogo2.common.exceptions import AeroGo2Error, BridgeError, TransitionRejected
from aerogo2.common.immutable import deep_thaw
from aerogo2.common.models import (
    F446Status,
    LandingCommand,
    LandingEstimate,
    OperatorRequest,
    RCStatus,
    SafetyViolation,
    SystemSnapshot,
    TransitionRecord,
    snapshot_to_dict,
)
from aerogo2.common.results import GuardResult, OperationResult
from aerogo2.landing.controller_base import LandingControllerBase
from aerogo2.landing.safety_filter import LandingSafetyFilter
from aerogo2.manager.state_machine import StateMachine
from aerogo2.manager.transition_guards import TRANSFORM_STATES, TransitionGuards
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.go2_contact import assess_foot_contact
from aerogo2.safety.interlocks import SafetyInterlocks
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
    ) -> None:
        self.config = config
        self._pixhawk = pixhawk
        self._f446 = f446
        self._go2 = go2
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
        self._active_violations: Dict[str, SafetyViolation] = {}
        self._violation_history: List[SafetyViolation] = []
        self._last_landing_command = LandingCommand(timestamp=now)
        self._last_landing_update: Optional[float] = None
        self._next_landing_update_at: Optional[float] = None
        self._touchdown_since: Optional[float] = None
        self._touchdown_height_reference: Optional[float] = None
        self._landing_contact_since: Optional[float] = None
        self._landing_compliant_since: Optional[float] = None
        self._go2_stationary_since: Optional[float] = None
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
        self._manual_motion_deadline: Optional[float] = None
        self._manual_last_direction: Optional[str] = None
        self._manual_motion_started = False
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
    def transitions(self) -> Tuple[Any, ...]:
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
        if (
            record.previous_state is SystemState.FLIGHT_READY
            and record.new_state is not SystemState.FLIGHT_READY
        ):
            await self._revoke_ground_arm_authorization(f"state changed to {record.new_state.name}")

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

    async def connect_all(self) -> OperationResult:
        """Connect injected bridges and adopt only a verified idle configuration."""

        try:
            await self._pixhawk.connect()
            await self._f446.connect()
            await self._go2.connect()
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
        pixhawk = self._pixhawk.get_status()
        f446 = self._f446.get_status()
        go2 = self._go2.get_status()
        now = self._clock.monotonic()
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
        operator = OperatorRequest(
            timestamp=self._rc.timestamp,
            flight_enable=self._rc.flight_enable,
            morphology_request=self._rc.morphology_request.value,
            auto_landing_request=self._rc.auto_landing_request.value,
            manual_override=self._rc.manual_override,
        )
        bridge_authorized = self._pixhawk.ground_arm_authorization_active()
        if self._ground_arm_authorized and not bridge_authorized:
            self._ground_arm_authorized = False
            self._ground_arm_authorization_expires_at = None
        ground_arm_authorized = self._ground_arm_authorized and bridge_authorized
        ground_arm_authorization_expires_at = (
            self._ground_arm_authorization_expires_at if ground_arm_authorized else None
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
            ground_arm_authorized=ground_arm_authorized,
            ground_arm_authorization_expires_at=ground_arm_authorization_expires_at,
            active_fault_codes=tuple(sorted(self._active_violations)),
        )
        return self._snapshot

    def accept_rc_status(self, status: RCStatus) -> None:
        """Accept telemetry produced by RCMonitor; this never sends RC data."""

        self._rc = status

    def accept_landing_estimate(self, estimate: LandingEstimate) -> None:
        """Accept simulated estimator/ground-sensor output."""

        self._estimate = estimate

    async def authorize_ground_arm(self) -> OperationResult:
        """Open a short, one-shot window; this method never arms Pixhawk."""

        if not self._control_writes_allowed():
            return OperationResult.failure(
                "HARDWARE_WRITE_DISABLED",
                "Ground-arm authorization requires an explicitly unlocked hardware process",
            )
        await self.refresh_snapshot()
        guard = self._flight_readiness_guard(require_flight_enable_low=True)
        if not guard.permitted:
            return OperationResult.failure(
                guard.codes[0] if guard.codes else "FLIGHT_NOT_READY",
                "; ".join(guard.messages) or "Flight readiness checks failed",
                data=self._flight_readiness_report(require_flight_enable_low=True),
            )
        try:
            result = await self._pixhawk.set_ground_arm_authorization(
                True,
                _GROUND_ARM_AUTHORIZATION_TTL_S,
            )
        except (BridgeError, OSError, RuntimeError, ValueError) as exc:
            return OperationResult.failure("GROUND_ARM_AUTHORIZATION_FAILED", str(exc))
        if not result.ok or not self._pixhawk.ground_arm_authorization_active():
            self._ground_arm_authorized = False
            self._ground_arm_authorization_expires_at = None
            return OperationResult.failure(
                result.code or "GROUND_ARM_AUTHORIZATION_FAILED",
                result.message or "Pixhawk arm gate did not become active",
                data=result.data,
            )
        now = self._clock.monotonic()
        self._ground_arm_authorized = True
        self._ground_arm_authorization_expires_at = now + _GROUND_ARM_AUTHORIZATION_TTL_S
        self._emit(
            "GROUND_ARM_AUTHORIZED",
            ttl_s=_GROUND_ARM_AUTHORIZATION_TTL_S,
            rc_channel=self.config.rc.flight_enable_channel,
        )
        await self.refresh_snapshot()
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
        return await self._revoke_ground_arm_authorization("operator request")

    async def _revoke_ground_arm_authorization(self, reason: str) -> OperationResult:
        was_active = self._ground_arm_authorized
        self._ground_arm_authorized = False
        self._ground_arm_authorization_expires_at = None
        try:
            result = await self._pixhawk.set_ground_arm_authorization(False, 0.0)
        except (BridgeError, OSError, RuntimeError, ValueError) as exc:
            self._emit("GROUND_ARM_AUTH_REVOKE_FAILED", reason=reason, error=str(exc))
            return OperationResult.failure("GROUND_ARM_AUTH_REVOKE_FAILED", str(exc))
        self._emit("GROUND_ARM_AUTH_REVOKED", reason=reason, was_active=was_active)
        if not result.ok:
            return result
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
        }:
            return OperationResult.failure(
                "INVALID_STATE",
                f"Manual positioning cannot start from {self.state.name}",
            )
        try:
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
                "Manual positioning active; use mf/mr or limf/limr, stop, then confirm walk/flight",
                code="F446_MANUAL_POSITIONING_ACTIVE",
                data={
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
                "next": "stop, then confirm walk or confirm flight",
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

    async def confirm_manual_configuration(
        self,
        target: Configuration,
        operator_confirmed: bool = False,
    ) -> OperationResult:
        """Accept a stopped manual position only after explicit operator confirmation."""

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

        await self.refresh_snapshot()
        if self._snapshot.f446.duty != 0:
            return OperationResult.failure(
                "F446_STILL_MOVING",
                "Run stop and verify duty=0 before confirming the physical position",
            )
        if not self._manual_motion_started:
            return OperationResult.failure(
                "F446_MANUAL_MOTION_REQUIRED",
                "No movement has been started in this manual positioning session",
            )
        required_direction = self.config.f446.direction_for(target.value)
        if self._manual_last_direction != required_direction:
            return OperationResult.failure(
                "F446_DIRECTION_TARGET_MISMATCH",
                (
                    f"{target.value} requires last motion direction {required_direction}, "
                    f"observed {self._manual_last_direction}"
                ),
            )
        expected_state = self.config.f446.expected_state_for(target.value)
        opposite = self.config.f446.expected_state_for(
            Configuration.FLIGHT.value if target is Configuration.WALK else Configuration.WALK.value
        )
        if self._snapshot.f446.state is opposite:
            return OperationResult.failure(
                "F446_LIMIT_TARGET_MISMATCH",
                (f"F446 reports {opposite.value}; it cannot be confirmed as {target.value}"),
            )
        if self._snapshot.f446.state not in {F446State.IDLE, expected_state}:
            return OperationResult.failure(
                "F446_FINAL_STATE_INVALID",
                (
                    "Manual confirmation requires IDLE or the matching limit state; "
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

        if target is Configuration.FLIGHT:
            try:
                if not await self._go2.request_flight_pose():
                    raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected the request")
            except (BridgeError, OSError, RuntimeError) as exc:
                return OperationResult.failure("GO2_JOINT_LOCK_FAILED", str(exc))
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                return OperationResult.failure(
                    "GO2_JOINT_LOCK_UNCONFIRMED",
                    "Go2 did not authoritatively report mode=6 JOINT_LOCK",
                )

        self._operator_confirmed_configuration = target
        await self.refresh_snapshot()
        target_state = (
            SystemState.WALK if target is Configuration.WALK else SystemState.FLIGHT_READY
        )
        try:
            await self._state_machine.transition_to(
                target_state,
                reason=f"operator confirmed stopped manual position as {target.value}",
                snapshot=self._snapshot,
            )
        except (AeroGo2Error, OSError, RuntimeError, ValueError) as exc:
            self._operator_confirmed_configuration = None
            await self.refresh_snapshot()
            return OperationResult.failure("F446_MANUAL_CONFIRM_REJECTED", str(exc))

        await self.refresh_snapshot()
        self._emit(
            "F446_MANUAL_CONFIGURATION_CONFIRMED",
            configuration=target.value,
            source=self._snapshot.configuration_source,
        )
        return OperationResult.success(
            f"Operator-confirmed {target.value}; system entered {target_state.name}",
            code="F446_MANUAL_CONFIGURATION_CONFIRMED",
            data={
                "configuration": target.value,
                "configuration_source": self._snapshot.configuration_source,
                "state": target_state.name,
            },
        )

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
            if not await self._go2.request_flight_pose():
                raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected the request")
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                raise BridgeError(
                    "GO2_JOINT_LOCK_UNCONFIRMED: Go2 did not report mode=6 JOINT_LOCK"
                )
            self._emit("FLIGHT_LIMIT_REACHED")
            await self._state_machine.transition_to(
                SystemState.FLIGHT_READY,
                reason="F446 FLIGHT limit and zero duty verified",
                snapshot=self._snapshot,
            )
            self._emit("FLIGHT_CONFIGURATION_VERIFIED")
            self._emit("FLIGHT_READY")
            await self.refresh_snapshot()
            return OperationResult.success("FLIGHT verified; only RadioMaster may arm")
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
        self._emit("AUTOLAND_READY")
        await self.refresh_snapshot()
        return OperationResult.success("Automatic landing ready; setpoints remain stopped")

    async def start_autoland(self) -> OperationResult:
        if self._runtime_mode is not RuntimeMode.DRY_RUN:
            return OperationResult.failure(
                "PHASE_NOT_AVAILABLE",
                "Phase 1 automatic landing is available only in DRY-RUN",
            )
        if self.state is not SystemState.AUTO_LANDING_READY:
            return OperationResult.failure(
                "INVALID_STATE", "autoland start requires AUTO_LANDING_READY"
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
        first = await self.update_autoland()
        if first.ok:
            self._emit("AUTOLAND_STARTED")
        return first

    async def update_autoland(self) -> OperationResult:
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
            return await self.abort_autoland("manual override or flight failsafe")

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
            return await self.abort_autoland("automatic landing controller timeout")

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
        interlock = self._landing_interlocks.can_send_landing_setpoint(self._snapshot)
        if not interlock.permitted:
            reason = "; ".join(interlock.messages) or "landing setpoint interlock rejected"
            self._last_landing_command = self._landing_safety_filter.invalid(
                self._snapshot.timestamp,
                reason,
            )
            return await self.abort_autoland(reason)
        if not command.valid:
            if "timeout" in command.reason.lower():
                violation = self._safety_monitor.controller_timeout_violation(self._snapshot)
                self._record_runtime_violation(violation)
                self._emit(
                    "AUTOLAND_CONTROLLER_TIMEOUT",
                    safety_violations=(violation,),
                )
            return await self.abort_autoland(command.reason)
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
            return await self.abort_autoland(f"setpoint rejected: {exc}")
        self._setpoint_active = True
        self._emit("LANDING_COMMAND", landing_command=command)
        await self.refresh_snapshot()
        return OperationResult.success("Simulated landing setpoint recorded")

    async def abort_autoland(self, reason: str = "operator request") -> OperationResult:
        try:
            await self._stop_setpoints()
        except (BridgeError, OSError, RuntimeError) as exc:
            self._autoland_active = False
            self._next_landing_update_at = None
            await self._fault("AUTOLAND_SETPOINT_STOP_FAILED", str(exc))
            return OperationResult.failure("AUTOLAND_ABORT_FAILED", str(exc))
        self._autoland_active = False
        self._next_landing_update_at = None
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
        """Refresh telemetry, evaluate safety, and advance passive state changes."""

        previously_armed = self._snapshot.pixhawk.armed
        await self.refresh_snapshot()
        if previously_armed and not self._snapshot.pixhawk.armed:
            self._emit("PIXHAWK_DISARMED")

        if self._ground_arm_authorized and not self._pixhawk.ground_arm_authorization_active():
            self._ground_arm_authorized = False
            self._ground_arm_authorization_expires_at = None

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

        violations = tuple(self._safety_monitor.evaluate(self._snapshot))
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
                await self.abort_autoland("manual takeover or automatic-landing input failure")
                return violations

        if self.state in TRANSFORM_STATES and blocking:
            stop_result = await self._stop_transform_outputs()
            message = blocking[0].message
            if not stop_result.ok:
                message += f"; F446/setpoint stop incomplete: {stop_result.message}"
            await self._fault(blocking[0].code, message, stop_attempted=True)
            return violations

        non_escalating_codes = {
            "AUTOLAND_CONTROLLER_TIMEOUT",
            "AUTOLAND_ESTIMATOR_INVALID",
            "MANUAL_OVERRIDE_REQUESTED",
            "PIXHAWK_TIMEOUT",
            "RC_FAILSAFE",
            "RC_TIMEOUT",
        }
        escalating = [item for item in blocking if item.code not in non_escalating_codes]
        if escalating and self.state not in (
            SystemState.BOOT_SAFE,
            SystemState.FAULT,
            SystemState.EMERGENCY_STOP,
        ):
            stop_result = await self.stop_supervised()
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
            await self.update_autoland()
        if self.state in (SystemState.FLIGHT_MANUAL, SystemState.AUTO_LANDING):
            await self._check_touchdown()
        if self.state is SystemState.TOUCHDOWN_VERIFY:
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
        """Stop F446, Go2 and external setpoints, but never arm/disarm rotors."""

        initial_state = self.state
        transform_result = await self._stop_transform_outputs()
        failures: List[str] = []
        if not transform_result.ok:
            failures.append(transform_result.message)
        if self._control_writes_allowed() and self._snapshot.go2.connected:
            try:
                if initial_state is SystemState.LANDING_COMPLIANT:
                    relock = await self._finish_landing_compliance("supervised stop")
                    if not relock.ok:
                        raise BridgeError(f"{relock.code}: {relock.message}")
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
            return OperationResult.failure("SUPERVISED_STOP_PARTIAL", message)

        if self._ground_arm_authorized:
            await self._revoke_ground_arm_authorization("supervised stop")
        if initial_state in (SystemState.AUTO_LANDING, SystemState.AUTO_LANDING_READY):
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
            return snapshot_to_dict(self._snapshot)
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
            return {"go2": go2}
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
            return {"mapping": dict(self.config.esc.slots)}
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
            return {
                "system_state": self.state.name,
                "configuration": self._snapshot.configuration.value,
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

    def _esc_telemetry_confirms_touchdown(self) -> bool:
        return assess_esc_telemetry(
            self._snapshot,
            self.config.esc.slots,
            maximum_abs_rpm=self.config.safety.touchdown_max_esc_rpm,
        ).safe

    async def _check_touchdown(self) -> None:
        status = self._snapshot.pixhawk
        height = status.relative_altitude_m
        if (
            self.state is SystemState.AUTO_LANDING
            and self._snapshot.landing_estimate.height_m is not None
        ):
            height = self._snapshot.landing_estimate.height_m
        touchdown = (
            status.landed
            and math.isfinite(height)
            and math.isfinite(status.vertical_velocity_mps)
            and math.isfinite(status.roll_rad)
            and math.isfinite(status.pitch_rad)
            and abs(status.vertical_velocity_mps)
            <= self.config.safety.touchdown_max_vertical_speed_mps
            and abs(status.roll_rad) <= self.config.safety.touchdown_max_tilt_rad
            and abs(status.pitch_rad) <= self.config.safety.touchdown_max_tilt_rad
            and self._esc_telemetry_confirms_touchdown()
        )
        now = self._clock.monotonic()
        if not touchdown:
            self._touchdown_since = None
            self._touchdown_height_reference = None
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
            return
        if self._touchdown_since is None:
            self._touchdown_since = now
            return
        if now - self._touchdown_since < self.config.safety.touchdown_confirm_s:
            return
        self._touchdown_confirmed = True
        await self._stop_setpoints()
        self._autoland_active = False
        self._next_landing_update_at = None
        await self.refresh_snapshot()
        await self._state_machine.transition_to(
            SystemState.TOUCHDOWN_VERIFY,
            reason="touchdown conditions held for configured duration",
            snapshot=self._snapshot,
        )
        self._emit("TOUCHDOWN_CONFIRMED")
        await self.refresh_snapshot()

    def _landing_compliance_entry_result(self) -> OperationResult:
        if not self.config.go2.landing_compliance_enabled:
            return OperationResult.failure(
                "LANDING_COMPLIANCE_DISABLED",
                "Calibrate all four foot-force thresholds before enabling landing compliance",
            )
        if self.state is not SystemState.TOUCHDOWN_VERIFY:
            return OperationResult.failure(
                "INVALID_STATE",
                "Landing compliance entry requires TOUCHDOWN_VERIFY",
            )
        snapshot = self._snapshot
        contact = assess_foot_contact(snapshot.go2, self.config.go2)
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
        if not snapshot.go2.joints_locked:
            return OperationResult.failure(
                "GO2_JOINT_LOCK_REQUIRED",
                "Go2 must remain JOINT_LOCKED until compliance entry",
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
            target = SystemState.WALK
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
            if not await self._go2.request_flight_pose():
                raise BridgeError("GO2_JOINT_LOCK_FAILED: Go2 rejected the request")
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                raise BridgeError(
                    "GO2_JOINT_LOCK_UNCONFIRMED: Go2 did not report mode=6 JOINT_LOCK"
                )
            target = SystemState.FLIGHT_READY
        else:
            self._emit("F446_CONFIGURATION_UNKNOWN", state=self._snapshot.f446.state.value)
            return
        await self._state_machine.transition_to(
            target,
            reason=f"devices verified in safe {configuration.value} configuration",
            snapshot=self._snapshot,
        )

    async def _prepare_disconnect(self) -> OperationResult:
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
        await self.refresh_snapshot()
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
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False

    async def _enter_manual_positioning(self, snapshot: SystemSnapshot) -> None:
        self._maintenance_mode = True
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False

    async def _leave_manual_positioning(self, snapshot: SystemSnapshot) -> None:
        self._maintenance_mode = False
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False

    async def _enter_landing_compliant(self, snapshot: SystemSnapshot) -> None:
        if not self._control_writes_allowed():
            raise BridgeError("Go2 landing compliance requires unlocked hardware writes")
        failure: Optional[Exception] = None
        try:
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
            if not await self._go2.request_flight_pose():
                raise BridgeError("Go2 rejected the JOINT_LOCK restoration request")
            await self.refresh_snapshot()
            if not self._snapshot.go2.joints_locked:
                raise BridgeError("Go2 did not authoritatively report mode=6 JOINT_LOCK")
            await self._state_machine.transition_to(
                SystemState.FLIGHT_READY,
                reason=f"landing compliance finished: {reason}",
                snapshot=self._snapshot,
            )
            self._landing_compliant_since = None
            self._emit("LANDING_COMPLIANCE_FINISHED", reason=reason)
            await self.refresh_snapshot()
        except (
            AeroGo2Error,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return OperationResult.failure("GO2_LANDING_RELOCK_FAILED", str(exc))
        if self._snapshot.state is not SystemState.FLIGHT_READY:
            return OperationResult.failure(
                "GO2_LANDING_RELOCK_FAILED",
                f"Expected FLIGHT_READY after relock, received {self._snapshot.state.name}",
            )
        return OperationResult.success("Go2 JOINT_LOCK restored after landing compliance")

    async def _enter_fault_state(self, snapshot: SystemSnapshot) -> None:
        self._autoland_active = False
        self._maintenance_mode = False
        self._operator_confirmed_configuration = None
        self._manual_motion_deadline = None
        self._manual_last_direction = None
        self._manual_motion_started = False
        if self._suppress_fault_entry_stop:
            return
        await self._stop_setpoints()
        stop_result = await self._safe_f446_stop()
        if not stop_result.ok:
            raise BridgeError(f"{stop_result.code}: {stop_result.message}")

    async def _enter_emergency_stop(self, snapshot: SystemSnapshot) -> None:
        result = await self.stop_supervised()
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
