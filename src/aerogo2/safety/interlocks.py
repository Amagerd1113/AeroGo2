"""Fail-closed transition and motion interlocks."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446State,
    SystemState,
)
from aerogo2.common.models import SystemSnapshot
from aerogo2.common.results import GuardResult
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.rc_interlock import assess_flight_enable
from aerogo2.safety.watchdog import timestamp_is_fresh


class SafetyInterlocks:
    """Evaluate safety prerequisites without performing any device operation."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def can_start_transform(self, snapshot: SystemSnapshot, target: Configuration) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []

        def reject(code: str, message: str) -> None:
            codes.append(code)
            messages.append(message)

        if target not in (Configuration.WALK, Configuration.FLIGHT):
            reject("TARGET_CONFIGURATION_UNKNOWN", "Transform target must be WALK or FLIGHT.")
            return self._result(codes, messages)

        legal_states: Tuple[SystemState, ...]
        expected_current: Configuration
        if target is Configuration.FLIGHT:
            legal_states = (SystemState.WALK, SystemState.WALK_TO_FLIGHT_PRECHECK)
            expected_current = Configuration.WALK
        else:
            legal_states = (
                SystemState.TOUCHDOWN_VERIFY,
                SystemState.LANDING_COMPLIANT,
                SystemState.FLIGHT_TO_WALK_PRECHECK,
            )
            expected_current = Configuration.FLIGHT
        if snapshot.state not in legal_states:
            reject(
                "INVALID_TRANSFORM_SOURCE_STATE",
                f"{target.value} cannot start from {snapshot.state.name}.",
            )
        if snapshot.configuration is not expected_current:
            reject(
                "CONFIGURATION_NOT_VERIFIED",
                f"Current configuration must be verified as {expected_current.value}.",
            )

        self._require_fresh_devices(snapshot, reject)
        if snapshot.pixhawk.armed:
            reject("PIXHAWK_ARMED", "Pixhawk must be disarmed before any transform.")
        if snapshot.pixhawk.failsafe or snapshot.pixhawk.rc_failsafe:
            reject("PIXHAWK_FAILSAFE", "Pixhawk failsafe is active.")
        exact_rotor_stop = target is Configuration.WALK or snapshot.state is not SystemState.WALK
        if not self._esc_telemetry_safe(snapshot, exact_zero=exact_rotor_stop):
            reject(
                "ESC_RPM_UNSAFE",
                "All configured ESCs must be uniquely present, online, finite, and at a safe RPM.",
            )
        flight_enable = assess_flight_enable(snapshot.rc, self._config.rc)
        if not flight_enable.valid:
            reject(
                "FLIGHT_ENABLE_INVALID",
                "RC CH5 is missing, ambiguous, malformed, or inconsistent with parsed state.",
            )
        elif not flight_enable.low:
            reject("FLIGHT_ENABLE_HIGH", "CH5 flight enable must be LOW.")
        if snapshot.rc.failsafe:
            reject("RC_FAILSAFE", "RC failsafe resets transform requests to HOLD.")
        if not self._go2_stationary(snapshot):
            reject("GO2_NOT_STATIONARY", "Go2 must be stable and below the velocity limit.")
        if snapshot.f446.faulted:
            reject("F446_FAULT", "F446 reports a fault.")
        if snapshot.f446.duty != 0:
            reject("F446_MOVING", "F446 duty must be zero before a new transform.")
        if self._f446_current_unsafe(snapshot):
            reject("F446_CURRENT_UNSAFE", "F446 current is above the configured safe limit.")

        if expected_current is Configuration.WALK:
            expected_state = self._config.f446.expected_walk_state
        else:
            expected_state = self._config.f446.expected_flight_state
        if not self._f446_configuration_state_is_verified(snapshot, expected_state):
            reject(
                "F446_CONFIGURATION_UNVERIFIED",
                (
                    f"F446 state {snapshot.f446.state.value} does not prove "
                    f"{expected_current.value} configuration by a limit or guarded operator mark."
                ),
            )

        if target is Configuration.WALK:
            if not snapshot.pixhawk.landed:
                reject("TOUCHDOWN_NOT_CONFIRMED", "Pixhawk must report landed.")
            if snapshot.autoland_active or snapshot.external_setpoint_active:
                reject(
                    "AUTOLAND_NOT_STOPPED",
                    "Automatic landing and external setpoints must be stopped.",
                )
        if snapshot.maintenance_mode:
            reject("MAINTENANCE_MODE_ACTIVE", "Exit F446 maintenance mode first.")
        if snapshot.active_fault_codes:
            reject("ACTIVE_FAULTS", "Active faults must be resolved before transforming.")
        return self._result(codes, messages)

    def can_enter_walk(self, snapshot: SystemSnapshot) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []

        def reject(code: str, message: str) -> None:
            codes.append(code)
            messages.append(message)

        self._require_fresh_devices(snapshot, reject)
        if snapshot.configuration is not Configuration.WALK:
            reject("WALK_CONFIGURATION_UNVERIFIED", "WALK configuration is not verified.")
        if not self._f446_configuration_state_is_verified(
            snapshot,
            self._config.f446.expected_walk_state,
        ):
            reject("F446_WALK_STATE_MISMATCH", "F446 is not in the expected WALK state.")
        if snapshot.f446.duty != 0 or snapshot.f446.faulted:
            reject("F446_NOT_SAFE", "F446 must be stopped and fault-free.")
        if snapshot.pixhawk.armed:
            reject("PIXHAWK_ARMED", "Pixhawk must be disarmed before enabling walking.")
        if not self._esc_telemetry_safe(snapshot, exact_zero=True):
            reject("ESC_RPM_UNSAFE", "Every configured ESC must report exactly zero RPM.")
        if snapshot.active_fault_codes:
            reject("ACTIVE_FAULTS", "Active faults prevent walking permission.")
        return self._result(codes, messages)

    def can_enter_flight_ready(
        self,
        snapshot: SystemSnapshot,
        *,
        require_flight_enable_low: bool = True,
    ) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []

        def reject(code: str, message: str) -> None:
            codes.append(code)
            messages.append(message)

        self._require_fresh_devices(snapshot, reject, require_go2=True)
        if snapshot.configuration is not Configuration.FLIGHT:
            reject(
                "FLIGHT_CONFIGURATION_UNVERIFIED",
                "FLIGHT configuration is not verified.",
            )
        if not self._f446_configuration_state_is_verified(
            snapshot,
            self._config.f446.expected_flight_state,
        ):
            reject("F446_FLIGHT_STATE_MISMATCH", "F446 is not in the expected FLIGHT state.")
        if snapshot.f446.duty != 0 or snapshot.f446.faulted:
            reject("F446_NOT_SAFE", "F446 must be stopped and fault-free.")
        if snapshot.pixhawk.armed:
            reject(
                "PIXHAWK_ALREADY_ARMED",
                "FLIGHT_READY must be established before operator arming.",
            )
        if snapshot.pixhawk.failsafe:
            reject("PIXHAWK_FAILSAFE", "Pixhawk failsafe prevents flight enable.")
        if not self._esc_telemetry_safe(snapshot, exact_zero=True):
            reject(
                "ESC_TELEMETRY_UNSAFE",
                "Every configured ESC must be online, consistent, healthy, and at zero RPM.",
            )
        if snapshot.rc.failsafe:
            reject("RC_FAILSAFE", "RC failsafe prevents flight enable.")
        flight_enable = assess_flight_enable(snapshot.rc, self._config.rc)
        if not flight_enable.valid:
            reject(
                "FLIGHT_ENABLE_INVALID",
                "RC CH5 is missing, ambiguous, malformed, or inconsistent with parsed state.",
            )
        elif require_flight_enable_low and not flight_enable.low:
            reject(
                "FLIGHT_ENABLE_NOT_LOW",
                "CH5 must remain LOW until the flight-enable precheck passes.",
            )
        if not self._go2_stationary(snapshot):
            reject("GO2_NOT_STATIONARY", "Go2 must remain stationary.")
        if not snapshot.joint_lock_confirmed:
            reject(
                "GO2_JOINT_LOCK_REQUIRED",
                "Go2 joint lock must be confirmed by telemetry or guarded operator assertion.",
            )
        if snapshot.active_fault_codes:
            reject("ACTIVE_FAULTS", "Active faults prevent flight readiness.")
        return self._result(codes, messages)

    def can_start_autoland(self, snapshot: SystemSnapshot) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []

        def reject(code: str, message: str) -> None:
            codes.append(code)
            messages.append(message)

        if snapshot.state not in (
            SystemState.FLIGHT_MANUAL,
            SystemState.AUTO_LANDING_READY,
            SystemState.AUTO_LANDING,
        ):
            reject(
                "INVALID_AUTOLAND_SOURCE_STATE",
                "Automatic landing can only start from flight control states.",
            )
        if snapshot.configuration is not Configuration.FLIGHT:
            reject("FLIGHT_CONFIGURATION_UNVERIFIED", "FLIGHT configuration is required.")
        if not snapshot.pixhawk.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.pixhawk.heartbeat_timestamp,
            self._config.safety.pixhawk_timeout_s,
        ):
            reject("PIXHAWK_TIMEOUT", "Pixhawk heartbeat is unavailable or stale.")
        if not snapshot.pixhawk.armed:
            reject("PIXHAWK_DISARMED", "Automatic landing requires an armed Pixhawk.")
        if snapshot.pixhawk.failsafe:
            reject("PIXHAWK_FAILSAFE", "Pixhawk failsafe is active.")
        if not snapshot.rc.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.rc.timestamp,
            self._config.safety.rc_timeout_s,
        ):
            reject("RC_TIMEOUT", "RC data is unavailable or stale.")
        if snapshot.rc.failsafe:
            reject("RC_FAILSAFE", "RC failsafe requires manual/Pixhawk fallback.")
        if snapshot.rc.manual_override:
            reject("MANUAL_OVERRIDE_REQUESTED", "RadioMaster manual override has priority.")
        if snapshot.rc.auto_landing_request is not AutoLandingRequest.AUTO_EXECUTE:
            reject("AUTO_EXECUTE_NOT_REQUESTED", "CH10 must be stable at AUTO_EXECUTE.")
        if (
            not snapshot.landing_estimate.valid
            or not snapshot.landing_estimate.ground_detected
            or not timestamp_is_fresh(
                snapshot.timestamp,
                snapshot.landing_estimate.timestamp,
                self._config.safety.controller_timeout_s,
            )
        ):
            reject("AUTOLAND_ESTIMATOR_INVALID", "Landing estimate is invalid or stale.")
        estimate_values = (
            snapshot.landing_estimate.height_m,
            snapshot.landing_estimate.vertical_velocity_mps,
            snapshot.landing_estimate.horizontal_velocity_mps,
        )
        if any(value is None or not math.isfinite(value) for value in estimate_values):
            reject("AUTOLAND_ESTIMATOR_INVALID", "Landing estimate values must be finite.")
        if snapshot.active_fault_codes:
            reject("ACTIVE_FAULTS", "Active faults prevent automatic landing.")
        return self._result(codes, messages)

    def can_send_landing_setpoint(self, snapshot: SystemSnapshot) -> GuardResult:
        if snapshot.state is not SystemState.AUTO_LANDING or not snapshot.autoland_active:
            return GuardResult.reject(
                "AUTOLAND_NOT_ACTIVE",
                "External setpoints are permitted only in active AUTO_LANDING.",
            )
        return self.can_start_autoland(snapshot)

    def can_move_go2(self, snapshot: SystemSnapshot) -> GuardResult:
        if snapshot.state is not SystemState.WALK:
            return GuardResult.reject("NOT_IN_WALK", "Go2 motion is permitted only in WALK.")
        return self.can_enter_walk(snapshot)

    def _require_fresh_devices(
        self,
        snapshot: SystemSnapshot,
        reject_result: Callable[[str, str], None],
        *,
        require_go2: bool = False,
    ) -> None:
        if not snapshot.pixhawk.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.pixhawk.heartbeat_timestamp,
            self._config.safety.pixhawk_timeout_s,
        ):
            reject_result("PIXHAWK_TIMEOUT", "Pixhawk status is unavailable or stale.")
        if not snapshot.f446.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.f446.timestamp,
            self._config.safety.f446_timeout_s,
        ):
            reject_result("F446_TIMEOUT", "F446 status is unavailable or stale.")
        if (require_go2 or self._config.go2.enabled) and (
            not snapshot.go2.connected
            or not timestamp_is_fresh(
                snapshot.timestamp,
                snapshot.go2.timestamp,
                self._config.safety.go2_timeout_s,
            )
        ):
            reject_result("GO2_TIMEOUT", "Go2 status is unavailable or stale.")
        if not snapshot.rc.connected or not timestamp_is_fresh(
            snapshot.timestamp,
            snapshot.rc.timestamp,
            self._config.safety.rc_timeout_s,
        ):
            reject_result("RC_TIMEOUT", "RC status is unavailable or stale.")

    def _esc_telemetry_safe(
        self,
        snapshot: SystemSnapshot,
        *,
        exact_zero: bool,
    ) -> bool:
        return assess_esc_telemetry(
            snapshot,
            self._config.esc.slots,
            exact_zero=exact_zero,
            maximum_abs_rpm=(
                None if exact_zero else self._config.safety.maximum_safe_esc_rpm_for_transform
            ),
        ).safe

    @staticmethod
    def _f446_configuration_state_is_verified(
        snapshot: SystemSnapshot,
        expected_state: F446State,
    ) -> bool:
        if snapshot.f446.state is expected_state:
            return True
        return (
            snapshot.configuration_source == "operator"
            and snapshot.f446.state is F446State.IDLE
            and snapshot.f446.duty == 0
            and not snapshot.f446.faulted
        )

    def _go2_stationary(self, snapshot: SystemSnapshot) -> bool:
        velocity_components = snapshot.go2.body_velocity
        return (
            math.isfinite(snapshot.go2.velocity_mps)
            and abs(snapshot.go2.velocity_mps) < self._config.safety.stationary_velocity_mps
            and len(velocity_components) == 3
            and all(
                math.isfinite(component)
                and abs(component) < self._config.safety.stationary_velocity_mps
                for component in velocity_components
            )
            and snapshot.go2.stable
            and not snapshot.go2.moving
            and not snapshot.go2.controller_active
        )

    def _f446_current_unsafe(self, snapshot: SystemSnapshot) -> bool:
        current = snapshot.f446.used_current_adc
        threshold = snapshot.f446.threshold_adc
        if (
            current is None
            or isinstance(current, bool)
            or threshold is None
            or isinstance(threshold, bool)
        ):
            return True
        try:
            current_value = float(current)
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            return True
        margin = float(self._config.f446.current_safe_margin_adc)
        return (
            not math.isfinite(current_value)
            or current_value < 0.0
            or not math.isfinite(threshold_value)
            or threshold_value <= margin
            or current_value > threshold_value - margin
        )

    @staticmethod
    def _result(codes: List[str], messages: List[str]) -> GuardResult:
        return GuardResult(
            permitted=not codes,
            codes=tuple(codes),
            messages=tuple(messages),
        )
