"""Snapshot-only safety evaluation.

``SafetyMonitor`` never calls a bridge and never mutates manager state.  Given
the same immutable configuration and snapshot it returns the same ordered list
of violations.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State, SafetySeverity, SystemState
from aerogo2.common.models import SafetyViolation, SystemSnapshot
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.go2_contact import assess_foot_contact
from aerogo2.safety.rc_interlock import assess_flight_enable
from aerogo2.safety.watchdog import timestamp_is_fresh

_ACTIVE_TRANSFORM_STATES = frozenset(
    (
        SystemState.HOMING_TO_WALK,
        SystemState.MANUAL_POSITIONING,
        SystemState.TRANSFORM_TO_FLIGHT,
        SystemState.TRANSFORM_TO_WALK,
    )
)
_TRANSFORM_INTERLOCK_STATES = frozenset(
    (
        SystemState.WALK_TO_FLIGHT_PRECHECK,
        SystemState.MANUAL_POSITIONING,
        SystemState.HOMING_TO_WALK,
        SystemState.TRANSFORM_TO_FLIGHT,
        SystemState.GO2_JOINT_LOCK_WAIT,
        SystemState.FLIGHT_TO_WALK_PRECHECK,
        SystemState.TRANSFORM_TO_WALK,
    )
)
_SAFE_UNKNOWN_CONFIGURATION_STATES = frozenset(
    (SystemState.BOOT_SAFE, SystemState.FAULT, SystemState.EMERGENCY_STOP)
)
_AUTOLAND_STATES = frozenset((SystemState.AUTO_LANDING_READY, SystemState.AUTO_LANDING))
_FLIGHT_JOINT_LOCK_STATES = frozenset(
    (
        SystemState.FLIGHT_READY,
        SystemState.FLIGHT_MANUAL,
        SystemState.AUTO_LANDING_READY,
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        SystemState.FLIGHT_TO_WALK_PRECHECK,
        SystemState.TRANSFORM_TO_WALK,
    )
)


class SafetyMonitor:
    """Evaluate current safety facts without producing side effects."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def evaluate(self, snapshot: SystemSnapshot) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        emitted: Dict[str, None] = {}

        def add(
            code: str,
            severity: SafetySeverity,
            message: str,
            recommended_action: str,
        ) -> None:
            if code in emitted:
                return
            emitted[code] = None
            violations.append(
                SafetyViolation(
                    code=code,
                    severity=severity,
                    message=message,
                    recommended_action=recommended_action,
                    timestamp=snapshot.timestamp,
                )
            )

        self._evaluate_freshness(snapshot, add)

        if snapshot.rc.failsafe:
            add(
                "RC_FAILSAFE",
                SafetySeverity.FAULT,
                "The RC receiver reports failsafe.",
                "Hold morphology requests, stop external setpoints, and use the "
                "independent RadioMaster/Pixhawk failsafe path.",
            )

        if (
            snapshot.state is SystemState.FLIGHT_READY
            and snapshot.pixhawk.armed
            and not snapshot.ground_arm_authorized
        ):
            add(
                "UNAUTHORIZED_PIXHAWK_ARM",
                SafetySeverity.EMERGENCY,
                "Pixhawk armed without a live AeroGo2 shell authorization.",
                "Stop supervised outputs; retain RadioMaster control and do not auto-disarm.",
            )
        if snapshot.state in _FLIGHT_JOINT_LOCK_STATES and not snapshot.go2.joints_locked:
            add(
                "GO2_JOINT_LOCK_LOST",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "Go2 no longer reports JOINT_LOCK in a flight-configuration state.",
                "Stop AeroGo2-owned outputs, retain RadioMaster/Pixhawk control, and do not auto-disarm.",
            )

        if snapshot.state is SystemState.LANDING_COMPLIANT:
            contact = assess_foot_contact(snapshot.go2, self._config.go2)
            if snapshot.pixhawk.armed:
                add(
                    "PIXHAWK_ARMED_DURING_LANDING_COMPLIANCE",
                    SafetySeverity.EMERGENCY,
                    "Pixhawk armed while Go2 joints were in the compliant landing posture.",
                    "Re-lock Go2 immediately; retain RadioMaster control and never auto-disarm.",
                )
            if self._esc_is_unsafe(snapshot):
                add(
                    "ESC_RPM_NONZERO_DURING_LANDING_COMPLIANCE",
                    SafetySeverity.EMERGENCY,
                    "ESC telemetry is not complete, healthy, finite, and exactly zero.",
                    "Re-lock Go2 immediately and keep rotor control with RadioMaster/Pixhawk.",
                )
            if not contact.valid:
                add(
                    "GO2_FOOT_FORCE_INVALID",
                    SafetySeverity.FAULT,
                    "One or more calibrated Go2 foot-force channels are invalid.",
                    "Re-lock Go2 and inspect live foot-force telemetry before retrying.",
                )
            elif not contact.safe:
                add(
                    "GO2_FOOT_CONTACT_LOST",
                    SafetySeverity.FAULT,
                    f"Only {contact.contact_count} feet report contact; "
                    f"{contact.required_count} are required.",
                    "Re-lock Go2 and stabilize the vehicle before retrying.",
                )
            if (
                snapshot.go2.locomotion_mode != "BALANCE_STAND"
                or snapshot.go2.joints_locked
                or not snapshot.go2.stable
                or snapshot.go2.moving
            ):
                add(
                    "GO2_LANDING_COMPLIANCE_LOST",
                    SafetySeverity.FAULT,
                    "Go2 no longer reports a stable BALANCE_STAND landing posture.",
                    "Re-lock Go2 before any morphology movement.",
                )

        if snapshot.state in _TRANSFORM_INTERLOCK_STATES:
            if snapshot.pixhawk.failsafe or snapshot.pixhawk.rc_failsafe:
                add(
                    "PIXHAWK_FAILSAFE",
                    SafetySeverity.FAULT,
                    "Pixhawk entered failsafe while the morphology mechanism was active.",
                    "Stop the F446 mechanism and keep Pixhawk control with RadioMaster.",
                )
            flight_enable = assess_flight_enable(snapshot.rc, self._config.rc)
            if not flight_enable.valid:
                add(
                    "FLIGHT_ENABLE_INVALID",
                    SafetySeverity.FAULT,
                    "RC CH5 is missing, ambiguous, malformed, or inconsistent.",
                    "Stop the F446 mechanism and restore valid independent RC telemetry.",
                )
            elif flight_enable.high:
                add(
                    "FLIGHT_ENABLE_HIGH",
                    SafetySeverity.FAULT,
                    "RC flight enable became active while the morphology mechanism was active.",
                    "Stop the F446 mechanism and inhibit any flight-unlock flow.",
                )
            if snapshot.pixhawk.armed:
                add(
                    "PIXHAWK_ARMED_DURING_TRANSFORM",
                    SafetySeverity.EMERGENCY,
                    "Pixhawk became armed while the morphology mechanism was active.",
                    "Stop the F446 mechanism and require RadioMaster control; never "
                    "issue an automatic disarm.",
                )
            if snapshot.state in _ACTIVE_TRANSFORM_STATES and self._esc_is_unsafe(snapshot):
                add(
                    "ESC_RPM_NONZERO_DURING_TRANSFORM",
                    SafetySeverity.EMERGENCY,
                    "ESC telemetry is incomplete, inconsistent, unhealthy, non-finite, or nonzero during active transformation.",
                    "Stop the F446 mechanism; do not command rotor shutdown.",
                )
            if (
                snapshot.state is SystemState.GO2_JOINT_LOCK_WAIT
                and self._go2_joint_lock_transition_is_unsafe(snapshot)
            ):
                add(
                    "GO2_UNSAFE_DURING_JOINT_LOCK",
                    SafetySeverity.FAULT,
                    "Go2 entered a locomotion/unsafe-speed state while waiting for mode=6 JOINT_LOCK.",
                    "Stop the F446 mechanism and keep the Go2 stationary; select Joint Lock in the Unitree app.",
                )
            elif snapshot.state is not SystemState.GO2_JOINT_LOCK_WAIT and self._go2_is_moving(snapshot):
                add(
                    "GO2_MOVING_DURING_TRANSFORM",
                    SafetySeverity.FAULT,
                    "Go2 is moving or unstable while the morphology mechanism is active.",
                    "Stop the F446 mechanism and request a Go2 stop.",
                )

        if snapshot.f446.faulted:
            add(
                "F446_FAULT",
                SafetySeverity.FAULT,
                "F446 reports a local fault.",
                "Stop the mechanism, inhibit new transforms, and require explicit "
                "operator fault recovery.",
            )

        if self._f446_overcurrent(snapshot):
            add(
                "F446_OVERCURRENT",
                SafetySeverity.FAULT,
                "F446 current is at or above the configured transform safety limit.",
                "Stop the mechanism and inspect the linkage before clearing the fault.",
            )

        if snapshot.f446.connected and snapshot.f446.state is F446State.UNKNOWN:
            add(
                "F446_STATE_UNKNOWN",
                SafetySeverity.FAULT,
                "F446 is connected but its state cannot be identified.",
                "Inhibit walking, flight readiness, and all new transforms.",
            )

        configuration_conflict = self._configuration_conflict(snapshot)
        if configuration_conflict is not None:
            add(
                "F446_CONFIGURATION_CONFLICT",
                SafetySeverity.FAULT,
                configuration_conflict,
                "Treat the physical configuration as UNKNOWN and inhibit motion.",
            )

        if snapshot.state in _AUTOLAND_STATES and self._landing_estimate_invalid(snapshot):
            add(
                "AUTOLAND_ESTIMATOR_INVALID",
                SafetySeverity.FAULT,
                "The landing estimate or ground observation is invalid or stale.",
                "Stop external setpoints and return control to RadioMaster/Pixhawk.",
            )

        if snapshot.state in _AUTOLAND_STATES and snapshot.rc.manual_override:
            add(
                "MANUAL_OVERRIDE_REQUESTED",
                SafetySeverity.WARNING,
                "RadioMaster manual override was requested.",
                "Stop external setpoints immediately and return to FLIGHT_MANUAL.",
            )

        externally_latched = set(snapshot.active_fault_codes)
        if "AUTOLAND_CONTROLLER_TIMEOUT" in externally_latched:
            add(
                "AUTOLAND_CONTROLLER_TIMEOUT",
                SafetySeverity.FAULT,
                "The manager detected an automatic-landing controller timeout.",
                "Stop external setpoints and return control to RadioMaster/Pixhawk.",
            )
        if "INVALID_STATE_TRANSITION" in externally_latched:
            add(
                "INVALID_STATE_TRANSITION",
                SafetySeverity.WARNING,
                "A requested state transition was rejected.",
                "Keep the current safe state and review the failed guard results.",
            )

        if snapshot.external_setpoint_active and snapshot.state is not SystemState.AUTO_LANDING:
            add(
                "EXTERNAL_SETPOINT_OUTSIDE_AUTOLAND",
                SafetySeverity.EMERGENCY,
                "An external setpoint is active outside AUTO_LANDING.",
                "Stop external setpoints immediately; do not disarm or stop rotors.",
            )

        return violations

    def invalid_transition_violation(
        self,
        snapshot: SystemSnapshot,
        requested_state: SystemState,
        reason: str,
    ) -> SafetyViolation:
        """Build the violation associated with a rejected transition attempt."""

        detail = reason.strip() or "guard rejected the request"
        return SafetyViolation(
            code="INVALID_STATE_TRANSITION",
            severity=SafetySeverity.WARNING,
            message=f"Transition {snapshot.state.name} -> {requested_state.name} rejected: {detail}.",
            recommended_action="Keep the current safe state and inspect state guards.",
            timestamp=snapshot.timestamp,
        )

    def controller_timeout_violation(self, snapshot: SystemSnapshot) -> SafetyViolation:
        """Build the violation produced by the manager's setpoint watchdog."""

        return SafetyViolation(
            code="AUTOLAND_CONTROLLER_TIMEOUT",
            severity=SafetySeverity.FAULT,
            message="Automatic-landing command production exceeded its timeout.",
            recommended_action=(
                "Stop external setpoints and return control to RadioMaster/Pixhawk."
            ),
            timestamp=snapshot.timestamp,
        )

    def _evaluate_freshness(
        self,
        snapshot: SystemSnapshot,
        add_violation: Callable[[str, SafetySeverity, str, str], None],
    ) -> None:
        freshness = (
            (
                "PIXHAWK_TIMEOUT",
                snapshot.pixhawk.connected,
                snapshot.pixhawk.heartbeat_timestamp,
                self._config.safety.pixhawk_timeout_s,
                "Pixhawk heartbeat is disconnected, invalid, or stale.",
                "Inhibit new transforms and external setpoints.",
            ),
            (
                "F446_TIMEOUT",
                snapshot.f446.connected,
                snapshot.f446.timestamp,
                self._config.safety.f446_timeout_s,
                "F446 status is disconnected, invalid, or stale.",
                "Inhibit new transforms; stop an active transform if possible.",
            ),
            (
                "RC_TIMEOUT",
                snapshot.rc.connected,
                snapshot.rc.timestamp,
                self._config.safety.rc_timeout_s,
                "RC channel data is disconnected, invalid, or stale.",
                "Reset high-level requests to HOLD/MANUAL and stop external setpoints.",
            ),
        )
        for code, connected, timestamp, timeout_s, message, action in freshness:
            if not connected or not timestamp_is_fresh(snapshot.timestamp, timestamp, timeout_s):
                add_violation(code, SafetySeverity.FAULT, message, action)

        if self._config.go2.enabled and (
            not snapshot.go2.connected
            or not timestamp_is_fresh(
                snapshot.timestamp,
                snapshot.go2.timestamp,
                self._config.safety.go2_timeout_s,
            )
        ):
            add_violation(
                "GO2_TIMEOUT",
                SafetySeverity.FAULT,
                "Go2 status is disconnected, invalid, or stale.",
                "Inhibit walking permission and all new transforms.",
            )

    def _esc_is_unsafe(self, snapshot: SystemSnapshot) -> bool:
        return not assess_esc_telemetry(
            snapshot,
            self._config.esc.slots,
            exact_zero=True,
        ).safe

    def _go2_is_moving(self, snapshot: SystemSnapshot) -> bool:
        velocity = snapshot.go2.velocity_mps
        return (
            not math.isfinite(velocity)
            or abs(velocity) >= self._config.safety.stationary_velocity_mps
            or not snapshot.go2.stable
            or snapshot.go2.controller_active
        )

    def _go2_joint_lock_transition_is_unsafe(self, snapshot: SystemSnapshot) -> bool:
        allowed_modes = {
            "IDLE_STAND",
            "BALANCE_STAND",
            "POSE",
            "JOINT_LOCK",
            # Simulation aliases; the hardware bridge emits the names above.
            "STAND",
            "STOPPED",
        }
        components = snapshot.go2.body_velocity
        limit = self._config.safety.stationary_velocity_mps
        return (
            snapshot.go2.locomotion_mode not in allowed_modes
            or not math.isfinite(snapshot.go2.velocity_mps)
            or len(components) != 3
            or any(not math.isfinite(value) or abs(value) >= limit for value in components)
            or abs(snapshot.go2.velocity_mps) >= limit
        )

    def _f446_overcurrent(self, snapshot: SystemSnapshot) -> bool:
        current = snapshot.f446.used_current_adc
        if current is None:
            return snapshot.state in _ACTIVE_TRANSFORM_STATES
        try:
            numeric_current = float(current)
        except (TypeError, ValueError):
            return True
        if (
            snapshot.configuration_source == "f446_limit"
            and math.isfinite(numeric_current)
            and snapshot.f446.duty == 0
            and snapshot.f446.state in {F446State.LIMIT_REACHED_FWD, F446State.LIMIT_REACHED_REV}
        ):
            return False
        return (
            not math.isfinite(numeric_current)
            or numeric_current < 0.0
            or numeric_current >= self._config.safety.maximum_transform_current_adc
        )

    def _configuration_conflict(self, snapshot: SystemSnapshot) -> Optional[str]:
        if not snapshot.f446.connected:
            return None
        if snapshot.state in _ACTIVE_TRANSFORM_STATES:
            return self._active_transform_configuration_conflict(snapshot)
        if snapshot.configuration is Configuration.UNKNOWN:
            if snapshot.state in _SAFE_UNKNOWN_CONFIGURATION_STATES:
                return None
            return "Logical configuration is UNKNOWN in an operational state."
        if snapshot.configuration is Configuration.WALK:
            expected = self._config.f446.expected_walk_state
        else:
            expected = self._config.f446.expected_flight_state
        operator_confirmed_idle = (
            snapshot.configuration_source == "operator"
            and snapshot.f446.state is F446State.IDLE
            and snapshot.f446.duty == 0
        )
        if snapshot.f446.state is not expected and not operator_confirmed_idle:
            return (
                f"Logical configuration {snapshot.configuration.value} conflicts with "
                f"F446 state {snapshot.f446.state.value}."
            )

        state_expected: Optional[Configuration] = None
        if snapshot.state in (SystemState.WALK, SystemState.WALK_TO_FLIGHT_PRECHECK):
            state_expected = Configuration.WALK
        elif snapshot.state in (
            SystemState.GO2_JOINT_LOCK_WAIT,
            SystemState.FLIGHT_READY,
            SystemState.FLIGHT_MANUAL,
            SystemState.AUTO_LANDING_READY,
            SystemState.AUTO_LANDING,
            SystemState.TOUCHDOWN_VERIFY,
            SystemState.LANDING_COMPLIANT,
            SystemState.FLIGHT_TO_WALK_PRECHECK,
        ):
            state_expected = Configuration.FLIGHT
        if state_expected is not None and snapshot.configuration is not state_expected:
            return f"System state {snapshot.state.name} requires {state_expected.value} configuration, not {snapshot.configuration.value}."
        return None

    def _active_transform_configuration_conflict(
        self,
        snapshot: SystemSnapshot,
    ) -> Optional[str]:
        if snapshot.f446.duty == 0:
            return None
        if snapshot.state is SystemState.MANUAL_POSITIONING:
            allowed_states = {
                F446State.MANUAL_FWD,
                F446State.MANUAL_REV,
                F446State.LIMIT_FWD,
                F446State.LIMIT_REV,
            }
            if snapshot.f446.state not in allowed_states:
                return (
                    "Manual positioning with nonzero duty requires a manual or "
                    f"limit-move F446 state, not {snapshot.f446.state.value}."
                )
            return None
        target = (
            Configuration.FLIGHT
            if snapshot.state is SystemState.TRANSFORM_TO_FLIGHT
            else Configuration.WALK
        )
        direction = self._config.f446.direction_for(target.value)
        expected_moving_state = (
            F446State.LIMIT_FWD if direction == "forward" else F446State.LIMIT_REV
        )
        if snapshot.f446.state is not expected_moving_state:
            return (
                f"Active transform toward {target.value} requires F446 state "
                f"{expected_moving_state.value}, not {snapshot.f446.state.value}."
            )
        return None

    def _landing_estimate_invalid(self, snapshot: SystemSnapshot) -> bool:
        estimate = snapshot.landing_estimate
        if not estimate.valid or not estimate.ground_detected:
            return True
        if not timestamp_is_fresh(
            snapshot.timestamp,
            estimate.timestamp,
            self._config.safety.controller_timeout_s,
        ):
            return True
        numeric_values = (
            estimate.height_m,
            estimate.vertical_velocity_mps,
            estimate.horizontal_velocity_mps,
        )
        return any(value is not None and not math.isfinite(value) for value in numeric_values)
