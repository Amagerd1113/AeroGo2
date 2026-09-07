"""Snapshot-only safety evaluation.

``SafetyMonitor`` never calls a bridge and never mutates manager state.  Given
the same immutable configuration and snapshot it returns the same ordered list
of violations.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    Configuration,
    F446State,
    Go2ControlAuthorityState,
    SafetySeverity,
    SystemState,
)
from aerogo2.common.models import LowCmdOwnershipState, SafetyViolation, SystemSnapshot
from aerogo2.common.numeric import finite_real
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.go2_contact import assess_foot_contact
from aerogo2.safety.pixhawk_freshness import (
    assess_pixhawk_source_freshness,
    pixhawk_ground_state_is_current,
    pixhawk_touchdown_payload_is_valid,
    timestamps_are_coherent,
)
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
_LOWCMD_STABLE_OWNER_STATES = frozenset(
    (
        LowCmdOwnershipState.HOLDING,
        LowCmdOwnershipState.MPC_ACTIVE,
        LowCmdOwnershipState.SAFE_HOLD,
    )
)
_LOWCMD_ALLOWED_SYSTEM_STATES = frozenset(
    (
        # FLIGHT_READY is the only normal state in which the ground-only
        # ownership handover may have completed before Pixhawk arming.
        SystemState.FLIGHT_READY,
        SystemState.FLIGHT_MANUAL,
        SystemState.AUTO_LANDING_READY,
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        # A fault must not make the sole writer disappear while airborne.
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    )
)
_LOWCMD_REQUIRED_SYSTEM_STATES = frozenset(
    (
        SystemState.FLIGHT_MANUAL,
        SystemState.AUTO_LANDING_READY,
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
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

        if snapshot.state in _AUTOLAND_STATES:
            source_freshness = assess_pixhawk_source_freshness(
                snapshot.pixhawk,
                snapshot.timestamp,
                self._config.safety.pixhawk_timeout_s,
                self._config.safety.touchdown_max_source_age_s,
            )
            if not source_freshness.touchdown:
                add(
                    "PIXHAWK_TOUCHDOWN_SOURCE_STALE",
                    SafetySeverity.FAULT,
                    "One or more independent Pixhawk touchdown sources are disconnected, invalid, future-dated, or stale.",
                    "Stop external descent setpoints and return control to RadioMaster/Pixhawk.",
                )
            elif not pixhawk_touchdown_payload_is_valid(snapshot.pixhawk):
                add(
                    "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID",
                    SafetySeverity.FAULT,
                    "Pixhawk touchdown telemetry contains a malformed boolean or non-finite numeric value.",
                    "Stop external descent setpoints and inspect the MAVLink telemetry source.",
                )
            elif not timestamps_are_coherent(
                (
                    snapshot.pixhawk.attitude_timestamp,
                    snapshot.pixhawk.kinematics_timestamp,
                    snapshot.pixhawk.landed_state_timestamp,
                    snapshot.landing_estimate.timestamp,
                ),
                self._config.safety.touchdown_max_source_skew_s,
            ):
                add(
                    "PIXHAWK_TOUCHDOWN_SOURCE_INCOHERENT",
                    SafetySeverity.FAULT,
                    "Pixhawk touchdown inputs and the landing estimate do not belong to one bounded observation window.",
                    "Stop external descent setpoints and inspect MAVLink stream rates and estimator timing.",
                )

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
        low_level = snapshot.go2.low_level_status
        lowcmd_owned = low_level.ownership_pending
        lowcmd_stable = low_level.ownership_state in _LOWCMD_STABLE_OWNER_STATES
        authority = snapshot.go2.control_authority
        authority_state = authority.state
        if lowcmd_owned and snapshot.state not in _LOWCMD_ALLOWED_SYSTEM_STATES:
            add(
                "GO2_LOWCMD_OUTSIDE_AUTHORIZED_STATE",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "The dedicated LowCmd writer owns Go2 outside an authorized flight/fault state.",
                "Keep the writer in safe-hold; release it only after verified ground, disarm, and zero-rotor checks.",
            )
        transition_authorized = self._authority_transition_is_authorized(snapshot)
        if authority.transition_pending:
            deadline = authority.transition_deadline
            if (
                not transition_authorized
                or deadline is None
                or not math.isfinite(deadline)
                or snapshot.timestamp >= deadline
            ):
                add(
                    "GO2_CONTROL_AUTHORITY_TRANSITION_TIMEOUT",
                    (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                    "The Go2 control-authority transition is unauthorized, invalid, or timed out.",
                    "Keep the current sole writer/ground handover fail-closed and inspect the ownership epoch before retrying.",
                )
        if self._authority_is_inconsistent(snapshot):
            add(
                "GO2_CONTROL_AUTHORITY_INCONSISTENT",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "The explicit Go2 control-authority state contradicts LowCmd ownership facts.",
                "Do not start SportClient or another LowCmd writer; retain the existing fail-closed owner and inspect the epoch.",
            )
        if (
            lowcmd_owned
            and not authority.transition_pending
            and (not lowcmd_stable or not self._lowcmd_owner_healthy(snapshot))
        ):
            add(
                "GO2_LOWCMD_OWNER_UNHEALTHY",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "The sole Go2 LowCmd owner is active but its stream, watchdog, mapping, or LowState health is invalid.",
                "Revoke the MPC lease so the same writer remains in safe-hold; do not stop LowCmd while airborne.",
            )
        if self._config.go2.low_level.enabled and snapshot.state in _LOWCMD_REQUIRED_SYSTEM_STATES:
            recovery_finalizing = self._impact_recovery_finalization_authorized(snapshot)
            if snapshot.state is SystemState.AUTO_LANDING and recovery_finalizing:
                authority_missing = authority_state not in {
                    Go2ControlAuthorityState.LOWCMD_ACTIVE,
                    Go2ControlAuthorityState.LOWCMD_SAFE_HOLD,
                }
            else:
                expected_authority = (
                    Go2ControlAuthorityState.LOWCMD_ACTIVE
                    if snapshot.state is SystemState.AUTO_LANDING
                    else Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
                )
                authority_missing = authority_state is not expected_authority
        else:
            authority_missing = False
        if authority_missing:
            add(
                "GO2_LOWCMD_OWNER_MISSING",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "This flight state does not have the required explicit LowCmd authority phase.",
                "Use the independent flight fallback; acquire LowCmd only after returning to a supported, disarmed, zero-RPM state.",
            )
        if (
            snapshot.state is SystemState.AUTO_LANDING
            and self._config.go2.low_level.enabled
            and (
                authority_state
                not in (
                    {Go2ControlAuthorityState.LOWCMD_ACTIVE}
                    if not self._impact_recovery_finalization_authorized(snapshot)
                    else {
                        Go2ControlAuthorityState.LOWCMD_ACTIVE,
                        Go2ControlAuthorityState.LOWCMD_SAFE_HOLD,
                    }
                )
                or not lowcmd_owned
                or (
                    low_level.ownership_state is not LowCmdOwnershipState.MPC_ACTIVE
                    and not (
                        self._impact_recovery_finalization_authorized(snapshot)
                        and low_level.ownership_state
                        in {LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.SAFE_HOLD}
                    )
                )
            )
        ):
            add(
                "GO2_LOWCMD_MPC_INACTIVE",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "AUTO_LANDING requires a healthy MPC target lease held by the sole LowCmd owner.",
                "Abort impact-aware landing, clear the rotor residual, and retain the Go2 safe-hold stream.",
            )
        if lowcmd_stable and authority_state in {
            Go2ControlAuthorityState.LOWCMD_ACTIVE,
            Go2ControlAuthorityState.LOWCMD_SAFE_HOLD,
        }:
            tracking_failure = self._lowcmd_joint_tracking_failure(snapshot)
            if tracking_failure is not None:
                add(
                    "GO2_JOINT_TRACKING_ERROR",
                    (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                    tracking_failure,
                    "Revoke the MPC lease into the same-owner safe hold and inspect mapping, timing, mechanics, gains, and joint limits.",
                )
        high_level_expected = (
            snapshot.state in _FLIGHT_JOINT_LOCK_STATES
            and not lowcmd_owned
            and (
                not self._config.go2.low_level.enabled
                or authority_state is Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK
            )
        )
        if high_level_expected and not snapshot.joint_lock_confirmed:
            add(
                "GO2_JOINT_LOCK_LOST",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "Go2 joint-lock confirmation was lost in a flight-configuration state.",
                "Stop AeroGo2-owned outputs, retain RadioMaster/Pixhawk control, and do not auto-disarm.",
            )
        elif (
            snapshot.state in _FLIGHT_JOINT_LOCK_STATES
            and snapshot.joint_lock_source == "operator"
            and (
                snapshot.go2.fault_code not in self._config.go2.accepted_state_codes
                or self._go2_joint_lock_transition_is_unsafe(snapshot)
                or self._go2_is_moving(snapshot)
                or snapshot.go2.moving
            )
        ):
            add(
                "GO2_OPERATOR_LOCK_UNSAFE",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "Go2 telemetry became unsafe after operator-confirmed joint lock.",
                "Retain RadioMaster/Pixhawk control and restore a stationary phone-app Lock On state.",
            )
        if (
            self._config.go2.low_level.enabled
            and snapshot.state in _FLIGHT_JOINT_LOCK_STATES
            and not lowcmd_owned
            and authority_state
            not in {
                Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK,
                Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
            }
        ):
            add(
                "GO2_CONTROL_AUTHORITY_UNKNOWN",
                (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                "No authoritative high-level JOINT_LOCK or exclusive LowCmd owner is proven.",
                "Keep all new actuation inhibited until the ownership transaction is reconciled.",
            )

        if snapshot.state is SystemState.LANDING_COMPLIANT:
            contact = assess_foot_contact(snapshot.go2, self._config.go2)
            if not self._pixhawk_ground_state_is_current(snapshot) or not snapshot.pixhawk.landed:
                add(
                    "PIXHAWK_GROUND_STATE_INVALID_DURING_LANDING_COMPLIANCE",
                    SafetySeverity.FAULT,
                    "Landing compliance lacks fresh landed-state evidence.",
                    "Re-lock Go2 and restore current Pixhawk ground-state telemetry.",
                )
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

        if snapshot.state is SystemState.GO2_GROUND_HANDOVER:
            if (
                not self._pixhawk_ground_state_is_current(snapshot)
                or snapshot.pixhawk.armed
                or not snapshot.pixhawk.landed
            ):
                add(
                    "GO2_HANDOVER_NOT_GROUNDED",
                    SafetySeverity.EMERGENCY,
                    "Go2 high-level handover is waiting while Pixhawk is armed or not landed.",
                    "Keep personnel clear, stop supervised outputs, and do not issue Sport commands.",
                )
            if self._esc_is_unsafe(snapshot):
                add(
                    "ESC_RPM_NONZERO_DURING_GO2_HANDOVER",
                    SafetySeverity.EMERGENCY,
                    "ESC telemetry is not complete, healthy, finite, and exactly zero during Go2 handover.",
                    "Keep the vehicle supported and do not continue the high-level handover.",
                )
            if snapshot.f446.duty != 0 or snapshot.f446.faulted:
                add(
                    "F446_UNSAFE_DURING_GO2_HANDOVER",
                    SafetySeverity.FAULT,
                    "F446 is moving or faulted during the Go2 high-level handover.",
                    "Stop the folding drive and keep the deployed structure mechanically locked.",
                )
            if snapshot.go2.low_level_status.ownership_pending:
                add(
                    "GO2_LOWCMD_RELEASE_UNCONFIRMED",
                    SafetySeverity.FAULT,
                    "LowCmd ownership reappeared or remained ambiguous during the high-level handover.",
                    "Do not issue Sport commands; retain the owner process and inspect the handover audit.",
                )
            if self._go2_joint_lock_transition_is_unsafe(snapshot):
                add(
                    "GO2_UNSAFE_DURING_GROUND_HANDOVER",
                    SafetySeverity.FAULT,
                    "Go2 entered a locomotion or unsafe-speed state while waiting for mode=6 JOINT_LOCK.",
                    "Keep the vehicle supported and select Joint Lock without commanding locomotion.",
                )

        if snapshot.state in _TRANSFORM_INTERLOCK_STATES:
            if snapshot.state in {
                SystemState.HOMING_TO_WALK,
                SystemState.FLIGHT_TO_WALK_PRECHECK,
                SystemState.TRANSFORM_TO_WALK,
            } and (
                not self._pixhawk_ground_state_is_current(snapshot) or not snapshot.pixhawk.landed
            ):
                add(
                    "PIXHAWK_GROUND_STATE_INVALID_DURING_WALK_TRANSFORM",
                    SafetySeverity.FAULT,
                    "A WALK-directed morphology transaction lacks fresh landed-state evidence.",
                    "Stop the F446 mechanism and restore current Pixhawk ground-state telemetry.",
                )
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
                    "Go2 entered a sustained locomotion/unsafe-speed state while waiting for Joint Lock.",
                    "Stop the F446 mechanism and keep the Go2 stationary; select Joint Lock in the Unitree app.",
                )
            elif snapshot.state is not SystemState.GO2_JOINT_LOCK_WAIT and self._go2_is_moving(
                snapshot
            ):
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

        high_level_reacquisition_wait = bool(
            snapshot.go2.control_authority.state is Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING
            and snapshot.go2.control_authority.transition_pending
            and self._authority_transition_is_authorized(snapshot)
        )
        if self._config.go2.enabled and (
            not snapshot.go2.low_level_status.ownership_pending
            and not high_level_reacquisition_wait
            and (
                not snapshot.go2.connected
                or not timestamp_is_fresh(
                    snapshot.timestamp,
                    snapshot.go2.timestamp,
                    self._config.safety.go2_timeout_s,
                )
            )
        ):
            add_violation(
                "GO2_TIMEOUT",
                SafetySeverity.FAULT,
                "Go2 status is disconnected, invalid, or stale.",
                "Inhibit walking permission and all new transforms.",
            )

        low_level = snapshot.go2.low_level_status
        if low_level.ownership_pending:
            maximum_age = self._config.go2.low_level.low_state_max_age_s
            if (
                maximum_age is None
                or not low_level.connected
                or not timestamp_is_fresh(
                    snapshot.timestamp,
                    low_level.low_state_timestamp,
                    maximum_age,
                )
            ):
                add_violation(
                    "GO2_LOWSTATE_TIMEOUT",
                    (SafetySeverity.EMERGENCY if snapshot.pixhawk.armed else SafetySeverity.FAULT),
                    "The active LowCmd owner has disconnected, invalid, or stale LowState feedback.",
                    "Keep the sole writer in its verified safe-hold path and use the independent flight fallback.",
                )

    def _lowcmd_owner_healthy(self, snapshot: SystemSnapshot) -> bool:
        status = snapshot.go2.low_level_status
        configured_hash = self._config.go2.low_level.mapping_hash
        if status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE:
            deadline = status.target_deadline
            phase_valid = (
                status.target_sequence is not None
                and deadline is not None
                and math.isfinite(deadline)
                and snapshot.timestamp < deadline
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
            status.connected
            and status.owner_epoch > 0
            and status.healthy
            and status.publisher_active
            and status.writer_alive
            and status.watchdog_healthy
            and status.high_level_released
            and status.network_exclusivity_verified
            and status.mapping_hash_verified
            and configured_hash is not None
            and status.active_mapping_hash == configured_hash
            and phase_valid
        )

    def _impact_recovery_finalization_authorized(self, snapshot: SystemSnapshot) -> bool:
        evidence = snapshot.impact_recovery
        status = snapshot.go2.low_level_status
        return bool(
            snapshot.state is SystemState.AUTO_LANDING
            and evidence.confirmed
            and evidence.landing_session_id == snapshot.impact_landing_session_id
            and evidence.go2_ownership_epoch == status.owner_epoch
            and timestamp_is_fresh(
                snapshot.timestamp,
                evidence.timestamp,
                self._config.safety.impact_recovery_status_max_age_s,
            )
            and timestamp_is_fresh(
                snapshot.timestamp,
                evidence.residual_zero_status_timestamp,
                self._config.safety.impact_recovery_status_max_age_s,
            )
            and snapshot.timestamp < evidence.valid_until
        )

    def _authority_transition_is_authorized(self, snapshot: SystemSnapshot) -> bool:
        authority = snapshot.go2.control_authority
        low_level = snapshot.go2.low_level_status
        ground_state_current = self._pixhawk_ground_state_is_current(snapshot)
        if authority.state is Go2ControlAuthorityState.LOWCMD_ACQUIRING:
            return bool(
                snapshot.state is SystemState.FLIGHT_READY
                and ground_state_current
                and not snapshot.pixhawk.armed
                and snapshot.pixhawk.landed
                and low_level.ownership_state is LowCmdOwnershipState.ACQUIRING
                and low_level.owner_epoch > 0
                and authority.ownership_epoch == low_level.owner_epoch
            )
        if authority.state is Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING:
            return bool(
                snapshot.state
                in {
                    SystemState.TOUCHDOWN_VERIFY,
                    SystemState.GO2_GROUND_HANDOVER,
                    SystemState.FLIGHT_READY,
                    SystemState.BOOT_SAFE,
                    SystemState.FAULT,
                    SystemState.EMERGENCY_STOP,
                }
                and ground_state_current
                and not snapshot.pixhawk.armed
                and snapshot.pixhawk.landed
                and low_level.ownership_state
                in {
                    LowCmdOwnershipState.RELEASING,
                    LowCmdOwnershipState.OBSERVE_ONLY,
                }
            )
        return False

    def _pixhawk_ground_state_is_current(self, snapshot: SystemSnapshot) -> bool:
        return pixhawk_ground_state_is_current(
            snapshot.pixhawk,
            snapshot.timestamp,
            self._config.safety.pixhawk_timeout_s,
            self._config.safety.touchdown_max_source_age_s,
        )

    @staticmethod
    def _authority_is_inconsistent(snapshot: SystemSnapshot) -> bool:
        authority = snapshot.go2.control_authority
        low_level = snapshot.go2.low_level_status
        state = authority.state
        if state is Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK:
            return low_level.ownership_pending or authority.ownership_epoch != 0
        if state is Go2ControlAuthorityState.LOWCMD_ACQUIRING:
            return bool(
                low_level.ownership_state is not LowCmdOwnershipState.ACQUIRING
                or low_level.owner_epoch <= 0
                or authority.ownership_epoch != low_level.owner_epoch
            )
        if state is Go2ControlAuthorityState.LOWCMD_ACTIVE:
            return bool(
                low_level.ownership_state is not LowCmdOwnershipState.MPC_ACTIVE
                or low_level.owner_epoch <= 0
                or authority.ownership_epoch != low_level.owner_epoch
            )
        if state is Go2ControlAuthorityState.LOWCMD_SAFE_HOLD:
            return bool(
                low_level.ownership_state
                not in {LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.SAFE_HOLD}
                or low_level.owner_epoch <= 0
                or authority.ownership_epoch != low_level.owner_epoch
            )
        if state is Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING:
            return bool(
                low_level.ownership_state
                not in {LowCmdOwnershipState.RELEASING, LowCmdOwnershipState.OBSERVE_ONLY}
                or (
                    low_level.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
                    and low_level.ownership_pending
                )
            )
        return False

    def _lowcmd_joint_tracking_failure(self, snapshot: SystemSnapshot) -> Optional[str]:
        status = snapshot.go2.low_level_status
        limits = self._config.go2.low_level.tracking_position_error_limit_rad
        references = status.tracking_reference_q_rad
        errors = status.position_error_rad
        motors = status.motors
        if limits is None or len(limits) != 12:
            return "The commissioned 12-joint tracking-error limits are unavailable."
        if len(references) != 12 or len(errors) != 12 or len(motors) != 12:
            return "The causally paired LowCmd/LowState tracking sample is incomplete."
        feedback_time = status.tracking_error_timestamp
        reference_time = status.tracking_reference_write_timestamp
        maximum_age = self._config.go2.low_level.low_state_max_age_s
        if maximum_age is None or not math.isfinite(maximum_age) or maximum_age <= 0.0:
            return "The commissioned LowState age limit is unavailable or invalid."
        if (
            not math.isfinite(feedback_time)
            or not math.isfinite(reference_time)
            or reference_time <= 0.0
            or feedback_time < reference_time
            or status.tracking_reference_write_generation <= 0
            or not timestamp_is_fresh(
                snapshot.timestamp,
                feedback_time,
                maximum_age,
            )
        ):
            return "The LowCmd/LowState tracking pair is stale, future-dated, or not causal."
        maximum_ratio = 0.0
        maximum_index = 0
        for index, (reference, error, limit, motor) in enumerate(
            zip(references, errors, limits, motors)
        ):
            measured_q = motor.q_rad
            values = (reference, error, limit, motor.timestamp)
            if (
                motor.lost
                or measured_q is None
                or any(value is None for value in values)
                or any(not math.isfinite(float(value)) for value in values if value is not None)
                or not math.isfinite(measured_q)
                or float(limit) <= 0.0
                or motor.timestamp != feedback_time
            ):
                return f"Joint {index} has invalid paired tracking feedback."
            measured_error = measured_q - float(reference)
            if not math.isclose(measured_error, float(error), rel_tol=0.0, abs_tol=1.0e-9):
                return (
                    f"Joint {index} tracking error does not match its paired command and feedback."
                )
            ratio = abs(float(error)) / float(limit)
            if ratio > maximum_ratio:
                maximum_ratio = ratio
                maximum_index = index
        if maximum_ratio > 1.0:
            return (
                f"Joint {maximum_index} tracking error exceeds its commissioned limit "
                f"({maximum_ratio:.3f} times the limit)."
            )
        return None

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
            or snapshot.go2.fault_code not in self._config.go2.accepted_state_codes
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
            SystemState.GO2_GROUND_HANDOVER,
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
        return any(finite_real(value) is None for value in numeric_values)
