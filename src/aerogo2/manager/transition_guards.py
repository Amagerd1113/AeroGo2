"""Pure transition guards for the AeroGo2 state machine."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Mapping, Set, Tuple

from aerogo2.common.config import AppConfig
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446State,
    Go2ControlAuthorityState,
    MorphologyRequest,
    SystemState,
)
from aerogo2.common.models import LowCmdOwnershipState, SystemSnapshot
from aerogo2.common.numeric import finite_real
from aerogo2.common.results import GuardResult
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

_MUTABLE_ALLOWED_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
    SystemState.BOOT_SAFE: {
        SystemState.MANUAL_POSITIONING,
        SystemState.HOMING_TO_WALK,
        SystemState.WALK,
        SystemState.FLIGHT_READY,
        SystemState.GO2_JOINT_LOCK_WAIT,
        SystemState.FAULT,
    },
    SystemState.MANUAL_POSITIONING: {
        SystemState.BOOT_SAFE,
        SystemState.WALK,
        SystemState.FLIGHT_READY,
        SystemState.GO2_JOINT_LOCK_WAIT,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.HOMING_TO_WALK: {
        SystemState.WALK,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.WALK: {
        SystemState.MANUAL_POSITIONING,
        SystemState.BOOT_SAFE,
        SystemState.WALK_TO_FLIGHT_PRECHECK,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.WALK_TO_FLIGHT_PRECHECK: {
        SystemState.TRANSFORM_TO_FLIGHT,
        SystemState.WALK,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.TRANSFORM_TO_FLIGHT: {
        SystemState.GO2_JOINT_LOCK_WAIT,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.GO2_JOINT_LOCK_WAIT: {
        SystemState.FLIGHT_READY,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.FLIGHT_READY: {
        SystemState.MANUAL_POSITIONING,
        SystemState.BOOT_SAFE,
        SystemState.FLIGHT_MANUAL,
        SystemState.FLIGHT_TO_WALK_PRECHECK,
        SystemState.LANDING_COMPLIANT,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.FLIGHT_MANUAL: {
        SystemState.AUTO_LANDING_READY,
        SystemState.TOUCHDOWN_VERIFY,
        SystemState.FLIGHT_TO_WALK_PRECHECK,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.AUTO_LANDING_READY: {
        SystemState.AUTO_LANDING,
        SystemState.FLIGHT_MANUAL,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.AUTO_LANDING: {
        SystemState.TOUCHDOWN_VERIFY,
        SystemState.FLIGHT_MANUAL,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.TOUCHDOWN_VERIFY: {
        SystemState.MANUAL_POSITIONING,
        SystemState.FLIGHT_MANUAL,
        SystemState.FLIGHT_TO_WALK_PRECHECK,
        SystemState.LANDING_COMPLIANT,
        SystemState.GO2_GROUND_HANDOVER,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.GO2_GROUND_HANDOVER: {
        SystemState.FLIGHT_READY,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.LANDING_COMPLIANT: {
        SystemState.MANUAL_POSITIONING,
        SystemState.GO2_JOINT_LOCK_WAIT,
        SystemState.FLIGHT_READY,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.FLIGHT_TO_WALK_PRECHECK: {
        SystemState.TRANSFORM_TO_WALK,
        SystemState.FLIGHT_READY,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.TRANSFORM_TO_WALK: {
        SystemState.WALK,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    },
    SystemState.FAULT: {SystemState.BOOT_SAFE, SystemState.EMERGENCY_STOP},
    SystemState.EMERGENCY_STOP: {SystemState.FAULT, SystemState.BOOT_SAFE},
}

ALLOWED_TRANSITIONS: Mapping[SystemState, FrozenSet[SystemState]] = MappingProxyType(
    {state: frozenset(targets) for state, targets in _MUTABLE_ALLOWED_TRANSITIONS.items()}
)
del _MUTABLE_ALLOWED_TRANSITIONS

TRANSFORM_PRECHECK_STATES: FrozenSet[SystemState] = frozenset(
    (SystemState.WALK_TO_FLIGHT_PRECHECK, SystemState.FLIGHT_TO_WALK_PRECHECK)
)
ACTIVE_TRANSFORM_STATES: FrozenSet[SystemState] = frozenset(
    (
        SystemState.MANUAL_POSITIONING,
        SystemState.HOMING_TO_WALK,
        SystemState.TRANSFORM_TO_FLIGHT,
        SystemState.TRANSFORM_TO_WALK,
    )
)
TRANSFORM_STATES: FrozenSet[SystemState] = frozenset(
    TRANSFORM_PRECHECK_STATES
    | ACTIVE_TRANSFORM_STATES
    | frozenset((SystemState.GO2_JOINT_LOCK_WAIT,))
)

# A live or ambiguous LowCmd handoff may only coexist with flight states that
# are explicitly supervised by the sole-writer watchdog.  In particular it
# must never leak into a morphology/manual state where F446 can move.  FAULT
# and EMERGENCY_STOP are handled by the unconditional fail-closed edge above.
_LOWCMD_OWNERSHIP_ALLOWED_STATES: FrozenSet[SystemState] = frozenset(
    (
        SystemState.FLIGHT_READY,
        SystemState.FLIGHT_MANUAL,
        SystemState.AUTO_LANDING_READY,
        SystemState.AUTO_LANDING,
        SystemState.TOUCHDOWN_VERIFY,
        SystemState.FAULT,
        SystemState.EMERGENCY_STOP,
    )
)


class TransitionGuards:
    """Evaluates state graph legality and state-specific safety conditions."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def evaluate(
        self,
        current: SystemState,
        new_state: SystemState,
        snapshot: SystemSnapshot,
    ) -> GuardResult:
        codes: List[str] = []
        messages: List[str] = []

        if new_state not in ALLOWED_TRANSITIONS.get(current, set()):
            self._reject(
                codes,
                messages,
                "INVALID_STATE_TRANSITION",
                f"{current.name} cannot transition directly to {new_state.name}",
            )
            return GuardResult(False, tuple(codes), tuple(messages))

        if new_state in (SystemState.FAULT, SystemState.EMERGENCY_STOP):
            return GuardResult.allow()

        if (
            snapshot.go2.low_level_status.ownership_pending
            and new_state not in _LOWCMD_OWNERSHIP_ALLOWED_STATES
        ):
            self._reject(
                codes,
                messages,
                "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
                "Complete the supported-ground LowCmd-to-high-level handover before "
                f"entering {new_state.name}",
            )

        if current in TRANSFORM_STATES or new_state in TRANSFORM_STATES:
            initial_flight_precheck = (
                current is SystemState.WALK and new_state is SystemState.WALK_TO_FLIGHT_PRECHECK
            )
            self._require_transform_safe(
                snapshot,
                codes,
                messages,
                exact_rotor_stop=not initial_flight_precheck,
            )

        if (
            current in {SystemState.TOUCHDOWN_VERIFY, SystemState.LANDING_COMPLIANT}
            and new_state is SystemState.MANUAL_POSITIONING
            and not snapshot.pixhawk.landed
        ):
            self._reject(
                codes,
                messages,
                "TOUCHDOWN_NOT_CONFIRMED",
                "Post-touchdown manual positioning requires Pixhawk landed=true",
            )

        if current is SystemState.BOOT_SAFE and new_state is SystemState.WALK:
            self._require_fresh_devices(snapshot, codes, messages, include_rc=False)
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            self._require_no_active_faults(snapshot, codes, messages)
            if snapshot.configuration is not Configuration.WALK:
                self._reject(
                    codes,
                    messages,
                    "WALK_CONFIGURATION_NOT_CONFIRMED",
                    "F446 has not confirmed the configured WALK limit state",
                )

        if current is SystemState.BOOT_SAFE and new_state is SystemState.FLIGHT_READY:
            self._require_fresh_devices(snapshot, codes, messages, include_rc=False)
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            self._require_no_active_faults(snapshot, codes, messages)
            if snapshot.configuration is not Configuration.FLIGHT:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                    "F446 has not confirmed the configured FLIGHT limit state",
                )
            if snapshot.f446.duty != 0:
                self._reject(
                    codes,
                    messages,
                    "F446_DUTY_NONZERO",
                    "F446 final duty must be zero",
                )

        if new_state is SystemState.HOMING_TO_WALK:
            if snapshot.configuration is not Configuration.UNKNOWN:
                self._reject(
                    codes,
                    messages,
                    "F446_HOME_NOT_REQUIRED",
                    "F446 homing is allowed only when physical configuration is UNKNOWN",
                )
            if snapshot.f446.state is not F446State.IDLE:
                self._reject(
                    codes,
                    messages,
                    "F446_HOME_START_STATE_INVALID",
                    (
                        "F446 homing requires IDLE with zero duty, received "
                        f"{snapshot.f446.state.value}"
                    ),
                )

        if current is SystemState.WALK and new_state is SystemState.WALK_TO_FLIGHT_PRECHECK:
            if snapshot.rc.morphology_request is not MorphologyRequest.FLIGHT_REQUEST:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_REQUEST_NOT_ACTIVE",
                    "RC CH9 is not a debounced FLIGHT_REQUEST",
                )
            if snapshot.configuration is not Configuration.WALK:
                self._reject(
                    codes,
                    messages,
                    "WALK_CONFIGURATION_NOT_CONFIRMED",
                    "Current morphology must be confirmed WALK",
                )
            self._require_go2_stationary(snapshot, codes, messages)

        if new_state is SystemState.TRANSFORM_TO_FLIGHT:
            if snapshot.state is not SystemState.WALK_TO_FLIGHT_PRECHECK:
                self._reject(
                    codes,
                    messages,
                    "PRECHECK_STATE_REQUIRED",
                    "Flight transformation requires WALK_TO_FLIGHT_PRECHECK",
                )
            if snapshot.configuration is not Configuration.WALK:
                self._reject(
                    codes,
                    messages,
                    "WALK_CONFIGURATION_NOT_CONFIRMED",
                    "The second-stage check requires the mechanism to remain in WALK",
                )
            self._require_go2_stationary(snapshot, codes, messages)

        if new_state is SystemState.GO2_JOINT_LOCK_WAIT:
            if snapshot.configuration is not Configuration.FLIGHT:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                    "Go2 joint-lock wait requires a verified FLIGHT endpoint",
                )
            if snapshot.f446.duty != 0:
                self._reject(
                    codes,
                    messages,
                    "F446_DUTY_NONZERO",
                    "F446 must be stopped before waiting for Go2 JOINT_LOCK",
                )
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)

        if new_state is SystemState.FLIGHT_READY:
            if snapshot.configuration is not Configuration.FLIGHT:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                    "F446 has not confirmed the configured FLIGHT limit state",
                )
            if snapshot.f446.duty != 0:
                self._reject(
                    codes,
                    messages,
                    "F446_DUTY_NONZERO",
                    "F446 final duty must be zero",
                )
            self._require_disarmed(snapshot, codes, messages)
            if snapshot.configuration is Configuration.FLIGHT:
                self._require_go2_joint_lock(snapshot, codes, messages)

        if new_state is SystemState.FLIGHT_MANUAL:
            if current is SystemState.FLIGHT_READY and snapshot.ground_arm_authorized is not True:
                self._reject(
                    codes,
                    messages,
                    "GROUND_ARM_AUTHORIZATION_REQUIRED",
                    "AeroGo2 shell authorization is required before RadioMaster arming",
                )
            if current is SystemState.FLIGHT_READY and snapshot.pixhawk.armed is not True:
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_NOT_ARMED",
                    "Only RadioMaster/Pixhawk arming may enter FLIGHT_MANUAL",
                )
            if current in (SystemState.AUTO_LANDING, SystemState.AUTO_LANDING_READY):
                if not (
                    snapshot.rc.manual_override
                    or snapshot.rc.auto_landing_request is AutoLandingRequest.MANUAL
                ):
                    self._reject(
                        codes,
                        messages,
                        "MANUAL_OVERRIDE_NOT_ACTIVE",
                        "Landing exit requires an operator override or controller abort",
                    )

        if new_state is SystemState.AUTO_LANDING_READY:
            self._require_autoland_sources_current(snapshot, codes, messages)
            if not snapshot.pixhawk.armed:
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_NOT_ARMED",
                    "Automatic landing preparation requires an armed aircraft",
                )
            if snapshot.rc.auto_landing_request not in (
                AutoLandingRequest.AUTO_READY,
                AutoLandingRequest.AUTO_EXECUTE,
            ):
                self._reject(
                    codes,
                    messages,
                    "AUTOLAND_READY_NOT_REQUESTED",
                    "RC CH10 is not AUTO_READY/AUTO_EXECUTE",
                )
            if (
                self._config.go2.low_level.enabled
                and snapshot.go2.control_authority.state
                is not Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
            ):
                self._reject(
                    codes,
                    messages,
                    "GO2_LOWCMD_SAFE_HOLD_REQUIRED",
                    "Automatic landing preparation requires the acquired LowCmd owner in verified safe-hold",
                )

        if new_state is SystemState.AUTO_LANDING:
            self._require_autoland_sources_current(snapshot, codes, messages)
            if snapshot.rc.auto_landing_request is not AutoLandingRequest.AUTO_EXECUTE:
                self._reject(
                    codes,
                    messages,
                    "AUTOLAND_EXECUTE_NOT_REQUESTED",
                    "RC CH10 must be AUTO_EXECUTE",
                )
            if not snapshot.pixhawk.armed:
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_NOT_ARMED",
                    "Automatic landing requires an armed aircraft",
                )
            if (
                self._config.go2.low_level.enabled
                and snapshot.go2.control_authority.state
                is not Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
            ):
                self._reject(
                    codes,
                    messages,
                    "GO2_LOWCMD_SAFE_HOLD_REQUIRED",
                    "AUTO_LANDING may start only from the same healthy LowCmd safe-hold epoch",
                )

        if new_state is SystemState.TOUCHDOWN_VERIFY:
            source_freshness = assess_pixhawk_source_freshness(
                snapshot.pixhawk,
                snapshot.timestamp,
                self._config.safety.pixhawk_timeout_s,
                self._config.safety.touchdown_max_source_age_s,
            )
            source_timestamps = [
                snapshot.pixhawk.attitude_timestamp,
                snapshot.pixhawk.kinematics_timestamp,
                snapshot.pixhawk.landed_state_timestamp,
            ]
            estimate_current = True
            if current is SystemState.AUTO_LANDING:
                estimate = snapshot.landing_estimate
                source_timestamps.append(estimate.timestamp)
                estimate_current = bool(
                    type(estimate.valid) is bool
                    and estimate.valid is True
                    and type(estimate.ground_detected) is bool
                    and estimate.ground_detected is True
                    and finite_real(estimate.height_m) is not None
                    and finite_real(estimate.vertical_velocity_mps) is not None
                    and finite_real(estimate.horizontal_velocity_mps) is not None
                    and timestamp_is_fresh(
                        snapshot.timestamp,
                        estimate.timestamp,
                        self._config.safety.controller_timeout_s,
                    )
                )
            if not source_freshness.touchdown:
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_TOUCHDOWN_EVIDENCE_STALE",
                    "One or more independently timed Pixhawk touchdown sources are stale",
                )
            if not pixhawk_touchdown_payload_is_valid(snapshot.pixhawk):
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID",
                    "Pixhawk touchdown telemetry contains malformed or non-finite values",
                )
            if not estimate_current:
                self._reject(
                    codes,
                    messages,
                    "AUTOLAND_ESTIMATOR_INVALID",
                    "The automatic-landing estimate is invalid, lacks ground detection, or is stale",
                )
            if not timestamps_are_coherent(
                source_timestamps,
                self._config.safety.touchdown_max_source_skew_s,
            ):
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_TOUCHDOWN_EVIDENCE_INCOHERENT",
                    "Touchdown sources are outside the allowed mutual-skew window",
                )
            if not snapshot.pixhawk.landed:
                self._reject(
                    codes,
                    messages,
                    "PIXHAWK_NOT_LANDED",
                    "Pixhawk landed state is not confirmed",
                )
            if current is SystemState.AUTO_LANDING and self._config.go2.low_level.enabled:
                evidence = snapshot.impact_recovery
                low_level = snapshot.go2.low_level_status
                maximum_low_state_age = self._config.go2.low_level.low_state_max_age_s
                retained_safe_hold = bool(
                    not self._config.go2.low_level.enabled
                    or (
                        snapshot.go2.control_authority.state
                        is Go2ControlAuthorityState.LOWCMD_SAFE_HOLD
                        and low_level.ownership_state
                        in {LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.SAFE_HOLD}
                        and low_level.owner_epoch > 0
                        and evidence.go2_ownership_epoch == low_level.owner_epoch
                        and low_level.healthy
                        and low_level.connected
                        and low_level.publisher_active
                        and low_level.writer_alive
                        and low_level.watchdog_healthy
                        and low_level.safe_hold_active
                        and low_level.safe_hold_settled
                        and low_level.target_sequence is None
                        and low_level.target_deadline is None
                        and low_level.high_level_released
                        and low_level.network_exclusivity_verified
                        and low_level.mapping_hash_verified
                        and low_level.active_mapping_hash == self._config.go2.low_level.mapping_hash
                        and maximum_low_state_age is not None
                        and timestamp_is_fresh(
                            snapshot.timestamp,
                            low_level.low_state_timestamp,
                            maximum_low_state_age,
                        )
                    )
                )
                if (
                    not snapshot.impact_landing_exit_ready
                    or not snapshot.post_touchdown_stable_dwell_complete
                    or snapshot.post_touchdown_last_stability_check_at is None
                    or not math.isfinite(snapshot.post_touchdown_last_stability_check_at)
                    or snapshot.timestamp < snapshot.post_touchdown_last_stability_check_at
                    or snapshot.timestamp - snapshot.post_touchdown_last_stability_check_at
                    > self._config.safety.post_touchdown_stability_max_check_gap_s
                    or snapshot.external_setpoint_active
                    or not evidence.confirmed
                    or not retained_safe_hold
                    or evidence.landing_session_id != snapshot.impact_landing_session_id
                    or snapshot.timestamp >= evidence.valid_until
                    or not timestamp_is_fresh(
                        snapshot.timestamp,
                        evidence.timestamp,
                        self._config.safety.impact_recovery_status_max_age_s,
                    )
                    or not timestamp_is_fresh(
                        snapshot.timestamp,
                        evidence.residual_zero_status_timestamp,
                        self._config.safety.impact_recovery_status_max_age_s,
                    )
                ):
                    self._reject(
                        codes,
                        messages,
                        "IMPACT_RECOVERY_EXIT_NOT_CONFIRMED",
                        "AUTO_LANDING must retain control until recovery, immutable FC CLEAR/execution proof, fresh persistent-zero status, and the post-touchdown stable dwell are all confirmed",
                    )

        if new_state is SystemState.GO2_GROUND_HANDOVER:
            # The SportModeState publisher may legitimately have been silent
            # while MotionSwitcher was released.  SystemManager waits for a
            # causally-new post-handover Go2 sample before leaving this state;
            # requiring the pre-handover sample here would make the handover
            # impossible after a normal flight-duration LowCmd session.
            self._require_fresh_devices(
                snapshot,
                codes,
                messages,
                include_rc=False,
                include_go2=False,
            )
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            self._require_pixhawk_ground_state_current(snapshot, codes, messages)
            if not snapshot.pixhawk.landed:
                self._reject(
                    codes,
                    messages,
                    "TOUCHDOWN_NOT_CONFIRMED",
                    "Pixhawk must report landed during the Go2 control handover",
                )
            low_level = snapshot.go2.low_level_status
            if (
                self._config.go2.low_level.enabled
                and snapshot.go2.control_authority.state
                is not Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING
            ):
                self._reject(
                    codes,
                    messages,
                    "GO2_HIGH_LEVEL_REACQUISITION_REQUIRED",
                    "The explicit authority transaction must be waiting for a post-handover JOINT_LOCK sample",
                )
            if low_level.ownership_pending:
                self._reject(
                    codes,
                    messages,
                    "GO2_LOWCMD_RELEASE_UNCONFIRMED",
                    "Ground handover requires a stopped writer, zero epoch, and confirmed high-level service",
                )
            if snapshot.f446.duty != 0 or snapshot.f446.faulted:
                self._reject(
                    codes,
                    messages,
                    "F446_NOT_SAFE",
                    "F446 must remain stopped and fault-free during ground handover",
                )

        if new_state is SystemState.LANDING_COMPLIANT:
            if not self._config.go2.landing_compliance_enabled:
                self._reject(
                    codes,
                    messages,
                    "LANDING_COMPLIANCE_DISABLED",
                    "Landing compliance is disabled until foot-force calibration is configured",
                )
            self._require_fresh_devices(snapshot, codes, messages, include_rc=False)
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            self._require_no_active_faults(snapshot, codes, messages)
            self._require_go2_joint_lock(snapshot, codes, messages)
            self._require_go2_foot_contact(snapshot, codes, messages)
            self._require_pixhawk_ground_state_current(snapshot, codes, messages)
            if not snapshot.pixhawk.landed:
                self._reject(
                    codes,
                    messages,
                    "TOUCHDOWN_NOT_CONFIRMED",
                    "Pixhawk must continue to report landed",
                )
            if snapshot.configuration is not Configuration.FLIGHT:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                    "Landing compliance requires the verified FLIGHT configuration",
                )
            if snapshot.f446.duty != 0 or snapshot.f446.faulted:
                self._reject(
                    codes,
                    messages,
                    "F446_NOT_SAFE",
                    "F446 must remain stopped and fault-free",
                )

        if new_state is SystemState.FLIGHT_TO_WALK_PRECHECK:
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            self._require_go2_stationary(snapshot, codes, messages)
            self._require_pixhawk_ground_state_current(snapshot, codes, messages)
            if not snapshot.pixhawk.landed:
                self._reject(
                    codes,
                    messages,
                    "TOUCHDOWN_NOT_CONFIRMED",
                    "Pixhawk must report landed before transforming to WALK",
                )

        if new_state is SystemState.TRANSFORM_TO_WALK:
            if snapshot.state is not SystemState.FLIGHT_TO_WALK_PRECHECK:
                self._reject(
                    codes,
                    messages,
                    "PRECHECK_STATE_REQUIRED",
                    "Walk transformation requires FLIGHT_TO_WALK_PRECHECK",
                )
            if snapshot.configuration is not Configuration.FLIGHT:
                self._reject(
                    codes,
                    messages,
                    "FLIGHT_CONFIGURATION_NOT_CONFIRMED",
                    "The second-stage check requires the mechanism to remain in FLIGHT",
                )

        if new_state is SystemState.WALK:
            if snapshot.configuration is not Configuration.WALK:
                self._reject(
                    codes,
                    messages,
                    "WALK_CONFIGURATION_NOT_CONFIRMED",
                    "F446 has not confirmed the configured WALK limit state",
                )
            if snapshot.f446.duty != 0:
                self._reject(
                    codes,
                    messages,
                    "F446_DUTY_NONZERO",
                    "F446 final duty must be zero",
                )

        if new_state is SystemState.BOOT_SAFE:
            self._require_disarmed(snapshot, codes, messages)
            self._require_rotors_stopped(snapshot, codes, messages)
            if snapshot.go2.low_level_status.ownership_pending:
                self._reject(
                    codes,
                    messages,
                    "GO2_LOWCMD_EXPLICIT_RELEASE_REQUIRED",
                    "BOOT_SAFE requires a completed LowCmd-to-high-level handover",
                )
            if snapshot.active_fault_codes:
                self._reject(
                    codes,
                    messages,
                    "ACTIVE_FAULTS_REMAIN",
                    "Active faults must be acknowledged before BOOT_SAFE",
                )

        return GuardResult(not codes, tuple(codes), tuple(messages))

    def manual_motion_guard(self, snapshot: SystemSnapshot) -> GuardResult:
        """Recheck all live interlocks immediately before a manual command."""

        codes: List[str] = []
        messages: List[str] = []
        if snapshot.state is not SystemState.MANUAL_POSITIONING:
            self._reject(
                codes,
                messages,
                "MANUAL_POSITIONING_REQUIRED",
                "Enter MANUAL_POSITIONING before commanding F446 directly",
            )
        if not snapshot.maintenance_mode:
            self._reject(
                codes,
                messages,
                "F446_MAINTENANCE_REQUIRED",
                "F446 maintenance mode is not active",
            )
        self._require_transform_safe(
            snapshot,
            codes,
            messages,
            exact_rotor_stop=True,
        )
        return GuardResult(not codes, tuple(codes), tuple(messages))

    def _require_transform_safe(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
        *,
        exact_rotor_stop: bool,
    ) -> None:
        self._require_fresh_devices(snapshot, codes, messages, include_rc=True)
        self._require_disarmed(snapshot, codes, messages)
        if exact_rotor_stop:
            self._require_rotors_stopped(snapshot, codes, messages)
        else:
            self._require_rotors_below_limit(snapshot, codes, messages)
        self._require_no_active_faults(snapshot, codes, messages)
        if snapshot.pixhawk.failsafe or snapshot.pixhawk.rc_failsafe:
            self._reject(
                codes,
                messages,
                "PIXHAWK_FAILSAFE",
                "Pixhawk failsafe is active; morphology movement is forbidden",
            )
        flight_enable = assess_flight_enable(snapshot.rc, self._config.rc)
        if not flight_enable.valid:
            self._reject(
                codes,
                messages,
                "FLIGHT_ENABLE_INVALID",
                "RC CH5 is missing, ambiguous, malformed, or inconsistent with parsed state",
            )
        elif not flight_enable.low:
            self._reject(
                codes,
                messages,
                "FLIGHT_ENABLE_HIGH",
                "RC CH5 FLIGHT_ENABLE must be LOW",
            )
        if snapshot.f446.faulted:
            self._reject(codes, messages, "F446_FAULT", "F446 reports FAULT")
        if snapshot.f446.state is F446State.UNKNOWN:
            self._reject(
                codes,
                messages,
                "F446_STATE_UNKNOWN",
                "F446 state is unknown",
            )
        if snapshot.f446.duty != 0:
            self._reject(
                codes,
                messages,
                "F446_DUTY_NONZERO",
                "F446 duty must be zero before a morphology transaction",
            )
        current = snapshot.f446.used_current_adc
        threshold = snapshot.f446.threshold_adc
        try:
            current_value = float(current) if current is not None else math.nan
            threshold_value = float(threshold) if threshold is not None else math.nan
        except (TypeError, ValueError):
            current_value = math.nan
            threshold_value = math.nan
        margin = float(self._config.f446.current_safe_margin_adc)
        if (
            current is None
            or isinstance(current, bool)
            or threshold is None
            or isinstance(threshold, bool)
            or not math.isfinite(current_value)
            or current_value < 0.0
            or not math.isfinite(threshold_value)
            or threshold_value <= margin
            or current_value > threshold_value - margin
        ):
            self._reject(
                codes,
                messages,
                "F446_OVERCURRENT",
                "F446 current must be no greater than threshold_raw minus current_safe_margin_adc",
            )
        self._require_go2_stationary(snapshot, codes, messages)

    def _require_fresh_devices(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
        include_rc: bool,
        include_go2: bool = True,
    ) -> None:
        now = snapshot.timestamp
        checks: Tuple[Tuple[str, bool, float, float], ...] = (
            (
                "PIXHAWK",
                snapshot.pixhawk.connected,
                snapshot.pixhawk.heartbeat_timestamp,
                self._config.safety.pixhawk_timeout_s,
            ),
            (
                "F446",
                snapshot.f446.connected,
                snapshot.f446.timestamp,
                self._config.safety.f446_timeout_s,
            ),
        )
        if include_go2:
            checks += (
                (
                    "GO2",
                    snapshot.go2.connected,
                    snapshot.go2.timestamp,
                    self._config.safety.go2_timeout_s,
                ),
            )
        for name, connected, timestamp, timeout in checks:
            if not connected or timestamp <= 0.0 or not timestamp_is_fresh(now, timestamp, timeout):
                self._reject(
                    codes,
                    messages,
                    f"{name}_TIMEOUT",
                    f"{name} status is disconnected, invalid, future-dated, or stale",
                )
        if include_rc:
            if (
                not snapshot.rc.connected
                or snapshot.rc.failsafe
                or snapshot.rc.timestamp <= 0.0
                or not timestamp_is_fresh(
                    now,
                    snapshot.rc.timestamp,
                    self._config.safety.rc_timeout_s,
                )
            ):
                self._reject(codes, messages, "RC_TIMEOUT", "RC status is failsafe or stale")

    def _require_pixhawk_ground_state_current(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
    ) -> None:
        if not pixhawk_ground_state_is_current(
            snapshot.pixhawk,
            snapshot.timestamp,
            self._config.safety.pixhawk_timeout_s,
            self._config.safety.touchdown_max_source_age_s,
        ):
            self._reject(
                codes,
                messages,
                "PIXHAWK_GROUND_STATE_STALE",
                "Pixhawk heartbeat and landed state must both be fresh",
            )

    def _require_autoland_sources_current(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
    ) -> None:
        freshness = assess_pixhawk_source_freshness(
            snapshot.pixhawk,
            snapshot.timestamp,
            self._config.safety.pixhawk_timeout_s,
            self._config.safety.touchdown_max_source_age_s,
        )
        if not freshness.touchdown:
            self._reject(
                codes,
                messages,
                "PIXHAWK_TOUCHDOWN_SOURCE_STALE",
                "Independent Pixhawk attitude, kinematics, or landed-state telemetry is stale",
            )
        if not pixhawk_touchdown_payload_is_valid(snapshot.pixhawk):
            self._reject(
                codes,
                messages,
                "PIXHAWK_TOUCHDOWN_PAYLOAD_INVALID",
                "Pixhawk touchdown telemetry contains malformed or non-finite values",
            )
        estimate = snapshot.landing_estimate
        if (
            not estimate.valid
            or not estimate.ground_detected
            or finite_real(estimate.height_m) is None
            or finite_real(estimate.vertical_velocity_mps) is None
            or finite_real(estimate.horizontal_velocity_mps) is None
            or not timestamp_is_fresh(
                snapshot.timestamp,
                estimate.timestamp,
                self._config.safety.controller_timeout_s,
            )
        ):
            self._reject(
                codes,
                messages,
                "AUTOLAND_ESTIMATOR_INVALID",
                "Landing estimate is invalid, lacks ground detection, or is stale",
            )
        elif freshness.touchdown and not timestamps_are_coherent(
            (
                snapshot.pixhawk.attitude_timestamp,
                snapshot.pixhawk.kinematics_timestamp,
                snapshot.pixhawk.landed_state_timestamp,
                estimate.timestamp,
            ),
            self._config.safety.touchdown_max_source_skew_s,
        ):
            self._reject(
                codes,
                messages,
                "PIXHAWK_TOUCHDOWN_SOURCE_INCOHERENT",
                "Pixhawk touchdown telemetry and landing estimate are outside the allowed mutual-skew window",
            )

    @staticmethod
    def _require_disarmed(snapshot: SystemSnapshot, codes: List[str], messages: List[str]) -> None:
        if snapshot.pixhawk.armed:
            TransitionGuards._reject(
                codes,
                messages,
                "PIXHAWK_ARMED_DURING_TRANSFORM",
                "Pixhawk is armed; F446 motion is forbidden",
            )

    def _require_rotors_stopped(
        self, snapshot: SystemSnapshot, codes: List[str], messages: List[str]
    ) -> None:
        if self._esc_telemetry_is_unsafe(snapshot, exact_zero=True):
            self._reject(
                codes,
                messages,
                "ESC_RPM_NONZERO_DURING_TRANSFORM",
                "Every configured ESC must be uniquely present, online, finite, and at zero RPM",
            )

    def _require_rotors_below_limit(
        self, snapshot: SystemSnapshot, codes: List[str], messages: List[str]
    ) -> None:
        if self._esc_telemetry_is_unsafe(snapshot, exact_zero=False):
            self._reject(
                codes,
                messages,
                "ESC_RPM_NONZERO_DURING_TRANSFORM",
                "Every configured ESC must be uniquely present, online, finite, and below the configured precheck RPM limit",
            )

    def _esc_telemetry_is_unsafe(
        self,
        snapshot: SystemSnapshot,
        *,
        exact_zero: bool,
    ) -> bool:
        return not assess_esc_telemetry(
            snapshot,
            self._config.esc.slots,
            exact_zero=exact_zero,
            maximum_abs_rpm=(
                None if exact_zero else self._config.safety.maximum_safe_esc_rpm_for_transform
            ),
        ).safe

    def _require_go2_stationary(
        self, snapshot: SystemSnapshot, codes: List[str], messages: List[str]
    ) -> None:
        components = snapshot.go2.body_velocity
        if (
            not math.isfinite(snapshot.go2.velocity_mps)
            or len(components) != 3
            or any(
                not math.isfinite(component)
                or abs(component) >= self._config.safety.stationary_velocity_mps
                for component in components
            )
            or not snapshot.go2.stable
            or snapshot.go2.moving
            or snapshot.go2.controller_active
            or abs(snapshot.go2.velocity_mps) >= self._config.safety.stationary_velocity_mps
        ):
            self._reject(
                codes,
                messages,
                "GO2_MOVING_DURING_TRANSFORM",
                "Go2 must be stable and stationary",
            )

    def _require_go2_joint_lock(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
    ) -> None:
        if not snapshot.joint_lock_confirmed or (
            self._config.go2.low_level.enabled
            and snapshot.go2.control_authority.state
            is not Go2ControlAuthorityState.HIGH_LEVEL_JOINT_LOCK
        ):
            TransitionGuards._reject(
                codes,
                messages,
                "GO2_JOINT_LOCK_REQUIRED",
                "Go2 joint lock must be confirmed before entering FLIGHT_READY",
            )

    def _require_go2_foot_contact(
        self,
        snapshot: SystemSnapshot,
        codes: List[str],
        messages: List[str],
    ) -> None:
        if not self._config.go2.landing_compliance_enabled:
            return
        assessment = assess_foot_contact(snapshot.go2, self._config.go2)
        if not assessment.valid:
            self._reject(
                codes,
                messages,
                "GO2_FOOT_FORCE_INVALID",
                "All four calibrated Go2 foot-force channels must be valid",
            )
        elif not assessment.safe:
            self._reject(
                codes,
                messages,
                "GO2_FOOT_CONTACT_INSUFFICIENT",
                f"Only {assessment.contact_count} feet report contact; "
                f"{assessment.required_count} are required",
            )

    @staticmethod
    def _require_no_active_faults(
        snapshot: SystemSnapshot, codes: List[str], messages: List[str]
    ) -> None:
        if snapshot.active_fault_codes:
            TransitionGuards._reject(
                codes,
                messages,
                "ACTIVE_FAULTS_PRESENT",
                "Active safety faults must be cleared before changing configuration",
            )

    @staticmethod
    def _reject(codes: List[str], messages: List[str], code: str, message: str) -> None:
        if code not in codes:
            codes.append(code)
            messages.append(message)
