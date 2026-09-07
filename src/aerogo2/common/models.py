"""Immutable snapshots crossing AeroGo2 subsystem boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from aerogo2.common.config import X8_ESC_SLOT_MAPPING
from aerogo2.common.enums import (
    AutoLandingRequest,
    Configuration,
    F446EventType,
    F446State,
    Go2ControlAuthorityState,
    ImpactLandingPhase,
    MorphologyRequest,
    SafetySeverity,
    SystemState,
)
from aerogo2.common.immutable import frozen_mapping
from aerogo2.common.numeric import finite_real


@dataclass(frozen=True)
class EscTelemetry:
    slot: int
    physical_position: str
    rpm: float = 0.0
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    temperature_c: Optional[float] = None
    healthy: bool = True
    timestamp: float = 0.0


def default_esc_telemetry() -> Tuple[EscTelemetry, ...]:
    return tuple(EscTelemetry(slot, position) for slot, position in X8_ESC_SLOT_MAPPING.items())


@dataclass(frozen=True)
class PixhawkStatus:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    landed: bool = True
    rc_failsafe: bool = False
    attitude_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery_voltage: Optional[float] = None
    rc_channels: Mapping[int, int] = field(default_factory=dict)
    esc_rpm: Mapping[int, float] = field(default_factory=dict)
    esc_online: Mapping[int, bool] = field(default_factory=dict)
    esc_raw_present_slots: Tuple[int, ...] = ()
    esc_mavlink_display_shift: int = 0

    # Compatibility/detail fields used by the Phase 1 safety implementation.
    failsafe: bool = False
    heartbeat_timestamp: float = 0.0
    # Source-specific receive times.  A fresh HEARTBEAT must not make an old
    # ATTITUDE, GLOBAL_POSITION_INT, or EXTENDED_SYS_STATE sample look current
    # enough to prove touchdown.
    attitude_timestamp: float = 0.0
    kinematics_timestamp: float = 0.0
    landed_state_timestamp: float = 0.0
    vertical_velocity_mps: float = 0.0
    relative_altitude_m: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    statustext: Tuple[str, ...] = ()
    esc: Tuple[EscTelemetry, ...] = field(default_factory=default_esc_telemetry)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attitude_rpy", tuple(self.attitude_rpy))
        object.__setattr__(self, "angular_velocity", tuple(self.angular_velocity))
        object.__setattr__(self, "local_position", tuple(self.local_position))
        object.__setattr__(self, "local_velocity", tuple(self.local_velocity))
        object.__setattr__(self, "rc_channels", frozen_mapping(self.rc_channels))
        object.__setattr__(self, "esc_rpm", frozen_mapping(self.esc_rpm))
        object.__setattr__(self, "esc_online", frozen_mapping(self.esc_online))
        object.__setattr__(self, "esc_raw_present_slots", tuple(self.esc_raw_present_slots))
        object.__setattr__(self, "statustext", tuple(self.statustext))
        object.__setattr__(self, "esc", tuple(self.esc))

    @property
    def maximum_esc_rpm(self) -> float:
        raw_values = [item.rpm for item in self.esc]
        raw_values.extend(self.esc_rpm.values())
        values = []
        for raw in raw_values:
            value = finite_real(raw)
            if value is None:
                return math.inf
            values.append(abs(value))
        return max(values, default=0.0)


@dataclass(frozen=True)
class F446Status:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    raw_state: str = "UNKNOWN"
    configuration: str = "UNKNOWN"
    duty: int = 0
    manual_limit: int = 0
    sense_mode: str = "unknown"
    r_is_raw: int = 0
    r_is_mv: int = 0
    l_is_raw: int = 0
    l_is_mv: int = 0
    used_raw: int = 0
    used_mv: int = 0
    threshold_raw: int = 0
    threshold_mv: int = 0
    blanking_ms: int = 0
    overcurrent_ms: int = 0
    timeout_ms: int = 0
    over_active: bool = False
    fault_message: Optional[str] = None

    # Compatibility/detail fields used by existing guards and diagnostics.
    state: F446State = F446State.UNKNOWN
    r_is_adc: Optional[int] = None
    l_is_adc: Optional[int] = None
    used_current_adc: Optional[int] = None
    threshold_adc: Optional[int] = None
    auto_status: Optional[bool] = None
    raw_lines: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_lines", tuple(self.raw_lines))

    @property
    def faulted(self) -> bool:
        return self.state is F446State.FAULT or self.fault_message is not None


class LowCmdOwnershipState(str, Enum):
    """Lifecycle of the process that may be the sole ``rt/lowcmd`` writer."""

    DISABLED = "DISABLED"
    DISCONNECTED = "DISCONNECTED"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    ACQUIRING = "ACQUIRING"
    HOLDING = "HOLDING"
    MPC_ACTIVE = "MPC_ACTIVE"
    SAFE_HOLD = "SAFE_HOLD"
    RELEASING = "RELEASING"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Go2MotorFeedback:
    """Public LowState feedback available for one mapped Go2 joint.

    The public Unitree LowState contract does not expose per-joint current, so
    no fabricated current field is included here.  ``tau_est_nm`` remains an
    estimate reported by the robot and is not treated as a current measurement.
    """

    motor_id: int = -1
    joint_name: str = "UNKNOWN"
    q_rad: Optional[float] = None
    dq_rad_s: Optional[float] = None
    tau_est_nm: Optional[float] = None
    temperature_c: Optional[float] = None
    lost: bool = True
    timestamp: float = 0.0


@dataclass(frozen=True)
class Go2FootForceFeedback:
    """Uncalibrated force-related fields from one Go2 ``LowState`` frame.

    Unitree's public Go2 IDL exposes two signed ``int16[4]`` arrays named
    ``foot_force`` and ``foot_force_est``.  This DTO deliberately preserves
    those SDK-indexed integers without labelling either array as newtons or
    assigning anatomical foot names: neither conversion is specified by the
    public Go2 IDL.  A separately reviewed mapping and robot-specific
    calibration are therefore required before this data can enter the
    impact-aware controller's ``normal_forces_n`` input.

    ``receipt_timestamp_s`` is the host monotonic callback-ingress time, not a
    robot timestamp.  ``source_tick`` is the uint32 value carried by LowState;
    its period and clock relation must be measured on the target robot.
    """

    receipt_timestamp_s: float = 0.0
    receipt_sequence: int = 0
    subscription_generation: int = 0
    source_tick: Optional[int] = None
    source_tick_valid: bool = False
    source_tick_monotonic: bool = False
    raw_sdk_int16: Tuple[int, int, int, int] = (0, 0, 0, 0)
    estimated_sdk_int16: Tuple[int, int, int, int] = (0, 0, 0, 0)
    raw_valid: bool = False
    estimated_valid: bool = False

    def __post_init__(self) -> None:
        timestamp = finite_real(self.receipt_timestamp_s)
        if timestamp is None or timestamp < 0.0:
            raise ValueError("receipt_timestamp_s must be finite and nonnegative")
        for name in ("receipt_sequence", "subscription_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.source_tick is not None and (
            isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 0
            or self.source_tick > 0xFFFFFFFF
        ):
            raise ValueError("source_tick must be a uint32 or None")
        for name in (
            "source_tick_valid",
            "source_tick_monotonic",
            "raw_valid",
            "estimated_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.source_tick_valid != (self.source_tick is not None):
            raise ValueError("source_tick_valid must agree with source_tick presence")
        if self.source_tick_monotonic and not self.source_tick_valid:
            raise ValueError("a monotonic source tick must also be valid")
        for name in ("raw_sdk_int16", "estimated_sdk_int16"):
            values = tuple(getattr(self, name))
            if len(values) != 4 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -32768
                or value > 32767
                for value in values
            ):
                raise ValueError(f"{name} must contain exactly four signed int16 values")
            object.__setattr__(self, name, values)

    @property
    def source_identity_valid(self) -> bool:
        """Whether this callback has a usable, forward LowState identity."""

        return bool(
            self.receipt_sequence > 0
            and self.subscription_generation > 0
            and self.source_tick_valid
            and self.source_tick_monotonic
        )


@dataclass(frozen=True)
class Go2LowLevelStatus:
    """Fail-closed status snapshot for low-level ownership and its watchdog."""

    timestamp: float = 0.0
    connected: bool = False
    ownership_state: LowCmdOwnershipState = LowCmdOwnershipState.DISABLED
    owner_epoch: int = 0
    healthy: bool = False
    low_state_timestamp: float = 0.0
    low_state_age_s: float = math.inf
    # ``target_sequence`` is retained as the legacy mailbox identity.  New
    # code must use the explicit staged/enqueued/applied fields below so a
    # mailbox replacement cannot be mistaken for physical execution.
    target_sequence: Optional[int] = None
    target_age_s: Optional[float] = None
    target_deadline: Optional[float] = None
    mailbox_staged_target_sequence: Optional[int] = None
    writer_enqueued_target_sequence: Optional[int] = None
    actuator_applied_target_sequence: Optional[int] = None
    # Monotonic local writer identity and the exact q values after software
    # position/slew/torque-envelope limiting that were accepted by DDS Write.
    # This is writer-enqueue evidence, not motor-side application evidence.
    writer_enqueue_generation: int = 0
    writer_enqueued_q_rad: Tuple[float, ...] = ()
    # True from successful ChannelPublisher construction until its Close()
    # returns without exception. A retained epoch with this flag false is a
    # handback-only recovery transaction: LowCmd cannot be restarted, so the
    # exact MotionSwitcher restore must remain possible if LowState disappears.
    publisher_active: bool = False
    writer_alive: bool = False
    last_write_timestamp: Optional[float] = None
    watchdog_healthy: bool = False
    safe_hold_active: bool = False
    safe_hold_settled: bool = False
    safe_hold_request_generation: int = 0
    safe_hold_write_generation: int = 0
    last_safe_hold_write_timestamp: Optional[float] = None
    high_level_released: bool = False
    high_level_restore_form: Optional[str] = None
    high_level_restore_mode: Optional[str] = None
    network_exclusivity_verified: bool = False
    # The current DDS graph verifier is an acquisition-time observation only.
    # These capability flags remain false until independent target-side
    # implementations provide continuous owner monitoring, an independent
    # watchdog, and motor-side command application feedback.
    continuous_owner_monitoring_active: bool = False
    independent_watchdog_active: bool = False
    writer_enqueue_ack_available: bool = False
    actuator_application_ack_available: bool = False
    mapping_hash_verified: bool = False
    active_mapping_hash: Optional[str] = None
    fault_reason: Optional[str] = "low-level control disabled"
    motors: Tuple[Go2MotorFeedback, ...] = ()
    foot_force_feedback: Go2FootForceFeedback = field(default_factory=Go2FootForceFeedback)
    # The paired tracking sample is computed only from a LowState callback
    # that entered after the referenced LowCmd write was enqueued.  It is not
    # the raw MPC target and must not be confused with the per-cycle slew
    # limit or the safe-hold settling tolerance.
    tracking_error_timestamp: float = 0.0
    tracking_reference_write_timestamp: float = 0.0
    tracking_reference_write_generation: int = 0
    tracking_reference_q_rad: Tuple[float, ...] = ()
    position_error_rad: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.foot_force_feedback, Go2FootForceFeedback):
            raise TypeError("foot_force_feedback must be Go2FootForceFeedback")
        object.__setattr__(self, "motors", tuple(self.motors))
        object.__setattr__(
            self,
            "tracking_reference_q_rad",
            tuple(self.tracking_reference_q_rad),
        )
        if (
            isinstance(self.writer_enqueue_generation, bool)
            or not isinstance(self.writer_enqueue_generation, int)
            or self.writer_enqueue_generation < 0
        ):
            raise ValueError("writer_enqueue_generation must be a nonnegative integer")
        writer_q = tuple(self.writer_enqueued_q_rad)
        if writer_q and (
            len(writer_q) != 12
            or any(finite_real(value) is None for value in writer_q)
        ):
            raise ValueError("writer_enqueued_q_rad must be empty or contain 12 finite values")
        if (self.writer_enqueue_generation == 0) is not (len(writer_q) == 0):
            raise ValueError(
                "writer_enqueue_generation and writer_enqueued_q_rad must appear together"
            )
        for name in (
            "writer_enqueue_ack_available",
            "actuator_application_ack_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in (
            "writer_enqueued_target_sequence",
            "actuator_applied_target_sequence",
        ):
            sequence = getattr(self, name)
            if sequence is not None and (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                raise ValueError(f"{name} must be None or a nonnegative integer")
        if self.writer_enqueued_target_sequence is not None and (
            not self.writer_enqueue_ack_available
            or self.writer_enqueue_generation <= 0
            or len(writer_q) != 12
        ):
            raise ValueError(
                "writer target identity requires writer capability, generation, and limited q"
            )
        if (
            self.actuator_applied_target_sequence is not None
            and not self.actuator_application_ack_available
        ):
            raise ValueError(
                "actuator target identity requires actuator application acknowledgement capability"
            )
        object.__setattr__(self, "writer_enqueued_q_rad", writer_q)
        object.__setattr__(self, "position_error_rad", tuple(self.position_error_rad))

    @property
    def owns_lowcmd(self) -> bool:
        # Transitional/fault states can still own the arbiter epoch and may
        # still be publishing.  Callers must never infer that SportClient is
        # safe merely because the writer's health state is FAULT.
        return self.owner_epoch > 0 and self.ownership_state in {
            LowCmdOwnershipState.ACQUIRING,
            LowCmdOwnershipState.HOLDING,
            LowCmdOwnershipState.MPC_ACTIVE,
            LowCmdOwnershipState.SAFE_HOLD,
            LowCmdOwnershipState.RELEASING,
            LowCmdOwnershipState.FAULT,
        }

    @property
    def ownership_pending(self) -> bool:
        """Whether SportClient use or process exit must remain inhibited.

        This deliberately treats internally inconsistent snapshots as owned:
        an epoch, a live writer, a released high-level service, or any owner
        lifecycle state is enough to fail closed.
        """

        return (
            self.owner_epoch != 0
            or self.publisher_active
            or self.writer_alive
            or self.high_level_released
            or self.safe_hold_active
            or self.safe_hold_settled
            or self.watchdog_healthy
            or self.network_exclusivity_verified
            or self.target_sequence is not None
            or self.mailbox_staged_target_sequence is not None
            or self.writer_enqueued_target_sequence is not None
            or self.actuator_applied_target_sequence is not None
            or self.target_deadline is not None
            or self.ownership_state
            not in {
                LowCmdOwnershipState.DISABLED,
                LowCmdOwnershipState.DISCONNECTED,
                LowCmdOwnershipState.OBSERVE_ONLY,
            }
        )


@dataclass(frozen=True)
class Go2ControlAuthorityStatus:
    """Manager-owned state for the SportMode/LowCmd authority transaction."""

    state: Go2ControlAuthorityState = Go2ControlAuthorityState.UNKNOWN
    timestamp: float = 0.0
    transition_started_at: Optional[float] = None
    transition_deadline: Optional[float] = None
    generation: int = 0
    ownership_epoch: int = 0
    reason: str = "control authority has not been established"

    def __post_init__(self) -> None:
        if not isinstance(self.state, Go2ControlAuthorityState):
            raise TypeError("state must be a Go2ControlAuthorityState")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")
        if isinstance(self.ownership_epoch, bool) or not isinstance(self.ownership_epoch, int):
            raise TypeError("ownership_epoch must be an integer")
        if self.generation < 0 or self.ownership_epoch < 0:
            raise ValueError("authority generations and epochs cannot be negative")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        transition_fields = (self.transition_started_at, self.transition_deadline)
        if (transition_fields[0] is None) != (transition_fields[1] is None):
            raise ValueError("authority transition start and deadline must appear together")
        if transition_fields[0] is not None:
            started = finite_real(transition_fields[0])
            deadline = finite_real(transition_fields[1])
            if started is None or deadline is None or deadline <= started:
                raise ValueError("authority transition deadline must follow its start")

    @property
    def transition_pending(self) -> bool:
        return self.state in {
            Go2ControlAuthorityState.LOWCMD_ACQUIRING,
            Go2ControlAuthorityState.HIGH_LEVEL_REACQUIRING,
        }


@dataclass(frozen=True)
class Go2Status:
    # Public specification fields.
    timestamp: float = 0.0
    connected: bool = False
    message_age_s: float = 0.0
    body_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    standing: bool = True
    moving: bool = False
    locomotion_mode: str = "UNKNOWN"
    fault_code: int = 0
    joints_locked: bool = False
    foot_force: Tuple[int, int, int, int] = (0, 0, 0, 0)
    foot_force_valid: bool = False

    # Compatibility/detail fields used by Phase 1 high-level motion guards.
    velocity_mps: float = 0.0
    stable: bool = True
    controller_active: bool = False
    low_level_status: Go2LowLevelStatus = field(default_factory=Go2LowLevelStatus)
    control_authority: Go2ControlAuthorityStatus = field(default_factory=Go2ControlAuthorityStatus)

    def __post_init__(self) -> None:
        object.__setattr__(self, "body_velocity", tuple(self.body_velocity))
        object.__setattr__(self, "body_rpy", tuple(self.body_rpy))
        object.__setattr__(self, "foot_force", tuple(self.foot_force))


@dataclass(frozen=True)
class RCStatus:
    connected: bool = False
    failsafe: bool = True
    channels: Mapping[int, int] = field(default_factory=dict)
    flight_enable: bool = False
    morphology_request: MorphologyRequest = MorphologyRequest.UNKNOWN
    auto_landing_request: AutoLandingRequest = AutoLandingRequest.MANUAL
    manual_override: bool = False
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", frozen_mapping(self.channels))


@dataclass(frozen=True)
class OperatorRequest:
    timestamp: float = 0.0
    flight_enable: bool = False
    morphology_request: str = "UNKNOWN"
    auto_landing_request: str = "MANUAL"
    manual_override: bool = False


@dataclass(frozen=True)
class LandingEstimate:
    valid: bool = False
    ground_detected: bool = False
    height_m: Optional[float] = None
    vertical_velocity_mps: Optional[float] = None
    horizontal_velocity_mps: Optional[float] = None
    timestamp: float = 0.0
    reason: str = "not initialized"


@dataclass(frozen=True)
class LandingCommand:
    timestamp: float = 0.0
    valid: bool = False
    vx_des: float = 0.0
    vy_des: float = 0.0
    vz_des: float = 0.0
    yaw_rate_des: float = 0.0
    reason: str = "inactive"


@dataclass(frozen=True)
class ImpactLandingRecoveryEvidence:
    """Fresh, identity-bound proof that normal paper-control recovery finished.

    The default value is deliberately unconfirmed.  A production composition
    root must create this from the high-rate loop, the asynchronous MPC worker,
    the FC residual sink and the exact LowCmd ownership epoch; Pixhawk
    ``landed`` alone can never create it.
    """

    timestamp: float = 0.0
    valid_until: float = 0.0
    landing_session_id: int = 0
    sequence: int = 0
    phase: ImpactLandingPhase = ImpactLandingPhase.INACTIVE
    healthy: bool = False
    controller_quiesced: bool = False
    recovery_complete: bool = False
    load_transfer_complete: bool = False
    body_state_stable: bool = False
    contacts: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    admittance_blends: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    go2_ownership_epoch: int = 0
    contact_epoch: int = 0
    residual_zero_acknowledged: bool = False
    residual_zero_ack_timestamp: float = 0.0
    residual_zero_execution_timestamp: float = 0.0
    residual_zero_status_timestamp: float = 0.0
    fc_session_id: Optional[int] = None
    fc_control_epoch: Optional[int] = None
    fc_transport_generation: Optional[int] = None
    last_residual_command_sequence: Optional[int] = None
    clear_through_command_sequence: Optional[int] = None
    residual_register_inactive: bool = False
    baseline_controller_retained: bool = False
    reason: str = "post-touchdown recovery has not been attested"

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ImpactLandingPhase):
            raise TypeError("phase must be an ImpactLandingPhase")
        for name in (
            "healthy",
            "controller_quiesced",
            "recovery_complete",
            "load_transfer_complete",
            "body_state_stable",
            "residual_zero_acknowledged",
            "residual_register_inactive",
            "baseline_controller_retained",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in (
            "landing_session_id",
            "sequence",
            "go2_ownership_epoch",
            "contact_epoch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "fc_session_id",
            "fc_control_epoch",
            "fc_transport_generation",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when present")
        for name in (
            "last_residual_command_sequence",
            "clear_through_command_sequence",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer when present")
        contacts = tuple(self.contacts)
        if len(contacts) != 4 or any(type(value) is not bool for value in contacts):
            raise TypeError("contacts must contain exactly four booleans")
        object.__setattr__(self, "contacts", cast(Tuple[bool, bool, bool, bool], contacts))
        blends = tuple(self.admittance_blends)
        if len(blends) != 4:
            raise ValueError("admittance_blends must contain exactly four values")
        normalized_blends = []
        for value in blends:
            normalized = finite_real(value)
            if normalized is None or not 0.0 <= normalized <= 1.0:
                raise ValueError("admittance_blends entries must be finite and in [0, 1]")
            normalized_blends.append(normalized)
        object.__setattr__(
            self,
            "admittance_blends",
            cast(Tuple[float, float, float, float], tuple(normalized_blends)),
        )
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")

    @property
    def confirmed(self) -> bool:
        watermark_safe = self.clear_through_command_sequence is not None and (
            self.last_residual_command_sequence is None
            or self.clear_through_command_sequence >= self.last_residual_command_sequence
        )
        identities_present = all(
            value is not None
            for value in (
                self.fc_session_id,
                self.fc_control_epoch,
                self.fc_transport_generation,
            )
        )
        temporal_order = (
            math.isfinite(self.timestamp)
            and math.isfinite(self.valid_until)
            and math.isfinite(self.residual_zero_ack_timestamp)
            and math.isfinite(self.residual_zero_execution_timestamp)
            and math.isfinite(self.residual_zero_status_timestamp)
            and 0.0
            < self.residual_zero_ack_timestamp
            <= self.residual_zero_execution_timestamp
            <= self.residual_zero_status_timestamp
            <= self.timestamp
            < self.valid_until
        )
        return bool(
            self.landing_session_id > 0
            and self.sequence > 0
            and self.contact_epoch > 0
            and self.phase is ImpactLandingPhase.POST_TOUCHDOWN_RECOVERY
            and self.healthy
            and self.controller_quiesced
            and self.recovery_complete
            and self.load_transfer_complete
            and self.body_state_stable
            and all(self.contacts)
            and all(value >= 1.0 - 1.0e-9 for value in self.admittance_blends)
            and self.residual_zero_acknowledged
            and self.residual_register_inactive
            and self.baseline_controller_retained
            and identities_present
            and watermark_safe
            and temporal_order
        )


@dataclass(frozen=True)
class SystemSnapshot:
    timestamp: float
    state: SystemState
    pixhawk: PixhawkStatus = field(default_factory=PixhawkStatus)
    f446: F446Status = field(default_factory=F446Status)
    go2: Go2Status = field(default_factory=Go2Status)
    operator: OperatorRequest = field(default_factory=OperatorRequest)
    rc: RCStatus = field(default_factory=RCStatus)
    configuration: Configuration = Configuration.UNKNOWN
    landing_estimate: LandingEstimate = field(default_factory=LandingEstimate)
    autoland_active: bool = False
    external_setpoint_active: bool = False
    maintenance_mode: bool = False
    joint_lock_confirmed: bool = False
    joint_lock_source: str = "none"
    ground_arm_authorized: bool = False
    ground_arm_authorization_expires_at: Optional[float] = None
    active_fault_codes: Tuple[str, ...] = ()
    configuration_source: str = "unconfirmed"
    impact_landing_session_id: int = 0
    impact_recovery: ImpactLandingRecoveryEvidence = field(
        default_factory=ImpactLandingRecoveryEvidence
    )
    impact_recovery_wait_started_at: Optional[float] = None
    impact_recovery_finalization_started_at: Optional[float] = None
    post_touchdown_stable_since: Optional[float] = None
    post_touchdown_last_stability_check_at: Optional[float] = None
    post_touchdown_stable_dwell_complete: bool = False
    impact_landing_exit_ready: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_fault_codes", tuple(self.active_fault_codes))
        if self.go2.connected and self.go2.joints_locked:
            object.__setattr__(self, "joint_lock_confirmed", True)
            object.__setattr__(self, "joint_lock_source", "telemetry")
        elif self.joint_lock_source == "telemetry":
            object.__setattr__(self, "joint_lock_confirmed", False)
            object.__setattr__(self, "joint_lock_source", "none")
        elif self.joint_lock_confirmed and self.joint_lock_source == "none":
            object.__setattr__(self, "joint_lock_source", "operator")
        elif not self.joint_lock_confirmed:
            object.__setattr__(self, "joint_lock_source", "none")

    def with_state(self, state: SystemState, timestamp: Optional[float] = None) -> SystemSnapshot:
        return replace(
            self, state=state, timestamp=self.timestamp if timestamp is None else timestamp
        )


@dataclass(frozen=True)
class SafetyViolation:
    code: str
    severity: SafetySeverity
    message: str
    recommended_action: str
    timestamp: float


@dataclass(frozen=True)
class TransitionRecord:
    timestamp: float
    previous_state: SystemState
    new_state: SystemState
    reason: str
    permitted: bool
    guard_codes: Tuple[str, ...] = ()
    entry_action_error: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "guard_codes", tuple(self.guard_codes))


@dataclass(frozen=True)
class F446Event:
    event_type: F446EventType
    line: str
    timestamp: float
    state: Optional[F446State] = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", frozen_mapping(self.values))


@dataclass(frozen=True)
class HealthReport:
    healthy: bool
    checks: Mapping[str, bool]
    messages: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", frozen_mapping(self.checks))
        object.__setattr__(self, "messages", tuple(self.messages))


def snapshot_to_dict(snapshot: SystemSnapshot) -> Dict[str, Any]:
    """Convert a snapshot to JSON-compatible primitives without mutating it."""

    def convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value if isinstance(value.value, str) else value.name
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, set, frozenset)):
            return [convert(item) for item in value]
        if is_dataclass(value):
            return {
                item.name: convert(getattr(value, item.name)) for item in fields(cast(Any, value))
            }
        return value

    return cast(Dict[str, Any], convert(snapshot))
