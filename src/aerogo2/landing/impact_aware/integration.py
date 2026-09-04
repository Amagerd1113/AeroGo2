"""Typed, fail-closed seams between the paper controller and AeroGo2 bridges.

The Unitree ``LowCmd`` path is implemented behind a separate exclusive-owner
interface and a low-rate executor.  This module also freezes and validates the
host side of the flight-controller rotor-residual protocol.  A real transport
still requires matching flight-controller firmware; the ordinary Pixhawk
velocity/actuator APIs are deliberately not reinterpreted as this interface.

中文说明：本文件冻结腿部命令与飞控旋翼残差的边界协议。旋翼侧唯一允许的语义是
``u_final = u_fc + delta_u_applied``，其中 ``delta_u_applied`` 已在主机侧乘过
安全系数 κ，飞控不得再次缩放或重复叠加。协议同时绑定会话、目标 tick、基线版本、
序号、TTL、ACK 和执行回读；过期或身份不一致时只清零残差，飞控自身姿态控制基线
必须继续工作。当前普通 MAVLink/PX4 接口不自动满足这一契约。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Callable, Optional, Protocol, Tuple

from aerogo2.common.async_utils import await_nonabandonable
from aerogo2.common.enums import ImpactLandingPhase, SystemState
from aerogo2.common.models import (
    Go2LowLevelStatus,
    ImpactLandingRecoveryEvidence,
    LowCmdOwnershipState,
)
from aerogo2.common.results import OperationResult

FC_RESIDUAL_PROTOCOL_VERSION = 1
FC_RESIDUAL_THRUST_UNIT = "N"
FC_RESIDUAL_ROTOR_ORDER = ("RR", "LF", "LR", "RF")
FC_RESIDUAL_REPLACE_SEMANTICS = "replace_register_with_already_kappa_scaled_residual_add_once"
_UINT64_MAX = (1 << 64) - 1
_SHA256_PREFIX = "sha256:"


def phase_for_system_state(state: SystemState) -> ImpactLandingPhase:
    """Return the only existing FSM state authorized to run paper control.

    All three physical paper phases must occur while ``AUTO_LANDING`` remains
    active and the aircraft may still be armed.  ``TOUCHDOWN_VERIFY`` stops
    the existing external controller, while ``LANDING_COMPLIANT`` requires a
    disarmed Pixhawk and zero ESC RPM, so neither may carry rotor residuals.
    The contact detector refines ``AUTO_LANDING`` into touchdown/recovery.
    """

    if state is SystemState.AUTO_LANDING:
        return ImpactLandingPhase.PRE_TOUCHDOWN
    return ImpactLandingPhase.INACTIVE


def _finite_tuple(name: str, values: object, length: int) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be an iterable of finite numbers")
    raw = tuple(values)
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in raw):
        raise TypeError(f"{name} must contain real numeric values")
    result = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_integer(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    if value > _UINT64_MAX:
        raise ValueError(f"{name} must fit an unsigned 64-bit field")
    return value


def _nonnegative_tuple(name: str, values: object, length: int) -> Tuple[float, ...]:
    result = _finite_tuple(name, values, length)
    if any(value < 0.0 for value in result):
        raise ValueError(f"{name} cannot contain negative values")
    return result


def _boolean_tuple(name: str, values: object, length: int) -> Tuple[bool, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be an iterable of booleans")
    result = tuple(values)
    if len(result) != length or any(type(value) is not bool for value in result):
        raise TypeError(f"{name} must contain exactly {length} booleans")
    return result


def _sha256_identity(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    digest = value[len(_SHA256_PREFIX) :] if value.startswith(_SHA256_PREFIX) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must use sha256:<64 lowercase hexadecimal digits>")
    return value


def _wire_digest(payload: bytes) -> str:
    return f"{_SHA256_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class Go2JointPositionCommand:
    """One short-lived joint frame for the dedicated LowCmd owner.

    ``sequence`` identifies the high-rate frame.  The optional source fields
    identify the slower force/residual policy from which the frame was
    generated; they are observability metadata and are deliberately not
    conflated with the writer sequence.

    中文：``joint_positions_rad`` 固定为 12 关节算法顺序；真正转换到 SDK motor ID、
    零位和方向由唯一 LowCmd owner 完成。该对象只携带短寿命目标和期望足力日志，
    不携带 CRC/Kp/Kd/tau，因为这些属于固定周期发送器的最终安全层。
    """

    sequence: int
    timestamp_s: float
    valid_until_s: float
    joint_positions_rad: Tuple[float, ...]
    desired_contact_forces_world_n: Tuple[float, ...]
    source_policy_sequence: Optional[int] = None
    source_policy_generation: Optional[int] = None
    source_contact_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        timestamp = _finite_real("timestamp_s", self.timestamp_s)
        valid_until = _finite_real("valid_until_s", self.valid_until_s)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "valid_until_s", valid_until)
        if valid_until <= timestamp:
            raise ValueError("valid_until_s must be later than timestamp_s")
        object.__setattr__(
            self,
            "joint_positions_rad",
            _finite_tuple("joint_positions_rad", self.joint_positions_rad, 12),
        )
        object.__setattr__(
            self,
            "desired_contact_forces_world_n",
            _finite_tuple(
                "desired_contact_forces_world_n",
                self.desired_contact_forces_world_n,
                12,
            ),
        )
        source_values = (
            self.source_policy_sequence,
            self.source_policy_generation,
            self.source_contact_epoch,
        )
        if any(value is not None for value in source_values):
            if self.source_contact_epoch is None:
                raise ValueError("source_contact_epoch is required with policy metadata")
            _strict_integer("source_contact_epoch", self.source_contact_epoch)
            if self.source_policy_sequence is None:
                if self.source_policy_generation is not None:
                    raise ValueError("source_policy_generation requires source_policy_sequence")
            else:
                _strict_integer("source_policy_sequence", self.source_policy_sequence)
                if self.source_policy_generation is None:
                    raise ValueError(
                        "source_policy_generation is required with source_policy_sequence"
                    )
                _strict_integer(
                    "source_policy_generation",
                    self.source_policy_generation,
                )

    def is_fresh(self, now_s: float) -> bool:
        return (
            not isinstance(now_s, bool)
            and isinstance(now_s, Real)
            and math.isfinite(float(now_s))
            and self.timestamp_s <= float(now_s) < self.valid_until_s
        )


@dataclass(frozen=True)
class FlightControllerRotorResidualCommand:
    """Correction command reserved for one future flight-controller tick.

    ``applied_residual_thrusts_n`` is the only payload for the future
    FC-specific adapter: it already equals ``kappa * delta_u_mpc``.  The FC
    replaces its residual register with this vector and adds that register
    exactly once to its own baseline; it must not multiply by
    ``correction_gain`` again and must never accumulate from a previous final
    output.  ``fc_session_id`` rejects a reboot/reconnect replay, while
    ``target_fc_tick`` and ``baseline_version`` bind the residual to an FC-side
    reference snapshot produced before this command; execution feedback carries
    the live same-tick baseline separately.  At positive gain,
    ``transport_raw_residual_thrusts_n`` is algebraic provenance whose scaled
    value equals that applied residual; it is not a separately optimized
    kappa=1 solution.  At exactly zero gain it is ``None`` so logs cannot
    invent a transport target.

    中文：四元素顺序固定为 ``[RR, LF, LR, RF]``，单位固定为 N。正 κ 时必须满足
    ``applied_residual = κ * raw_residual`` 且
    ``applied_total = baseline + applied_residual``；κ=0 时 raw 目标刻意设为 None，
    防止日志或飞控把未授权的理论目标误当成可执行命令。
    """

    sequence: int
    timestamp_s: float
    valid_until_s: float
    fc_session_id: int
    target_fc_tick: int
    baseline_version: int
    baseline_timestamp_s: float
    baseline_thrusts_n: Tuple[float, ...]
    transport_raw_residual_thrusts_n: Optional[Tuple[float, ...]]
    applied_residual_thrusts_n: Tuple[float, ...]
    applied_total_thrusts_n: Tuple[float, ...]
    correction_gain: float
    transport_target_semantics: str
    thrust_unit: str = field(default=FC_RESIDUAL_THRUST_UNIT, init=False)
    rotor_order: Tuple[str, ...] = field(default=FC_RESIDUAL_ROTOR_ORDER, init=False)
    residual_application_semantics: str = field(
        default=FC_RESIDUAL_REPLACE_SEMANTICS,
        init=False,
    )
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        _strict_integer("sequence", self.sequence)
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("target_fc_tick", self.target_fc_tick, positive=True)
        _strict_integer("baseline_version", self.baseline_version, positive=True)
        timestamp = _finite_real("timestamp_s", self.timestamp_s)
        valid_until = _finite_real("valid_until_s", self.valid_until_s)
        baseline_timestamp = _finite_real(
            "baseline_timestamp_s",
            self.baseline_timestamp_s,
        )
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "valid_until_s", valid_until)
        object.__setattr__(self, "baseline_timestamp_s", baseline_timestamp)
        if valid_until <= timestamp:
            raise ValueError("valid_until_s must be later than timestamp_s")
        if baseline_timestamp > timestamp:
            raise ValueError("baseline_timestamp_s cannot be later than timestamp_s")
        correction_gain = _finite_real("correction_gain", self.correction_gain)
        object.__setattr__(self, "correction_gain", correction_gain)
        if not 0.0 <= correction_gain <= 1.0:
            raise ValueError("correction_gain must be within [0, 1]")
        if (
            not isinstance(self.transport_target_semantics, str)
            or not self.transport_target_semantics.strip()
        ):
            raise ValueError("transport_target_semantics must be a nonempty string")
        object.__setattr__(
            self,
            "baseline_thrusts_n",
            _nonnegative_tuple("baseline_thrusts_n", self.baseline_thrusts_n, 4),
        )
        object.__setattr__(
            self,
            "applied_residual_thrusts_n",
            _finite_tuple(
                "applied_residual_thrusts_n",
                self.applied_residual_thrusts_n,
                4,
            ),
        )
        object.__setattr__(
            self,
            "applied_total_thrusts_n",
            _nonnegative_tuple(
                "applied_total_thrusts_n",
                self.applied_total_thrusts_n,
                4,
            ),
        )
        tolerance = 1.0e-12
        raw = self.transport_raw_residual_thrusts_n
        if self.correction_gain == 0.0:
            if raw is not None:
                raise ValueError(
                    "zero correction_gain requires an undefined transport raw residual"
                )
            if self.transport_target_semantics != "zero_gain_no_transport_target":
                raise ValueError("zero correction_gain requires zero-gain transport semantics")
            if any(abs(item) > tolerance for item in self.applied_residual_thrusts_n):
                raise ValueError("an undefined zero-gain target requires a zero payload")
        else:
            if raw is None:
                raise ValueError("a positive correction_gain requires a transport raw residual")
            expected_semantics = (
                "active_gain_one_transport_target"
                if self.correction_gain == 1.0
                else "gain_limited_algebraic_reconstruction"
            )
            if self.transport_target_semantics != expected_semantics:
                raise ValueError(
                    "transport target semantics must exactly match the correction-gain mode"
                )
            raw = _finite_tuple("transport_raw_residual_thrusts_n", raw, 4)
            object.__setattr__(self, "transport_raw_residual_thrusts_n", raw)
        for index in range(4):
            if raw is not None:
                expected_scaled = self.correction_gain * raw[index]
                if not math.isclose(
                    self.applied_residual_thrusts_n[index],
                    expected_scaled,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        "applied_residual_thrusts_n must equal correction_gain * "
                        "transport_raw_residual_thrusts_n componentwise"
                    )
            expected_final = self.baseline_thrusts_n[index] + self.applied_residual_thrusts_n[index]
            if not math.isclose(
                self.applied_total_thrusts_n[index],
                expected_final,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    "applied_total_thrusts_n must equal baseline_thrusts_n + "
                    "applied_residual_thrusts_n componentwise"
                )

    def is_fresh(self, now_s: float) -> bool:
        return (
            not isinstance(now_s, bool)
            and isinstance(now_s, Real)
            and math.isfinite(float(now_s))
            and self.timestamp_s <= float(now_s) < self.valid_until_s
        )


@dataclass(frozen=True)
class CoordinatedLandingCommand:
    """Coherent command pair for a future synchronized transport.

    Matching metadata does not itself make two independent hardware writes
    atomic.  A real adapter must stage both sides, activate them at one agreed
    deadline, reject replayed sequences, and clear/revoke both on any failure.

    中文：腿与旋翼命令必须共享序号、时间戳和截止时间。这个数据类只证明逻辑配对，
    并不能让两个独立网络写入自动获得原子性；生产桥必须实现预备、共同激活和失败时
    双侧回退，不能简单先后调用两个 send。
    """

    phase: ImpactLandingPhase
    leg: Go2JointPositionCommand
    rotor: FlightControllerRotorResidualCommand
    solver_succeeded: bool
    solver_status: str
    solver_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ImpactLandingPhase):
            raise TypeError("phase must be an ImpactLandingPhase")
        if self.phase is ImpactLandingPhase.INACTIVE:
            raise ValueError("an active command cannot use the INACTIVE phase")
        if not isinstance(self.leg, Go2JointPositionCommand):
            raise TypeError("leg must be a Go2JointPositionCommand")
        if not isinstance(self.rotor, FlightControllerRotorResidualCommand):
            raise TypeError("rotor must be a FlightControllerRotorResidualCommand")
        if type(self.solver_succeeded) is not bool or not self.solver_succeeded:
            raise ValueError("an active command requires solver_succeeded=True")
        if self.leg.sequence != self.rotor.sequence:
            raise ValueError("leg and rotor commands must share one sequence")
        if self.leg.timestamp_s != self.rotor.timestamp_s:
            raise ValueError("leg and rotor commands must share one timestamp")
        if self.leg.valid_until_s != self.rotor.valid_until_s:
            raise ValueError("leg and rotor commands must share one validity deadline")
        solver_time = _finite_real("solver_time_s", self.solver_time_s)
        object.__setattr__(self, "solver_time_s", solver_time)
        if solver_time < 0.0:
            raise ValueError("solver_time_s must be finite and nonnegative")
        if not isinstance(self.solver_status, str) or not self.solver_status.strip():
            raise ValueError("solver_status cannot be empty")


class Go2LowLevelCommandSink(Protocol):
    """Future exclusive owner of Unitree ``rt/lowcmd`` at a fixed high rate."""

    async def send_joint_position_command(
        self,
        command: Go2JointPositionCommand,
    ) -> OperationResult: ...

    async def revoke_mpc_control(self, reason: str) -> OperationResult: ...


class FlightControllerResidualOperation(Enum):
    """Operations acknowledged independently from execution feedback."""

    STAGE = "stage"
    CLEAR = "clear"


class FlightControllerResidualExecutionResult(Enum):
    """Fixed fast-loop outcome; diagnostic text is never state-machine input."""

    APPLIED = "applied"
    CLEARED = "cleared"
    HEADROOM_REJECTED = "headroom_rejected"
    BASELINE_DEVIATION_REJECTED = "baseline_deviation_rejected"
    GATE_REJECTED = "gate_rejected"
    EXPIRED = "expired"


class FlightControllerResidualState(Enum):
    """What the host can prove about the FC residual register."""

    CONFIRMED_ZERO = "confirmed_zero"
    STAGE_PENDING = "stage_pending"
    ACTIVE = "active"
    CLEAR_PENDING = "clear_pending"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FlightControllerBaselineReservation:
    """FC reference snapshot and admission envelope for one future activation.

    The baseline vector is the reference used by the MPC, not a promise that a
    closed-loop attitude controller will produce that exact vector at a future
    tick.  At ``target_fc_tick`` the FC must form its live baseline and live
    headroom atomically, then apply the complete residual or zero.  The
    execution feedback reports that same-tick baseline separately.  Timestamps
    here are companion-monotonic observations produced by the transport after
    a bounded TIMESYNC conversion; FC-local activation/expiry uses tick IDs.
    """

    fc_session_id: int
    control_epoch: int
    transport_generation: int
    timesync_generation: int
    target_fc_tick: int
    baseline_version: int
    timestamp_s: float
    valid_until_s: float
    baseline_thrusts_n: Tuple[float, ...]
    negative_headroom_n: Tuple[float, ...]
    positive_headroom_n: Tuple[float, ...]
    reference_digest: str = field(init=False)
    thrust_unit: str = field(default=FC_RESIDUAL_THRUST_UNIT, init=False)
    rotor_order: Tuple[str, ...] = field(default=FC_RESIDUAL_ROTOR_ORDER, init=False)
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        _strict_integer("timesync_generation", self.timesync_generation, positive=True)
        _strict_integer("target_fc_tick", self.target_fc_tick, positive=True)
        _strict_integer("baseline_version", self.baseline_version, positive=True)
        timestamp = _finite_real("timestamp_s", self.timestamp_s)
        deadline = _finite_real("valid_until_s", self.valid_until_s)
        if deadline <= timestamp:
            raise ValueError("valid_until_s must be later than timestamp_s")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "valid_until_s", deadline)
        object.__setattr__(
            self,
            "baseline_thrusts_n",
            _nonnegative_tuple("baseline_thrusts_n", self.baseline_thrusts_n, 4),
        )
        object.__setattr__(
            self,
            "negative_headroom_n",
            _nonnegative_tuple("negative_headroom_n", self.negative_headroom_n, 4),
        )
        object.__setattr__(
            self,
            "positive_headroom_n",
            _nonnegative_tuple("positive_headroom_n", self.positive_headroom_n, 4),
        )
        digest_payload = struct.pack(
            ">7Q14d",
            self.protocol_version,
            self.fc_session_id,
            self.control_epoch,
            self.transport_generation,
            self.timesync_generation,
            self.target_fc_tick,
            self.baseline_version,
            self.timestamp_s,
            self.valid_until_s,
            *self.baseline_thrusts_n,
            *self.negative_headroom_n,
            *self.positive_headroom_n,
        )
        object.__setattr__(self, "reference_digest", _wire_digest(digest_payload))

    def is_fresh(self, now_s: float) -> bool:
        return (
            not isinstance(now_s, bool)
            and isinstance(now_s, Real)
            and math.isfinite(float(now_s))
            and self.timestamp_s <= float(now_s) < self.valid_until_s
        )


@dataclass(frozen=True)
class FlightControllerResidualTransportStatus:
    """Evidence required before the host may stage any non-baseline residual."""

    timestamp_s: float
    connected: bool
    healthy: bool
    transport_generation: int
    fc_session_id: Optional[int]
    control_epoch: Optional[int]
    current_fc_tick: Optional[int]
    fc_tick_period_s: Optional[float]
    timesync_generation: Optional[int]
    timesync_age_s: Optional[float]
    firmware_watchdog_timeout_s: Optional[float]
    clock_sync_uncertainty_s: Optional[float]
    firmware_hash: str
    dialect_hash: str
    mixer_hash: str
    mapping_hash: str
    calibration_hash: str
    residual_register_active: bool
    active_command_sequence: Optional[int]
    active_request_digest: Optional[str]
    active_valid_until_fc_tick: Optional[int]
    pending_stage_present: bool
    pending_command_sequence: Optional[int]
    pending_request_digest: Optional[str]
    pending_target_fc_tick: Optional[int]
    clear_through_command_sequence: Optional[int]
    residual_enabled: bool
    baseline_controller_active: bool
    allocator_ready: bool
    replacement_semantics_verified: bool
    autonomous_expiry_clear_verified: bool
    disconnect_clear_verified: bool
    baseline_preservation_verified: bool
    execution_feedback_verified: bool
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)
    thrust_unit: str = field(default=FC_RESIDUAL_THRUST_UNIT, init=False)
    rotor_order: Tuple[str, ...] = field(default=FC_RESIDUAL_ROTOR_ORDER, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        for name in (
            "connected",
            "healthy",
            "residual_register_active",
            "pending_stage_present",
            "residual_enabled",
            "baseline_controller_active",
            "allocator_ready",
            "replacement_semantics_verified",
            "autonomous_expiry_clear_verified",
            "disconnect_clear_verified",
            "baseline_preservation_verified",
            "execution_feedback_verified",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.fc_session_id is not None:
            _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        if self.control_epoch is not None:
            _strict_integer("control_epoch", self.control_epoch, positive=True)
        if self.current_fc_tick is not None:
            _strict_integer("current_fc_tick", self.current_fc_tick)
        if self.fc_tick_period_s is not None:
            period = _finite_real("fc_tick_period_s", self.fc_tick_period_s)
            if period <= 0.0:
                raise ValueError("fc_tick_period_s must be positive")
            object.__setattr__(self, "fc_tick_period_s", period)
        if self.timesync_generation is not None:
            _strict_integer("timesync_generation", self.timesync_generation, positive=True)
        if self.timesync_age_s is not None:
            age = _finite_real("timesync_age_s", self.timesync_age_s)
            if age < 0.0:
                raise ValueError("timesync_age_s cannot be negative")
            object.__setattr__(self, "timesync_age_s", age)
        if self.firmware_watchdog_timeout_s is not None:
            timeout = _finite_real(
                "firmware_watchdog_timeout_s",
                self.firmware_watchdog_timeout_s,
            )
            if timeout <= 0.0:
                raise ValueError("firmware_watchdog_timeout_s must be positive")
            object.__setattr__(self, "firmware_watchdog_timeout_s", timeout)
        if self.clock_sync_uncertainty_s is not None:
            uncertainty = _finite_real(
                "clock_sync_uncertainty_s",
                self.clock_sync_uncertainty_s,
            )
            if uncertainty < 0.0:
                raise ValueError("clock_sync_uncertainty_s cannot be negative")
            object.__setattr__(self, "clock_sync_uncertainty_s", uncertainty)
        for name in (
            "firmware_hash",
            "dialect_hash",
            "mixer_hash",
            "mapping_hash",
            "calibration_hash",
        ):
            object.__setattr__(self, name, _sha256_identity(name, getattr(self, name)))
        if self.active_command_sequence is not None:
            _strict_integer("active_command_sequence", self.active_command_sequence)
        if self.active_request_digest is not None:
            object.__setattr__(
                self,
                "active_request_digest",
                _sha256_identity("active_request_digest", self.active_request_digest),
            )
        if self.active_valid_until_fc_tick is not None:
            _strict_integer(
                "active_valid_until_fc_tick",
                self.active_valid_until_fc_tick,
                positive=True,
            )
        active_metadata = (
            self.active_command_sequence,
            self.active_request_digest,
            self.active_valid_until_fc_tick,
        )
        if (
            self.residual_register_active
            and not all(value is not None for value in active_metadata)
        ) or (
            not self.residual_register_active
            and any(value is not None for value in active_metadata)
        ):
            raise ValueError(
                "active residual metadata must be complete exactly when the register is active"
            )
        if self.pending_command_sequence is not None:
            _strict_integer("pending_command_sequence", self.pending_command_sequence)
        if self.pending_request_digest is not None:
            object.__setattr__(
                self,
                "pending_request_digest",
                _sha256_identity("pending_request_digest", self.pending_request_digest),
            )
        if self.pending_target_fc_tick is not None:
            _strict_integer("pending_target_fc_tick", self.pending_target_fc_tick, positive=True)
        if self.clear_through_command_sequence is not None:
            _strict_integer(
                "clear_through_command_sequence",
                self.clear_through_command_sequence,
            )
        pending_metadata = (
            self.pending_command_sequence,
            self.pending_request_digest,
            self.pending_target_fc_tick,
        )
        if (
            self.pending_stage_present and not all(value is not None for value in pending_metadata)
        ) or (
            not self.pending_stage_present and any(value is not None for value in pending_metadata)
        ):
            raise ValueError(
                "pending stage metadata must be complete exactly when a stage is pending"
            )
        if self.clear_through_command_sequence is not None and (
            (
                self.active_command_sequence is not None
                and self.active_command_sequence <= self.clear_through_command_sequence
            )
            or (
                self.pending_command_sequence is not None
                and self.pending_command_sequence <= self.clear_through_command_sequence
            )
        ):
            raise ValueError("active/pending sequence cannot be covered by the clear watermark")


@dataclass(frozen=True)
class FlightControllerResidualStageRequest:
    """Wire-level SET payload; deliberately contains no raw target or kappa.

    The FC receives only the already gain-scaled residual and implements an
    idempotent *replacement* write keyed by session/sequence.  Repeating an
    identical packet may refresh neither its TTL nor its value; it can never
    increment the residual register.
    """

    fc_session_id: int
    control_epoch: int
    transport_generation: int
    sequence: int
    timestamp_s: float
    timesync_generation: int
    target_fc_tick: int
    valid_until_fc_tick: int
    baseline_version: int
    baseline_reference_digest: str
    applied_residual_thrusts_n: Tuple[float, ...]
    required_headroom_reserve_n: Tuple[float, ...]
    maximum_baseline_deviation_n: Tuple[float, ...]
    request_digest: str = field(init=False)
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)
    thrust_unit: str = field(default=FC_RESIDUAL_THRUST_UNIT, init=False)
    rotor_order: Tuple[str, ...] = field(default=FC_RESIDUAL_ROTOR_ORDER, init=False)
    application_semantics: str = field(default=FC_RESIDUAL_REPLACE_SEMANTICS, init=False)

    def __post_init__(self) -> None:
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        _strict_integer("sequence", self.sequence)
        _strict_integer("timesync_generation", self.timesync_generation, positive=True)
        _strict_integer("target_fc_tick", self.target_fc_tick, positive=True)
        _strict_integer("valid_until_fc_tick", self.valid_until_fc_tick, positive=True)
        if self.valid_until_fc_tick <= self.target_fc_tick:
            raise ValueError("valid_until_fc_tick must be later than target_fc_tick")
        _strict_integer("baseline_version", self.baseline_version, positive=True)
        object.__setattr__(
            self,
            "baseline_reference_digest",
            _sha256_identity(
                "baseline_reference_digest",
                self.baseline_reference_digest,
            ),
        )
        timestamp = _finite_real("timestamp_s", self.timestamp_s)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "applied_residual_thrusts_n",
            _finite_tuple(
                "applied_residual_thrusts_n",
                self.applied_residual_thrusts_n,
                4,
            ),
        )
        object.__setattr__(
            self,
            "required_headroom_reserve_n",
            _nonnegative_tuple(
                "required_headroom_reserve_n",
                self.required_headroom_reserve_n,
                4,
            ),
        )
        object.__setattr__(
            self,
            "maximum_baseline_deviation_n",
            _nonnegative_tuple(
                "maximum_baseline_deviation_n",
                self.maximum_baseline_deviation_n,
                4,
            ),
        )
        digest_payload = struct.pack(
            ">9Q13d",
            self.protocol_version,
            self.fc_session_id,
            self.control_epoch,
            self.transport_generation,
            self.sequence,
            self.timesync_generation,
            self.target_fc_tick,
            self.valid_until_fc_tick,
            self.baseline_version,
            self.timestamp_s,
            *self.applied_residual_thrusts_n,
            *self.required_headroom_reserve_n,
            *self.maximum_baseline_deviation_n,
        ) + self.baseline_reference_digest.encode("ascii")
        object.__setattr__(self, "request_digest", _wire_digest(digest_payload))


@dataclass(frozen=True)
class FlightControllerResidualClearRequest:
    """Idempotent request to clear only the additive residual register."""

    fc_session_id: int
    control_epoch: int
    transport_generation: int
    clear_sequence: int
    clear_through_command_sequence: int
    timestamp_s: float
    reason: str
    request_digest: str = field(init=False)
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        _strict_integer("clear_sequence", self.clear_sequence, positive=True)
        _strict_integer(
            "clear_through_command_sequence",
            self.clear_through_command_sequence,
        )
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a nonempty string")
        reason_bytes = self.reason.encode("utf-8")
        if len(reason_bytes) > 96:
            raise ValueError("reason cannot exceed 96 UTF-8 bytes")
        digest_payload = (
            struct.pack(
                ">6Qd",
                self.protocol_version,
                self.fc_session_id,
                self.control_epoch,
                self.transport_generation,
                self.clear_sequence,
                self.clear_through_command_sequence,
                self.timestamp_s,
            )
            + reason_bytes
        )
        object.__setattr__(self, "request_digest", _wire_digest(digest_payload))


@dataclass(frozen=True)
class FlightControllerResidualAck:
    """Receipt/staging ACK; it is never treated as proof of execution."""

    operation: FlightControllerResidualOperation
    operation_sequence: int
    command_sequence: Optional[int]
    fc_session_id: int
    control_epoch: int
    transport_generation: int
    target_fc_tick: Optional[int]
    valid_until_fc_tick: Optional[int]
    baseline_version: Optional[int]
    clear_through_command_sequence: Optional[int]
    request_digest: str
    timestamp_s: float
    accepted: bool
    result_code: str
    message: str
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, FlightControllerResidualOperation):
            raise TypeError("operation must be a FlightControllerResidualOperation")
        _strict_integer("operation_sequence", self.operation_sequence)
        if self.command_sequence is not None:
            _strict_integer("command_sequence", self.command_sequence)
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        if self.target_fc_tick is not None:
            _strict_integer("target_fc_tick", self.target_fc_tick, positive=True)
        if self.valid_until_fc_tick is not None:
            _strict_integer("valid_until_fc_tick", self.valid_until_fc_tick, positive=True)
        if self.baseline_version is not None:
            _strict_integer("baseline_version", self.baseline_version, positive=True)
        if self.clear_through_command_sequence is not None:
            _strict_integer(
                "clear_through_command_sequence",
                self.clear_through_command_sequence,
            )
        object.__setattr__(
            self,
            "request_digest",
            _sha256_identity("request_digest", self.request_digest),
        )
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a bool")
        if not isinstance(self.result_code, str) or not self.result_code.strip():
            raise ValueError("result_code must be a nonempty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.operation is FlightControllerResidualOperation.STAGE:
            if (
                self.command_sequence is None
                or self.target_fc_tick is None
                or self.valid_until_fc_tick is None
                or self.baseline_version is None
                or self.clear_through_command_sequence is not None
            ):
                raise ValueError("STAGE ACK metadata is incomplete or contains clear fields")
            if self.valid_until_fc_tick <= self.target_fc_tick:
                raise ValueError("STAGE ACK FC lease is empty")
        elif (
            self.command_sequence is not None
            or self.target_fc_tick is not None
            or self.valid_until_fc_tick is not None
            or self.baseline_version is not None
            or self.clear_through_command_sequence is None
        ):
            raise ValueError("CLEAR ACK metadata is incomplete or contains stage fields")


@dataclass(frozen=True)
class FlightControllerResidualExecutionFeedback:
    """Authoritative fast-loop readback after STAGE or CLEAR.

    ``residual_addition_terms`` is one for an active stage and zero after a
    clear.  It describes the algebra in each FC output evaluation, not the
    number of fast-loop ticks for which a short-lived residual is held.
    """

    operation: FlightControllerResidualOperation
    execution_result: FlightControllerResidualExecutionResult
    operation_sequence: int
    command_sequence: Optional[int]
    fc_session_id: int
    control_epoch: int
    transport_generation: int
    timesync_generation: int
    execution_fc_tick: int
    valid_until_fc_tick: Optional[int]
    baseline_version: int
    execution_baseline_version: int
    clear_through_command_sequence: Optional[int]
    request_digest: str
    timestamp_s: float
    baseline_thrusts_n: Tuple[float, ...]
    requested_residual_thrusts_n: Tuple[float, ...]
    applied_residual_thrusts_n: Tuple[float, ...]
    final_thrusts_n: Tuple[float, ...]
    negative_headroom_n: Tuple[float, ...]
    positive_headroom_n: Tuple[float, ...]
    required_headroom_reserve_n: Tuple[float, ...]
    maximum_baseline_deviation_n: Tuple[float, ...]
    saturation_mask: Tuple[bool, ...]
    saturation_scale: float
    residual_active: bool
    baseline_controller_active: bool
    residual_addition_terms: int
    protocol_version: int = field(default=FC_RESIDUAL_PROTOCOL_VERSION, init=False)
    thrust_unit: str = field(default=FC_RESIDUAL_THRUST_UNIT, init=False)
    rotor_order: Tuple[str, ...] = field(default=FC_RESIDUAL_ROTOR_ORDER, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, FlightControllerResidualOperation):
            raise TypeError("operation must be a FlightControllerResidualOperation")
        if not isinstance(self.execution_result, FlightControllerResidualExecutionResult):
            raise TypeError("execution_result must be a FlightControllerResidualExecutionResult")
        _strict_integer("operation_sequence", self.operation_sequence)
        if self.command_sequence is not None:
            _strict_integer("command_sequence", self.command_sequence)
        _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        _strict_integer("transport_generation", self.transport_generation, positive=True)
        _strict_integer("timesync_generation", self.timesync_generation, positive=True)
        _strict_integer("execution_fc_tick", self.execution_fc_tick)
        if self.valid_until_fc_tick is not None:
            _strict_integer("valid_until_fc_tick", self.valid_until_fc_tick, positive=True)
        _strict_integer("baseline_version", self.baseline_version, positive=True)
        _strict_integer(
            "execution_baseline_version",
            self.execution_baseline_version,
            positive=True,
        )
        if self.clear_through_command_sequence is not None:
            _strict_integer(
                "clear_through_command_sequence",
                self.clear_through_command_sequence,
            )
        object.__setattr__(
            self,
            "request_digest",
            _sha256_identity("request_digest", self.request_digest),
        )
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        object.__setattr__(
            self,
            "baseline_thrusts_n",
            _nonnegative_tuple("baseline_thrusts_n", self.baseline_thrusts_n, 4),
        )
        for name in ("requested_residual_thrusts_n", "applied_residual_thrusts_n"):
            object.__setattr__(self, name, _finite_tuple(name, getattr(self, name), 4))
        object.__setattr__(
            self,
            "final_thrusts_n",
            _nonnegative_tuple("final_thrusts_n", self.final_thrusts_n, 4),
        )
        for name in ("negative_headroom_n", "positive_headroom_n"):
            object.__setattr__(
                self,
                name,
                _nonnegative_tuple(name, getattr(self, name), 4),
            )
        object.__setattr__(
            self,
            "required_headroom_reserve_n",
            _nonnegative_tuple(
                "required_headroom_reserve_n",
                self.required_headroom_reserve_n,
                4,
            ),
        )
        object.__setattr__(
            self,
            "maximum_baseline_deviation_n",
            _nonnegative_tuple(
                "maximum_baseline_deviation_n",
                self.maximum_baseline_deviation_n,
                4,
            ),
        )
        object.__setattr__(
            self,
            "saturation_mask",
            _boolean_tuple("saturation_mask", self.saturation_mask, 4),
        )
        scale = _finite_real("saturation_scale", self.saturation_scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError("saturation_scale must be within [0, 1]")
        object.__setattr__(self, "saturation_scale", scale)
        for name in ("residual_active", "baseline_controller_active"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        _strict_integer("residual_addition_terms", self.residual_addition_terms)
        if self.residual_addition_terms not in (0, 1):
            raise ValueError("residual_addition_terms must be zero or one")
        # Algebraic equality is checked by the sink with the configured wire
        # quantisation/calibration tolerance.  Enforcing 1e-12 here would reject
        # ordinary MAVLink float32 round trips before policy can inspect them.
        if self.operation is FlightControllerResidualOperation.STAGE:
            if (
                self.command_sequence is None
                or self.valid_until_fc_tick is None
                or self.clear_through_command_sequence is not None
            ):
                raise ValueError("STAGE feedback metadata is incomplete")
            if self.execution_result is FlightControllerResidualExecutionResult.CLEARED:
                raise ValueError("STAGE feedback cannot use the CLEAR outcome")
            if (
                self.execution_result is FlightControllerResidualExecutionResult.APPLIED
                and self.valid_until_fc_tick <= self.execution_fc_tick
            ):
                raise ValueError("APPLIED feedback cannot describe an expired FC-tick lease")
            if (
                self.execution_result is FlightControllerResidualExecutionResult.EXPIRED
                and self.execution_fc_tick < self.valid_until_fc_tick
            ):
                raise ValueError("EXPIRED feedback must be sampled at or after lease expiry")
        elif (
            self.command_sequence is not None
            or self.valid_until_fc_tick is not None
            or self.clear_through_command_sequence is None
        ):
            raise ValueError("CLEAR feedback metadata is incomplete or contains stage fields")
        elif self.execution_result is not FlightControllerResidualExecutionResult.CLEARED:
            raise ValueError("CLEAR feedback must use the CLEARED outcome")


class FlightControllerResidualTransport(Protocol):
    """Low-level transport implemented only with matching custom FC firmware."""

    def status(self) -> FlightControllerResidualTransportStatus: ...

    def latest_baseline_reservation(
        self,
    ) -> Optional[FlightControllerBaselineReservation]: ...

    async def stage_residual(
        self,
        request: FlightControllerResidualStageRequest,
    ) -> FlightControllerResidualAck: ...

    async def clear_residual(
        self,
        request: FlightControllerResidualClearRequest,
    ) -> FlightControllerResidualAck: ...

    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback: ...


@dataclass(frozen=True)
class FlightControllerResidualSinkConfig:
    """Host-side deadlines and independent residual safety margins."""

    maximum_command_ttl_s: float
    maximum_baseline_age_s: float
    maximum_status_age_s: float
    acknowledgement_timeout_s: float
    execution_feedback_timeout_s: float
    maximum_clock_sync_uncertainty_s: float
    maximum_timesync_age_s: float
    minimum_target_lead_ticks: int
    maximum_target_lead_ticks: int
    residual_lease_ticks: int
    control_epoch: int
    expected_firmware_hash: str
    expected_dialect_hash: str
    expected_mixer_hash: str
    expected_mapping_hash: str
    expected_calibration_hash: str
    headroom_reserve_n: Tuple[float, ...]
    maximum_baseline_deviation_n: Tuple[float, ...]
    thrust_match_tolerance_n: float

    def __post_init__(self) -> None:
        for name in (
            "maximum_command_ttl_s",
            "maximum_baseline_age_s",
            "maximum_status_age_s",
            "acknowledgement_timeout_s",
            "execution_feedback_timeout_s",
            "maximum_timesync_age_s",
            "thrust_match_tolerance_n",
        ):
            value = _finite_real(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        uncertainty = _finite_real(
            "maximum_clock_sync_uncertainty_s",
            self.maximum_clock_sync_uncertainty_s,
        )
        if uncertainty < 0.0:
            raise ValueError("maximum_clock_sync_uncertainty_s cannot be negative")
        object.__setattr__(self, "maximum_clock_sync_uncertainty_s", uncertainty)
        _strict_integer(
            "minimum_target_lead_ticks",
            self.minimum_target_lead_ticks,
            positive=True,
        )
        _strict_integer(
            "maximum_target_lead_ticks",
            self.maximum_target_lead_ticks,
            positive=True,
        )
        if self.maximum_target_lead_ticks < self.minimum_target_lead_ticks:
            raise ValueError(
                "maximum_target_lead_ticks cannot be smaller than minimum_target_lead_ticks"
            )
        _strict_integer("residual_lease_ticks", self.residual_lease_ticks, positive=True)
        _strict_integer("control_epoch", self.control_epoch, positive=True)
        for name in (
            "expected_firmware_hash",
            "expected_dialect_hash",
            "expected_mixer_hash",
            "expected_mapping_hash",
            "expected_calibration_hash",
        ):
            object.__setattr__(self, name, _sha256_identity(name, getattr(self, name)))
        object.__setattr__(
            self,
            "headroom_reserve_n",
            _nonnegative_tuple("headroom_reserve_n", self.headroom_reserve_n, 4),
        )
        object.__setattr__(
            self,
            "maximum_baseline_deviation_n",
            _nonnegative_tuple(
                "maximum_baseline_deviation_n",
                self.maximum_baseline_deviation_n,
                4,
            ),
        )


@dataclass(frozen=True)
class FlightControllerResidualSinkStatus:
    """Host view; an ambiguous clear remains a visible latched fault."""

    timestamp_s: float
    healthy: bool
    fault_latched: bool
    residual_state: FlightControllerResidualState
    residual_active: bool
    clear_confirmed: bool
    fc_session_id: Optional[int]
    last_sequence: Optional[int]
    active_valid_until_s: Optional[float]
    last_error: str
    active_command_sequence: Optional[int] = None
    pending_command_sequence: Optional[int] = None
    pending_started_s: Optional[float] = None
    pending_valid_until_s: Optional[float] = None
    control_epoch: Optional[int] = None
    transport_generation: Optional[int] = None
    clear_through_command_sequence: Optional[int] = None
    residual_register_inactive: bool = False
    baseline_controller_retained: bool = False
    clear_ack_timestamp_s: Optional[float] = None
    clear_execution_timestamp_s: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        if not isinstance(self.residual_state, FlightControllerResidualState):
            raise TypeError("residual_state must be a FlightControllerResidualState")
        for name in (
            "healthy",
            "fault_latched",
            "residual_active",
            "clear_confirmed",
            "residual_register_inactive",
            "baseline_controller_retained",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.fc_session_id is not None:
            _strict_integer("fc_session_id", self.fc_session_id, positive=True)
        for name in (
            "last_sequence",
            "active_command_sequence",
            "pending_command_sequence",
            "clear_through_command_sequence",
        ):
            if getattr(self, name) is not None:
                _strict_integer(name, getattr(self, name))
        for name in ("control_epoch", "transport_generation"):
            value = getattr(self, name)
            if value is not None:
                _strict_integer(name, value, positive=True)
        for name in (
            "active_valid_until_s",
            "pending_started_s",
            "pending_valid_until_s",
            "clear_ack_timestamp_s",
            "clear_execution_timestamp_s",
        ):
            if getattr(self, name) is None:
                continue
            object.__setattr__(
                self,
                name,
                _finite_real(name, getattr(self, name)),
            )
        if (self.clear_ack_timestamp_s is None) != (self.clear_execution_timestamp_s is None):
            raise ValueError("clear ACK and execution timestamps must appear together")
        if (
            self.clear_ack_timestamp_s is not None
            and self.clear_execution_timestamp_s is not None
            and not (
                0.0
                < self.clear_ack_timestamp_s
                <= self.clear_execution_timestamp_s
                <= self.timestamp_s
            )
        ):
            raise ValueError(
                "clear ACK/execution timestamps must precede the current zero-status observation"
            )
        if (self.pending_command_sequence is None) != (self.pending_started_s is None):
            raise ValueError("pending command identity and start time must appear together")
        if (self.pending_command_sequence is None) != (self.pending_valid_until_s is None):
            raise ValueError("pending command identity and deadline must appear together")
        if (
            self.pending_started_s is not None
            and self.pending_valid_until_s is not None
            and self.pending_started_s >= self.pending_valid_until_s
        ):
            raise ValueError("pending command deadline must follow its start time")
        if (self.active_command_sequence is None) != (self.active_valid_until_s is None):
            raise ValueError("active command identity and deadline must appear together")
        if self.last_sequence is not None and any(
            sequence is not None and sequence > self.last_sequence
            for sequence in (self.active_command_sequence, self.pending_command_sequence)
        ):
            raise ValueError("active/pending sequence cannot exceed the consumed watermark")
        if self.residual_state is FlightControllerResidualState.CONFIRMED_ZERO:
            if (
                self.active_command_sequence is not None
                or self.pending_command_sequence is not None
                or self.residual_active
                or not self.clear_confirmed
            ):
                raise ValueError("confirmed-zero status cannot retain active/pending residuals")
        elif self.residual_state is FlightControllerResidualState.ACTIVE:
            if (
                self.active_command_sequence is None
                or self.pending_command_sequence is not None
                or not self.residual_active
                or self.clear_confirmed
            ):
                raise ValueError("active status requires only one confirmed active identity")
        elif self.residual_state is FlightControllerResidualState.STAGE_PENDING:
            if (
                self.pending_command_sequence is None
                or not self.residual_active
                or self.clear_confirmed
            ):
                raise ValueError("stage-pending status requires a pending identity")
        elif not self.residual_active or self.clear_confirmed:
            raise ValueError("clear-pending/ambiguous status must expose residual uncertainty")
        if not isinstance(self.last_error, str):
            raise TypeError("last_error must be a string")


def attest_post_touchdown_recovery(
    *,
    timestamp_s: float,
    valid_until_s: float,
    maximum_residual_status_age_s: float,
    landing_session_id: int,
    sequence: int,
    contact_epoch: int,
    contacts: Tuple[bool, bool, bool, bool],
    admittance_blends: Tuple[float, float, float, float],
    controller_quiesced: bool,
    recovery_complete: bool,
    load_transfer_complete: bool,
    body_state_stable: bool,
    go2_status: Go2LowLevelStatus,
    residual_status: FlightControllerResidualSinkStatus,
    reason: str = "normal post-touchdown recovery completed",
) -> ImpactLandingRecoveryEvidence:
    """Build completion evidence only from exact owner and FC status snapshots.

    This function does not quiesce either controller.  The production session
    owner must first fence new MPC snapshots/frames, drain any in-flight FC
    stage, call the residual CLEAR barrier, and then pass the resulting status
    here.  Invalid or ambiguous status is rejected instead of being encoded as
    a successful attestation.
    """

    observed_at = _finite_real("timestamp_s", timestamp_s)
    valid_until = _finite_real("valid_until_s", valid_until_s)
    maximum_residual_age = _finite_real(
        "maximum_residual_status_age_s",
        maximum_residual_status_age_s,
    )
    if maximum_residual_age <= 0.0:
        raise ValueError("maximum_residual_status_age_s must be positive")
    if valid_until <= observed_at:
        raise ValueError("recovery evidence deadline must follow its timestamp")
    if not isinstance(go2_status, Go2LowLevelStatus):
        raise TypeError("go2_status must be a Go2LowLevelStatus")
    if not isinstance(residual_status, FlightControllerResidualSinkStatus):
        raise TypeError("residual_status must be a FlightControllerResidualSinkStatus")
    if (
        go2_status.owner_epoch <= 0
        or not go2_status.ownership_pending
        or go2_status.ownership_state
        not in {
            LowCmdOwnershipState.MPC_ACTIVE,
            LowCmdOwnershipState.HOLDING,
            LowCmdOwnershipState.SAFE_HOLD,
        }
    ):
        raise ValueError("recovery evidence requires the exact retained LowCmd epoch")
    if (
        residual_status.timestamp_s > observed_at
        or observed_at - residual_status.timestamp_s > maximum_residual_age
        or not residual_status.healthy
        or residual_status.fault_latched
        or residual_status.residual_state is not FlightControllerResidualState.CONFIRMED_ZERO
        or residual_status.residual_active
        or not residual_status.clear_confirmed
        or residual_status.active_command_sequence is not None
        or residual_status.pending_command_sequence is not None
        or residual_status.clear_through_command_sequence is None
        or residual_status.clear_ack_timestamp_s is None
        or residual_status.clear_execution_timestamp_s is None
        or (
            residual_status.last_sequence is not None
            and residual_status.clear_through_command_sequence < residual_status.last_sequence
        )
        or not residual_status.residual_register_inactive
        or not residual_status.baseline_controller_retained
        or residual_status.fc_session_id is None
        or residual_status.control_epoch is None
        or residual_status.transport_generation is None
    ):
        raise ValueError(
            "FC status does not prove a persistent zero residual with its baseline retained"
        )
    return ImpactLandingRecoveryEvidence(
        timestamp=observed_at,
        valid_until=valid_until,
        landing_session_id=_strict_integer("landing_session_id", landing_session_id, positive=True),
        sequence=_strict_integer("sequence", sequence, positive=True),
        phase=ImpactLandingPhase.POST_TOUCHDOWN_RECOVERY,
        healthy=True,
        controller_quiesced=controller_quiesced,
        recovery_complete=recovery_complete,
        load_transfer_complete=load_transfer_complete,
        body_state_stable=body_state_stable,
        contacts=contacts,
        admittance_blends=admittance_blends,
        go2_ownership_epoch=go2_status.owner_epoch,
        contact_epoch=_strict_integer("contact_epoch", contact_epoch, positive=True),
        residual_zero_acknowledged=True,
        residual_zero_ack_timestamp=residual_status.clear_ack_timestamp_s,
        residual_zero_execution_timestamp=(residual_status.clear_execution_timestamp_s),
        residual_zero_status_timestamp=residual_status.timestamp_s,
        fc_session_id=residual_status.fc_session_id,
        fc_control_epoch=residual_status.control_epoch,
        fc_transport_generation=residual_status.transport_generation,
        last_residual_command_sequence=residual_status.last_sequence,
        clear_through_command_sequence=residual_status.clear_through_command_sequence,
        residual_register_inactive=True,
        baseline_controller_retained=True,
        reason=reason,
    )


class FlightControllerResidualSink:
    """Validate, stage, ACK and verify one already-scaled FC residual lease.

    This host state machine is deliberately unusable with stock MAVLink.  Its
    transport must prove an FC-local watchdog, replacement semantics and
    baseline-preserving clear behavior before the first residual is accepted.
    Any protocol ambiguity latches the session and requests a clear.  A new
    instance (and normally a new coordinated ownership epoch) is required to
    resume after such a fault.

    中文：一次发送包含“取得同 tick 基线预留 → stage ACK → 等待飞控高速环执行回读
    → 核对最终推力/基线版本/headroom”的完整事务。网络超时、飞控重启、时钟同步
    generation 改变或反馈不一致都会锁存会话并请求 clear；clear 只删除残差寄存器，
    不得关闭飞控基线姿态控制。
    """

    def __init__(
        self,
        transport: FlightControllerResidualTransport,
        config: FlightControllerResidualSinkConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for method_name in (
            "status",
            "latest_baseline_reservation",
            "stage_residual",
            "clear_residual",
            "wait_execution_feedback",
        ):
            if not callable(getattr(transport, method_name, None)):
                raise TypeError(f"transport must provide callable {method_name}()")
        if not isinstance(config, FlightControllerResidualSinkConfig):
            raise TypeError("config must be a FlightControllerResidualSinkConfig")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._transport = transport
        self._config = config
        self._clock = monotonic_clock
        self._lock = asyncio.Lock()
        self._last_sequence: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None
        self._active_command_sequence: Optional[int] = None
        self._active_valid_until_s: Optional[float] = None
        self._active_valid_until_fc_tick: Optional[int] = None
        self._pending_command_sequence: Optional[int] = None
        self._pending_started_s: Optional[float] = None
        self._pending_valid_until_s: Optional[float] = None
        self._residual_state = FlightControllerResidualState.CONFIRMED_ZERO
        self._fault_latched = False
        self._clear_confirmed = True
        self._last_error = ""
        self._clear_sequence = 0
        self._last_execution_fc_tick: Optional[int] = None
        self._last_execution_baseline_version: Optional[int] = None
        self._active_fc_session_id: Optional[int] = None
        self._active_transport_generation: Optional[int] = None
        self._active_timesync_generation: Optional[int] = None
        self._active_request_digest: Optional[str] = None
        self._pending_fc_session_id: Optional[int] = None
        self._pending_transport_generation: Optional[int] = None
        self._pending_timesync_generation: Optional[int] = None
        self._pending_request_digest: Optional[str] = None
        self._pending_target_fc_tick: Optional[int] = None
        self._pending_valid_until_fc_tick: Optional[int] = None
        self._confirmed_clear_domain: Optional[Tuple[int, int, int]] = None
        self._confirmed_clear_through_sequence: Optional[int] = None
        self._confirmed_clear_ack_timestamp_s: Optional[float] = None
        self._confirmed_clear_execution_timestamp_s: Optional[float] = None
        self._clear_intent_generation = 0
        self._clear_task: Optional[
            asyncio.Task[Tuple[OperationResult, Optional[BaseException]]]
        ] = None

    @property
    def simulation_only(self) -> bool:
        """Propagate the transport's explicit no-hardware capability marker."""

        return getattr(self._transport, "simulation_only", False) is True

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        """Stage one replacement residual and require separate execution proof."""

        async with self._lock:
            clear_intent_at_start = self._clear_intent_generation
            now = self._now()
            if (
                self._residual_state is FlightControllerResidualState.ACTIVE
                and self._active_valid_until_s is not None
                and now >= self._active_valid_until_s
            ):
                return await self._fail_and_clear(
                    "FC_RESIDUAL_PREVIOUS_LEASE_EXPIRED",
                    "Previous residual lease expired before a confirmed clear",
                )
            if self._fault_latched:
                return OperationResult.failure(
                    "FC_RESIDUAL_SESSION_LATCHED",
                    "The residual session is fault-latched; construct a new coordinated session",
                    {"clear_confirmed": self._clear_confirmed},
                )
            if self._residual_state not in {
                FlightControllerResidualState.CONFIRMED_ZERO,
                FlightControllerResidualState.ACTIVE,
            }:
                return await self._fail_and_clear(
                    "FC_RESIDUAL_STATE_NOT_STAGEABLE",
                    "A pending or ambiguous residual state must be reconciled before staging",
                )
            try:
                preflight_error = self._validate_preflight(command, now)
                if preflight_error is not None:
                    return await self._fail_and_clear(*preflight_error)
                transport_status = self._transport.status()
                now = self._now()
                preflight_error = self._validate_preflight(command, now)
                if preflight_error is not None:
                    return await self._fail_and_clear(*preflight_error)
                if (
                    self._residual_state is FlightControllerResidualState.ACTIVE
                    and isinstance(
                        transport_status,
                        FlightControllerResidualTransportStatus,
                    )
                    and not self._active_state_matches(transport_status)
                ):
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_ACTIVE_IDENTITY_CHANGED",
                        "Active residual session or transport/TIMESYNC generation changed",
                    )
                status_error = self._validate_transport_status(transport_status, now, command)
                if status_error is not None:
                    return await self._fail_and_clear(*status_error)
                if transport_status.timesync_generation is None:
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_STATUS_INVALID",
                        "TIMESYNC generation disappeared after validation",
                    )
                reservation = self._transport.latest_baseline_reservation()
                now = self._now()
                preflight_error = self._validate_preflight(command, now)
                if preflight_error is not None:
                    return await self._fail_and_clear(*preflight_error)
                status_error = self._validate_transport_status(transport_status, now, command)
                if status_error is not None:
                    return await self._fail_and_clear(*status_error)
                reservation_error = self._validate_reservation(
                    reservation,
                    command,
                    transport_status,
                    now,
                )
                if reservation_error is not None:
                    return await self._fail_and_clear(*reservation_error)
                if not isinstance(reservation, FlightControllerBaselineReservation):
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_BASELINE_UNAVAILABLE",
                        "Typed FC baseline reference disappeared after validation",
                    )

                if command.target_fc_tick > _UINT64_MAX - self._config.residual_lease_ticks:
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_TICK_OVERFLOW",
                        "Target tick plus residual lease would overflow uint64",
                    )
                valid_until_fc_tick = command.target_fc_tick + self._config.residual_lease_ticks

                # Consume the application sequence before an async write.  A
                # lost ACK is ambiguous and the sequence must never be replayed.
                self._last_sequence = command.sequence
                self._last_timestamp_s = command.timestamp_s
                request = FlightControllerResidualStageRequest(
                    fc_session_id=command.fc_session_id,
                    control_epoch=self._config.control_epoch,
                    transport_generation=transport_status.transport_generation,
                    sequence=command.sequence,
                    timestamp_s=command.timestamp_s,
                    timesync_generation=transport_status.timesync_generation,
                    target_fc_tick=command.target_fc_tick,
                    valid_until_fc_tick=valid_until_fc_tick,
                    baseline_version=command.baseline_version,
                    baseline_reference_digest=reservation.reference_digest,
                    applied_residual_thrusts_n=command.applied_residual_thrusts_n,
                    required_headroom_reserve_n=self._config.headroom_reserve_n,
                    maximum_baseline_deviation_n=(self._config.maximum_baseline_deviation_n),
                )
                pending_started_s = self._now()
                if pending_started_s >= command.valid_until_s:
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_COMMAND_STALE",
                        "Residual command expired immediately before transport staging",
                    )
                self._pending_command_sequence = command.sequence
                self._pending_started_s = pending_started_s
                self._pending_valid_until_s = command.valid_until_s
                self._pending_fc_session_id = request.fc_session_id
                self._pending_transport_generation = request.transport_generation
                self._pending_timesync_generation = request.timesync_generation
                self._pending_request_digest = request.request_digest
                self._pending_target_fc_tick = request.target_fc_tick
                self._pending_valid_until_fc_tick = request.valid_until_fc_tick
                self._clear_confirmed = False
                self._confirmed_clear_ack_timestamp_s = None
                self._confirmed_clear_execution_timestamp_s = None
                self._residual_state = FlightControllerResidualState.STAGE_PENDING
                ack = await asyncio.wait_for(
                    self._transport.stage_residual(request),
                    timeout=self._config.acknowledgement_timeout_s,
                )
                ack_observed_at = self._now()
                ack_error = self._validate_stage_ack(
                    ack,
                    command,
                    request,
                    ack_observed_at,
                )
                if ack_error is not None:
                    return await self._fail_and_clear(*ack_error)
                feedback = await asyncio.wait_for(
                    self._transport.wait_execution_feedback(
                        FlightControllerResidualOperation.STAGE,
                        command.sequence,
                    ),
                    timeout=self._config.execution_feedback_timeout_s,
                )
                completed_at = self._now()
                feedback_error = self._validate_stage_feedback(
                    feedback,
                    command,
                    request,
                    completed_at,
                )
                if feedback_error is not None:
                    return await self._fail_and_clear(*feedback_error)
                if completed_at >= command.valid_until_s:
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_EXECUTION_DEADLINE_MISSED",
                        "Execution feedback arrived after the residual command expired",
                    )
                post_status = self._transport.status()
                post_status_observed_at = self._now()
                post_status_error = self._validate_post_execution_status(
                    post_status,
                    post_status_observed_at,
                    command,
                    feedback,
                )
                if post_status_error is not None:
                    return await self._fail_and_clear(*post_status_error)
                if self._fault_latched or self._clear_intent_generation != clear_intent_at_start:
                    return await self._fail_and_clear(
                        "FC_RESIDUAL_CONCURRENT_FAULT_LATCHED",
                        "Residual state became ambiguous or an explicit clear was requested while staging",
                    )
            except asyncio.CancelledError:
                self._fault_latched = True
                self._last_error = "FC_RESIDUAL_SEND_CANCELLED"
                clear_task = self._get_or_start_clear_task(
                    "FC_RESIDUAL_SEND_CANCELLED: staging or feedback wait was cancelled"
                )
                await await_nonabandonable(clear_task)
                raise
            except asyncio.TimeoutError:
                return await self._fail_and_clear(
                    "FC_RESIDUAL_PROTOCOL_TIMEOUT",
                    "Residual staging ACK or execution feedback timed out",
                )
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                return await self._fail_and_clear(
                    "FC_RESIDUAL_PROTOCOL_INVALID",
                    f"Residual boundary rejected malformed data: {exc}",
                )
            except Exception as exc:
                return await self._fail_and_clear(
                    "FC_RESIDUAL_TRANSPORT_EXCEPTION",
                    f"Residual transport raised {type(exc).__name__}: {exc}",
                )

            self._residual_state = FlightControllerResidualState.ACTIVE
            self._active_command_sequence = command.sequence
            self._active_valid_until_s = command.valid_until_s
            self._active_valid_until_fc_tick = request.valid_until_fc_tick
            self._active_fc_session_id = self._pending_fc_session_id
            self._active_transport_generation = self._pending_transport_generation
            self._active_timesync_generation = self._pending_timesync_generation
            self._active_request_digest = self._pending_request_digest
            self._pending_command_sequence = None
            self._pending_started_s = None
            self._pending_valid_until_s = None
            self._pending_fc_session_id = None
            self._pending_transport_generation = None
            self._pending_timesync_generation = None
            self._pending_request_digest = None
            self._pending_target_fc_tick = None
            self._pending_valid_until_fc_tick = None
            self._clear_confirmed = False
            self._last_execution_fc_tick = feedback.execution_fc_tick
            self._last_execution_baseline_version = feedback.execution_baseline_version
            self._last_error = ""
            return OperationResult.success(
                "Flight controller staged and executed the already-scaled rotor residual",
                {
                    "sequence": command.sequence,
                    "fc_session_id": command.fc_session_id,
                    "target_fc_tick": command.target_fc_tick,
                    "baseline_version": command.baseline_version,
                    "execution_baseline_version": feedback.execution_baseline_version,
                    "valid_until_fc_tick": request.valid_until_fc_tick,
                    "control_epoch": request.control_epoch,
                    "request_digest": request.request_digest,
                    "valid_until_s": command.valid_until_s,
                    "thrust_unit": FC_RESIDUAL_THRUST_UNIT,
                    "rotor_order": FC_RESIDUAL_ROTOR_ORDER,
                    "fc_multiplies_kappa": False,
                },
                code="FC_RESIDUAL_EXECUTION_VERIFIED",
            )

    async def clear_rotor_residual(self, reason: str) -> OperationResult:
        """Clear only the residual; the FC attitude-control baseline remains active."""

        clear_reason = "host requested residual clear (invalid diagnostic reason)"
        if type(reason) is str and reason.strip():
            try:
                clear_reason = _truncate_utf8(reason, 96)
            except UnicodeEncodeError:
                # A malformed diagnostic string must never suppress the safety
                # action.  The wire request still receives a bounded reason.
                pass
        # Register the abort synchronously, before the first await.  An
        # emergency clear must not disappear merely because send() currently
        # owns the staging lock and the caller is cancelled while waiting.
        self._clear_intent_generation += 1
        self._residual_state = FlightControllerResidualState.CLEAR_PENDING
        self._clear_confirmed = False
        clear_task = self._get_or_start_clear_task(clear_reason)
        (result, clear_error), cancellation_seen = await await_nonabandonable(clear_task)
        if clear_error is not None:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            self._clear_confirmed = False
            result = OperationResult.failure(
                "FC_RESIDUAL_CLEAR_EXCEPTION",
                f"{type(clear_error).__name__}: {clear_error}",
            )
        if not result.ok:
            self._fault_latched = True
            self._clear_confirmed = False
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            self._last_error = f"{result.code}: {result.message}"
        if cancellation_seen:
            self._fault_latched = True
            self._last_error = "FC_RESIDUAL_CLEAR_CANCELLED"
            raise asyncio.CancelledError
        return result

    async def watchdog(self) -> OperationResult:
        """Host-side observer; FC-local expiry/disconnect clearing is authoritative."""

        async with self._lock:
            status_error = ""
            try:
                status = self._transport.status()
            except Exception as exc:
                status = None
                status_error = f"{type(exc).__name__}: {exc}"
            now = self._now()
            expired = self._active_valid_until_s is not None and now >= self._active_valid_until_s
            tick_expired = (
                isinstance(status, FlightControllerResidualTransportStatus)
                and status.current_fc_tick is not None
                and self._active_valid_until_fc_tick is not None
                and status.current_fc_tick >= self._active_valid_until_fc_tick
            )
            stale_or_lost = (
                not isinstance(status, FlightControllerResidualTransportStatus)
                or not self._status_identity_is_safe(status, now)
                or (
                    isinstance(status, FlightControllerResidualTransportStatus)
                    and self._has_tracked_residual_identity()
                    and not self._tracked_state_matches(status)
                )
            )
            clear_watermark_regressed = isinstance(
                status, FlightControllerResidualTransportStatus
            ) and not self._confirmed_clear_watermark_is_safe(status)
            unexpected_residual_state = (
                isinstance(status, FlightControllerResidualTransportStatus)
                and self._residual_state is FlightControllerResidualState.CONFIRMED_ZERO
                and (
                    status.residual_register_active
                    or status.pending_stage_present
                    or clear_watermark_regressed
                )
            )
            residual_uncertain = unexpected_residual_state or self._residual_state in {
                FlightControllerResidualState.STAGE_PENDING,
                FlightControllerResidualState.ACTIVE,
                FlightControllerResidualState.CLEAR_PENDING,
                FlightControllerResidualState.AMBIGUOUS,
            }
            state_requires_clear = unexpected_residual_state or self._residual_state in {
                FlightControllerResidualState.CLEAR_PENDING,
                FlightControllerResidualState.AMBIGUOUS,
            }
            if state_requires_clear or (
                residual_uncertain and (expired or tick_expired or stale_or_lost)
            ):
                self._fault_latched = True
                self._residual_state = FlightControllerResidualState.CLEAR_PENDING
                self._clear_confirmed = False
                if expired or tick_expired:
                    reason = "host observed residual TTL expiry"
                elif stale_or_lost:
                    reason = f"host observed residual transport loss/staleness: {status_error}"
                else:
                    reason = "host is reconciling an ambiguous residual state"
                clear_task = self._get_or_start_clear_task(reason)
                (result, clear_error), cancellation_seen = await await_nonabandonable(clear_task)
                if clear_error is not None:
                    result = OperationResult.failure(
                        "FC_RESIDUAL_CLEAR_EXCEPTION",
                        f"{type(clear_error).__name__}: {clear_error}",
                    )
                if result.ok:
                    observed = OperationResult.failure(
                        "FC_RESIDUAL_WATCHDOG_CLEARED",
                        f"{reason}; FC clear was confirmed and the session is latched",
                        {"clear_confirmed": True},
                    )
                else:
                    self._residual_state = FlightControllerResidualState.AMBIGUOUS
                    observed = OperationResult.failure(
                        "FC_RESIDUAL_WATCHDOG_AMBIGUOUS",
                        f"{reason}; explicit clear was not confirmed, so the verified FC-local watchdog must clear it",
                        {
                            "clear_confirmed": False,
                            "transport_code": result.code,
                        },
                    )
                if cancellation_seen:
                    self._last_error = "FC_RESIDUAL_WATCHDOG_CANCELLED"
                    raise asyncio.CancelledError
                return observed
            if stale_or_lost:
                return OperationResult.failure(
                    "FC_RESIDUAL_WATCHDOG_STATUS_UNAVAILABLE",
                    f"Residual transport status is unavailable or unhealthy: {status_error}",
                )
            if self._fault_latched:
                return OperationResult.failure(
                    "FC_RESIDUAL_WATCHDOG_SESSION_LATCHED",
                    "Residual register is confirmed zero, but this coordinated session remains fault-latched",
                    {"clear_confirmed": self._clear_confirmed},
                )
            return OperationResult.success(
                "Residual watchdog observer is healthy",
                {
                    "residual_state": self._residual_state.value,
                    "residual_may_be_active": residual_uncertain,
                },
                code="FC_RESIDUAL_WATCHDOG_HEALTHY",
            )

    def status(self) -> FlightControllerResidualSinkStatus:
        """Return immutable host state without treating link loss as a clear ACK."""

        fc_session_id: Optional[int] = None
        transport_healthy = False
        transport_status: Optional[FlightControllerResidualTransportStatus] = None
        try:
            candidate = self._transport.status()
            if not isinstance(candidate, FlightControllerResidualTransportStatus):
                raise TypeError("transport returned an invalid status object")
            transport_status = candidate
            fc_session_id = transport_status.fc_session_id
        except Exception:
            transport_status = None
        now = self._now()
        clear_watermark_regressed = (
            transport_status is not None
            and not self._confirmed_clear_watermark_is_safe(transport_status)
        )
        if transport_status is not None:
            transport_healthy = self._status_identity_is_safe(transport_status, now)
            if (
                self._has_tracked_residual_identity()
                and self._residual_state is not FlightControllerResidualState.CONFIRMED_ZERO
                and not self._tracked_state_matches(transport_status)
            ):
                transport_healthy = False
            if self._residual_state is FlightControllerResidualState.CONFIRMED_ZERO and (
                transport_status.residual_register_active
                or transport_status.pending_stage_present
                or clear_watermark_regressed
            ):
                transport_healthy = False
        expired = self._active_valid_until_s is not None and now >= self._active_valid_until_s
        tick_expired = (
            transport_status is not None
            and transport_status.current_fc_tick is not None
            and self._active_valid_until_fc_tick is not None
            and transport_status.current_fc_tick >= self._active_valid_until_fc_tick
        )
        residual_uncertain = self._residual_state in {
            FlightControllerResidualState.STAGE_PENDING,
            FlightControllerResidualState.ACTIVE,
            FlightControllerResidualState.CLEAR_PENDING,
            FlightControllerResidualState.AMBIGUOUS,
        }
        unexpected_residual_state = (
            transport_status is not None
            and self._residual_state is FlightControllerResidualState.CONFIRMED_ZERO
            and (
                transport_status.residual_register_active
                or transport_status.pending_stage_present
                or clear_watermark_regressed
            )
        )
        if unexpected_residual_state or (
            residual_uncertain and (expired or tick_expired or not transport_healthy)
        ):
            self._fault_latched = True
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            self._clear_confirmed = False
            if not self._last_error:
                self._last_error = "FC_RESIDUAL_STATE_BECAME_AMBIGUOUS"
        residual_uncertain = (
            self._residual_state is not FlightControllerResidualState.CONFIRMED_ZERO
        )
        # Preserve the observation time produced by the flight-controller
        # transport.  Re-stamping a cached status with the host read time would
        # turn repeated reads of one FC packet into apparently new evidence and
        # could falsely satisfy the post-touchdown persistent-zero gate.
        status_timestamp_s = now if transport_status is None else transport_status.timestamp_s
        return FlightControllerResidualSinkStatus(
            timestamp_s=status_timestamp_s,
            healthy=transport_healthy and not self._fault_latched,
            fault_latched=self._fault_latched,
            residual_state=self._residual_state,
            residual_active=residual_uncertain,
            clear_confirmed=self._clear_confirmed,
            fc_session_id=fc_session_id,
            last_sequence=self._last_sequence,
            active_valid_until_s=self._active_valid_until_s,
            last_error=self._last_error,
            active_command_sequence=self._active_command_sequence,
            pending_command_sequence=self._pending_command_sequence,
            pending_started_s=self._pending_started_s,
            pending_valid_until_s=self._pending_valid_until_s,
            control_epoch=(None if transport_status is None else transport_status.control_epoch),
            transport_generation=(
                None if transport_status is None else transport_status.transport_generation
            ),
            clear_through_command_sequence=(self._confirmed_clear_through_sequence),
            residual_register_inactive=bool(
                transport_status is not None
                and not transport_status.residual_register_active
                and not transport_status.pending_stage_present
            ),
            baseline_controller_retained=bool(
                transport_status is not None
                and transport_status.baseline_controller_active
                and transport_status.baseline_preservation_verified
            ),
            clear_ack_timestamp_s=self._confirmed_clear_ack_timestamp_s,
            clear_execution_timestamp_s=(self._confirmed_clear_execution_timestamp_s),
        )

    def _validate_preflight(
        self,
        command: object,
        now: float,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(command, FlightControllerRotorResidualCommand):
            return (
                "FC_RESIDUAL_COMMAND_INVALID",
                "command must be a FlightControllerRotorResidualCommand",
            )
        if command.timestamp_s > now:
            return "FC_RESIDUAL_COMMAND_FROM_FUTURE", "Command timestamp is in the future"
        if now >= command.valid_until_s:
            return "FC_RESIDUAL_COMMAND_STALE", "Command is already expired"
        ttl = command.valid_until_s - command.timestamp_s
        if ttl > self._config.maximum_command_ttl_s and not math.isclose(
            ttl,
            self._config.maximum_command_ttl_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return (
                "FC_RESIDUAL_COMMAND_TTL_INVALID",
                "Command TTL exceeds the configured maximum",
            )
        if self._last_sequence is not None and command.sequence <= self._last_sequence:
            return (
                "FC_RESIDUAL_COMMAND_REPLAY",
                "Residual sequence is duplicate or out of order",
            )
        if self._last_timestamp_s is not None and command.timestamp_s <= self._last_timestamp_s:
            return (
                "FC_RESIDUAL_COMMAND_REPLAY",
                "Residual source timestamp did not advance",
            )
        return None

    def _validate_transport_status(
        self,
        status: object,
        now: float,
        command: FlightControllerRotorResidualCommand,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(status, FlightControllerResidualTransportStatus):
            return "FC_RESIDUAL_STATUS_INVALID", "Transport returned an invalid status object"
        if not self._status_identity_is_safe(status, now):
            return (
                "FC_RESIDUAL_STATUS_OR_IDENTITY_UNSAFE",
                "FC status, control epoch, firmware identities, gates or TIMESYNC are unsafe",
            )
        if status.fc_session_id != command.fc_session_id:
            return "FC_RESIDUAL_SESSION_MISMATCH", "Flight-controller session/reboot ID changed"
        if (
            status.clear_through_command_sequence is not None
            and command.sequence <= status.clear_through_command_sequence
        ):
            return (
                "FC_RESIDUAL_COMMAND_REPLAY",
                "Residual sequence is already covered by the FC clear watermark",
            )
        if self._residual_state is FlightControllerResidualState.CONFIRMED_ZERO and (
            status.residual_register_active or status.pending_stage_present
        ):
            return (
                "FC_RESIDUAL_UNEXPECTED_REGISTER_STATE",
                "FC reports an active or pending residual while the host expects confirmed zero",
            )
        if (
            self._residual_state is FlightControllerResidualState.ACTIVE
            and not self._active_state_matches(status)
        ):
            return (
                "FC_RESIDUAL_ACTIVE_REGISTER_MISMATCH",
                "FC active residual sequence, digest or lease no longer matches the host",
            )
        if (
            status.current_fc_tick is None
            or status.fc_tick_period_s is None
            or status.timesync_generation is None
            or status.clock_sync_uncertainty_s is None
        ):
            return "FC_RESIDUAL_STATUS_INVALID", "FC tick or TIMESYNC metadata is unavailable"
        lead = command.target_fc_tick - status.current_fc_tick
        if (
            not self._config.minimum_target_lead_ticks
            <= lead
            <= self._config.maximum_target_lead_ticks
        ):
            return (
                "FC_RESIDUAL_TARGET_TICK_INVALID",
                "Target FC tick is outside the configured staging lead window",
            )
        remaining_ttl_s = command.valid_until_s - now
        if status.firmware_watchdog_timeout_s is None or (
            status.firmware_watchdog_timeout_s > remaining_ttl_s
            and not math.isclose(
                status.firmware_watchdog_timeout_s,
                remaining_ttl_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            return (
                "FC_RESIDUAL_WATCHDOG_UNVERIFIED",
                "FC-local watchdog timeout is absent or exceeds this command's remaining TTL",
            )
        if (
            status.clock_sync_uncertainty_s is None
            or status.clock_sync_uncertainty_s > self._config.maximum_clock_sync_uncertainty_s
        ):
            return (
                "FC_RESIDUAL_CLOCK_SYNC_UNSAFE",
                "FC/companion clock uncertainty exceeds the configured bound",
            )
        protocol_budget_s = (
            self._config.acknowledgement_timeout_s
            + self._config.execution_feedback_timeout_s
            + status.clock_sync_uncertainty_s
        )
        if remaining_ttl_s <= protocol_budget_s:
            return (
                "FC_RESIDUAL_DEADLINE_BUDGET_UNSAFE",
                "Remaining TTL cannot cover ACK, execution-feedback and clock-error budgets",
            )
        final_tick = command.target_fc_tick + self._config.residual_lease_ticks
        fc_expiry_delay_s = (
            final_tick - status.current_fc_tick
        ) * status.fc_tick_period_s + status.clock_sync_uncertainty_s
        if fc_expiry_delay_s > remaining_ttl_s:
            return (
                "FC_RESIDUAL_FC_TTL_EXCEEDS_HOST_TTL",
                "FC-local activation/lease would outlive the host command deadline",
            )
        guarantees = (
            status.replacement_semantics_verified,
            status.autonomous_expiry_clear_verified,
            status.disconnect_clear_verified,
            status.baseline_preservation_verified,
            status.execution_feedback_verified,
        )
        if not all(guarantees):
            return (
                "FC_RESIDUAL_FIRMWARE_CONTRACT_UNVERIFIED",
                "FC firmware has not proved replacement, watchdog, baseline-preservation and readback semantics",
            )
        return None

    def _status_identity_is_safe(
        self,
        status: FlightControllerResidualTransportStatus,
        now: float,
    ) -> bool:
        if status.timestamp_s > now or now - status.timestamp_s > self._config.maximum_status_age_s:
            return False
        if not status.connected or not status.healthy:
            return False
        if (
            status.control_epoch != self._config.control_epoch
            or status.current_fc_tick is None
            or status.fc_tick_period_s is None
            or status.timesync_generation is None
            or status.timesync_age_s is None
            or status.timesync_age_s + (now - status.timestamp_s)
            > self._config.maximum_timesync_age_s
            or status.clock_sync_uncertainty_s is None
            or status.clock_sync_uncertainty_s > self._config.maximum_clock_sync_uncertainty_s
            or status.firmware_watchdog_timeout_s is None
            or status.firmware_watchdog_timeout_s > self._config.maximum_command_ttl_s
            or not self._confirmed_clear_watermark_is_safe(status)
        ):
            return False
        if not (
            status.residual_enabled
            and status.baseline_controller_active
            and status.allocator_ready
            and status.replacement_semantics_verified
            and status.autonomous_expiry_clear_verified
            and status.disconnect_clear_verified
            and status.baseline_preservation_verified
            and status.execution_feedback_verified
        ):
            return False
        expected_identities = (
            (status.firmware_hash, self._config.expected_firmware_hash),
            (status.dialect_hash, self._config.expected_dialect_hash),
            (status.mixer_hash, self._config.expected_mixer_hash),
            (status.mapping_hash, self._config.expected_mapping_hash),
            (status.calibration_hash, self._config.expected_calibration_hash),
        )
        return all(actual == expected for actual, expected in expected_identities)

    @staticmethod
    def _status_identity_domain(
        status: FlightControllerResidualTransportStatus,
    ) -> Optional[Tuple[int, int, int]]:
        fc_session_id = status.fc_session_id
        control_epoch = status.control_epoch
        if fc_session_id is None or control_epoch is None:
            return None
        return (
            fc_session_id,
            control_epoch,
            status.transport_generation,
        )

    def _confirmed_clear_watermark_is_safe(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        """Reject a clear barrier regression within its exact FC identity domain."""

        expected = self._confirmed_clear_through_sequence
        if expected is None:
            return True
        if self._status_identity_domain(status) != self._confirmed_clear_domain:
            # FC-session/transport-domain changes reject old packets
            # independently and begin a new watermark domain.  A TIMESYNC
            # re-solve is deliberately not a replay-domain change.
            return True
        observed = status.clear_through_command_sequence
        return observed is not None and observed >= expected

    def _has_tracked_residual_identity(self) -> bool:
        return bool(
            self._active_fc_session_id is not None or self._pending_fc_session_id is not None
        )

    def _active_identity_matches(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        return (
            self._active_fc_session_id is not None
            and self._active_transport_generation is not None
            and self._active_timesync_generation is not None
            and status.fc_session_id == self._active_fc_session_id
            and status.control_epoch == self._config.control_epoch
            and status.transport_generation == self._active_transport_generation
            and status.timesync_generation == self._active_timesync_generation
        )

    def _active_state_matches(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        return (
            self._active_identity_matches(status)
            and self._active_command_sequence is not None
            and self._active_request_digest is not None
            and self._active_valid_until_fc_tick is not None
            and status.residual_register_active
            and status.active_command_sequence == self._active_command_sequence
            and status.active_request_digest == self._active_request_digest
            and status.active_valid_until_fc_tick == self._active_valid_until_fc_tick
            and not status.pending_stage_present
        )

    def _pending_identity_matches(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        return (
            self._pending_fc_session_id is not None
            and self._pending_transport_generation is not None
            and self._pending_timesync_generation is not None
            and status.fc_session_id == self._pending_fc_session_id
            and status.control_epoch == self._config.control_epoch
            and status.transport_generation == self._pending_transport_generation
            and status.timesync_generation == self._pending_timesync_generation
        )

    def _pending_state_matches(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        if (
            not self._pending_identity_matches(status)
            or self._pending_command_sequence is None
            or self._pending_request_digest is None
            or self._pending_target_fc_tick is None
            or self._pending_valid_until_fc_tick is None
        ):
            return False
        old_active_matches = (
            not status.residual_register_active
            if self._active_command_sequence is None
            else self._active_state_matches_allowing_pending(status)
        )
        pending_matches = bool(
            status.pending_stage_present
            and status.pending_command_sequence == self._pending_command_sequence
            and status.pending_request_digest == self._pending_request_digest
            and status.pending_target_fc_tick == self._pending_target_fc_tick
        )
        promoted_matches = bool(
            not status.pending_stage_present
            and status.residual_register_active
            and status.active_command_sequence == self._pending_command_sequence
            and status.active_request_digest == self._pending_request_digest
            and status.active_valid_until_fc_tick == self._pending_valid_until_fc_tick
        )
        request_not_visible_yet = bool(not status.pending_stage_present and old_active_matches)
        return bool(
            (pending_matches and old_active_matches) or promoted_matches or request_not_visible_yet
        )

    def _active_state_matches_allowing_pending(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        return bool(
            self._active_identity_matches(status)
            and self._active_command_sequence is not None
            and self._active_request_digest is not None
            and self._active_valid_until_fc_tick is not None
            and status.residual_register_active
            and status.active_command_sequence == self._active_command_sequence
            and status.active_request_digest == self._active_request_digest
            and status.active_valid_until_fc_tick == self._active_valid_until_fc_tick
        )

    def _tracked_state_matches(
        self,
        status: FlightControllerResidualTransportStatus,
    ) -> bool:
        if self._residual_state is FlightControllerResidualState.ACTIVE:
            return self._active_state_matches(status)
        if self._residual_state is FlightControllerResidualState.STAGE_PENDING:
            return self._pending_state_matches(status)
        return True

    def _validate_post_execution_status(
        self,
        status: object,
        now: float,
        command: FlightControllerRotorResidualCommand,
        feedback: FlightControllerResidualExecutionFeedback,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(status, FlightControllerResidualTransportStatus):
            return "FC_RESIDUAL_POST_STATUS_INVALID", "Post-execution FC status is invalid"
        if not self._status_identity_is_safe(status, now):
            return (
                "FC_RESIDUAL_POST_STATUS_UNSAFE",
                "FC disconnected, changed identity/epoch, or lost a required gate after ACK",
            )
        if now >= command.valid_until_s:
            return (
                "FC_RESIDUAL_POST_EXECUTION_DEADLINE_MISSED",
                "Post-execution status arrived after the host command deadline",
            )
        if (
            status.fc_session_id != command.fc_session_id
            or status.transport_generation != feedback.transport_generation
            or status.timesync_generation != feedback.timesync_generation
            or status.current_fc_tick is None
            or status.current_fc_tick < feedback.execution_fc_tick
        ):
            return (
                "FC_RESIDUAL_POST_EXECUTION_FENCE_FAILED",
                "Post-execution session, TIMESYNC generation or FC tick does not fence the feedback",
            )
        if not (
            status.residual_register_active
            and status.active_command_sequence == command.sequence
            and status.active_request_digest == feedback.request_digest
            and status.active_valid_until_fc_tick == feedback.valid_until_fc_tick
            and not status.pending_stage_present
        ):
            return (
                "FC_RESIDUAL_POST_EXECUTION_REGISTER_MISMATCH",
                "Post-execution status does not expose the staged active residual identity",
            )
        if status.timestamp_s < feedback.timestamp_s:
            return (
                "FC_RESIDUAL_POST_EXECUTION_FENCE_FAILED",
                "Post-execution status timestamp predates the execution feedback",
            )
        if (
            feedback.valid_until_fc_tick is None
            or status.current_fc_tick >= feedback.valid_until_fc_tick
        ):
            return (
                "FC_RESIDUAL_POST_EXECUTION_LEASE_EXPIRED",
                "Residual lease expired before host verification completed",
            )
        if (
            status.firmware_watchdog_timeout_s is None
            or status.fc_tick_period_s is None
            or status.clock_sync_uncertainty_s is None
        ):
            return (
                "FC_RESIDUAL_POST_STATUS_UNSAFE",
                "Post-execution watchdog, FC tick period or clock quality is unavailable",
            )
        remaining_ttl_s = command.valid_until_s - now
        remaining_fc_lease_s = (
            feedback.valid_until_fc_tick - status.current_fc_tick
        ) * status.fc_tick_period_s + status.clock_sync_uncertainty_s
        if (
            status.firmware_watchdog_timeout_s > remaining_ttl_s
            or remaining_fc_lease_s > remaining_ttl_s
        ):
            return (
                "FC_RESIDUAL_POST_EXECUTION_TTL_UNSAFE",
                "Post-execution FC watchdog or tick lease can outlive the host deadline",
            )
        return None

    def _validate_reservation(
        self,
        reservation: object,
        command: FlightControllerRotorResidualCommand,
        status: FlightControllerResidualTransportStatus,
        now: float,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(reservation, FlightControllerBaselineReservation):
            return (
                "FC_RESIDUAL_BASELINE_UNAVAILABLE",
                "No typed FC baseline reference/admission envelope is available",
            )
        if not reservation.is_fresh(now):
            return "FC_RESIDUAL_BASELINE_STALE", "FC baseline reservation is stale"
        if now - reservation.timestamp_s > self._config.maximum_baseline_age_s:
            return "FC_RESIDUAL_BASELINE_STALE", "FC baseline reservation exceeds maximum age"
        identity = (
            reservation.fc_session_id == command.fc_session_id
            and reservation.control_epoch == self._config.control_epoch
            and reservation.transport_generation == status.transport_generation
            and reservation.timesync_generation == status.timesync_generation
            and reservation.target_fc_tick == command.target_fc_tick
            and reservation.baseline_version == command.baseline_version
            and reservation.timestamp_s == command.baseline_timestamp_s
        )
        if not identity:
            return (
                "FC_RESIDUAL_BASELINE_VERSION_MISMATCH",
                "Command does not reference the exact FC session/tick/reference snapshot",
            )
        if status.current_fc_tick is None or reservation.target_fc_tick <= status.current_fc_tick:
            return "FC_RESIDUAL_BASELINE_LATE", "The reserved FC execution tick has passed"
        tolerance = self._config.thrust_match_tolerance_n
        for index in range(4):
            if not math.isclose(
                reservation.baseline_thrusts_n[index],
                command.baseline_thrusts_n[index],
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                return (
                    "FC_RESIDUAL_BASELINE_VALUE_MISMATCH",
                    "Command baseline thrust does not match the reserved FC baseline",
                )
            residual = command.applied_residual_thrusts_n[index]
            reserve = self._config.headroom_reserve_n[index]
            positive_limit = max(0.0, reservation.positive_headroom_n[index] - reserve)
            negative_limit = max(0.0, reservation.negative_headroom_n[index] - reserve)
            if residual > positive_limit + tolerance or residual < -negative_limit - tolerance:
                return (
                    "FC_RESIDUAL_HEADROOM_EXCEEDED",
                    f"Rotor {FC_RESIDUAL_ROTOR_ORDER[index]} residual exceeds reserved headroom",
                )
        return None

    def _validate_stage_ack(
        self,
        ack: object,
        command: FlightControllerRotorResidualCommand,
        request: FlightControllerResidualStageRequest,
        observed_at: float,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(ack, FlightControllerResidualAck):
            return "FC_RESIDUAL_ACK_INVALID", "Transport returned an invalid staging ACK"
        exact = (
            ack.operation is FlightControllerResidualOperation.STAGE
            and ack.operation_sequence == command.sequence
            and ack.command_sequence == command.sequence
            and ack.fc_session_id == command.fc_session_id
            and ack.control_epoch == self._config.control_epoch
            and ack.transport_generation == request.transport_generation
            and ack.target_fc_tick == command.target_fc_tick
            and ack.valid_until_fc_tick
            == command.target_fc_tick + self._config.residual_lease_ticks
            and ack.baseline_version == command.baseline_version
            and ack.clear_through_command_sequence is None
            and ack.request_digest == request.request_digest
        )
        if not exact:
            return "FC_RESIDUAL_ACK_MISMATCH", "Staging ACK metadata does not match the command"
        if not (command.timestamp_s <= ack.timestamp_s <= observed_at < command.valid_until_s):
            return "FC_RESIDUAL_ACK_STALE", "Staging ACK timestamp is outside command validity"
        if not ack.accepted:
            return (
                "FC_RESIDUAL_STAGE_REJECTED",
                f"FC rejected residual staging: {ack.result_code}: {ack.message}",
            )
        return None

    def _validate_stage_feedback(
        self,
        feedback: object,
        command: FlightControllerRotorResidualCommand,
        request: FlightControllerResidualStageRequest,
        observed_at: float,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(feedback, FlightControllerResidualExecutionFeedback):
            return "FC_RESIDUAL_FEEDBACK_INVALID", "Transport returned invalid execution feedback"
        exact = (
            feedback.operation is FlightControllerResidualOperation.STAGE
            and feedback.operation_sequence == command.sequence
            and feedback.command_sequence == command.sequence
            and feedback.fc_session_id == command.fc_session_id
            and feedback.control_epoch == request.control_epoch
            and feedback.transport_generation == request.transport_generation
            and feedback.timesync_generation == request.timesync_generation
            and feedback.valid_until_fc_tick == request.valid_until_fc_tick
            and feedback.baseline_version == command.baseline_version
            and feedback.clear_through_command_sequence is None
            and feedback.request_digest == request.request_digest
        )
        if not exact:
            return (
                "FC_RESIDUAL_EXECUTION_MISMATCH",
                "Execution feedback does not match the staged session/sequence/tick/baseline",
            )
        if not (command.timestamp_s <= feedback.timestamp_s <= observed_at < command.valid_until_s):
            return (
                "FC_RESIDUAL_EXECUTION_DEADLINE_MISSED",
                "Execution feedback timestamp is outside command validity",
            )
        if feedback.execution_result is FlightControllerResidualExecutionResult.EXPIRED:
            if feedback.execution_fc_tick < request.valid_until_fc_tick:
                return (
                    "FC_RESIDUAL_EXECUTION_MISMATCH",
                    "EXPIRED feedback was emitted before the exclusive FC-tick lease deadline",
                )
        elif feedback.execution_result is FlightControllerResidualExecutionResult.GATE_REJECTED:
            if not (
                command.target_fc_tick <= feedback.execution_fc_tick < request.valid_until_fc_tick
            ):
                return (
                    "FC_RESIDUAL_EXECUTION_MISMATCH",
                    "GATE_REJECTED feedback is outside the target-to-expiry FC-tick window",
                )
        elif feedback.execution_fc_tick != command.target_fc_tick:
            return (
                "FC_RESIDUAL_EXECUTION_MISMATCH",
                "Non-expiry execution feedback was not sampled at the staged target tick",
            )
        if feedback.execution_result is not FlightControllerResidualExecutionResult.APPLIED:
            tolerance = self._config.thrust_match_tolerance_n
            rejected_zero_is_proved = (
                not feedback.residual_active
                and feedback.baseline_controller_active
                and feedback.residual_addition_terms == 0
                and not any(feedback.saturation_mask)
                and math.isclose(
                    feedback.saturation_scale,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and all(
                    math.isclose(value, 0.0, rel_tol=0.0, abs_tol=tolerance)
                    for value in feedback.applied_residual_thrusts_n
                )
                and all(
                    math.isclose(final, baseline, rel_tol=0.0, abs_tol=tolerance)
                    for final, baseline in zip(
                        feedback.final_thrusts_n,
                        feedback.baseline_thrusts_n,
                    )
                )
            )
            if not rejected_zero_is_proved:
                return (
                    "FC_RESIDUAL_REJECTION_SEMANTICS_INVALID",
                    "FC rejected the residual without proving an atomic zero-residual fallback",
                )
            return (
                "FC_RESIDUAL_EXECUTION_REJECTED",
                f"FC fast loop rejected the residual: {feedback.execution_result.value}",
            )
        if (
            not feedback.residual_active
            or not feedback.baseline_controller_active
            or feedback.residual_addition_terms != 1
        ):
            return (
                "FC_RESIDUAL_EXECUTION_SEMANTICS_INVALID",
                "FC did not prove one additive residual term over an active attitude baseline",
            )
        if any(feedback.saturation_mask) or not math.isclose(
            feedback.saturation_scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return (
                "FC_RESIDUAL_SATURATED",
                "FC saturated the residual; strict paper-equation execution is rejected",
            )
        tolerance = self._config.thrust_match_tolerance_n
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
            for actual, expected in zip(
                feedback.required_headroom_reserve_n,
                request.required_headroom_reserve_n,
            )
        ):
            return (
                "FC_RESIDUAL_EXECUTION_HEADROOM_INVALID",
                "FC did not echo the required same-tick headroom reserve",
            )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
            for actual, expected in zip(
                feedback.maximum_baseline_deviation_n,
                request.maximum_baseline_deviation_n,
            )
        ):
            return (
                "FC_RESIDUAL_BASELINE_DEVIATION_POLICY_INVALID",
                "FC did not echo the staged live-baseline deviation policy",
            )
        if any(
            abs(live - reference) > limit + tolerance
            for live, reference, limit in zip(
                feedback.baseline_thrusts_n,
                command.baseline_thrusts_n,
                request.maximum_baseline_deviation_n,
            )
        ):
            return (
                "FC_RESIDUAL_BASELINE_DEVIATION_EXCEEDED",
                "Execution-tick baseline deviated beyond the bound used by the MPC",
            )
        expected_vectors = (
            (
                feedback.requested_residual_thrusts_n,
                command.applied_residual_thrusts_n,
            ),
            (feedback.applied_residual_thrusts_n, command.applied_residual_thrusts_n),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
            for actual_values, expected_values in expected_vectors
            for actual, expected in zip(actual_values, expected_values)
        ):
            return (
                "FC_RESIDUAL_EXECUTION_VALUE_MISMATCH",
                "FC execution readback does not equal baseline plus the already-scaled payload",
            )
        if any(
            not math.isclose(
                final,
                baseline + applied,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for final, baseline, applied in zip(
                feedback.final_thrusts_n,
                feedback.baseline_thrusts_n,
                feedback.applied_residual_thrusts_n,
            )
        ):
            return (
                "FC_RESIDUAL_EXECUTION_VALUE_MISMATCH",
                "Same-tick commanded final thrust does not equal live baseline plus residual",
            )
        if any(
            abs(final - modelled) > limit + tolerance
            for final, modelled, limit in zip(
                feedback.final_thrusts_n,
                command.applied_total_thrusts_n,
                request.maximum_baseline_deviation_n,
            )
        ):
            return (
                "FC_RESIDUAL_MODEL_EXECUTION_MISMATCH",
                "Commanded total thrust deviated beyond the MPC execution bound",
            )
        for index, applied in enumerate(feedback.applied_residual_thrusts_n):
            reserve = self._config.headroom_reserve_n[index]
            positive_limit = max(0.0, feedback.positive_headroom_n[index] - reserve)
            negative_limit = max(0.0, feedback.negative_headroom_n[index] - reserve)
            if applied > positive_limit + tolerance or applied < -negative_limit - tolerance:
                return (
                    "FC_RESIDUAL_EXECUTION_HEADROOM_INVALID",
                    "FC execution readback exceeds its same-tick reported headroom",
                )
        return None

    async def _fail_and_clear(self, code: str, message: str) -> OperationResult:
        self._fault_latched = True
        self._last_error = f"{code}: {message}"
        clear_task = self._get_or_start_clear_task(self._last_error)
        (clear_result, clear_error), cancellation_seen = await await_nonabandonable(clear_task)
        if clear_error is None:
            clear_ok = clear_result.ok
            clear_code = clear_result.code
            clear_message = clear_result.message
        else:
            clear_ok = False
            clear_code = "FC_RESIDUAL_CLEAR_EXCEPTION"
            clear_message = f"{type(clear_error).__name__}: {clear_error}"
        detail = message
        if not clear_ok:
            detail += (
                "; explicit residual clear was not confirmed; rely on the verified "
                f"FC-local watchdog: {clear_code}: {clear_message}"
            )
        result = OperationResult.failure(
            code,
            detail,
            {
                "clear_requested": True,
                "clear_confirmed": clear_ok,
                "clear_code": clear_code,
                "last_sequence": self._last_sequence,
            },
        )
        if cancellation_seen:
            raise asyncio.CancelledError
        return result

    def _get_or_start_clear_task(
        self,
        reason: str,
    ) -> asyncio.Task[Tuple[OperationResult, Optional[BaseException]]]:
        task = self._clear_task
        if task is None or task.done():
            task = asyncio.create_task(self._invoke_clear(reason))
            self._clear_task = task
        return task

    async def _invoke_clear(
        self,
        reason: str,
    ) -> Tuple[OperationResult, Optional[BaseException]]:
        try:
            return await self._clear_internal(reason), None
        except (Exception, asyncio.CancelledError) as exc:
            return (
                OperationResult.failure(
                    "FC_RESIDUAL_CLEAR_EXCEPTION",
                    f"{type(exc).__name__}: {exc}",
                ),
                exc,
            )

    async def _clear_internal(self, reason: str) -> OperationResult:
        self._residual_state = FlightControllerResidualState.CLEAR_PENDING
        self._clear_confirmed = False
        self._confirmed_clear_ack_timestamp_s = None
        self._confirmed_clear_execution_timestamp_s = None
        try:
            status = self._transport.status()
        except Exception as exc:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_STATUS_FAILED",
                f"Cannot inspect FC session before clear: {type(exc).__name__}: {exc}",
            )
        now = self._now()
        if (
            not isinstance(status, FlightControllerResidualTransportStatus)
            or not status.connected
            or status.fc_session_id is None
            or status.control_epoch != self._config.control_epoch
            or status.timesync_generation is None
        ):
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_LINK_UNAVAILABLE",
                "Cannot explicitly clear a disconnected FC link; FC-local watchdog is authoritative",
            )
        self._clear_sequence += 1
        request = FlightControllerResidualClearRequest(
            fc_session_id=status.fc_session_id,
            control_epoch=self._config.control_epoch,
            transport_generation=status.transport_generation,
            clear_sequence=self._clear_sequence,
            clear_through_command_sequence=max(
                0 if self._last_sequence is None else self._last_sequence,
                (0 if status.active_command_sequence is None else status.active_command_sequence),
                (0 if status.pending_command_sequence is None else status.pending_command_sequence),
                (
                    0
                    if status.clear_through_command_sequence is None
                    else status.clear_through_command_sequence
                ),
            ),
            timestamp_s=now,
            reason=_truncate_utf8(reason, 96),
        )
        try:
            ack = await asyncio.wait_for(
                self._transport.clear_residual(request),
                timeout=self._config.acknowledgement_timeout_s,
            )
            ack_observed_at = self._now()
            if not isinstance(ack, FlightControllerResidualAck):
                raise ValueError("transport returned an invalid clear ACK")
            if not (
                ack.operation is FlightControllerResidualOperation.CLEAR
                and ack.operation_sequence == request.clear_sequence
                and ack.command_sequence is None
                and ack.fc_session_id == request.fc_session_id
                and ack.control_epoch == request.control_epoch
                and ack.transport_generation == request.transport_generation
                and ack.target_fc_tick is None
                and ack.valid_until_fc_tick is None
                and ack.baseline_version is None
                and ack.clear_through_command_sequence is not None
                and ack.clear_through_command_sequence >= request.clear_through_command_sequence
                and ack.request_digest == request.request_digest
                and request.timestamp_s <= ack.timestamp_s <= ack_observed_at
                and ack.accepted
            ):
                raise ValueError("clear ACK metadata or acceptance does not match")
            effective_clear_through = ack.clear_through_command_sequence
            if effective_clear_through is None:
                raise ValueError("clear ACK omitted its effective sequence watermark")
            feedback = await asyncio.wait_for(
                self._transport.wait_execution_feedback(
                    FlightControllerResidualOperation.CLEAR,
                    request.clear_sequence,
                ),
                timeout=self._config.execution_feedback_timeout_s,
            )
        except asyncio.TimeoutError:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_TIMEOUT",
                "Clear ACK or zero-residual execution feedback timed out",
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure("FC_RESIDUAL_CLEAR_PROTOCOL_INVALID", str(exc))
        except Exception as exc:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_FAILED",
                f"{type(exc).__name__}: {exc}",
            )
        completed_at = self._now()
        if not self._clear_feedback_is_safe(
            feedback,
            request,
            effective_clear_through,
            completed_at,
        ):
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_FEEDBACK_INVALID",
                "FC did not prove zero residual while retaining its baseline controller",
            )
        try:
            post_status = self._transport.status()
        except Exception as exc:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_POST_STATUS_FAILED",
                f"Cannot fence clear feedback: {type(exc).__name__}: {exc}",
            )
        post_status_observed_at = self._now()
        if not (
            isinstance(post_status, FlightControllerResidualTransportStatus)
            and self._status_identity_is_safe(post_status, post_status_observed_at)
            and post_status.fc_session_id == request.fc_session_id
            and post_status.control_epoch == request.control_epoch
            and post_status.transport_generation == feedback.transport_generation
            and post_status.timesync_generation == feedback.timesync_generation
            and post_status.current_fc_tick is not None
            and post_status.current_fc_tick >= feedback.execution_fc_tick
            and post_status.timestamp_s >= feedback.timestamp_s
            and not post_status.residual_register_active
            and not post_status.pending_stage_present
            and post_status.clear_through_command_sequence is not None
            and post_status.clear_through_command_sequence >= effective_clear_through
        ):
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_POST_FENCE_FAILED",
                "FC status changed or became unhealthy after zero-residual feedback",
            )
        confirmed_domain = self._status_identity_domain(post_status)
        confirmed_clear_through = post_status.clear_through_command_sequence
        if confirmed_domain is None or confirmed_clear_through is None:
            self._residual_state = FlightControllerResidualState.AMBIGUOUS
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_POST_FENCE_FAILED",
                "FC clear status omitted its identity domain or persistent watermark",
            )
        self._residual_state = FlightControllerResidualState.CONFIRMED_ZERO
        self._active_command_sequence = None
        self._active_valid_until_s = None
        self._active_valid_until_fc_tick = None
        self._pending_command_sequence = None
        self._pending_started_s = None
        self._pending_valid_until_s = None
        self._active_fc_session_id = None
        self._active_transport_generation = None
        self._active_timesync_generation = None
        self._active_request_digest = None
        self._pending_fc_session_id = None
        self._pending_transport_generation = None
        self._pending_timesync_generation = None
        self._pending_request_digest = None
        self._pending_target_fc_tick = None
        self._pending_valid_until_fc_tick = None
        self._confirmed_clear_domain = confirmed_domain
        self._confirmed_clear_through_sequence = confirmed_clear_through
        self._confirmed_clear_ack_timestamp_s = ack.timestamp_s
        self._confirmed_clear_execution_timestamp_s = feedback.timestamp_s
        self._last_sequence = max(
            confirmed_clear_through,
            0 if self._last_sequence is None else self._last_sequence,
        )
        self._clear_confirmed = True
        return OperationResult.success(
            "Flight-controller residual cleared; FC baseline remains active",
            {
                "clear_sequence": request.clear_sequence,
                "fc_session_id": request.fc_session_id,
                "control_epoch": request.control_epoch,
                "requested_clear_through_command_sequence": (
                    request.clear_through_command_sequence
                ),
                "clear_through_command_sequence": effective_clear_through,
                "request_digest": request.request_digest,
                "baseline_preserved": True,
            },
            code="FC_RESIDUAL_CLEAR_VERIFIED",
        )

    def _clear_feedback_is_safe(
        self,
        feedback: object,
        request: FlightControllerResidualClearRequest,
        effective_clear_through: int,
        now: float,
    ) -> bool:
        if not isinstance(feedback, FlightControllerResidualExecutionFeedback):
            return False
        return (
            feedback.operation is FlightControllerResidualOperation.CLEAR
            and feedback.execution_result is FlightControllerResidualExecutionResult.CLEARED
            and feedback.operation_sequence == request.clear_sequence
            and feedback.command_sequence is None
            and feedback.fc_session_id == request.fc_session_id
            and feedback.control_epoch == request.control_epoch
            and feedback.transport_generation == request.transport_generation
            and feedback.valid_until_fc_tick is None
            and feedback.clear_through_command_sequence == effective_clear_through
            and feedback.request_digest == request.request_digest
            and request.timestamp_s <= feedback.timestamp_s <= now
            and not feedback.residual_active
            and feedback.baseline_controller_active
            and feedback.residual_addition_terms == 0
            and all(
                math.isclose(value, 0.0, rel_tol=0.0, abs_tol=self._config.thrust_match_tolerance_n)
                for value in (
                    feedback.required_headroom_reserve_n + feedback.maximum_baseline_deviation_n
                )
            )
            and all(
                math.isclose(value, 0.0, rel_tol=0.0, abs_tol=self._config.thrust_match_tolerance_n)
                for value in feedback.requested_residual_thrusts_n
                + feedback.applied_residual_thrusts_n
            )
            and all(
                math.isclose(
                    final, baseline, rel_tol=0.0, abs_tol=self._config.thrust_match_tolerance_n
                )
                for final, baseline in zip(
                    feedback.final_thrusts_n,
                    feedback.baseline_thrusts_n,
                )
            )
            and not any(feedback.saturation_mask)
            and feedback.saturation_scale == 1.0
        )

    def _now(self) -> float:
        return _finite_real("monotonic clock", self._clock())


class UnavailableGo2LowLevelSink:
    """Default adapter that proves real joint output has not been configured."""

    async def send_joint_position_command(
        self,
        command: Go2JointPositionCommand,
    ) -> OperationResult:
        del command
        return OperationResult.failure(
            "GO2_LOW_LEVEL_NOT_CONFIGURED",
            "Dedicated Unitree LowCmd ownership, CRC, ordering, and watchdog are not configured",
        )

    async def revoke_mpc_control(self, reason: str) -> OperationResult:
        return OperationResult.success(f"No Go2 LowCmd stream existed: {reason}")


class UnavailableFlightControllerResidualSink:
    """Default adapter that refuses to reinterpret residuals as MAVLink velocity."""

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        del command
        return OperationResult.failure(
            "FC_ROTOR_RESIDUAL_NOT_CONFIGURED",
            "The flight-controller mixer-residual protocol and unit calibration are not configured",
        )

    async def clear_rotor_residual(self, reason: str) -> OperationResult:
        return OperationResult.success(f"No flight-controller residual stream existed: {reason}")


__all__ = [
    "FC_RESIDUAL_PROTOCOL_VERSION",
    "FC_RESIDUAL_REPLACE_SEMANTICS",
    "FC_RESIDUAL_ROTOR_ORDER",
    "FC_RESIDUAL_THRUST_UNIT",
    "CoordinatedLandingCommand",
    "FlightControllerBaselineReservation",
    "FlightControllerResidualAck",
    "FlightControllerResidualClearRequest",
    "FlightControllerResidualExecutionFeedback",
    "FlightControllerResidualExecutionResult",
    "FlightControllerResidualOperation",
    "FlightControllerResidualSink",
    "FlightControllerResidualSinkConfig",
    "FlightControllerResidualSinkStatus",
    "FlightControllerResidualState",
    "FlightControllerResidualStageRequest",
    "FlightControllerResidualTransport",
    "FlightControllerResidualTransportStatus",
    "FlightControllerRotorResidualCommand",
    "Go2JointPositionCommand",
    "Go2LowLevelCommandSink",
    "ImpactLandingPhase",
    "UnavailableFlightControllerResidualSink",
    "UnavailableGo2LowLevelSink",
    "attest_post_touchdown_recovery",
    "phase_for_system_state",
]
