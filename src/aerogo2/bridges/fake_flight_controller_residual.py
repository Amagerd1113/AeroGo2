"""Deterministic FC-residual loopback for protocol and fault-injection tests.

This class is not a Pixhawk emulator and is permanently unsuitable for
physical output.  It models the safety contract that custom flight-controller
    firmware must implement: a reference baseline snapshot, idempotent replacement
of one already-scaled residual register, execution readback, and autonomous
baseline-preserving clear on TTL expiry or link loss.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

from aerogo2.landing.impact_aware.integration import (
    FlightControllerBaselineReservation,
    FlightControllerResidualAck,
    FlightControllerResidualClearRequest,
    FlightControllerResidualExecutionFeedback,
    FlightControllerResidualExecutionResult,
    FlightControllerResidualOperation,
    FlightControllerResidualStageRequest,
    FlightControllerResidualTransportStatus,
)

FAKE_FC_FIRMWARE_HASH = "sha256:" + "1" * 64
FAKE_FC_DIALECT_HASH = "sha256:" + "2" * 64
FAKE_FC_MIXER_HASH = "sha256:" + "3" * 64
FAKE_FC_MAPPING_HASH = "sha256:" + "4" * 64
FAKE_FC_CALIBRATION_HASH = "sha256:" + "5" * 64


class FakeFlightControllerResidualTransport:
    """In-memory reference semantics with no hardware or MAVLink writes."""

    simulation_only = True

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        fc_session_id: int = 1,
        control_epoch: int = 1,
        current_fc_tick: int = 100,
        target_fc_tick: int = 102,
        baseline_version: int = 1,
        reservation_valid_until_s: Optional[float] = None,
        baseline_thrusts_n: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0),
        negative_headroom_n: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0),
        positive_headroom_n: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0),
        fc_tick_period_s: float = 0.002,
        timesync_generation: int = 1,
        timesync_age_s: float = 0.0,
        firmware_watchdog_timeout_s: float = 0.02,
        clock_sync_uncertainty_s: float = 0.0001,
    ) -> None:
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._clock = monotonic_clock
        now = float(self._clock())
        if isinstance(control_epoch, bool) or not isinstance(control_epoch, int):
            raise TypeError("control_epoch must be an integer")
        if control_epoch <= 0:
            raise ValueError("control_epoch must be positive")
        if not math.isfinite(fc_tick_period_s) or fc_tick_period_s <= 0.0:
            raise ValueError("fc_tick_period_s must be finite and positive")
        if isinstance(timesync_generation, bool) or not isinstance(timesync_generation, int):
            raise TypeError("timesync_generation must be an integer")
        if timesync_generation <= 0:
            raise ValueError("timesync_generation must be positive")
        if not math.isfinite(timesync_age_s) or timesync_age_s < 0.0:
            raise ValueError("timesync_age_s must be finite and nonnegative")
        deadline = now + 1.0 if reservation_valid_until_s is None else reservation_valid_until_s
        self._reservation = FlightControllerBaselineReservation(
            fc_session_id=fc_session_id,
            control_epoch=control_epoch,
            transport_generation=1,
            timesync_generation=timesync_generation,
            target_fc_tick=target_fc_tick,
            baseline_version=baseline_version,
            timestamp_s=now,
            valid_until_s=deadline,
            baseline_thrusts_n=baseline_thrusts_n,
            negative_headroom_n=negative_headroom_n,
            positive_headroom_n=positive_headroom_n,
        )
        if isinstance(current_fc_tick, bool) or not isinstance(current_fc_tick, int):
            raise TypeError("current_fc_tick must be an integer")
        if current_fc_tick < 0 or current_fc_tick >= target_fc_tick:
            raise ValueError("current_fc_tick must be nonnegative and precede target_fc_tick")
        if not math.isfinite(firmware_watchdog_timeout_s) or firmware_watchdog_timeout_s <= 0.0:
            raise ValueError("firmware_watchdog_timeout_s must be finite and positive")
        if not math.isfinite(clock_sync_uncertainty_s) or clock_sync_uncertainty_s < 0.0:
            raise ValueError("clock_sync_uncertainty_s must be finite and nonnegative")
        self._current_fc_tick = current_fc_tick
        self._last_tick_clock_s = now
        self._transport_generation = 1
        self._control_epoch = control_epoch
        self._fc_tick_period_s = float(fc_tick_period_s)
        self._timesync_generation = timesync_generation
        self._timesync_age_s = float(timesync_age_s)
        self._firmware_watchdog_timeout_s = float(firmware_watchdog_timeout_s)
        self._clock_sync_uncertainty_s = float(clock_sync_uncertainty_s)
        self._connected = True
        self._healthy = True
        self._replacement_semantics_verified = True
        self._autonomous_expiry_clear_verified = True
        self._disconnect_clear_verified = True
        self._baseline_preservation_verified = True
        self._execution_feedback_verified = True
        self._live_baseline_thrusts_n = self._reservation.baseline_thrusts_n
        self._live_negative_headroom_n = self._reservation.negative_headroom_n
        self._live_positive_headroom_n = self._reservation.positive_headroom_n
        self._requested_residual: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        self._applied_residual: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        self._active = False
        self._active_valid_until_fc_tick: Optional[int] = None
        self._active_sequence: Optional[int] = None
        self._active_request_digest: Optional[str] = None
        self._active_headroom_reserve_n: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        self._active_reference_baseline_n: Tuple[float, ...] = baseline_thrusts_n
        self._active_maximum_baseline_deviation_n: Tuple[float, ...] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # Internal sentinel only; every wire-visible watermark remains uint64.
        self._clear_through_sequence = -1
        self._pending_stage: Optional[FlightControllerResidualStageRequest] = None
        self._last_stage_request: Optional[FlightControllerResidualStageRequest] = None
        self._last_stage_ack: Optional[FlightControllerResidualAck] = None
        self._last_clear_request: Optional[FlightControllerResidualClearRequest] = None
        self._last_clear_ack: Optional[FlightControllerResidualAck] = None
        self._feedback: Dict[
            Tuple[FlightControllerResidualOperation, int],
            FlightControllerResidualExecutionFeedback,
        ] = {}

    @property
    def physical_use_prohibited(self) -> bool:
        return True

    @property
    def last_stage_request(self) -> Optional[FlightControllerResidualStageRequest]:
        return self._last_stage_request

    @property
    def applied_residual_thrusts_n(self) -> Tuple[float, ...]:
        self._enforce_local_watchdog()
        return self._applied_residual

    @property
    def baseline_thrusts_n(self) -> Tuple[float, ...]:
        return self._live_baseline_thrusts_n

    @property
    def final_thrusts_n(self) -> Tuple[float, ...]:
        self._enforce_local_watchdog()
        return tuple(
            baseline + residual
            for baseline, residual in zip(
                self._live_baseline_thrusts_n,
                self._applied_residual,
            )
        )

    def status(self) -> FlightControllerResidualTransportStatus:
        self._enforce_local_watchdog()
        return FlightControllerResidualTransportStatus(
            timestamp_s=float(self._clock()),
            connected=self._connected,
            healthy=self._connected and self._healthy,
            transport_generation=self._transport_generation,
            fc_session_id=self._reservation.fc_session_id,
            control_epoch=self._control_epoch,
            current_fc_tick=self._current_fc_tick,
            fc_tick_period_s=self._fc_tick_period_s,
            timesync_generation=self._timesync_generation,
            timesync_age_s=self._timesync_age_s,
            firmware_watchdog_timeout_s=self._firmware_watchdog_timeout_s,
            clock_sync_uncertainty_s=self._clock_sync_uncertainty_s,
            firmware_hash=FAKE_FC_FIRMWARE_HASH,
            dialect_hash=FAKE_FC_DIALECT_HASH,
            mixer_hash=FAKE_FC_MIXER_HASH,
            mapping_hash=FAKE_FC_MAPPING_HASH,
            calibration_hash=FAKE_FC_CALIBRATION_HASH,
            residual_register_active=self._active,
            active_command_sequence=self._active_sequence,
            active_request_digest=self._active_request_digest,
            active_valid_until_fc_tick=self._active_valid_until_fc_tick,
            pending_stage_present=self._pending_stage is not None,
            pending_command_sequence=(
                None if self._pending_stage is None else self._pending_stage.sequence
            ),
            pending_request_digest=(
                None if self._pending_stage is None else self._pending_stage.request_digest
            ),
            pending_target_fc_tick=(
                None if self._pending_stage is None else self._pending_stage.target_fc_tick
            ),
            clear_through_command_sequence=(
                None if self._clear_through_sequence < 0 else self._clear_through_sequence
            ),
            residual_enabled=True,
            baseline_controller_active=True,
            allocator_ready=True,
            replacement_semantics_verified=self._replacement_semantics_verified,
            autonomous_expiry_clear_verified=self._autonomous_expiry_clear_verified,
            disconnect_clear_verified=self._disconnect_clear_verified,
            baseline_preservation_verified=self._baseline_preservation_verified,
            execution_feedback_verified=self._execution_feedback_verified,
        )

    def latest_baseline_reservation(
        self,
    ) -> Optional[FlightControllerBaselineReservation]:
        self._enforce_local_watchdog()
        if not self._connected or not self._healthy:
            return None
        return self._reservation

    async def stage_residual(
        self,
        request: FlightControllerResidualStageRequest,
    ) -> FlightControllerResidualAck:
        self._enforce_local_watchdog()
        now = float(self._clock())
        if not isinstance(request, FlightControllerResidualStageRequest):
            raise TypeError("request must be a FlightControllerResidualStageRequest")
        if request.control_epoch != self._control_epoch:
            return self._rejected_stage_ack(request, "CONTROL_EPOCH_MISMATCH")
        if request.transport_generation != self._transport_generation:
            return self._rejected_stage_ack(request, "TRANSPORT_GENERATION_MISMATCH")
        if request.sequence <= self._clear_through_sequence:
            return self._rejected_stage_ack(request, "CLEARED_SEQUENCE_REPLAY")
        if (
            self._last_stage_request is not None
            and request.sequence == self._last_stage_request.sequence
        ):
            if request == self._last_stage_request and self._last_stage_ack is not None:
                # Idempotent packet replay: return the original ACK without
                # re-staging, applying, or extending the residual lease.
                return self._last_stage_ack
            return self._rejected_stage_ack(request, "SEQUENCE_COLLISION")
        if (
            self._last_stage_request is not None
            and request.sequence < self._last_stage_request.sequence
        ):
            return self._rejected_stage_ack(request, "SEQUENCE_REPLAY")
        if not self._connected or not self._healthy:
            return self._rejected_stage_ack(request, "LINK_UNHEALTHY")
        reservation = self._reservation
        if not reservation.is_fresh(now):
            return self._rejected_stage_ack(request, "REFERENCE_RESERVATION_STALE")
        identity_matches = (
            request.fc_session_id == reservation.fc_session_id
            and request.control_epoch == reservation.control_epoch
            and request.transport_generation == reservation.transport_generation
            and request.timesync_generation == reservation.timesync_generation
            and request.target_fc_tick == reservation.target_fc_tick
            and request.baseline_version == reservation.baseline_version
            and request.baseline_reference_digest == reservation.reference_digest
        )
        if not identity_matches:
            return self._rejected_stage_ack(request, "BASELINE_IDENTITY_MISMATCH")
        if request.target_fc_tick <= self._current_fc_tick:
            return self._rejected_stage_ack(request, "TARGET_TICK_PASSED")
        if request.timestamp_s > now:
            return self._rejected_stage_ack(request, "SOURCE_TIMESTAMP_FROM_FUTURE")
        if request.valid_until_fc_tick <= request.target_fc_tick:
            return self._rejected_stage_ack(request, "FC_TICK_LEASE_INVALID")
        for index, residual in enumerate(request.applied_residual_thrusts_n):
            reserve = request.required_headroom_reserve_n[index]
            if residual > max(
                0.0, reservation.positive_headroom_n[index] - reserve
            ) or residual < -max(0.0, reservation.negative_headroom_n[index] - reserve):
                self._clear_register()
                return self._rejected_stage_ack(request, "HEADROOM_EXCEEDED")

        self._last_stage_request = request
        self._pending_stage = request
        ack = FlightControllerResidualAck(
            operation=FlightControllerResidualOperation.STAGE,
            operation_sequence=request.sequence,
            command_sequence=request.sequence,
            fc_session_id=request.fc_session_id,
            control_epoch=request.control_epoch,
            transport_generation=request.transport_generation,
            target_fc_tick=request.target_fc_tick,
            valid_until_fc_tick=request.valid_until_fc_tick,
            baseline_version=request.baseline_version,
            clear_through_command_sequence=None,
            request_digest=request.request_digest,
            timestamp_s=now,
            accepted=True,
            result_code="STAGED",
            message="replacement residual staged for the reserved FC tick",
        )
        self._last_stage_ack = ack
        return ack

    async def clear_residual(
        self,
        request: FlightControllerResidualClearRequest,
    ) -> FlightControllerResidualAck:
        self._enforce_local_watchdog()
        if not isinstance(request, FlightControllerResidualClearRequest):
            raise TypeError("request must be a FlightControllerResidualClearRequest")
        if request.control_epoch != self._control_epoch:
            return self._rejected_clear_ack(request, "CONTROL_EPOCH_MISMATCH")
        if request.transport_generation != self._transport_generation:
            return self._rejected_clear_ack(request, "TRANSPORT_GENERATION_MISMATCH")
        if (
            self._last_clear_request is not None
            and request.clear_sequence == self._last_clear_request.clear_sequence
        ):
            if request == self._last_clear_request and self._last_clear_ack is not None:
                return self._last_clear_ack
            return self._rejected_clear_ack(request, "CLEAR_SEQUENCE_COLLISION")
        if (
            self._last_clear_request is not None
            and request.clear_sequence < self._last_clear_request.clear_sequence
        ):
            return self._rejected_clear_ack(request, "CLEAR_SEQUENCE_REPLAY")
        if not self._connected or not self._healthy:
            return self._rejected_clear_ack(request, "LINK_UNHEALTHY")
        if request.fc_session_id != self._reservation.fc_session_id:
            return self._rejected_clear_ack(request, "SESSION_MISMATCH")
        self._last_clear_request = request
        highest_received_sequence = (
            -1 if self._last_stage_request is None else self._last_stage_request.sequence
        )
        self._clear_through_sequence = max(
            self._clear_through_sequence,
            request.clear_through_command_sequence,
            highest_received_sequence,
        )
        # CLEAR is an FC-local barrier: atomically discard every stage already
        # pending, even when the host had not yet observed its sequence.  The
        # promoted watermark additionally rejects any delayed replay already
        # received before this barrier.
        self._pending_stage = None
        self._clear_register()
        self._current_fc_tick += 1
        self._last_tick_clock_s = float(self._clock())
        ack = FlightControllerResidualAck(
            operation=FlightControllerResidualOperation.CLEAR,
            operation_sequence=request.clear_sequence,
            command_sequence=None,
            fc_session_id=request.fc_session_id,
            control_epoch=request.control_epoch,
            transport_generation=request.transport_generation,
            target_fc_tick=None,
            valid_until_fc_tick=None,
            baseline_version=None,
            clear_through_command_sequence=self._clear_through_sequence,
            request_digest=request.request_digest,
            timestamp_s=float(self._clock()),
            accepted=True,
            result_code="CLEARED",
            message="residual register cleared; baseline controller retained",
        )
        self._last_clear_ack = ack
        self._feedback[(FlightControllerResidualOperation.CLEAR, request.clear_sequence)] = (
            self._make_feedback(
                operation=FlightControllerResidualOperation.CLEAR,
                execution_result=FlightControllerResidualExecutionResult.CLEARED,
                operation_sequence=request.clear_sequence,
                command_sequence=None,
                request_digest=request.request_digest,
                required_headroom_reserve_n=(0.0, 0.0, 0.0, 0.0),
                maximum_baseline_deviation_n=(0.0, 0.0, 0.0, 0.0),
                valid_until_fc_tick=None,
                clear_through_command_sequence=self._clear_through_sequence,
                residual_active=False,
            )
        )
        return ack

    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        self._enforce_local_watchdog()
        if operation is FlightControllerResidualOperation.STAGE:
            request = self._pending_stage
            if request is not None and request.sequence == operation_sequence:
                if not self._connected or not self._healthy:
                    raise RuntimeError("FC link became unhealthy before activation")
                if request.sequence <= self._clear_through_sequence:
                    self._pending_stage = None
                    raise RuntimeError("staged residual was superseded by a clear watermark")
                if request.target_fc_tick < self._current_fc_tick:
                    self._pending_stage = None
                    self._requested_residual = request.applied_residual_thrusts_n
                    expired = self._current_fc_tick >= request.valid_until_fc_tick
                    self._clear_register()
                    feedback = self._make_feedback(
                        operation=FlightControllerResidualOperation.STAGE,
                        execution_result=(
                            FlightControllerResidualExecutionResult.EXPIRED
                            if expired
                            else FlightControllerResidualExecutionResult.GATE_REJECTED
                        ),
                        operation_sequence=request.sequence,
                        command_sequence=request.sequence,
                        request_digest=request.request_digest,
                        required_headroom_reserve_n=request.required_headroom_reserve_n,
                        maximum_baseline_deviation_n=(request.maximum_baseline_deviation_n),
                        valid_until_fc_tick=request.valid_until_fc_tick,
                        clear_through_command_sequence=None,
                        residual_active=False,
                    )
                    self._feedback[(operation, operation_sequence)] = feedback
                    return feedback
                now = float(self._clock())
                reservation = self._reservation
                if not reservation.is_fresh(now) or not (
                    request.fc_session_id == reservation.fc_session_id
                    and request.target_fc_tick == reservation.target_fc_tick
                    and request.baseline_version == reservation.baseline_version
                    and request.control_epoch == reservation.control_epoch
                    and request.transport_generation == reservation.transport_generation
                    and request.timesync_generation == reservation.timesync_generation
                    and request.baseline_reference_digest == reservation.reference_digest
                ):
                    self._pending_stage = None
                    raise RuntimeError("reference reservation changed or expired before activation")
                self._current_fc_tick = request.target_fc_tick
                self._last_tick_clock_s = float(self._clock())
                self._requested_residual = request.applied_residual_thrusts_n
                baseline_deviation_ok = all(
                    abs(live - reference) <= request.maximum_baseline_deviation_n[index]
                    for index, (live, reference) in enumerate(
                        zip(
                            self._live_baseline_thrusts_n,
                            reservation.baseline_thrusts_n,
                        )
                    )
                )
                if not baseline_deviation_ok:
                    self._clear_register()
                    feedback = self._make_feedback(
                        operation=FlightControllerResidualOperation.STAGE,
                        execution_result=(
                            FlightControllerResidualExecutionResult.BASELINE_DEVIATION_REJECTED
                        ),
                        operation_sequence=request.sequence,
                        command_sequence=request.sequence,
                        request_digest=request.request_digest,
                        required_headroom_reserve_n=request.required_headroom_reserve_n,
                        maximum_baseline_deviation_n=(request.maximum_baseline_deviation_n),
                        valid_until_fc_tick=request.valid_until_fc_tick,
                        clear_through_command_sequence=None,
                        residual_active=False,
                    )
                    self._feedback[(operation, operation_sequence)] = feedback
                    self._pending_stage = None
                    return feedback
                headroom_ok = all(
                    -max(
                        0.0,
                        self._live_negative_headroom_n[index]
                        - request.required_headroom_reserve_n[index],
                    )
                    <= residual
                    <= max(
                        0.0,
                        self._live_positive_headroom_n[index]
                        - request.required_headroom_reserve_n[index],
                    )
                    for index, residual in enumerate(request.applied_residual_thrusts_n)
                )
                if not headroom_ok:
                    self._clear_register()
                    feedback = self._make_feedback(
                        operation=FlightControllerResidualOperation.STAGE,
                        execution_result=(
                            FlightControllerResidualExecutionResult.HEADROOM_REJECTED
                        ),
                        operation_sequence=request.sequence,
                        command_sequence=request.sequence,
                        request_digest=request.request_digest,
                        required_headroom_reserve_n=request.required_headroom_reserve_n,
                        maximum_baseline_deviation_n=(request.maximum_baseline_deviation_n),
                        valid_until_fc_tick=request.valid_until_fc_tick,
                        clear_through_command_sequence=None,
                        residual_active=False,
                    )
                    self._feedback[(operation, operation_sequence)] = feedback
                    self._pending_stage = None
                    return feedback
                # This assignment is the key contract.  It is never ``+=`` and
                # the already-scaled payload is not multiplied by another gain.
                self._applied_residual = request.applied_residual_thrusts_n
                self._active = True
                self._active_sequence = request.sequence
                self._active_request_digest = request.request_digest
                self._active_headroom_reserve_n = request.required_headroom_reserve_n
                self._active_reference_baseline_n = reservation.baseline_thrusts_n
                self._active_maximum_baseline_deviation_n = request.maximum_baseline_deviation_n
                self._active_valid_until_fc_tick = request.valid_until_fc_tick
                feedback = self._make_feedback(
                    operation=FlightControllerResidualOperation.STAGE,
                    execution_result=FlightControllerResidualExecutionResult.APPLIED,
                    operation_sequence=request.sequence,
                    command_sequence=request.sequence,
                    request_digest=request.request_digest,
                    required_headroom_reserve_n=request.required_headroom_reserve_n,
                    maximum_baseline_deviation_n=request.maximum_baseline_deviation_n,
                    valid_until_fc_tick=request.valid_until_fc_tick,
                    clear_through_command_sequence=None,
                    residual_active=True,
                )
                self._feedback[(operation, operation_sequence)] = feedback
                self._pending_stage = None
        try:
            return self._feedback[(operation, operation_sequence)]
        except KeyError as exc:
            raise RuntimeError("matching execution feedback is unavailable") from exc

    def reserve_baseline(
        self,
        *,
        target_fc_tick: int,
        baseline_version: int,
        valid_until_s: float,
        baseline_thrusts_n: Optional[Tuple[float, ...]] = None,
        negative_headroom_n: Optional[Tuple[float, ...]] = None,
        positive_headroom_n: Optional[Tuple[float, ...]] = None,
    ) -> FlightControllerBaselineReservation:
        """Publish the next synthetic future reservation for a test cycle."""

        self._enforce_local_watchdog()
        previous = self._reservation
        if target_fc_tick <= self._current_fc_tick:
            raise ValueError("target_fc_tick must be later than the current FC tick")
        self._reservation = FlightControllerBaselineReservation(
            fc_session_id=previous.fc_session_id,
            control_epoch=self._control_epoch,
            transport_generation=self._transport_generation,
            timesync_generation=self._timesync_generation,
            target_fc_tick=target_fc_tick,
            baseline_version=baseline_version,
            timestamp_s=float(self._clock()),
            valid_until_s=valid_until_s,
            baseline_thrusts_n=(
                previous.baseline_thrusts_n if baseline_thrusts_n is None else baseline_thrusts_n
            ),
            negative_headroom_n=(
                previous.negative_headroom_n if negative_headroom_n is None else negative_headroom_n
            ),
            positive_headroom_n=(
                previous.positive_headroom_n if positive_headroom_n is None else positive_headroom_n
            ),
        )
        self._live_baseline_thrusts_n = self._reservation.baseline_thrusts_n
        self._live_negative_headroom_n = self._reservation.negative_headroom_n
        self._live_positive_headroom_n = self._reservation.positive_headroom_n
        return self._reservation

    def inject_live_allocation(
        self,
        *,
        baseline_thrusts_n: Optional[Tuple[float, ...]] = None,
        negative_headroom_n: Optional[Tuple[float, ...]] = None,
        positive_headroom_n: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """Change execution-tick values without rewriting the old reference."""

        reference = self._reservation
        candidate = FlightControllerBaselineReservation(
            fc_session_id=reference.fc_session_id,
            control_epoch=reference.control_epoch,
            transport_generation=reference.transport_generation,
            timesync_generation=reference.timesync_generation,
            target_fc_tick=reference.target_fc_tick,
            baseline_version=reference.baseline_version,
            timestamp_s=reference.timestamp_s,
            valid_until_s=reference.valid_until_s,
            baseline_thrusts_n=(
                self._live_baseline_thrusts_n if baseline_thrusts_n is None else baseline_thrusts_n
            ),
            negative_headroom_n=(
                self._live_negative_headroom_n
                if negative_headroom_n is None
                else negative_headroom_n
            ),
            positive_headroom_n=(
                self._live_positive_headroom_n
                if positive_headroom_n is None
                else positive_headroom_n
            ),
        )
        self._live_baseline_thrusts_n = candidate.baseline_thrusts_n
        self._live_negative_headroom_n = candidate.negative_headroom_n
        self._live_positive_headroom_n = candidate.positive_headroom_n

    def inject_connection(self, connected: bool) -> None:
        if type(connected) is not bool:
            raise TypeError("connected must be a bool")
        if connected != self._connected:
            self._transport_generation += 1
        self._connected = connected
        self._healthy = connected
        if not connected:
            # Model the mandatory FC-side link-lease watchdog.  The baseline
            # vector itself is intentionally unchanged.
            self._pending_stage = None
            self._clear_register()

    def inject_contract_verification(
        self,
        *,
        replacement: bool = True,
        expiry_clear: bool = True,
        disconnect_clear: bool = True,
        baseline_preservation: bool = True,
        execution_feedback: bool = True,
    ) -> None:
        values = (
            replacement,
            expiry_clear,
            disconnect_clear,
            baseline_preservation,
            execution_feedback,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("contract-verification flags must be booleans")
        self._replacement_semantics_verified = replacement
        self._autonomous_expiry_clear_verified = expiry_clear
        self._disconnect_clear_verified = disconnect_clear
        self._baseline_preservation_verified = baseline_preservation
        self._execution_feedback_verified = execution_feedback

    def inject_timing_quality(
        self,
        *,
        timesync_age_s: Optional[float] = None,
        clock_sync_uncertainty_s: Optional[float] = None,
        firmware_watchdog_timeout_s: Optional[float] = None,
        advance_timesync_generation: bool = False,
    ) -> None:
        """Inject transport timing metadata for fail-closed host tests."""

        if type(advance_timesync_generation) is not bool:
            raise TypeError("advance_timesync_generation must be a bool")
        if timesync_age_s is not None:
            if not math.isfinite(timesync_age_s) or timesync_age_s < 0.0:
                raise ValueError("timesync_age_s must be finite and nonnegative")
            self._timesync_age_s = float(timesync_age_s)
        if clock_sync_uncertainty_s is not None:
            if not math.isfinite(clock_sync_uncertainty_s) or clock_sync_uncertainty_s < 0.0:
                raise ValueError("clock_sync_uncertainty_s must be finite and nonnegative")
            self._clock_sync_uncertainty_s = float(clock_sync_uncertainty_s)
        if firmware_watchdog_timeout_s is not None:
            if not math.isfinite(firmware_watchdog_timeout_s) or firmware_watchdog_timeout_s <= 0.0:
                raise ValueError("firmware_watchdog_timeout_s must be finite and positive")
            self._firmware_watchdog_timeout_s = float(firmware_watchdog_timeout_s)
        if advance_timesync_generation:
            self._timesync_generation += 1

    def reboot(self) -> FlightControllerBaselineReservation:
        """Change the FC session ID and invalidate every staged command."""

        old = self._reservation
        self._clear_register()
        self._transport_generation += 1
        self._current_fc_tick = 0
        self._last_stage_request = None
        self._last_stage_ack = None
        self._last_clear_request = None
        self._last_clear_ack = None
        self._pending_stage = None
        self._clear_through_sequence = -1
        self._feedback.clear()
        self._reservation = FlightControllerBaselineReservation(
            fc_session_id=old.fc_session_id + 1,
            control_epoch=self._control_epoch,
            transport_generation=self._transport_generation,
            timesync_generation=self._timesync_generation,
            target_fc_tick=2,
            baseline_version=1,
            timestamp_s=float(self._clock()),
            valid_until_s=float(self._clock()) + 1.0,
            baseline_thrusts_n=old.baseline_thrusts_n,
            negative_headroom_n=old.negative_headroom_n,
            positive_headroom_n=old.positive_headroom_n,
        )
        self._live_baseline_thrusts_n = self._reservation.baseline_thrusts_n
        self._live_negative_headroom_n = self._reservation.negative_headroom_n
        self._live_positive_headroom_n = self._reservation.positive_headroom_n
        return self._reservation

    def _enforce_local_watchdog(self) -> None:
        self._synchronise_fc_tick()
        if self._active and (
            not self._connected
            or self._active_valid_until_fc_tick is None
            or self._current_fc_tick >= self._active_valid_until_fc_tick
            or any(
                residual
                > max(
                    0.0,
                    self._live_positive_headroom_n[index] - self._active_headroom_reserve_n[index],
                )
                or residual
                < -max(
                    0.0,
                    self._live_negative_headroom_n[index] - self._active_headroom_reserve_n[index],
                )
                for index, residual in enumerate(self._applied_residual)
            )
            or any(
                abs(live - reference) > self._active_maximum_baseline_deviation_n[index]
                for index, (live, reference) in enumerate(
                    zip(
                        self._live_baseline_thrusts_n,
                        self._active_reference_baseline_n,
                    )
                )
            )
        ):
            self._clear_register()

    def _synchronise_fc_tick(self) -> None:
        now = float(self._clock())
        elapsed = now - self._last_tick_clock_s
        if elapsed <= 0.0:
            return
        ticks = int(elapsed / self._fc_tick_period_s)
        if ticks <= 0:
            return
        self._current_fc_tick += ticks
        self._last_tick_clock_s += ticks * self._fc_tick_period_s

    def _clear_register(self) -> None:
        self._requested_residual = (0.0, 0.0, 0.0, 0.0)
        self._applied_residual = (0.0, 0.0, 0.0, 0.0)
        self._active = False
        self._active_valid_until_fc_tick = None
        self._active_sequence = None
        self._active_request_digest = None
        self._active_headroom_reserve_n = (0.0, 0.0, 0.0, 0.0)
        self._active_reference_baseline_n = self._live_baseline_thrusts_n
        self._active_maximum_baseline_deviation_n = (0.0, 0.0, 0.0, 0.0)

    def _make_feedback(
        self,
        *,
        operation: FlightControllerResidualOperation,
        execution_result: FlightControllerResidualExecutionResult,
        operation_sequence: int,
        command_sequence: Optional[int],
        request_digest: str,
        required_headroom_reserve_n: Tuple[float, ...],
        maximum_baseline_deviation_n: Tuple[float, ...],
        valid_until_fc_tick: Optional[int],
        clear_through_command_sequence: Optional[int],
        residual_active: bool,
    ) -> FlightControllerResidualExecutionFeedback:
        baseline = self._live_baseline_thrusts_n
        applied = self._applied_residual
        return FlightControllerResidualExecutionFeedback(
            operation=operation,
            execution_result=execution_result,
            operation_sequence=operation_sequence,
            command_sequence=command_sequence,
            fc_session_id=self._reservation.fc_session_id,
            control_epoch=self._control_epoch,
            transport_generation=self._transport_generation,
            timesync_generation=self._timesync_generation,
            execution_fc_tick=self._current_fc_tick,
            valid_until_fc_tick=valid_until_fc_tick,
            baseline_version=self._reservation.baseline_version,
            execution_baseline_version=max(1, self._current_fc_tick),
            clear_through_command_sequence=clear_through_command_sequence,
            request_digest=request_digest,
            timestamp_s=float(self._clock()),
            baseline_thrusts_n=baseline,
            requested_residual_thrusts_n=self._requested_residual,
            applied_residual_thrusts_n=applied,
            final_thrusts_n=tuple(
                base_value + residual for base_value, residual in zip(baseline, applied)
            ),
            negative_headroom_n=self._live_negative_headroom_n,
            positive_headroom_n=self._live_positive_headroom_n,
            required_headroom_reserve_n=required_headroom_reserve_n,
            maximum_baseline_deviation_n=maximum_baseline_deviation_n,
            saturation_mask=(False, False, False, False),
            saturation_scale=1.0,
            residual_active=residual_active,
            baseline_controller_active=True,
            residual_addition_terms=1 if residual_active else 0,
        )

    def _rejected_stage_ack(
        self,
        request: FlightControllerResidualStageRequest,
        code: str,
    ) -> FlightControllerResidualAck:
        return FlightControllerResidualAck(
            operation=FlightControllerResidualOperation.STAGE,
            operation_sequence=request.sequence,
            command_sequence=request.sequence,
            fc_session_id=request.fc_session_id,
            control_epoch=request.control_epoch,
            transport_generation=request.transport_generation,
            target_fc_tick=request.target_fc_tick,
            valid_until_fc_tick=request.valid_until_fc_tick,
            baseline_version=request.baseline_version,
            clear_through_command_sequence=None,
            request_digest=request.request_digest,
            timestamp_s=float(self._clock()),
            accepted=False,
            result_code=code,
            message="synthetic FC rejected the residual",
        )

    def _rejected_clear_ack(
        self,
        request: FlightControllerResidualClearRequest,
        code: str,
    ) -> FlightControllerResidualAck:
        return FlightControllerResidualAck(
            operation=FlightControllerResidualOperation.CLEAR,
            operation_sequence=request.clear_sequence,
            command_sequence=None,
            fc_session_id=request.fc_session_id,
            control_epoch=request.control_epoch,
            transport_generation=request.transport_generation,
            target_fc_tick=None,
            valid_until_fc_tick=None,
            baseline_version=None,
            clear_through_command_sequence=request.clear_through_command_sequence,
            request_digest=request.request_digest,
            timestamp_s=float(self._clock()),
            accepted=False,
            result_code=code,
            message="synthetic FC rejected the clear request",
        )


__all__ = [
    "FAKE_FC_CALIBRATION_HASH",
    "FAKE_FC_DIALECT_HASH",
    "FAKE_FC_FIRMWARE_HASH",
    "FAKE_FC_MAPPING_HASH",
    "FAKE_FC_MIXER_HASH",
    "FakeFlightControllerResidualTransport",
]
