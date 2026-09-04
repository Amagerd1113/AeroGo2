from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Optional, Tuple

import pytest

from aerogo2.bridges.fake_flight_controller_residual import (
    FAKE_FC_CALIBRATION_HASH,
    FAKE_FC_DIALECT_HASH,
    FAKE_FC_FIRMWARE_HASH,
    FAKE_FC_MAPPING_HASH,
    FAKE_FC_MIXER_HASH,
    FakeFlightControllerResidualTransport,
)
from aerogo2.common.clock import ManualClock
from aerogo2.landing.impact_aware.integration import (
    FC_RESIDUAL_REPLACE_SEMANTICS,
    FC_RESIDUAL_ROTOR_ORDER,
    FC_RESIDUAL_THRUST_UNIT,
    FlightControllerResidualClearRequest,
    FlightControllerResidualExecutionFeedback,
    FlightControllerResidualExecutionResult,
    FlightControllerResidualOperation,
    FlightControllerResidualSink,
    FlightControllerResidualSinkConfig,
    FlightControllerResidualStageRequest,
    FlightControllerResidualState,
    FlightControllerRotorResidualCommand,
)


def _config() -> FlightControllerResidualSinkConfig:
    return FlightControllerResidualSinkConfig(
        maximum_command_ttl_s=0.05,
        maximum_baseline_age_s=0.01,
        maximum_status_age_s=0.01,
        acknowledgement_timeout_s=0.02,
        execution_feedback_timeout_s=0.02,
        maximum_clock_sync_uncertainty_s=0.001,
        maximum_timesync_age_s=0.1,
        minimum_target_lead_ticks=1,
        maximum_target_lead_ticks=5,
        residual_lease_ticks=10,
        control_epoch=1,
        expected_firmware_hash=FAKE_FC_FIRMWARE_HASH,
        expected_dialect_hash=FAKE_FC_DIALECT_HASH,
        expected_mixer_hash=FAKE_FC_MIXER_HASH,
        expected_mapping_hash=FAKE_FC_MAPPING_HASH,
        expected_calibration_hash=FAKE_FC_CALIBRATION_HASH,
        headroom_reserve_n=(0.05, 0.05, 0.05, 0.05),
        maximum_baseline_deviation_n=(0.25, 0.25, 0.25, 0.25),
        thrust_match_tolerance_n=1.0e-6,
    )


def _command(
    *,
    sequence: int = 1,
    timestamp_s: float = 10.0,
    valid_until_s: float = 10.05,
    fc_session_id: int = 1,
    target_fc_tick: int = 102,
    baseline_version: int = 1,
    baseline_timestamp_s: float = 10.0,
    residual_n: float = 0.2,
) -> FlightControllerRotorResidualCommand:
    return FlightControllerRotorResidualCommand(
        sequence=sequence,
        timestamp_s=timestamp_s,
        valid_until_s=valid_until_s,
        fc_session_id=fc_session_id,
        target_fc_tick=target_fc_tick,
        baseline_version=baseline_version,
        baseline_timestamp_s=baseline_timestamp_s,
        baseline_thrusts_n=(5.0, 5.0, 5.0, 5.0),
        transport_raw_residual_thrusts_n=(2.0, 2.0, 2.0, 2.0),
        applied_residual_thrusts_n=(residual_n,) * 4,
        applied_total_thrusts_n=(5.0 + residual_n,) * 4,
        correction_gain=residual_n / 2.0,
        transport_target_semantics="gain_limited_algebraic_reconstruction",
    )


def _transport(
    clock: ManualClock,
    *,
    positive_headroom_n: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0),
) -> FakeFlightControllerResidualTransport:
    return FakeFlightControllerResidualTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
        positive_headroom_n=positive_headroom_n,
    )


def _stage_request(
    command: FlightControllerRotorResidualCommand,
    transport: FakeFlightControllerResidualTransport,
) -> FlightControllerResidualStageRequest:
    reference = transport.latest_baseline_reservation()
    assert reference is not None
    return FlightControllerResidualStageRequest(
        fc_session_id=command.fc_session_id,
        control_epoch=1,
        transport_generation=1,
        sequence=command.sequence,
        timestamp_s=command.timestamp_s,
        timesync_generation=1,
        target_fc_tick=command.target_fc_tick,
        valid_until_fc_tick=command.target_fc_tick + 10,
        baseline_version=command.baseline_version,
        baseline_reference_digest=reference.reference_digest,
        applied_residual_thrusts_n=command.applied_residual_thrusts_n,
        required_headroom_reserve_n=(0.05, 0.05, 0.05, 0.05),
        maximum_baseline_deviation_n=(0.25, 0.25, 0.25, 0.25),
    )


@pytest.mark.asyncio
async def test_success_freezes_units_order_and_already_scaled_replace_semantics() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )

    result = await sink.send_rotor_residual(_command())

    assert result.ok, result.message
    assert result.code == "FC_RESIDUAL_EXECUTION_VERIFIED"
    assert result.data["thrust_unit"] == FC_RESIDUAL_THRUST_UNIT == "N"
    assert (
        result.data["rotor_order"]
        == FC_RESIDUAL_ROTOR_ORDER
        == (
            "RR",
            "LF",
            "LR",
            "RF",
        )
    )
    assert result.data["fc_multiplies_kappa"] is False
    assert transport.applied_residual_thrusts_n == pytest.approx((0.2,) * 4)
    assert transport.final_thrusts_n == pytest.approx((5.2,) * 4)
    request = transport.last_stage_request
    assert isinstance(request, FlightControllerResidualStageRequest)
    assert request.application_semantics == FC_RESIDUAL_REPLACE_SEMANTICS
    assert request.applied_residual_thrusts_n == pytest.approx((0.2,) * 4)
    assert not hasattr(request, "correction_gain")
    assert not hasattr(request, "transport_raw_residual_thrusts_n")


@pytest.mark.asyncio
async def test_explicit_clear_zeros_only_residual_and_preserves_fc_baseline() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    baseline = transport.baseline_thrusts_n
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    assert (await sink.send_rotor_residual(_command())).ok

    result = await sink.clear_rotor_residual("operator abort")

    assert result.ok, result.message
    assert result.code == "FC_RESIDUAL_CLEAR_VERIFIED"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    assert transport.baseline_thrusts_n == baseline
    assert transport.final_thrusts_n == baseline
    status = sink.status()
    assert status.clear_confirmed
    assert status.clear_ack_timestamp_s is not None
    assert status.clear_execution_timestamp_s is not None
    assert status.clear_ack_timestamp_s <= status.clear_execution_timestamp_s <= status.timestamp_s


@pytest.mark.asyncio
async def test_fake_fc_packet_replay_is_idempotent_and_never_adds_twice() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    command = _command()
    request = _stage_request(command, transport)

    first = await transport.stage_residual(request)
    second = await transport.stage_residual(request)
    await transport.wait_execution_feedback(
        FlightControllerResidualOperation.STAGE,
        request.sequence,
    )

    assert first == second
    assert transport.applied_residual_thrusts_n == pytest.approx((0.2,) * 4)
    assert transport.final_thrusts_n == pytest.approx((5.2,) * 4)


@pytest.mark.asyncio
async def test_residual_ttl_expires_locally_on_fc_without_removing_baseline() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    baseline = transport.baseline_thrusts_n
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    assert (await sink.send_rotor_residual(_command())).ok

    clock.advance(0.05)

    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    assert transport.baseline_thrusts_n == baseline
    observed = await sink.watchdog()
    assert not observed.ok
    assert observed.code == "FC_RESIDUAL_WATCHDOG_CLEARED"


@pytest.mark.asyncio
async def test_fc_disconnect_clears_residual_locally_and_keeps_baseline() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    baseline = transport.baseline_thrusts_n
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    assert (await sink.send_rotor_residual(_command())).ok

    transport.inject_connection(False)

    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    assert transport.baseline_thrusts_n == baseline
    observed = await sink.watchdog()
    assert not observed.ok
    assert observed.code == "FC_RESIDUAL_WATCHDOG_AMBIGUOUS"
    assert observed.data["clear_confirmed"] is False


@pytest.mark.asyncio
async def test_headroom_violation_rejects_whole_vector_instead_of_silent_clip() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock, positive_headroom_n=(0.20,) * 4)
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )

    result = await sink.send_rotor_residual(_command(residual_n=0.2))

    assert not result.ok
    assert result.code == "FC_RESIDUAL_HEADROOM_EXCEEDED"
    assert result.data["clear_confirmed"] is True
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_session_or_baseline_identity_mismatch_fails_closed() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )

    mismatch = await sink.send_rotor_residual(_command(baseline_version=2))

    assert not mismatch.ok
    assert mismatch.code == "FC_RESIDUAL_BASELINE_VERSION_MISMATCH"
    assert mismatch.data["clear_confirmed"] is True


@pytest.mark.asyncio
async def test_unverified_fc_local_watchdog_contract_is_never_armed() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    transport.inject_contract_verification(disconnect_clear=False)
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_STATUS_OR_IDENTITY_UNSAFE"
    assert result.data["clear_confirmed"] is False


class _SaturatingFeedbackTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        feedback = await super().wait_execution_feedback(operation, operation_sequence)
        if operation is not FlightControllerResidualOperation.STAGE:
            return feedback
        half = tuple(value * 0.5 for value in feedback.applied_residual_thrusts_n)
        return replace(
            feedback,
            applied_residual_thrusts_n=half,
            final_thrusts_n=tuple(
                baseline + residual for baseline, residual in zip(feedback.baseline_thrusts_n, half)
            ),
            saturation_mask=(True, True, True, True),
            saturation_scale=0.5,
        )


@pytest.mark.asyncio
async def test_saturated_execution_readback_is_rejected_and_cleared() -> None:
    clock = ManualClock(10.0)
    transport = _SaturatingFeedbackTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_SATURATED"
    assert result.data["clear_confirmed"] is True
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _BlockingStageTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stage_started = asyncio.Event()

    async def stage_residual(
        self,
        request: FlightControllerResidualStageRequest,
    ):  # type: ignore[no-untyped-def]
        del request
        self.stage_started.set()
        await asyncio.Future()


class _AckHeldStageTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.pending_visible = asyncio.Event()
        self.release_ack = asyncio.Event()

    async def stage_residual(
        self,
        request: FlightControllerResidualStageRequest,
    ):  # type: ignore[no-untyped-def]
        ack = await super().stage_residual(request)
        self.pending_visible.set()
        await self.release_ack.wait()
        return ack


@pytest.mark.asyncio
async def test_status_separates_consumed_pending_and_executed_sequences() -> None:
    clock = ManualClock(10.0)
    transport = _AckHeldStageTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    operation = asyncio.create_task(sink.send_rotor_residual(_command(sequence=7)))
    pending_wait = asyncio.create_task(transport.pending_visible.wait())
    done, _ = await asyncio.wait(
        (operation, pending_wait),
        timeout=1.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if operation in done:
        result = operation.result()
        pending_wait.cancel()
        await asyncio.gather(pending_wait, return_exceptions=True)
        pytest.fail(
            "residual staging returned before its pending state was observable: "
            f"{result.code}: {result.message}"
        )
    assert pending_wait in done, "residual staging did not expose pending state within 1 s"

    pending = sink.status()
    assert pending.healthy
    assert pending.residual_state is FlightControllerResidualState.STAGE_PENDING
    assert pending.last_sequence == 7  # consumed anti-replay watermark
    assert pending.active_command_sequence is None
    assert pending.pending_command_sequence == 7
    assert pending.pending_started_s == 10.0
    assert pending.pending_valid_until_s == 10.05

    transport.release_ack.set()
    assert (await operation).ok
    active = sink.status()
    assert active.residual_state is FlightControllerResidualState.ACTIVE
    assert active.last_sequence == 7
    assert active.active_command_sequence == 7
    assert active.pending_command_sequence is None


@pytest.mark.asyncio
async def test_cancelled_stage_waits_for_nonabandonable_clear() -> None:
    clock = ManualClock(10.0)
    transport = _BlockingStageTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    task = asyncio.create_task(sink.send_rotor_residual(_command()))
    await transport.stage_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    status = sink.status()
    assert status.fault_latched
    assert status.clear_confirmed
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_replay_is_rejected_by_sink_and_latches_session() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(
        transport,
        _config(),
        monotonic_clock=clock.monotonic,
    )
    assert (await sink.send_rotor_residual(_command())).ok
    await sink.clear_rotor_residual("cycle complete")
    transport.reserve_baseline(
        target_fc_tick=105,
        baseline_version=2,
        valid_until_s=10.2,
    )

    replay = await sink.send_rotor_residual(_command())

    assert not replay.ok
    assert replay.code == "FC_RESIDUAL_COMMAND_REPLAY"


def test_protocol_types_reject_boolean_sequences_and_negative_thrust() -> None:
    with pytest.raises(TypeError, match="sequence"):
        replace(_command(), sequence=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="baseline_thrusts_n"):
        replace(_command(), baseline_thrusts_n=(-1.0, 5.0, 5.0, 5.0))


def test_sink_config_requires_bounded_target_tick_window() -> None:
    with pytest.raises(ValueError, match="maximum_target_lead_ticks"):
        replace(_config(), minimum_target_lead_ticks=5, maximum_target_lead_ticks=4)


def test_fake_transport_is_explicitly_prohibited_for_physical_use() -> None:
    clock = ManualClock(10.0)
    assert _transport(clock).physical_use_prohibited is True


@pytest.mark.asyncio
async def test_stage_ack_does_not_apply_until_target_tick_feedback() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    request = _stage_request(_command(), transport)

    ack = await transport.stage_residual(request)

    assert ack.accepted
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    feedback = await transport.wait_execution_feedback(
        FlightControllerResidualOperation.STAGE,
        request.sequence,
    )
    assert feedback.execution_fc_tick == request.target_fc_tick
    assert feedback.request_digest == request.request_digest
    assert transport.applied_residual_thrusts_n == pytest.approx((0.2,) * 4)


@pytest.mark.asyncio
async def test_clear_watermark_rejects_a_delayed_stage_and_prevents_reactivation() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    stage = _stage_request(_command(), transport)
    clear = FlightControllerResidualClearRequest(
        fc_session_id=1,
        control_epoch=1,
        transport_generation=1,
        clear_sequence=1,
        clear_through_command_sequence=stage.sequence,
        timestamp_s=clock.monotonic(),
        reason="cancelled stage",
    )
    assert (await transport.clear_residual(clear)).accepted

    late_ack = await transport.stage_residual(stage)

    assert not late_ack.accepted
    assert late_ack.result_code == "CLEARED_SEQUENCE_REPLAY"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _BlockingClearTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.clear_started = asyncio.Event()
        self.release_clear = asyncio.Event()

    async def clear_residual(self, request: FlightControllerResidualClearRequest):  # type: ignore[no-untyped-def]
        self.clear_started.set()
        await self.release_clear.wait()
        return await super().clear_residual(request)


@pytest.mark.asyncio
async def test_cancelled_public_clear_finishes_nonabandonable_transaction() -> None:
    clock = ManualClock(10.0)
    transport = _BlockingClearTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    task = asyncio.create_task(sink.clear_rotor_residual("operator clear"))
    await transport.clear_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    transport.release_clear.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    status = sink.status()
    assert status.fault_latched
    assert status.residual_state is FlightControllerResidualState.CONFIRMED_ZERO
    assert status.clear_confirmed
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_cancelled_watchdog_clear_finishes_before_cancellation_propagates() -> None:
    clock = ManualClock(10.0)
    transport = _BlockingClearTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    clock.advance(0.05)
    task = asyncio.create_task(sink.watchdog())
    await transport.clear_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    transport.release_clear.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    status = sink.status()
    assert status.fault_latched
    assert status.clear_confirmed
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_expired_lease_latches_and_cannot_be_washed_away_by_next_send() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    clock.advance(0.05)

    assert sink.status().fault_latched
    current_tick = transport.status().current_fc_tick
    assert current_tick is not None
    transport.reserve_baseline(
        target_fc_tick=current_tick + 2,
        baseline_version=2,
        valid_until_s=10.2,
    )
    result = await sink.send_rotor_residual(
        _command(
            sequence=2,
            timestamp_s=10.05,
            valid_until_s=10.10,
            target_fc_tick=current_tick + 2,
            baseline_version=2,
            baseline_timestamp_s=10.05,
        )
    )

    assert not result.ok
    assert result.code == "FC_RESIDUAL_SESSION_LATCHED"


class _DisconnectAfterFeedbackTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        feedback = await super().wait_execution_feedback(operation, operation_sequence)
        if operation is FlightControllerResidualOperation.STAGE:
            self.inject_connection(False)
        return feedback


@pytest.mark.asyncio
async def test_disconnect_after_feedback_is_rejected_by_post_execution_fence() -> None:
    clock = ManualClock(10.0)
    transport = _DisconnectAfterFeedbackTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_POST_STATUS_UNSAFE"
    assert result.data["clear_confirmed"] is False
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_fake_fc_rejects_reference_that_expires_between_host_check_and_stage() -> None:
    clock = ManualClock(10.0)
    transport = FakeFlightControllerResidualTransport(
        monotonic_clock=clock.monotonic,
        fc_tick_period_s=1.0,
        reservation_valid_until_s=10.001,
    )
    request = _stage_request(_command(), transport)
    clock.advance(0.002)

    ack = await transport.stage_residual(request)

    assert not ack.accepted
    assert ack.result_code == "REFERENCE_RESERVATION_STALE"


@pytest.mark.asyncio
async def test_fake_fc_rejects_out_of_order_clear_sequence() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)

    async def clear(sequence: int):  # type: ignore[no-untyped-def]
        request = FlightControllerResidualClearRequest(
            fc_session_id=1,
            control_epoch=1,
            transport_generation=1,
            clear_sequence=sequence,
            clear_through_command_sequence=0,
            timestamp_s=clock.monotonic(),
            reason=f"clear {sequence}",
        )
        return await transport.clear_residual(request)

    assert (await clear(2)).accepted
    assert (await clear(3)).accepted
    replay = await clear(1)
    assert not replay.accepted
    assert replay.result_code == "CLEAR_SEQUENCE_REPLAY"


class _ChangedLiveBaselineTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        if operation is FlightControllerResidualOperation.STAGE:
            self.inject_live_allocation(
                baseline_thrusts_n=(5.1, 5.1, 5.1, 5.1),
            )
        return await super().wait_execution_feedback(operation, operation_sequence)


@pytest.mark.asyncio
async def test_execution_uses_same_tick_live_baseline_not_frozen_reference_value() -> None:
    clock = ManualClock(10.0)
    transport = _ChangedLiveBaselineTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert result.ok, result.message
    assert transport.final_thrusts_n == pytest.approx((5.3,) * 4)
    assert result.data["execution_baseline_version"] == 102


class _ReducedLiveHeadroomTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        if operation is FlightControllerResidualOperation.STAGE:
            self.inject_live_allocation(
                positive_headroom_n=(0.2,) * 4,
            )
        return await super().wait_execution_feedback(operation, operation_sequence)


@pytest.mark.asyncio
async def test_same_tick_headroom_must_still_include_host_reserve() -> None:
    clock = ManualClock(10.0)
    transport = _ReducedLiveHeadroomTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_EXECUTION_REJECTED"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _AdvancingStatusClockTransport(FakeFlightControllerResidualTransport):
    def __init__(self, clock: ManualClock, **kwargs: object) -> None:
        self._test_clock = clock
        super().__init__(monotonic_clock=clock.monotonic, **kwargs)

    def status(self):  # type: ignore[no-untyped-def]
        self._test_clock.advance(0.0001)
        return super().status()


@pytest.mark.asyncio
async def test_status_timestamp_sampled_inside_transport_is_not_false_future() -> None:
    clock = ManualClock(10.0)
    transport = _AdvancingStatusClockTransport(
        clock,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    staged = await sink.send_rotor_residual(_command())
    cleared = await sink.clear_rotor_residual("timestamp-race regression")

    assert staged.ok, staged.message
    assert cleared.ok, cleared.message


@pytest.mark.asyncio
async def test_reconnect_invalidates_old_baseline_reference_until_republished() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    transport.inject_connection(False)
    transport.inject_connection(True)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_BASELINE_VERSION_MISMATCH"
    assert transport.last_stage_request is None


@pytest.mark.asyncio
async def test_active_residual_quick_reconnect_is_latched_before_next_send() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok

    transport.inject_connection(False)
    transport.inject_connection(True)
    observed = sink.status()

    assert not observed.healthy
    assert observed.fault_latched
    assert observed.residual_state is FlightControllerResidualState.AMBIGUOUS
    reconciled = await sink.watchdog()
    assert not reconciled.ok
    assert reconciled.code == "FC_RESIDUAL_WATCHDOG_CLEARED"
    assert sink.status().fault_latched


class _StatusFaultDuringStageTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stage_started = asyncio.Event()
        self.release_stage = asyncio.Event()
        self.report_unhealthy_once = False

    async def stage_residual(self, request: FlightControllerResidualStageRequest):  # type: ignore[no-untyped-def]
        self.stage_started.set()
        await self.release_stage.wait()
        return await super().stage_residual(request)

    def status(self):  # type: ignore[no-untyped-def]
        observed = super().status()
        if self.report_unhealthy_once:
            self.report_unhealthy_once = False
            return replace(observed, healthy=False)
        return observed


@pytest.mark.asyncio
async def test_concurrent_status_fault_cannot_be_washed_out_by_stage_success() -> None:
    clock = ManualClock(10.0)
    transport = _StatusFaultDuringStageTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    send_task = asyncio.create_task(sink.send_rotor_residual(_command()))
    await transport.stage_started.wait()

    transport.report_unhealthy_once = True
    concurrent_status = sink.status()
    transport.release_stage.set()
    result = await send_task

    assert concurrent_status.fault_latched
    assert not result.ok
    assert result.code == "FC_RESIDUAL_CONCURRENT_FAULT_LATCHED"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _NeverCompletesClearTransport(FakeFlightControllerResidualTransport):
    async def clear_residual(self, request: FlightControllerResidualClearRequest):  # type: ignore[no-untyped-def]
        del request
        await asyncio.Future()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_explicit_clear_timeout_latches_and_blocks_next_stage() -> None:
    clock = ManualClock(10.0)
    transport = _NeverCompletesClearTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok

    clear_result = await sink.clear_rotor_residual("force timeout")
    retry_result = await sink.send_rotor_residual(
        _command(sequence=2, timestamp_s=10.001, valid_until_s=10.049)
    )

    assert not clear_result.ok
    assert clear_result.code == "FC_RESIDUAL_CLEAR_TIMEOUT"
    assert sink.status().fault_latched
    assert not retry_result.ok
    assert retry_result.code == "FC_RESIDUAL_SESSION_LATCHED"


class _TimingDegradesAfterFeedbackTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        feedback = await super().wait_execution_feedback(operation, operation_sequence)
        if operation is FlightControllerResidualOperation.STAGE:
            self.inject_timing_quality(clock_sync_uncertainty_s=1.0)
        return feedback


@pytest.mark.asyncio
async def test_timing_quality_degradation_after_feedback_fails_post_fence() -> None:
    clock = ManualClock(10.0)
    transport = _TimingDegradesAfterFeedbackTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_POST_STATUS_UNSAFE"
    assert sink.status().fault_latched
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _ExcessiveBaselineDeviationTransport(FakeFlightControllerResidualTransport):
    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        if operation is FlightControllerResidualOperation.STAGE:
            self.inject_live_allocation(baseline_thrusts_n=(0.1, 0.1, 0.1, 0.1))
        return await super().wait_execution_feedback(operation, operation_sequence)


@pytest.mark.asyncio
async def test_excessive_live_baseline_model_error_is_atomically_rejected() -> None:
    clock = ManualClock(10.0)
    transport = _ExcessiveBaselineDeviationTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_EXECUTION_REJECTED"
    assert "baseline_deviation_rejected" in result.message
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_active_register_early_clear_is_detected_and_latched() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok

    transport.inject_live_allocation(baseline_thrusts_n=(6.0, 6.0, 6.0, 6.0))
    observed = await sink.watchdog()

    assert not observed.ok
    assert observed.code == "FC_RESIDUAL_WATCHDOG_CLEARED"
    assert sink.status().fault_latched
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    still_latched = await sink.watchdog()
    assert not still_latched.ok
    assert still_latched.code == "FC_RESIDUAL_WATCHDOG_SESSION_LATCHED"


@pytest.mark.asyncio
async def test_active_contract_degradation_cannot_pass_watchdog_as_healthy() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok

    transport.inject_contract_verification(execution_feedback=False)
    observed = await sink.watchdog()

    assert not observed.ok
    assert observed.code == "FC_RESIDUAL_WATCHDOG_AMBIGUOUS"
    status = sink.status()
    assert status.fault_latched
    assert not status.healthy
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_new_sink_detects_unknown_preexisting_active_residual() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    request = _stage_request(_command(), transport)
    assert (await transport.stage_residual(request)).accepted
    await transport.wait_execution_feedback(
        FlightControllerResidualOperation.STAGE,
        request.sequence,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    observed = sink.status()

    assert observed.fault_latched
    assert observed.residual_state is FlightControllerResidualState.AMBIGUOUS
    cleared = await sink.watchdog()
    assert not cleared.ok
    assert cleared.code == "FC_RESIDUAL_WATCHDOG_CLEARED"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    delayed_replay = await transport.stage_residual(request)
    assert not delayed_replay.accepted
    assert delayed_replay.result_code == "CLEARED_SEQUENCE_REPLAY"


@pytest.mark.asyncio
async def test_clear_barrier_cancels_unknown_high_sequence_pending_stage() -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    pending = _stage_request(_command(sequence=100), transport)
    assert (await transport.stage_residual(pending)).accepted
    assert transport.status().pending_stage_present
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    cleared = await sink.clear_rotor_residual("cancel unknown pending stage")

    assert cleared.ok, cleared.message
    assert cleared.data["clear_through_command_sequence"] == 100
    assert sink.status().last_sequence == 100
    status = transport.status()
    assert not status.pending_stage_present
    assert not status.residual_register_active
    assert status.clear_through_command_sequence == 100
    with pytest.raises(RuntimeError, match="feedback is unavailable"):
        await transport.wait_execution_feedback(
            FlightControllerResidualOperation.STAGE,
            pending.sequence,
        )
    replay = await transport.stage_residual(pending)
    assert not replay.accepted
    assert replay.result_code == "CLEARED_SEQUENCE_REPLAY"


class _BlockingStageAndClearTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stage_started = asyncio.Event()
        self.release_stage = asyncio.Event()
        self.clear_started = asyncio.Event()
        self.release_clear = asyncio.Event()
        self.clear_calls = 0

    async def stage_residual(self, request: FlightControllerResidualStageRequest):  # type: ignore[no-untyped-def]
        self.stage_started.set()
        await self.release_stage.wait()
        return await super().stage_residual(request)

    async def clear_residual(self, request: FlightControllerResidualClearRequest):  # type: ignore[no-untyped-def]
        self.clear_calls += 1
        self.clear_started.set()
        await self.release_clear.wait()
        return await super().clear_residual(request)


@pytest.mark.asyncio
async def test_cancelled_clear_while_stage_lock_busy_still_executes_barrier() -> None:
    clock = ManualClock(10.0)
    transport = _BlockingStageAndClearTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    send_task = asyncio.create_task(sink.send_rotor_residual(_command()))
    await transport.stage_started.wait()
    clear_task = asyncio.create_task(sink.clear_rotor_residual("concurrent emergency clear"))
    await transport.clear_started.wait()

    clear_task.cancel()
    await asyncio.sleep(0)
    clear_task.cancel()
    transport.release_clear.set()
    with pytest.raises(asyncio.CancelledError):
        await clear_task
    assert transport.clear_calls >= 1

    transport.release_stage.set()
    send_result = await send_task
    assert not send_result.ok
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    assert sink.status().fault_latched


class _ExpireBeforeFeedbackTransport(FakeFlightControllerResidualTransport):
    def __init__(self, clock: ManualClock, **kwargs: object) -> None:
        self._test_clock = clock
        super().__init__(monotonic_clock=clock.monotonic, **kwargs)

    async def wait_execution_feedback(
        self,
        operation: FlightControllerResidualOperation,
        operation_sequence: int,
    ) -> FlightControllerResidualExecutionFeedback:
        if operation is FlightControllerResidualOperation.STAGE:
            self._test_clock.advance(0.026)
        return await super().wait_execution_feedback(operation, operation_sequence)


@pytest.mark.asyncio
async def test_expired_fc_tick_lease_has_typed_zero_residual_feedback() -> None:
    clock = ManualClock(10.0)
    transport = _ExpireBeforeFeedbackTransport(
        clock,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    result = await sink.send_rotor_residual(_command())

    assert not result.ok
    assert result.code == "FC_RESIDUAL_EXECUTION_REJECTED"
    assert FlightControllerResidualExecutionResult.EXPIRED.value in result.message
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


class _CachedTimingStatusTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.frozen_status = None

    def status(self):  # type: ignore[no-untyped-def]
        if self.frozen_status is not None:
            return self.frozen_status
        return super().status()


@pytest.mark.asyncio
async def test_cached_fc_status_is_not_restamped_as_new_zero_evidence() -> None:
    clock = ManualClock(10.0)
    transport = _CachedTimingStatusTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    assert (await sink.clear_rotor_residual("freeze persistent-zero status")).ok
    transport.frozen_status = transport.status()

    first = sink.status()
    clock.advance(0.001)
    repeated = sink.status()

    assert repeated.healthy
    assert repeated.timestamp_s == first.timestamp_s
    assert repeated.timestamp_s < clock.monotonic()
    assert repeated.clear_ack_timestamp_s == first.clear_ack_timestamp_s
    assert repeated.clear_execution_timestamp_s == first.clear_execution_timestamp_s


@pytest.mark.asyncio
async def test_cached_status_age_is_added_to_reported_timesync_age() -> None:
    clock = ManualClock(10.0)
    transport = _CachedTimingStatusTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    transport.inject_timing_quality(timesync_age_s=0.095)
    transport.frozen_status = transport.status()
    clock.advance(0.006)

    observed = await sink.watchdog()

    assert not observed.ok
    assert observed.code == "FC_RESIDUAL_WATCHDOG_AMBIGUOUS"
    assert sink.status().fault_latched
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", (None, "", "着陆中止" * 40))
async def test_invalid_or_oversize_clear_reason_never_suppresses_active_clear(
    reason: object,
) -> None:
    clock = ManualClock(10.0)
    transport = _transport(clock)
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok

    cleared = await sink.clear_rotor_residual(reason)  # type: ignore[arg-type]

    assert cleared.ok, cleared.message
    assert cleared.code == "FC_RESIDUAL_CLEAR_VERIFIED"
    assert transport.applied_residual_thrusts_n == (0.0,) * 4
    assert sink.status().clear_confirmed


class _WatermarkOverrideTransport(FakeFlightControllerResidualTransport):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.override_enabled = False
        self.watermark_override: Optional[int] = None
        self.clear_calls = 0

    def status(self):  # type: ignore[no-untyped-def]
        observed = super().status()
        if not self.override_enabled:
            return observed
        return replace(
            observed,
            clear_through_command_sequence=self.watermark_override,
        )

    async def clear_residual(self, request: FlightControllerResidualClearRequest):  # type: ignore[no-untyped-def]
        self.clear_calls += 1
        return await super().clear_residual(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("watermark", (None, 0))
async def test_confirmed_clear_watermark_cannot_disappear_or_regress(
    watermark: Optional[int],
) -> None:
    clock = ManualClock(10.0)
    transport = _WatermarkOverrideTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    assert (await sink.clear_rotor_residual("establish persistent barrier")).ok
    assert transport.clear_calls == 1
    transport.watermark_override = watermark
    transport.override_enabled = True

    unsafe = sink.status()
    reconciled = await sink.watchdog()

    assert unsafe.fault_latched
    assert unsafe.residual_state is FlightControllerResidualState.AMBIGUOUS
    assert not reconciled.ok
    assert reconciled.code == "FC_RESIDUAL_WATCHDOG_AMBIGUOUS"
    assert transport.clear_calls == 2
    assert transport.applied_residual_thrusts_n == (0.0,) * 4


@pytest.mark.asyncio
async def test_timesync_generation_change_does_not_reset_clear_watermark_domain() -> None:
    clock = ManualClock(10.0)
    transport = _WatermarkOverrideTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)

    assert (await sink.send_rotor_residual(_command())).ok
    assert (await sink.clear_rotor_residual("establish persistent barrier")).ok
    transport.inject_timing_quality(advance_timesync_generation=True)
    transport.override_enabled = True
    transport.watermark_override = None

    observed = sink.status()

    assert observed.fault_latched
    assert observed.residual_state is FlightControllerResidualState.AMBIGUOUS


@pytest.mark.asyncio
async def test_new_transport_identity_domain_does_not_inherit_old_watermark() -> None:
    clock = ManualClock(10.0)
    transport = _WatermarkOverrideTransport(
        monotonic_clock=clock.monotonic,
        reservation_valid_until_s=10.2,
    )
    sink = FlightControllerResidualSink(transport, _config(), monotonic_clock=clock.monotonic)
    assert (await sink.send_rotor_residual(_command())).ok
    assert (await sink.clear_rotor_residual("establish old-domain barrier")).ok
    transport.inject_connection(False)
    transport.inject_connection(True)
    transport.override_enabled = True
    transport.watermark_override = None

    observed = sink.status()

    assert observed.healthy
    assert not observed.fault_latched
    assert observed.residual_state is FlightControllerResidualState.CONFIRMED_ZERO
