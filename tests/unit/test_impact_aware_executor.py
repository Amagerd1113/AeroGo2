from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import List, Optional, Tuple, Type

import pytest

from aerogo2.common.models import Go2LowLevelStatus, LowCmdOwnershipState
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.executor import ImpactAwareLowCmdExecutor
from aerogo2.landing.impact_aware.integration import (
    CoordinatedLandingCommand,
    FlightControllerRotorResidualCommand,
    Go2JointPositionCommand,
    ImpactLandingPhase,
)

MAPPING_HASH = "sha256:" + "a" * 64


class MutableClock:
    def __init__(self, now_s: float) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


class RealMonotonicClock(MutableClock):
    def __init__(self) -> None:
        super().__init__(time.monotonic())

    def __call__(self) -> float:
        return time.monotonic()


class FakeLowCmdOwner:
    def __init__(self, clock: MutableClock, *, epoch: int = 17) -> None:
        self.clock = clock
        self.submissions: List[Tuple[Go2JointPositionCommand, int, str]] = []
        self.revocations: List[Tuple[str, Optional[int]]] = []
        self.release_calls = 0
        self.reject_submit = False
        self.raise_submit = False
        self.acknowledge_target = True
        self.revoke_ok = True
        self.revoke_enters_safe_hold = True
        self.revoke_settles_safe_hold = True
        self._status = Go2LowLevelStatus(
            timestamp=clock.now_s,
            connected=True,
            ownership_state=LowCmdOwnershipState.HOLDING,
            owner_epoch=epoch,
            healthy=True,
            low_state_timestamp=clock.now_s,
            low_state_age_s=0.0,
            publisher_active=True,
            writer_alive=True,
            watchdog_healthy=True,
            safe_hold_active=False,
            high_level_released=True,
            network_exclusivity_verified=True,
            mapping_hash_verified=True,
            active_mapping_hash=MAPPING_HASH,
            fault_reason="",
        )

    def status(self) -> Go2LowLevelStatus:
        return self._status

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        self.submissions.append((command, ownership_epoch, mapping_hash))
        if self.raise_submit:
            raise RuntimeError("transport failed")
        if self.reject_submit:
            return OperationResult.failure("OWNER_REJECTED", "mailbox rejected")
        if self.acknowledge_target:
            self._status = replace(
                self._status,
                timestamp=self.clock.now_s,
                ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
                target_sequence=command.sequence,
                target_age_s=0.0,
                target_deadline=command.valid_until_s,
                mailbox_staged_target_sequence=command.sequence,
            )
        return OperationResult.success("mailbox accepted")

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        self.revocations.append((reason, ownership_epoch))
        if not self.revoke_ok:
            return OperationResult.failure("REVOKE_FAILED", "safe-hold ACK missing")
        if self.revoke_enters_safe_hold:
            safe_hold_generation = self._status.safe_hold_request_generation + 1
            self._status = replace(
                self._status,
                timestamp=self.clock.now_s,
                ownership_state=LowCmdOwnershipState.SAFE_HOLD,
                healthy=True,
                safe_hold_active=True,
                safe_hold_settled=self.revoke_settles_safe_hold,
                target_sequence=None,
                target_age_s=None,
                target_deadline=None,
                mailbox_staged_target_sequence=None,
                writer_enqueued_target_sequence=None,
                actuator_applied_target_sequence=None,
                safe_hold_request_generation=safe_hold_generation,
                safe_hold_write_generation=safe_hold_generation,
            )
        return OperationResult.success("MPC lease revoked; publisher remains in safe-hold")

    async def release(self, *args: object, **kwargs: object) -> OperationResult:
        del args, kwargs
        self.release_calls += 1
        return OperationResult.failure("FORBIDDEN", "executor must never release LowCmd")


class LiveTimestampLowCmdOwner(FakeLowCmdOwner):
    """Return status timestamps sampled inside status(), like the real owner."""

    def status(self) -> Go2LowLevelStatus:
        call_started_at = time.monotonic()
        sampled_at = time.monotonic()
        while sampled_at <= call_started_at:
            sampled_at = time.monotonic()
        self._status = replace(
            self._status,
            timestamp=sampled_at,
            low_state_timestamp=sampled_at,
            low_state_age_s=0.0,
        )
        return self._status


class OneShotAdvancingStatusOwner(FakeLowCmdOwner):
    def __init__(
        self,
        clock: MutableClock,
        *,
        advance_s: float,
        epoch: int = 17,
    ) -> None:
        super().__init__(clock, epoch=epoch)
        self._advance_s = advance_s

    def status(self) -> Go2LowLevelStatus:
        self.clock.now_s += self._advance_s
        self._advance_s = 0.0
        self._status = replace(
            self._status,
            timestamp=self.clock.now_s,
            low_state_timestamp=self.clock.now_s,
            low_state_age_s=0.0,
        )
        return self._status


class BlockingRevokeOwner(FakeLowCmdOwner):
    def __init__(self, clock: MutableClock, *, epoch: int = 17) -> None:
        super().__init__(clock, epoch=epoch)
        self.revoke_started = asyncio.Event()
        self.allow_revoke = asyncio.Event()

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        self.revocations.append((reason, ownership_epoch))
        self.revoke_started.set()
        await self.allow_revoke.wait()
        safe_hold_generation = self._status.safe_hold_request_generation + 1
        self._status = replace(
            self._status,
            timestamp=self.clock.now_s,
            ownership_state=LowCmdOwnershipState.SAFE_HOLD,
            healthy=True,
            safe_hold_active=True,
            safe_hold_settled=True,
            target_sequence=None,
            target_age_s=None,
            target_deadline=None,
            mailbox_staged_target_sequence=None,
            writer_enqueued_target_sequence=None,
            actuator_applied_target_sequence=None,
            safe_hold_request_generation=safe_hold_generation,
            safe_hold_write_generation=safe_hold_generation,
        )
        return OperationResult.success("MPC lease revoked; publisher remains in safe-hold")


class BlockingSubmitOwner(FakeLowCmdOwner):
    def __init__(self, clock: MutableClock, *, epoch: int = 17) -> None:
        super().__init__(clock, epoch=epoch)
        self.submit_started = asyncio.Event()
        self.allow_submit = asyncio.Event()

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        self.submit_started.set()
        await self.allow_submit.wait()
        return await super().submit(
            command,
            ownership_epoch=ownership_epoch,
            mapping_hash=mapping_hash,
        )


class StaleLowStateAfterRevokeOwner(FakeLowCmdOwner):
    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        result = await super().revoke(reason, ownership_epoch=ownership_epoch)
        self.clock.now_s += 0.1
        self._status = replace(
            self._status,
            timestamp=self.clock.now_s,
            low_state_age_s=0.1,
        )
        return result


class StagedTargetAfterRevokeOwner(FakeLowCmdOwner):
    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        result = await super().revoke(reason, ownership_epoch=ownership_epoch)
        self._status = replace(self._status, mailbox_staged_target_sequence=3)
        return result


class WriterCapabilityFlipOwner(FakeLowCmdOwner):
    def __init__(
        self,
        clock: MutableClock,
        *,
        before_submit: bool,
        after_submit: bool,
        epoch: int = 17,
    ) -> None:
        super().__init__(clock, epoch=epoch)
        self._after_submit = after_submit
        self._status = replace(
            self._status,
            writer_enqueue_ack_available=before_submit,
        )

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        result = await super().submit(
            command,
            ownership_epoch=ownership_epoch,
            mapping_hash=mapping_hash,
        )
        self._status = replace(
            self._status,
            writer_enqueue_ack_available=self._after_submit,
        )
        return result


class ContaminatedSafeHoldOwner(FakeLowCmdOwner):
    def __init__(
        self,
        clock: MutableClock,
        *,
        safe_hold_changes: dict[str, object],
        epoch: int = 17,
    ) -> None:
        super().__init__(clock, epoch=epoch)
        self._safe_hold_changes = safe_hold_changes

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        result = await super().revoke(reason, ownership_epoch=ownership_epoch)
        if result.ok:
            self._status = replace(self._status, **self._safe_hold_changes)
        return result


class DelayedWriterAckOwner(FakeLowCmdOwner):
    """Expose a writer ACK only after the post-mailbox status observation.

    This models the real fixed-rate bridge: ``submit`` replaces the mailbox,
    then a later writer cycle applies software limiting and calls DDS Write.
    """

    def __init__(
        self,
        clock: MutableClock,
        *,
        publish_after_deadline: bool = False,
        epoch: int = 17,
    ) -> None:
        super().__init__(clock, epoch=epoch)
        self._post_submit_status_calls = 0
        self._writer_ack_published = False
        self._publish_after_deadline = publish_after_deadline
        self._status = replace(self._status, writer_enqueue_ack_available=True)

    def status(self) -> Go2LowLevelStatus:
        if self.submissions:
            self._post_submit_status_calls += 1
            # The first call is the immediate mailbox-stage observation.  The
            # next call represents consumption by a subsequent writer tick.
            if self._post_submit_status_calls >= 2 and not self._writer_ack_published:
                command = self.submissions[-1][0]
                if self._publish_after_deadline:
                    self.clock.now_s = command.valid_until_s + 1.0e-6
                limited_q = tuple(value * 0.5 for value in command.joint_positions_rad)
                self._status = replace(
                    self._status,
                    timestamp=self.clock.now_s,
                    writer_enqueued_target_sequence=command.sequence,
                    writer_enqueue_generation=self._status.writer_enqueue_generation + 1,
                    writer_enqueued_q_rad=limited_q,
                )
                self._writer_ack_published = True
        return self._status


class WriterCapabilityDropsWhileWaitingOwner(DelayedWriterAckOwner):
    """Lose the advertised writer evidence capability after mailbox ACK."""

    def status(self) -> Go2LowLevelStatus:
        if self.submissions:
            self._post_submit_status_calls += 1
            # Immediate post-submit observation remains capable; the next
            # observation is made inside the bounded writer wait.
            if self._post_submit_status_calls >= 2:
                self._status = replace(
                    self._status,
                    writer_enqueue_ack_available=False,
                )
        return self._status


def _leg(sequence: int = 1, timestamp_s: float = 10.0) -> Go2JointPositionCommand:
    return Go2JointPositionCommand(
        sequence=sequence,
        timestamp_s=timestamp_s,
        valid_until_s=timestamp_s + 0.05,
        joint_positions_rad=tuple(0.01 * index for index in range(12)),
        desired_contact_forces_world_n=(0.0,) * 12,
    )


def _coordinated(sequence: int = 1, timestamp_s: float = 10.0) -> CoordinatedLandingCommand:
    leg = _leg(sequence, timestamp_s)
    rotor = FlightControllerRotorResidualCommand(
        sequence=sequence,
        timestamp_s=timestamp_s,
        valid_until_s=timestamp_s + 0.05,
        fc_session_id=11,
        target_fc_tick=102,
        baseline_version=7,
        baseline_timestamp_s=timestamp_s,
        baseline_thrusts_n=(4.0,) * 4,
        transport_raw_residual_thrusts_n=None,
        applied_residual_thrusts_n=(0.0,) * 4,
        applied_total_thrusts_n=(4.0,) * 4,
        correction_gain=0.0,
        transport_target_semantics="zero_gain_no_transport_target",
    )
    return CoordinatedLandingCommand(
        phase=ImpactLandingPhase.PRE_TOUCHDOWN,
        leg=leg,
        rotor=rotor,
        solver_succeeded=True,
        solver_status="ok",
        solver_time_s=0.001,
    )


def _executor(
    owner: FakeLowCmdOwner,
    clock: MutableClock,
    *,
    epoch: int = 17,
    maximum_ttl_s: float = 0.05,
) -> ImpactAwareLowCmdExecutor:
    return ImpactAwareLowCmdExecutor(
        owner,
        mapping_hash=MAPPING_HASH,
        ownership_epoch=epoch,
        maximum_command_ttl_s=maximum_ttl_s,
        maximum_low_state_age_s=maximum_ttl_s,
        monotonic_clock=clock,
    )


@pytest.mark.asyncio
async def test_rejects_full_pair_instead_of_silently_dropping_rotor_half() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    executor = _executor(owner, clock)

    result = await executor.submit(_coordinated())

    assert not result.ok
    assert result.code == "LOWCMD_ATOMIC_COMMITTER_REQUIRED"
    assert not owner.submissions
    assert executor.last_sequence is None
    assert executor.revoked
    assert len(owner.revocations) == 1
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_accepts_monotonic_direct_leg_targets_without_running_a_send_loop() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    executor = _executor(owner, clock)
    assert (await executor.submit(_leg())).ok

    clock.now_s = 10.04
    owner._status = replace(owner._status, timestamp=clock.now_s)
    second = _leg(sequence=2, timestamp_s=10.03)
    assert (await executor.submit(second)).ok
    assert [item[0].sequence for item in owner.submissions] == [1, 2]


@pytest.mark.asyncio
async def test_real_monotonic_status_is_validated_at_its_observation_time() -> None:
    clock = RealMonotonicClock()
    owner = LiveTimestampLowCmdOwner(clock)
    executor = _executor(owner, clock, maximum_ttl_s=1.0)
    issued_at = clock()
    command = replace(
        _leg(timestamp_s=issued_at),
        valid_until_s=issued_at + 1.0,
    )

    accepted = await executor.submit(command)
    revoked = await executor.revoke_mpc_control("real-clock regression cleanup")

    assert accepted.ok
    assert accepted.code == "LOWCMD_TARGET_STAGED"
    assert accepted.data["mailbox_stage_acknowledged"] is True
    assert accepted.data["writer_enqueue_acknowledged"] is False
    assert accepted.data["actuator_application_acknowledged"] is False
    assert revoked.ok
    assert revoked.code == "LOWCMD_SAFE_HOLD_CONFIRMED"
    assert len(owner.submissions) == 1


@pytest.mark.asyncio
async def test_waits_for_delayed_writer_enqueue_ack_and_returns_exact_limited_q() -> None:
    clock = MutableClock(10.01)
    owner = DelayedWriterAckOwner(clock)
    executor = _executor(owner, clock)
    command = _leg()

    accepted = await executor.submit(command)

    expected_limited_q = tuple(value * 0.5 for value in command.joint_positions_rad)
    assert accepted.ok
    assert accepted.code == "LOWCMD_TARGET_STAGED"
    assert accepted.data["mailbox_stage_acknowledged"] is True
    assert accepted.data["writer_enqueue_acknowledged"] is True
    assert accepted.data["writer_enqueued_target_sequence"] == command.sequence
    assert accepted.data["writer_enqueue_generation"] == 1
    assert accepted.data["writer_enqueued_q_rad"] == expected_limited_q
    assert accepted.data["actuator_application_acknowledged"] is False
    assert owner.revocations == []


@pytest.mark.asyncio
async def test_writer_ack_first_observed_after_target_ttl_is_rejected_and_revoked() -> None:
    clock = MutableClock(10.01)
    owner = DelayedWriterAckOwner(clock, publish_after_deadline=True)
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_WRITER_ACK_TIMEOUT"
    assert result.data["safe_hold_acknowledged"] is True
    assert executor.revoked
    assert owner._writer_ack_published
    assert len(owner.revocations) == 1
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.release_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before_submit", "after_submit"),
    ((False, True), (True, False)),
)
async def test_writer_capability_flip_during_submit_fails_closed_and_revokes(
    before_submit: bool,
    after_submit: bool,
) -> None:
    clock = MutableClock(10.01)
    owner = WriterCapabilityFlipOwner(
        clock,
        before_submit=before_submit,
        after_submit=after_submit,
    )
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_WRITER_CAPABILITY_CHANGED"
    assert result.data["safe_hold_acknowledged"] is True
    assert executor.revoked
    assert len(owner.submissions) == 1
    assert len(owner.revocations) == 1
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.status().safe_hold_settled
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_writer_capability_drop_during_writer_wait_fails_closed() -> None:
    clock = MutableClock(10.01)
    owner = WriterCapabilityDropsWhileWaitingOwner(clock)
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_WRITER_CAPABILITY_CHANGED"
    assert result.data["safe_hold_acknowledged"] is True
    assert executor.revoked
    assert len(owner.submissions) == 1
    assert len(owner.revocations) == 1


@pytest.mark.asyncio
async def test_pre_submit_status_call_cannot_consume_command_deadline() -> None:
    clock = MutableClock(10.01)
    owner = OneShotAdvancingStatusOwner(clock, advance_s=0.041)
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_OWNER_STATUS_DEADLINE_MISSED"
    assert not owner.submissions
    assert result.data["safe_hold_acknowledged"] is True


@pytest.mark.asyncio
async def test_owner_status_call_duration_is_bounded() -> None:
    clock = MutableClock(10.01)
    owner = OneShotAdvancingStatusOwner(clock, advance_s=0.051)
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_OWNER_STATUS_TIMEOUT"
    assert not owner.submissions
    assert result.data["safe_hold_acknowledged"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "now_s", "expected_code"),
    [
        (_leg(timestamp_s=10.0), 10.05, "LOWCMD_COMMAND_STALE"),
        (_leg(timestamp_s=10.1), 10.0, "LOWCMD_COMMAND_FROM_FUTURE"),
        (_leg(timestamp_s=10.0), 10.01, "LOWCMD_COMMAND_TTL_INVALID"),
    ],
)
async def test_bad_time_or_ttl_revokes_to_safe_hold(
    command: Go2JointPositionCommand,
    now_s: float,
    expected_code: str,
) -> None:
    clock = MutableClock(now_s)
    owner = FakeLowCmdOwner(clock)
    maximum_ttl = 0.04 if expected_code == "LOWCMD_COMMAND_TTL_INVALID" else 0.05
    executor = _executor(owner, clock, maximum_ttl_s=maximum_ttl)

    result = await executor.submit(command)

    assert not result.ok
    assert result.code == expected_code
    assert executor.revoked
    assert owner.status().safe_hold_active
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert not owner.submissions
    assert len(owner.revocations) == 1
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_replay_is_consumed_once_then_revokes_the_session() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    executor = _executor(owner, clock)
    command = _leg()
    assert (await executor.submit(command)).ok

    replay = await executor.submit(command)

    assert not replay.ok
    assert replay.code == "LOWCMD_COMMAND_OUT_OF_ORDER"
    assert executor.revoked
    assert len(owner.submissions) == 1
    assert owner.status().safe_hold_active
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_invalid_joint_vector_is_rechecked_at_the_hardware_boundary() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    executor = _executor(owner, clock)
    command = _leg()
    object.__setattr__(command, "joint_positions_rad", (float("nan"),) + (0.0,) * 11)

    result = await executor.submit(command)

    assert not result.ok
    assert result.code == "LOWCMD_COMMAND_INVALID"
    assert not owner.submissions
    assert owner.status().safe_hold_active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_change", "expected_code"),
    [
        ({"owner_epoch": 18}, "LOWCMD_OWNERSHIP_EPOCH_MISMATCH"),
        ({"active_mapping_hash": "sha256:" + "b" * 64}, "LOWCMD_MAPPING_MISMATCH"),
        ({"mapping_hash_verified": False}, "LOWCMD_MAPPING_MISMATCH"),
        ({"publisher_active": False}, "LOWCMD_OWNER_UNHEALTHY"),
        ({"watchdog_healthy": False}, "LOWCMD_OWNER_UNHEALTHY"),
        ({"high_level_released": False}, "LOWCMD_OWNER_UNHEALTHY"),
        (
            {"ownership_state": LowCmdOwnershipState.SAFE_HOLD, "safe_hold_active": True},
            "LOWCMD_OWNERSHIP_INACTIVE",
        ),
    ],
)
async def test_owner_identity_and_health_must_match(
    status_change: object,
    expected_code: str,
) -> None:
    assert isinstance(status_change, dict)
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    owner._status = replace(owner._status, **status_change)
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == expected_code
    assert not owner.submissions
    assert executor.revoked
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_new_executor_rejects_an_epoch_that_already_contains_a_target() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    owner._status = replace(
        owner._status,
        ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
        target_sequence=8,
        target_age_s=0.001,
        target_deadline=10.04,
        mailbox_staged_target_sequence=8,
    )
    executor = _executor(owner, clock)

    result = await executor.submit(_leg(sequence=9))

    assert not result.ok
    assert result.code == "LOWCMD_OWNERSHIP_EPOCH_REUSED"
    assert not owner.submissions
    assert owner.status().safe_hold_active


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_exception", [False, True])
async def test_submit_failure_or_exception_revokes_but_never_releases_owner(
    raise_exception: bool,
) -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    owner.reject_submit = not raise_exception
    owner.raise_submit = raise_exception
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code in {"LOWCMD_SUBMIT_REJECTED", "LOWCMD_SUBMIT_EXCEPTION"}
    assert len(owner.submissions) == 1
    assert len(owner.revocations) == 1
    assert owner.status().safe_hold_active
    assert executor.last_sequence == 1
    assert executor.revoked
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_repeated_submit_cancellation_waits_for_same_revoke_transaction() -> None:
    clock = MutableClock(10.01)
    owner = BlockingRevokeOwner(clock)
    owner.reject_submit = True
    executor = _executor(owner, clock)

    submit_task = asyncio.create_task(executor.submit(_leg()))
    await asyncio.wait_for(owner.revoke_started.wait(), timeout=0.2)

    submit_task.cancel()
    await asyncio.sleep(0)
    assert not submit_task.done()
    submit_task.cancel()
    await asyncio.sleep(0)

    assert not submit_task.done()
    assert len(owner.revocations) == 1
    assert not owner.status().safe_hold_active

    owner.allow_revoke.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(submit_task, timeout=0.2)

    assert len(owner.revocations) == 1
    assert executor.revoked
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.status().safe_hold_active
    assert owner.status().safe_hold_settled
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_missing_mailbox_ack_revokes_and_reports_safe_hold_failure() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    owner.acknowledge_target = False
    owner.revoke_ok = False
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_SUBMIT_ACK_MISMATCH"
    assert result.data["safe_hold_requested"] is True
    assert result.data["safe_hold_acknowledged"] is False
    assert "safe-hold revoke was not acknowledged" in result.message
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_successful_revoke_reply_without_safe_hold_state_is_not_trusted() -> None:
    clock = MutableClock(10.05)
    owner = FakeLowCmdOwner(clock)
    owner.revoke_enters_safe_hold = False
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_COMMAND_STALE"
    assert result.data["safe_hold_acknowledged"] is False
    assert result.data["revoke_code"] == "LOWCMD_SAFE_HOLD_UNCONFIRMED"
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_successful_revoke_reply_without_settled_feedback_is_not_trusted() -> None:
    clock = MutableClock(10.05)
    owner = FakeLowCmdOwner(clock)
    owner.revoke_settles_safe_hold = False
    executor = _executor(owner, clock)

    result = await executor.submit(_leg())

    assert not result.ok
    assert result.code == "LOWCMD_COMMAND_STALE"
    assert result.data["safe_hold_acknowledged"] is False
    assert result.data["revoke_code"] == "LOWCMD_SAFE_HOLD_UNCONFIRMED"
    assert owner.status().safe_hold_active
    assert not owner.status().safe_hold_settled
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_safe_hold_reply_with_a_staged_target_is_not_trusted() -> None:
    clock = MutableClock(10.01)
    owner = StagedTargetAfterRevokeOwner(clock)
    executor = _executor(owner, clock)

    result = await executor.revoke_mpc_control("synthetic retained mailbox target")

    assert not result.ok
    assert result.code == "LOWCMD_SAFE_HOLD_UNCONFIRMED"
    assert len(owner.revocations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "safe_hold_changes",
    (
        {
            "writer_enqueue_ack_available": True,
            "writer_enqueued_target_sequence": 7,
            "writer_enqueue_generation": 1,
            "writer_enqueued_q_rad": (0.0,) * 12,
        },
        {
            "actuator_application_ack_available": True,
            "actuator_applied_target_sequence": 7,
        },
        {
            "safe_hold_request_generation": 2,
            "safe_hold_write_generation": 1,
        },
    ),
    ids=("writer-target-remains", "applied-target-remains", "write-generation-lags"),
)
async def test_safe_hold_rejects_residual_target_identity_or_lagging_generation(
    safe_hold_changes: dict[str, object],
) -> None:
    clock = MutableClock(10.01)
    owner = ContaminatedSafeHoldOwner(
        clock,
        safe_hold_changes=safe_hold_changes,
    )
    executor = _executor(owner, clock)

    result = await executor.revoke_mpc_control("synthetic incomplete safe-hold")

    assert not result.ok
    assert result.code == "LOWCMD_SAFE_HOLD_UNCONFIRMED"
    assert executor.revoked
    assert len(owner.revocations) == 1
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_public_safety_revoke_confirms_same_writer_safe_hold_without_release() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    executor = _executor(owner, clock)

    first = await executor.revoke_mpc_control("independent safety trip")
    repeated = await executor.revoke_mpc_control("duplicate trip")

    assert first.ok and repeated.ok
    assert first.code == repeated.code == "LOWCMD_SAFE_HOLD_CONFIRMED"
    assert executor.revoked
    assert len(owner.revocations) == 1
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.status().safe_hold_active
    assert owner.status().safe_hold_settled
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_failed_public_revoke_can_be_retried_idempotently() -> None:
    clock = MutableClock(10.01)
    owner = FakeLowCmdOwner(clock)
    owner.revoke_ok = False
    executor = _executor(owner, clock)

    first = await executor.revoke_mpc_control("first safety trip")
    owner.revoke_ok = True
    second = await executor.revoke_mpc_control("retry safety trip")

    assert not first.ok
    assert first.code == "REVOKE_FAILED"
    assert second.ok
    assert second.code == "LOWCMD_SAFE_HOLD_CONFIRMED"
    assert len(owner.revocations) == 2
    assert executor.revoked


@pytest.mark.asyncio
async def test_safe_hold_reply_with_stale_lowstate_is_not_confirmed() -> None:
    clock = MutableClock(10.01)
    owner = StaleLowStateAfterRevokeOwner(clock)
    executor = _executor(owner, clock)

    result = await executor.revoke_mpc_control("synthetic stale LowState")

    assert not result.ok
    assert result.code == "LOWCMD_SAFE_HOLD_UNCONFIRMED"
    assert "LOWCMD_LOWSTATE_STALE" in result.message
    assert executor.revoked
    assert owner.status().safe_hold_active
    assert owner.status().safe_hold_settled
    assert owner.release_calls == 0


@pytest.mark.asyncio
async def test_public_revoke_fences_a_target_already_blocked_inside_owner_submit() -> None:
    clock = MutableClock(10.01)
    owner = BlockingSubmitOwner(clock)
    executor = _executor(owner, clock)
    submit_task = asyncio.create_task(executor.submit(_leg()))
    await owner.submit_started.wait()

    revoke_task = asyncio.create_task(executor.revoke_mpc_control("manual safety trip"))
    await asyncio.sleep(0)
    assert executor.revoked
    assert not revoke_task.done()

    owner.allow_submit.set()
    submit_result, revoke_result = await asyncio.gather(submit_task, revoke_task)

    assert not submit_result.ok
    assert submit_result.code == "LOWCMD_REVOKED_DURING_SUBMIT"
    assert revoke_result.ok
    assert len(owner.revocations) == 2
    assert owner.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert owner.status().safe_hold_active
    assert owner.status().safe_hold_settled
    assert owner.status().target_sequence is None
    assert owner.release_calls == 0


@pytest.mark.parametrize(
    ("mapping_hash", "epoch", "ttl", "error"),
    [
        ("not-a-hash", 1, 0.05, ValueError),
        (MAPPING_HASH, 0, 0.05, ValueError),
        (MAPPING_HASH, True, 0.05, TypeError),
        (MAPPING_HASH, 1, 0.0, ValueError),
    ],
)
def test_constructor_rejects_ambiguous_session_binding(
    mapping_hash: str,
    epoch: object,
    ttl: float,
    error: Type[Exception],
) -> None:
    clock = MutableClock(10.0)
    owner = FakeLowCmdOwner(clock)
    with pytest.raises(error):
        ImpactAwareLowCmdExecutor(
            owner,
            mapping_hash=mapping_hash,
            ownership_epoch=epoch,  # type: ignore[arg-type]
            maximum_command_ttl_s=ttl,
            maximum_low_state_age_s=ttl,
            monotonic_clock=clock,
        )


@pytest.mark.parametrize(
    ("generation", "writer_q"),
    (
        (0, (0.0,) * 12),
        (1, ()),
        (1, (0.0,) * 11),
        (1, (float("nan"),) + (0.0,) * 11),
    ),
)
def test_lowcmd_status_rejects_unpaired_or_invalid_writer_generation_and_q(
    generation: int,
    writer_q: Tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="writer_enqueue"):
        Go2LowLevelStatus(
            writer_enqueue_generation=generation,
            writer_enqueued_q_rad=writer_q,
        )


@pytest.mark.parametrize(
    ("generation", "writer_q"),
    ((0, ()), (1, (0.0,) * 12)),
)
def test_lowcmd_status_accepts_exact_writer_generation_and_q_pairs(
    generation: int,
    writer_q: Tuple[float, ...],
) -> None:
    status = Go2LowLevelStatus(
        writer_enqueue_generation=generation,
        writer_enqueued_q_rad=writer_q,
    )

    assert status.writer_enqueue_generation == generation
    assert status.writer_enqueued_q_rad == writer_q
