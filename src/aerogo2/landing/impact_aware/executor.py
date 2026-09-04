"""Fail-closed target ingress to the exclusive Go2 LowCmd owner.

The serial reference coordinator produces one paired command per MPC cycle;
the multi-rate runtime instead produces a new joint frame from every accepted
high-rate LowState sample.  This module accepts either leg-frame source only
after its rotor-side activation contract has been handled, then deposits the
frame into the already running, fixed-rate LowCmd owner.
Passing the whole pair directly is rejected so its rotor half can never be
silently discarded.  This module creates no publisher thread and never
acquires or releases Unitree control authority.  Acquisition and final release
remain ground-only operations handled through ``Go2OwnershipPermit`` by the
system manager.

Any rejected, ambiguous, or failed submission revokes the MPC lease.  Revoke
means that the sole LowCmd publisher continues in its validated safe-hold
policy; it does not stop the publisher or transfer authority to SportClient.

中文说明：本文件是算法到 Go2 唯一 LowCmd owner 的窄接口。它不创建第二个发送
线程，也不负责获取/释放运动控制权；只把已通过飞控侧配对条件的最新腿部目标提交
给正在固定周期运行的 owner。序号、epoch、TTL、状态和提交结果任一不一致都会撤销
当前租约，使同一个 owner 转入保守持姿，而不是在空中停止 LowCmd 发布。
"""

from __future__ import annotations

import asyncio
import math
import time
from numbers import Real
from typing import Callable, Optional, Tuple, Union

from aerogo2.bridges.go2_lowlevel_interface import Go2LowLevelInterface
from aerogo2.common.async_utils import await_nonabandonable
from aerogo2.common.models import Go2LowLevelStatus, LowCmdOwnershipState
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.integration import (
    CoordinatedLandingCommand,
    Go2JointPositionCommand,
)

_ACTIVE_OWNER_STATES = frozenset(
    {
        LowCmdOwnershipState.HOLDING,
        LowCmdOwnershipState.MPC_ACTIVE,
    }
)


def _finite_real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validate_mapping_hash(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("mapping_hash must be a string")
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("mapping_hash must use sha256:<64 lowercase hexadecimal digits>")
    return value


def _command_vectors_are_finite(command: Go2JointPositionCommand) -> bool:
    vectors = (command.joint_positions_rad, command.desired_contact_forces_world_n)
    for values in vectors:
        if len(values) != 12:
            return False
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


class ImpactAwareLowCmdExecutor:
    """Submit expiring MPC joint targets to an existing LowCmd owner.

    One instance is bound to exactly one reviewed mapping and one nonzero owner
    epoch.  It is intentionally single-session: after any rejection or owner
    failure it remains revoked.  Starting another session requires a new
    ground-authorized ownership acquisition and a new executor instance.

    中文：提交前同时检查命令 TTL/序号、owner epoch、映射 hash、LowState 新鲜度
    和 owner 工作状态。提交成功后再次读取 owner 状态，确认同一目标已进入邮箱；
    这仍是 host-side ACK，并非电机侧执行回读。任一步失败都进入统一 revoke 路径。
    """

    def __init__(
        self,
        owner: Go2LowLevelInterface,
        *,
        mapping_hash: str,
        ownership_epoch: int,
        maximum_command_ttl_s: float,
        maximum_low_state_age_s: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for method_name in ("submit", "revoke", "status"):
            if not callable(getattr(owner, method_name, None)):
                raise TypeError(f"owner must provide callable {method_name}()")
        if type(ownership_epoch) is not int:
            raise TypeError("ownership_epoch must be an integer")
        if ownership_epoch <= 0:
            raise ValueError("ownership_epoch must be positive")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")

        self._owner = owner
        self._mapping_hash = _validate_mapping_hash(mapping_hash)
        self._ownership_epoch = ownership_epoch
        self._maximum_command_ttl_s = _finite_real(
            "maximum_command_ttl_s",
            maximum_command_ttl_s,
            positive=True,
        )
        self._maximum_low_state_age_s = _finite_real(
            "maximum_low_state_age_s",
            maximum_low_state_age_s,
            positive=True,
        )
        self._monotonic_clock = monotonic_clock
        self._last_sequence: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None
        self._revoked = False
        self._revoke_intent_generation = 0
        self._revoke_task: Optional[asyncio.Task[Tuple[object, Optional[BaseException]]]] = None
        self._lock = asyncio.Lock()

    @property
    def revoked(self) -> bool:
        """Whether this executor has permanently revoked its MPC lease."""

        return self._revoked

    @property
    def last_sequence(self) -> Optional[int]:
        """Highest sequence passed to the owner, accepted or ACK-ambiguous."""

        return self._last_sequence

    async def submit(
        self,
        command: Union[CoordinatedLandingCommand, Go2JointPositionCommand],
    ) -> OperationResult:
        """Validate and deposit one expiring target into the owner mailbox.

        Direct ``Go2JointPositionCommand`` input is reserved for a higher-level
        atomic cross-device committer that has already staged/acknowledged the
        rotor half.  A raw ``CoordinatedLandingCommand`` is rejected because
        this Go2-only component cannot safely commit its rotor payload.
        """

        async with self._lock:
            if self._revoked:
                return OperationResult.failure(
                    "LOWCMD_EXECUTOR_REVOKED",
                    "This MPC LowCmd lease was revoked; a new ground-authorized session is required",
                    {
                        "ownership_epoch": self._ownership_epoch,
                        "safe_hold_requested": True,
                    },
                )

            if isinstance(command, CoordinatedLandingCommand):
                return await self._fail_and_revoke(
                    "LOWCMD_ATOMIC_COMMITTER_REQUIRED",
                    "The paired coordinator output requires an FC-residual/Go2 atomic committer; the Go2 executor will not discard its rotor half",
                )

            try:
                revoke_generation = self._revoke_intent_generation
                leg = self._extract_leg_command(command)
                now = self._now()
                validation_error = self._validate_command(leg, now)
                if validation_error is not None:
                    return await self._fail_and_revoke(*validation_error)

                (
                    owner_status,
                    status_observed_at,
                    status_call_elapsed_s,
                    status_timing_error,
                ) = self._observe_owner_status()
                if status_timing_error is not None:
                    return await self._fail_and_revoke(*status_timing_error)
                if status_observed_at >= leg.valid_until_s:
                    return await self._fail_and_revoke(
                        "LOWCMD_OWNER_STATUS_DEADLINE_MISSED",
                        "The pre-submit LowCmd owner status read consumed the target "
                        f"validity window ({status_call_elapsed_s:.9g}s)",
                    )
                # status() is a synchronous boundary and may itself take time.
                # Revalidate at the observation instant rather than trusting the
                # clock sample taken before the call.
                validation_error = self._validate_command(leg, status_observed_at)
                if validation_error is not None:
                    return await self._fail_and_revoke(*validation_error)
                status_error = self._validate_owner_status(
                    owner_status,
                    status_observed_at,
                )
                if status_error is not None:
                    return await self._fail_and_revoke(*status_error)
                assert isinstance(owner_status, Go2LowLevelStatus)
                writer_generation_before_submit = owner_status.writer_enqueue_generation

                # Consume the sequence before crossing the async boundary.  If
                # the owner accepts the mailbox update but its ACK is lost, a
                # retry of the same command must never be treated as new.
                self._last_sequence = leg.sequence
                self._last_timestamp_s = leg.timestamp_s
                owner_result = await self._owner.submit(
                    leg,
                    ownership_epoch=self._ownership_epoch,
                    mapping_hash=self._mapping_hash,
                )
                if self._revoked or self._revoke_intent_generation != revoke_generation:
                    return await self._fail_and_revoke(
                        "LOWCMD_REVOKED_DURING_SUBMIT",
                        "A safety revoke raced the target submission; the target ACK is not executable",
                        force_new_revoke=True,
                    )
                if not isinstance(owner_result, OperationResult):
                    return await self._fail_and_revoke(
                        "LOWCMD_SUBMIT_PROTOCOL_ERROR",
                        "LowCmd owner returned an invalid submission result",
                    )
                if not owner_result.ok:
                    return await self._fail_and_revoke(
                        "LOWCMD_SUBMIT_REJECTED",
                        f"LowCmd owner rejected sequence {leg.sequence}: "
                        f"{owner_result.code}: {owner_result.message}",
                    )

                submit_completed_at = self._now()
                if submit_completed_at >= leg.valid_until_s:
                    return await self._fail_and_revoke(
                        "LOWCMD_SUBMIT_DEADLINE_MISSED",
                        "LowCmd owner acknowledged the target after its validity deadline",
                    )
                (
                    acknowledged_status,
                    status_observed_at,
                    status_call_elapsed_s,
                    status_timing_error,
                ) = self._observe_owner_status()
                if status_timing_error is not None:
                    return await self._fail_and_revoke(*status_timing_error)
                if status_observed_at >= leg.valid_until_s:
                    return await self._fail_and_revoke(
                        "LOWCMD_SUBMIT_DEADLINE_MISSED",
                        "The post-ACK LowCmd owner status read completed after the target "
                        f"validity deadline ({status_call_elapsed_s:.9g}s status call)",
                    )
                status_error = self._validate_owner_status(
                    acknowledged_status,
                    status_observed_at,
                    expected_sequence=leg.sequence,
                    expected_deadline=leg.valid_until_s,
                )
                if status_error is not None:
                    return await self._fail_and_revoke(*status_error)
                assert isinstance(acknowledged_status, Go2LowLevelStatus)
                if (
                    acknowledged_status.writer_enqueue_ack_available
                    is not owner_status.writer_enqueue_ack_available
                ):
                    return await self._fail_and_revoke(
                        "LOWCMD_WRITER_CAPABILITY_CHANGED",
                        "LowCmd writer-enqueue capability changed across target submission; "
                        "the target acknowledgement is ambiguous",
                    )
                if (
                    acknowledged_status.writer_enqueued_target_sequence != leg.sequence
                    and acknowledged_status.writer_enqueue_ack_available is True
                ):
                    acknowledged_status, writer_wait_error = await self._wait_for_writer_enqueue(
                        leg,
                        writer_generation_before_submit,
                        acknowledged_status,
                    )
                    if writer_wait_error is not None:
                        return await self._fail_and_revoke(*writer_wait_error)
                writer_enqueued = (
                    acknowledged_status.writer_enqueued_target_sequence == leg.sequence
                )
                writer_generation = acknowledged_status.writer_enqueue_generation
                writer_q = acknowledged_status.writer_enqueued_q_rad
                if writer_enqueued and (
                    writer_generation <= writer_generation_before_submit
                    or len(writer_q) != 12
                    or any(not math.isfinite(float(value)) for value in writer_q)
                ):
                    return await self._fail_and_revoke(
                        "LOWCMD_WRITER_ACK_INVALID",
                        "Matching writer sequence lacked a newer generation and 12 finite limited q values",
                    )
            except asyncio.CancelledError:
                revoke_task = self._register_revoke_intent(
                    "LOWCMD_SUBMIT_CANCELLED: submission task was cancelled"
                )
                # Cancellation makes the mailbox ACK ambiguous.  Do not let
                # repeated Task.cancel() calls abandon the corresponding
                # safe-hold transition.  The same per-session revoke task is
                # reused if cancellation arrived while _fail_and_revoke was
                # already waiting for it.
                await await_nonabandonable(revoke_task)
                raise
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                return await self._fail_and_revoke(
                    "LOWCMD_COMMAND_INVALID",
                    f"LowCmd command boundary rejected the target: {exc}",
                )
            except Exception as exc:
                return await self._fail_and_revoke(
                    "LOWCMD_SUBMIT_EXCEPTION",
                    f"LowCmd owner raised {type(exc).__name__}: {exc}",
                )

            return OperationResult.success(
                "LowCmd target staged in the exclusive fixed-rate owner mailbox",
                {
                    "sequence": leg.sequence,
                    "ownership_epoch": self._ownership_epoch,
                    "mapping_hash": self._mapping_hash,
                    "valid_until_s": leg.valid_until_s,
                    # The owner boundary above acknowledges only replacement
                    # of its capacity-one mailbox.  It does not prove that the
                    # fixed-rate writer has enqueued this target, still less
                    # that a motor applied it.
                    "mailbox_stage_acknowledged": True,
                    "mailbox_staged_target_sequence": leg.sequence,
                    "writer_enqueue_acknowledged": writer_enqueued,
                    "writer_enqueued_target_sequence": (
                        leg.sequence if writer_enqueued else None
                    ),
                    "writer_enqueue_generation": (
                        writer_generation if writer_enqueued else None
                    ),
                    "writer_enqueued_q_rad": writer_q if writer_enqueued else None,
                    "actuator_application_acknowledged": False,
                    "actuator_applied_target_sequence": None,
                },
                code="LOWCMD_TARGET_STAGED",
            )

    async def revoke_mpc_control(self, reason: str) -> OperationResult:
        """Public safety action: revoke once and prove settled safe-hold.

        This does not release Unitree control authority or stop the sole DDS
        publisher.  It leaves the same writer in its verified conservative
        hold, which is the only permitted airborne fallback.
        """

        detail = (
            reason.strip()
            if isinstance(reason, str) and reason.strip()
            else "external safety supervisor requested MPC revoke"
        )
        self._register_revoke_intent(detail)
        # Registration above is synchronous, so an in-flight submit observes
        # the fence.  Waiting for the submit lock then proves that any raced
        # target has completed its post-submit revoke barrier before return.
        async with self._lock:
            return await self._request_confirmed_revoke(detail)

    @staticmethod
    def _extract_leg_command(
        command: Union[CoordinatedLandingCommand, Go2JointPositionCommand],
    ) -> Go2JointPositionCommand:
        if isinstance(command, Go2JointPositionCommand):
            return command
        raise TypeError("command must be a Go2JointPositionCommand from the atomic committer")

    def _validate_command(
        self,
        command: Go2JointPositionCommand,
        now_s: float,
    ) -> Optional[Tuple[str, str]]:
        sequence = command.sequence
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            return "LOWCMD_COMMAND_INVALID", "LowCmd sequence must be a nonnegative integer"
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return (
                "LOWCMD_COMMAND_OUT_OF_ORDER",
                f"LowCmd sequence {sequence} is not newer than {self._last_sequence}",
            )

        timestamp = _finite_real("command.timestamp_s", command.timestamp_s)
        deadline = _finite_real("command.valid_until_s", command.valid_until_s)
        if deadline <= timestamp:
            return "LOWCMD_COMMAND_INVALID", "LowCmd validity deadline must follow its timestamp"
        ttl = deadline - timestamp
        if ttl > self._maximum_command_ttl_s and not math.isclose(
            ttl,
            self._maximum_command_ttl_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return (
                "LOWCMD_COMMAND_TTL_INVALID",
                f"LowCmd TTL {ttl:.9g}s exceeds the configured maximum",
            )
        if timestamp > now_s:
            return "LOWCMD_COMMAND_FROM_FUTURE", "LowCmd command timestamp is in the future"
        if now_s >= deadline:
            return "LOWCMD_COMMAND_STALE", "LowCmd command is already expired"
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            return (
                "LOWCMD_COMMAND_OUT_OF_ORDER",
                "LowCmd command timestamp did not advance monotonically",
            )
        if not _command_vectors_are_finite(command):
            return (
                "LOWCMD_COMMAND_INVALID",
                "LowCmd command must contain exactly 12 finite joint positions and 12 finite forces",
            )
        return None

    def _validate_owner_status(
        self,
        status: object,
        now_s: float,
        *,
        expected_sequence: Optional[int] = None,
        expected_deadline: Optional[float] = None,
    ) -> Optional[Tuple[str, str]]:
        freshness_error = self._validate_status_freshness(status, now_s)
        if freshness_error is not None:
            return freshness_error
        assert isinstance(status, Go2LowLevelStatus)
        if (
            isinstance(status.owner_epoch, bool)
            or not isinstance(status.owner_epoch, int)
            or status.owner_epoch != self._ownership_epoch
        ):
            return (
                "LOWCMD_OWNERSHIP_EPOCH_MISMATCH",
                "LowCmd owner epoch changed or does not match this executor",
            )
        if not isinstance(status.ownership_state, LowCmdOwnershipState):
            return "LOWCMD_OWNER_STATUS_INVALID", "LowCmd ownership state is invalid"
        if status.ownership_state not in _ACTIVE_OWNER_STATES:
            return (
                "LOWCMD_OWNERSHIP_INACTIVE",
                f"LowCmd owner state {status.ownership_state.value} cannot accept MPC targets",
            )
        health_values = (
            status.connected,
            status.healthy,
            status.publisher_active,
            status.writer_alive,
            status.watchdog_healthy,
            status.high_level_released,
            status.network_exclusivity_verified,
        )
        if any(type(value) is not bool for value in health_values) or not all(health_values):
            return (
                "LOWCMD_OWNER_UNHEALTHY",
                "LowCmd owner is disconnected, has no active publisher, is unhealthy, "
                "not exclusive, or lacks its watchdog",
            )
        if (
            type(status.mapping_hash_verified) is not bool
            or not status.mapping_hash_verified
            or status.active_mapping_hash != self._mapping_hash
        ):
            return (
                "LOWCMD_MAPPING_MISMATCH",
                "LowCmd owner has not verified the exact reviewed joint mapping hash",
            )
        staged_sequence = status.mailbox_staged_target_sequence
        legacy_sequence = status.target_sequence
        writer_sequence = status.writer_enqueued_target_sequence
        writer_generation = status.writer_enqueue_generation
        writer_q = status.writer_enqueued_q_rad
        if type(status.writer_enqueue_ack_available) is not bool:
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "Owner writer-enqueue capability flag is invalid",
            )
        if isinstance(writer_generation, bool) or not isinstance(writer_generation, int):
            return "LOWCMD_OWNER_STATUS_INVALID", "Owner writer generation is invalid"
        if writer_generation < 0:
            return "LOWCMD_OWNER_STATUS_INVALID", "Owner writer generation cannot be negative"
        if (writer_generation == 0) is not (len(writer_q) == 0):
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "Owner writer generation and limited-q evidence must appear together",
            )
        if writer_generation > 0 and (
            len(writer_q) != 12
            or any(not math.isfinite(float(value)) for value in writer_q)
        ):
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "Owner writer generation lacks 12 finite limited q values",
            )
        if writer_sequence is None:
            pass
        elif (
            isinstance(writer_sequence, bool)
            or not isinstance(writer_sequence, int)
            or writer_sequence < 0
            or not status.writer_enqueue_ack_available
            or writer_generation <= 0
            or len(writer_q) != 12
            or any(not math.isfinite(float(value)) for value in writer_q)
        ):
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "Owner writer evidence lacks a valid sequence, generation, or limited q vector",
            )
        if (
            staged_sequence is not None
            and legacy_sequence is not None
            and staged_sequence != legacy_sequence
        ):
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "Owner legacy target sequence disagrees with its explicit mailbox stage",
            )
        if staged_sequence is not None:
            if (
                isinstance(staged_sequence, bool)
                or not isinstance(staged_sequence, int)
                or staged_sequence < 0
            ):
                return "LOWCMD_OWNER_STATUS_INVALID", "Owner staged target sequence is invalid"
            if expected_sequence is None:
                if self._last_sequence is None:
                    # A newly constructed executor must not replay an epoch that
                    # already consumed targets.  A new session needs a new epoch.
                    return (
                        "LOWCMD_OWNERSHIP_EPOCH_REUSED",
                        "The ownership epoch already contains a target from another executor",
                    )
                if staged_sequence != self._last_sequence:
                    return (
                        "LOWCMD_OWNER_SEQUENCE_MISMATCH",
                        "LowCmd owner mailbox stage diverged from this executor",
                    )
            elif expected_sequence is not None and staged_sequence != expected_sequence:
                return (
                    "LOWCMD_SUBMIT_ACK_MISMATCH",
                    "LowCmd owner status did not acknowledge the submitted mailbox stage",
                )
        elif expected_sequence is not None:
            return (
                "LOWCMD_SUBMIT_ACK_MISMATCH",
                "LowCmd owner accepted the target without exposing an explicit mailbox stage",
            )
        deadline = status.target_deadline
        if staged_sequence is not None and expected_deadline is None:
            if (
                deadline is None
                or isinstance(deadline, bool)
                or not isinstance(deadline, Real)
                or not math.isfinite(float(deadline))
            ):
                return "LOWCMD_OWNER_STATUS_INVALID", "Owner target deadline is invalid"
            if now_s >= float(deadline):
                return (
                    "LOWCMD_PREVIOUS_TARGET_EXPIRED",
                    "The previous LowCmd target expired before this update arrived",
                )
        if expected_deadline is not None:
            if (
                deadline is None
                or isinstance(deadline, bool)
                or not isinstance(deadline, Real)
                or not math.isfinite(float(deadline))
                or not math.isclose(
                    float(deadline),
                    expected_deadline,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                return (
                    "LOWCMD_SUBMIT_ACK_MISMATCH",
                    "LowCmd owner status did not acknowledge the submitted deadline",
                )
        return None

    def _validate_status_freshness(
        self,
        status: object,
        now_s: float,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(status, Go2LowLevelStatus):
            return "LOWCMD_OWNER_STATUS_INVALID", "LowCmd owner returned an invalid status object"
        try:
            status_timestamp = _finite_real("owner status timestamp", status.timestamp)
            low_state_timestamp = _finite_real(
                "owner LowState timestamp",
                status.low_state_timestamp,
            )
            reported_low_state_age = _finite_real(
                "owner reported LowState age",
                status.low_state_age_s,
            )
        except (TypeError, ValueError):
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "LowCmd owner status/LowState age evidence is invalid",
            )
        if status_timestamp > now_s or low_state_timestamp > now_s:
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "LowCmd owner status or LowState timestamp is in the future",
            )
        if low_state_timestamp > status_timestamp:
            return (
                "LOWCMD_OWNER_STATUS_INVALID",
                "LowCmd owner LowState timestamp is later than its status snapshot",
            )
        if now_s - status_timestamp > self._maximum_command_ttl_s:
            return "LOWCMD_OWNER_STATUS_STALE", "LowCmd owner status is stale"
        effective_reported_age = reported_low_state_age + (now_s - status_timestamp)
        if (
            reported_low_state_age < 0.0
            or now_s - low_state_timestamp > self._maximum_low_state_age_s
            or effective_reported_age > self._maximum_low_state_age_s
        ):
            return "LOWCMD_LOWSTATE_STALE", "LowCmd owner LowState evidence is stale"
        return None

    def _now(self) -> float:
        return _finite_real("monotonic clock", self._monotonic_clock())

    def _observe_owner_status(
        self,
    ) -> Tuple[object, float, float, Optional[Tuple[str, str]]]:
        """Read one status and timestamp when that synchronous call returned."""

        call_started_at = self._now()
        status = self._owner.status()
        observed_at = self._now()
        call_elapsed_s = observed_at - call_started_at
        if call_elapsed_s < 0.0:
            return (
                status,
                observed_at,
                call_elapsed_s,
                (
                    "LOWCMD_MONOTONIC_CLOCK_INVALID",
                    "The monotonic clock moved backwards during the LowCmd owner status read",
                ),
            )
        if call_elapsed_s > self._maximum_command_ttl_s:
            return (
                status,
                observed_at,
                call_elapsed_s,
                (
                    "LOWCMD_OWNER_STATUS_TIMEOUT",
                    "The LowCmd owner status call took "
                    f"{call_elapsed_s:.9g}s, exceeding the configured "
                    f"{self._maximum_command_ttl_s:.9g}s safety observation budget",
                ),
            )
        return status, observed_at, call_elapsed_s, None

    async def _wait_for_writer_enqueue(
        self,
        command: Go2JointPositionCommand,
        generation_before_submit: int,
        initial_status: Go2LowLevelStatus,
    ) -> Tuple[Go2LowLevelStatus, Optional[Tuple[str, str]]]:
        """Wait boundedly for the fixed-rate writer to consume one mailbox frame.

        The bridge writer is asynchronous, so a single status snapshot after
        ``submit`` is inherently racy.  Both the command's monotonic TTL and an
        independent wall-clock deadline bound this wait.  External revoke sets
        its synchronous intent fence before waiting for the executor lock, so
        it also interrupts this loop at the next observation boundary.
        """

        status = initial_status
        try:
            remaining = command.valid_until_s - self._now()
        except (TypeError, ValueError) as exc:
            return status, ("LOWCMD_MONOTONIC_CLOCK_INVALID", str(exc))
        wall_deadline = time.monotonic() + max(0.0, remaining)
        while status.writer_enqueued_target_sequence != command.sequence:
            if self._revoked:
                return status, (
                    "LOWCMD_REVOKED_DURING_WRITER_WAIT",
                    "A safety revoke raced the LowCmd writer-enqueue wait",
                )
            now = self._now()
            wall_remaining = wall_deadline - time.monotonic()
            logical_remaining = command.valid_until_s - now
            if logical_remaining <= 0.0 or wall_remaining <= 0.0:
                return status, (
                    "LOWCMD_WRITER_ACK_TIMEOUT",
                    "The fixed-rate writer did not confirm the target before its deadline",
                )
            await asyncio.sleep(min(0.001, logical_remaining, wall_remaining))
            raw_status, observed_at, _elapsed, timing_error = self._observe_owner_status()
            if timing_error is not None:
                return status, timing_error
            # A blocking ``status()`` may cross either deadline and return a
            # matching sequence.  Never accept that late evidence merely
            # because the loop checked the budget before entering the call.
            if (
                observed_at >= command.valid_until_s
                or time.monotonic() >= wall_deadline
            ):
                return status, (
                    "LOWCMD_WRITER_ACK_TIMEOUT",
                    "The fixed-rate writer ACK was observed only after the target deadline",
                )
            status_error = self._validate_owner_status(
                raw_status,
                observed_at,
                expected_sequence=command.sequence,
                expected_deadline=command.valid_until_s,
            )
            if status_error is not None:
                return status, status_error
            assert isinstance(raw_status, Go2LowLevelStatus)
            if (
                raw_status.writer_enqueue_ack_available
                is not initial_status.writer_enqueue_ack_available
            ):
                return status, (
                    "LOWCMD_WRITER_CAPABILITY_CHANGED",
                    "LowCmd writer-enqueue capability changed during the bounded "
                    "writer wait; the target acknowledgement is ambiguous",
                )
            status = raw_status

        if status.writer_enqueue_generation <= generation_before_submit:
            return status, (
                "LOWCMD_WRITER_ACK_INVALID",
                "Writer sequence matched without a newer writer generation",
            )
        if len(status.writer_enqueued_q_rad) != 12:
            return status, (
                "LOWCMD_WRITER_ACK_INVALID",
                "Writer sequence matched without 12 limited q values",
            )
        return status, None

    def _get_or_start_revoke_task(
        self,
        reason: str,
    ) -> asyncio.Task[Tuple[object, Optional[BaseException]]]:
        """Return the one non-abandonable revoke transaction for this session."""

        task = self._revoke_task
        if task is not None and task.done() and self._raw_revoke_failed(task):
            # A failed/invalid owner transaction is not a permanent cached
            # answer.  Revocation is idempotent at the owner boundary, so the
            # next safety request must be allowed to establish a new barrier.
            # Successful transactions remain cached and their postcondition is
            # re-observed below without generating needless owner traffic.
            task = None
            self._revoke_task = None
        if task is None:
            task = asyncio.create_task(self._invoke_owner_revoke(reason))
            self._revoke_task = task
        return task

    @staticmethod
    def _raw_revoke_failed(
        task: asyncio.Task[Tuple[object, Optional[BaseException]]],
    ) -> bool:
        """Return whether a completed owner revoke can safely be retried."""

        try:
            revoke_result, revoke_error = task.result()
        except BaseException:
            return True
        return bool(
            revoke_error is not None
            or not isinstance(revoke_result, OperationResult)
            or not revoke_result.ok
        )

    def _register_revoke_intent(
        self,
        reason: str,
        *,
        force_new: bool = False,
    ) -> asyncio.Task[Tuple[object, Optional[BaseException]]]:
        """Synchronously fence all submissions before awaiting safe-hold ACK."""

        self._revoked = True
        self._revoke_intent_generation += 1
        if force_new:
            previous = self._revoke_task
            task = asyncio.create_task(self._invoke_owner_revoke_after(previous, reason))
            self._revoke_task = task
            return task
        return self._get_or_start_revoke_task(reason)

    async def _invoke_owner_revoke_after(
        self,
        previous: Optional[asyncio.Task[Tuple[object, Optional[BaseException]]]],
        reason: str,
    ) -> Tuple[object, Optional[BaseException]]:
        """Place a new revoke barrier after a submission that raced an older one."""

        if previous is not None:
            await await_nonabandonable(previous)
        return await self._invoke_owner_revoke(reason)

    async def _invoke_owner_revoke(
        self,
        reason: str,
    ) -> Tuple[object, Optional[BaseException]]:
        """Capture owner failures so caller cancellation remains observable."""

        try:
            return (
                await self._owner.revoke(
                    reason,
                    ownership_epoch=self._ownership_epoch,
                ),
                None,
            )
        except (Exception, asyncio.CancelledError) as exc:
            return None, exc

    async def _fail_and_revoke(
        self,
        code: str,
        message: str,
        *,
        force_new_revoke: bool = False,
    ) -> OperationResult:
        self._register_revoke_intent(
            f"{code}: {message}",
            force_new=force_new_revoke,
        )
        revoke = await self._request_confirmed_revoke(f"{code}: {message}")
        revoke_ok = revoke.ok
        revoke_code = revoke.code
        revoke_message = revoke.message

        detail = message
        if not revoke_ok:
            detail += f"; safe-hold revoke was not acknowledged: {revoke_code}: {revoke_message}"
        failure = OperationResult.failure(
            code,
            detail,
            {
                "ownership_epoch": self._ownership_epoch,
                "mapping_hash": self._mapping_hash,
                "safe_hold_requested": True,
                "safe_hold_acknowledged": revoke_ok,
                "revoke_code": revoke_code,
                "last_sequence": self._last_sequence,
            },
        )
        return failure

    async def _request_confirmed_revoke(self, reason: str) -> OperationResult:
        """Await the per-session revoke and validate its postcondition."""

        revoke_task = self._get_or_start_revoke_task(reason)
        (revoke_result, revoke_error), cancellation_seen = await await_nonabandonable(revoke_task)
        if revoke_error is not None:
            outcome = OperationResult.failure(
                "LOWCMD_REVOKE_EXCEPTION",
                f"{type(revoke_error).__name__}: {revoke_error}",
            )
        elif not isinstance(revoke_result, OperationResult):
            outcome = OperationResult.failure(
                "LOWCMD_REVOKE_PROTOCOL_ERROR",
                "LowCmd owner returned an invalid revoke result",
            )
        elif not revoke_result.ok:
            outcome = OperationResult.failure(
                revoke_result.code,
                revoke_result.message,
                revoke_result.data,
            )
        else:
            try:
                (
                    safe_status,
                    safe_now,
                    _status_call_elapsed_s,
                    status_timing_error,
                ) = self._observe_owner_status()
            except Exception as exc:
                safe_status = None
                status_error = f"{type(exc).__name__}: {exc}"
            else:
                if status_timing_error is not None:
                    status_error = f"{status_timing_error[0]}: {status_timing_error[1]}"
                else:
                    freshness_error = self._validate_status_freshness(safe_status, safe_now)
                    status_error = (
                        ""
                        if freshness_error is None
                        else f"{freshness_error[0]}: {freshness_error[1]}"
                    )
            safe_hold_confirmed = (
                isinstance(safe_status, Go2LowLevelStatus)
                and not status_error
                and safe_status.owner_epoch == self._ownership_epoch
                and safe_status.ownership_state is LowCmdOwnershipState.SAFE_HOLD
                and safe_status.safe_hold_active is True
                and safe_status.safe_hold_settled is True
                and safe_status.connected is True
                and safe_status.healthy is True
                and safe_status.publisher_active is True
                and safe_status.writer_alive is True
                and safe_status.watchdog_healthy is True
                and safe_status.high_level_released is True
                and safe_status.network_exclusivity_verified is True
                and safe_status.mapping_hash_verified is True
                and safe_status.active_mapping_hash == self._mapping_hash
                and safe_status.target_sequence is None
                and safe_status.mailbox_staged_target_sequence is None
                and safe_status.writer_enqueued_target_sequence is None
                and safe_status.actuator_applied_target_sequence is None
                and safe_status.safe_hold_request_generation > 0
                and safe_status.safe_hold_write_generation
                >= safe_status.safe_hold_request_generation
                and safe_status.target_deadline is None
            )
            if safe_hold_confirmed:
                outcome = OperationResult.success(
                    "LowCmd MPC lease revoked; sole writer confirmed settled safe-hold",
                    {
                        "ownership_epoch": self._ownership_epoch,
                        "mapping_hash": self._mapping_hash,
                        "last_sequence": self._last_sequence,
                    },
                    code="LOWCMD_SAFE_HOLD_CONFIRMED",
                )
            else:
                outcome = OperationResult.failure(
                    "LOWCMD_SAFE_HOLD_UNCONFIRMED",
                    "Revoke returned success but owner status did not confirm safe-hold"
                    + (f": {status_error}" if status_error else ""),
                )
        if not outcome.ok and self._revoke_task is revoke_task:
            # The owner may have returned success while the independently
            # observed safe-hold postcondition was absent/stale.  Do not let
            # that ambiguous result poison every later idempotent retry.
            self._revoke_task = None
        if cancellation_seen:
            # The postcondition has now been observed (or definitively failed),
            # so cancellation may cross this safety transaction boundary.
            raise asyncio.CancelledError
        return outcome


__all__ = ["ImpactAwareLowCmdExecutor"]
