"""Exclusive Unitree SDK2 LowCmd owner with a fixed-rate safety stream.

This module intentionally lazy-loads ``unitree_sdk2py``.  Importing AeroGo2 on
a development PC therefore cannot initialize DDS or write motor commands.
Calling :meth:`acquire` additionally requires the process hardware-write gate,
a short-lived ground permit, fresh LowState, verified mapping hash, successful
MotionSwitcher release, and the arbiter's single-instance lock.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, cast

from aerogo2.bridges.go2_control_arbiter import Go2ControlArbiter
from aerogo2.bridges.go2_lowlevel_interface import Go2OwnershipPermit
from aerogo2.common.async_utils import await_nonabandonable, run_blocking
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import (
    Go2Config,
    Go2LowLevelConfig,
    compute_go2_joint_mapping_hash,
)
from aerogo2.common.models import (
    Go2FootForceFeedback,
    Go2LowLevelStatus,
    Go2MotorFeedback,
    LowCmdOwnershipState,
)
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.integration import Go2JointPositionCommand

_JOINT_COUNT = 12
_MOTOR_COMMAND_COUNT = 20
_POSITION_STOP_F = 2.146e9
_VELOCITY_STOP_F = 16000.0


class _OwnerGuardTimeout(RuntimeError):
    """Raised when an event-loop operation cannot linearize with the writer."""

    def __init__(self, operation_name: str, waited_s: float) -> None:
        super().__init__(operation_name)
        self.operation_name = operation_name
        self.waited_s = waited_s


def _read_value(obj: Any, name: str, default: Any) -> Any:
    raw = getattr(obj, name, default)
    return raw() if callable(raw) else raw


def _finite_optional(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sdk_int16_quad(value: Any) -> Tuple[Tuple[int, int, int, int], bool]:
    """Parse one public SDK ``int16[4]`` field without numeric coercion."""

    try:
        raw = tuple(value)
    except (TypeError, ValueError):
        return (0, 0, 0, 0), False
    if len(raw) != 4:
        return (0, 0, 0, 0), False
    parsed = []
    for item in raw:
        if (
            isinstance(item, bool)
            or not isinstance(item, Integral)
            or int(item) < -32768
            or int(item) > 32767
        ):
            return (0, 0, 0, 0), False
        parsed.append(int(item))
    return (parsed[0], parsed[1], parsed[2], parsed[3]), True


def _sdk_uint32(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    parsed = int(value)
    return parsed if 0 <= parsed <= 0xFFFFFFFF else None


def compute_go2_mapping_hash(
    mapping_version: str,
    joint_names: Sequence[str],
    motor_ids: Sequence[int],
    directions: Sequence[int],
    zero_offsets_rad: Sequence[float],
) -> str:
    """Hash the exact algorithm-to-SDK coordinate transform.

    The canonical payload and ``sha256:`` prefix make independently generated
    commissioning records reproducible.  Limits and gains are deliberately
    excluded: changing them does not change joint identity or coordinates.
    """

    return compute_go2_joint_mapping_hash(
        mapping_version,
        tuple(joint_names),
        tuple(motor_ids),
        tuple(directions),
        tuple(zero_offsets_rad),
    )


@dataclass(frozen=True)
class Go2SdkBindings:
    """Injectable SDK seam used by hardware-free unit tests."""

    channel_factory_initialize: Callable[..., Any]
    subscriber_factory: Callable[..., Any]
    publisher_factory: Callable[..., Any]
    low_state_type: Any
    low_cmd_factory: Callable[[], Any]
    motion_switcher_factory: Callable[[], Any]
    crc_factory: Callable[[], Any]
    # Acquisition can close a partially initialized object only if publisher
    # construction itself creates no DDS endpoint; endpoint creation must be
    # deferred to Init(). Hardware bindings below encode the audited Unitree
    # SDK contract, while injected adapters must opt in explicitly.
    publisher_constructor_deferred_until_init: bool
    # A Close() exception has an ambiguous endpoint outcome.  Retrying Close
    # is permitted only for an adapter whose exact SDK/firmware combination
    # has been qualified as idempotent.
    publisher_close_retry_idempotency_verified: bool

    def __post_init__(self) -> None:
        if type(self.publisher_constructor_deferred_until_init) is not bool:
            raise TypeError("publisher_constructor_deferred_until_init must be a bool")
        if type(self.publisher_close_retry_idempotency_verified) is not bool:
            raise TypeError("publisher_close_retry_idempotency_verified must be a bool")

    @classmethod
    def load(cls) -> Go2SdkBindings:
        channel = importlib.import_module("unitree_sdk2py.core.channel")
        defaults = importlib.import_module("unitree_sdk2py.idl.default")
        dds = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")
        low_cmd_factory = getattr(defaults, "unitree_go_msg_dds__LowCmd_", None)
        if not callable(low_cmd_factory):
            low_cmd_type = dds.LowCmd_
            low_cmd_factory = low_cmd_type

        def motion_switcher_factory() -> Any:
            motion = importlib.import_module(
                "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"
            )
            return motion.MotionSwitcherClient()

        def crc_factory() -> Any:
            crc = importlib.import_module("unitree_sdk2py.utils.crc")
            return crc.CRC()

        return cls(
            channel_factory_initialize=channel.ChannelFactoryInitialize,
            subscriber_factory=channel.ChannelSubscriber,
            publisher_factory=channel.ChannelPublisher,
            low_state_type=dds.LowState_,
            low_cmd_factory=low_cmd_factory,
            motion_switcher_factory=motion_switcher_factory,
            crc_factory=crc_factory,
            publisher_constructor_deferred_until_init=True,
            # The public SDK example does not establish idempotent Close after
            # an exception.  Keep automatic retry disabled until that exact
            # target binding is commissioned.
            publisher_close_retry_idempotency_verified=False,
        )


class UnitreeGo2LowLevelSdkBridge:
    """Own LowCmd continuously; accept only fresh, epoch-bound MPC targets."""

    def __init__(
        self,
        config: Go2Config,
        *,
        arbiter: Go2ControlArbiter,
        clock: Optional[Clock] = None,
        allow_hardware_write: bool = False,
        sdk_bindings: Optional[Go2SdkBindings] = None,
        ground_transfer_verifier: Optional[
            Callable[[str, Go2OwnershipPermit], OperationResult]
        ] = None,
    ) -> None:
        if not isinstance(config, Go2Config):
            raise TypeError("config must be a Go2Config")
        if not isinstance(arbiter, Go2ControlArbiter):
            raise TypeError("arbiter must be a Go2ControlArbiter")
        if type(allow_hardware_write) is not bool:
            raise TypeError("allow_hardware_write must be a bool")
        if ground_transfer_verifier is not None and not callable(ground_transfer_verifier):
            raise TypeError("ground_transfer_verifier must be callable")
        self._go2_config = config
        self._config: Go2LowLevelConfig = config.low_level
        # Configuration is immutable, so these readiness tiers may be cached.
        # This also keeps the high-rate LowState callback free of YAML-style
        # validation work.
        self._actuation_config_ready = self.actuation_readiness().ok
        self._arbiter = arbiter
        self._clock = clock or RealClock()
        self._allow_hardware_write = allow_hardware_write
        self._injected_sdk = sdk_bindings
        self._ground_transfer_verifier = ground_transfer_verifier
        initial_state = (
            LowCmdOwnershipState.DISCONNECTED
            if self._config.observation_enabled
            else LowCmdOwnershipState.DISABLED
        )
        initial_reason = (
            "low-level transport is not connected"
            if self._config.observation_enabled
            else "low-level control disabled by configuration"
        )
        self._status = Go2LowLevelStatus(
            timestamp=self._clock.monotonic(),
            ownership_state=initial_state,
            fault_reason=initial_reason,
        )
        self._lifecycle_guard = threading.Lock()
        self._guard = threading.RLock()
        self._safe_hold_condition = threading.Condition(self._guard)
        self._low_state_ingress_condition = threading.Condition(threading.Lock())
        # Every subscription callback captures one generation. Disconnect and
        # each new connect advance it, so a delayed callback from an old DDS
        # reader can never satisfy freshness or mutate a newer connection.
        self._subscription_generation = 0
        self._low_state_ingress_sequence = 0
        self._low_state_inflight: Dict[int, int] = {}
        self._first_state = threading.Event()
        self._first_write = threading.Event()
        self._writer_stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        # Separate from _guard so status() can detect a writer blocked while
        # that writer deliberately holds _guard across the final command
        # construction-to-Write boundary.
        self._writer_health_guard = threading.Lock()
        self._writer_started_s: Optional[float] = None
        self._writer_write_started_s: Optional[float] = None
        self._writer_heartbeat_s: Optional[float] = None
        self._sdk: Optional[Go2SdkBindings] = None
        self._subscriber: Optional[Any] = None
        self._publisher: Optional[Any] = None
        self._publisher_close_task: Optional[asyncio.Future[OperationResult]] = None
        self._publisher_close_target: Optional[Any] = None
        self._motion_switcher: Optional[Any] = None
        self._crc: Optional[Any] = None
        self._connected = False
        self._owner_epoch = 0
        self._target: Optional[Go2JointPositionCommand] = None
        self._last_accepted_sequence: Optional[int] = None
        self._last_commanded_q: Optional[Tuple[float, ...]] = None
        self._command_reference_generation = 0
        self._command_reference_q: Optional[Tuple[float, ...]] = None
        self._command_reference_write_s: Optional[float] = None
        self._command_reference_ingress_cutoff = 0
        self._safe_hold_q: Optional[Tuple[float, ...]] = None
        self._expected_deadline_s: Optional[float] = None
        self._last_successful_write_s: Optional[float] = None
        self._safe_hold_request_generation = 0
        self._safe_hold_write_generation = 0
        self._last_safe_hold_write_s: Optional[float] = None
        self._safe_hold_command_reached_generation = 0
        self._safe_hold_command_reached_write_s: Optional[float] = None
        self._valid_low_state_sequence = 0
        self._valid_low_state_ingress_token = 0
        self._last_foot_force_source_tick: Optional[int] = None
        self._safe_hold_feedback_sequence_required = 0
        self._safe_hold_feedback_ingress_token_required = 0
        self._valid_motors: Tuple[Go2MotorFeedback, ...] = ()
        # Latest q/dq frame usable for the next command's instantaneous PD
        # envelope.  This is intentionally independent of _valid_motors:
        # finite position/velocity from an otherwise limit-faulted frame is
        # safer than silently reusing an older nominal frame.  Lost/non-finite
        # q/dq clears this cache and forces the commissioned degraded path.
        self._envelope_motors: Tuple[Go2MotorFeedback, ...] = ()
        self._envelope_ingress_token = 0
        self._high_level_restore_form: Optional[str] = None
        self._high_level_restore_mode: Optional[str] = None
        # This is deliberately distinct from ``high_level_released``.  Once
        # the ReleaseMode RPC boundary has been crossed, even a timeout is an
        # ambiguous handover and later cleanup must restore/confirm the saved
        # high-level service before releasing the local epoch.  Before that
        # boundary, a failed publisher cleanup needs no high-level handback.
        self._release_rpc_attempted = False
        self._handoff_committing = False
        self._handoff_commit_epoch = 0
        self._handoff_commit_generation = 0
        self._handoff_feedback_fault: Optional[str] = None
        self._handoff_transaction_counter = 0
        self._handoff_active_transaction_id = 0
        self._handoff_ingress_cutoff = 0
        self._handoff_ingress_open_transaction = 0

    async def _run_lifecycle_operation(
        self,
        operation_name: str,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> OperationResult:
        if not self._lifecycle_guard.acquire(blocking=False):
            return OperationResult.failure(
                "GO2_LOW_LEVEL_LIFECYCLE_BUSY",
                f"Cannot start {operation_name}; another owner lifecycle transaction is active",
            )
        try:
            try:
                result = await operation(*args, **kwargs)
            except _OwnerGuardTimeout as exc:
                return self._owner_guard_timeout_failure(exc)
            if not isinstance(result, OperationResult):
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_LIFECYCLE_INVALID_RESULT",
                    f"{operation_name} did not return an OperationResult",
                )
            return result
        finally:
            self._lifecycle_guard.release()

    def _maximum_owner_guard_wait_s(self) -> float:
        """Bound event-loop contention with a possibly stuck DDS ``Write``."""

        period = _finite_optional(self._config.send_period_s)
        jitter = _finite_optional(self._config.maximum_jitter_s)
        if period is None or period <= 0.0 or jitter is None or jitter < 0.0:
            return 0.01
        return max(0.001, period + jitter)

    @contextmanager
    def _event_loop_guard(
        self,
        operation_name: str,
        real_deadline_s: Optional[float] = None,
    ) -> Iterator[None]:
        """Acquire the writer linearization lock without unbounded loop stalls.

        The writer intentionally keeps ``_guard`` across ``publisher.Write`` so
        an older MPC frame cannot be enqueued after a newer revoke generation.
        Consequently every event-loop-side acquisition must be bounded.  A
        timeout retains the publisher/epoch and is reported as an unhealthy,
        externally observable fail-closed condition; it is never interpreted
        as a successful safe-hold or high-level handback.
        """

        wait_s = self._maximum_owner_guard_wait_s()
        if real_deadline_s is not None:
            wait_s = min(wait_s, max(0.0, real_deadline_s - time.monotonic()))
        started_s = time.monotonic()
        if not self._guard.acquire(timeout=wait_s):
            raise _OwnerGuardTimeout(
                operation_name,
                max(0.0, time.monotonic() - started_s),
            )
        try:
            yield
        finally:
            self._guard.release()

    def _owner_guard_timeout_failure(
        self,
        timeout: _OwnerGuardTimeout,
    ) -> OperationResult:
        status = self._status
        thread = self._writer_thread
        writer_alive = thread is not None and thread.is_alive()
        return OperationResult.failure(
            "GO2_OWNER_GUARD_TIMEOUT",
            f"{timeout.operation_name} could not linearize with the sole LowCmd "
            f"writer within {timeout.waited_s:.6f}s; a DDS Write may be stuck",
            {
                "ownership_epoch": self._owner_epoch,
                "owner_lock_retained": self._owner_epoch != 0,
                "publisher_may_be_active": self._publisher is not None,
                "writer_alive": writer_alive,
                "safe_hold_confirmed": False,
                "high_level_handoff_confirmed": False,
                "status_ownership_state": status.ownership_state.value,
            },
        )

    async def acquire(self, permit: Go2OwnershipPermit) -> OperationResult:
        return await self._run_lifecycle_operation("acquire", self._acquire_unlocked, permit)

    async def _acquire_unlocked(self, permit: Go2OwnershipPermit) -> OperationResult:
        """Acquire ownership on the ground and start continuous safe-hold."""

        if not isinstance(permit, Go2OwnershipPermit):
            return OperationResult.failure(
                "GO2_INVALID_OWNERSHIP_PERMIT", "A typed Go2OwnershipPermit is required"
            )
        if not self._allow_hardware_write:
            return OperationResult.failure(
                "GO2_LOW_LEVEL_WRITE_LOCKED",
                "The per-process hardware-write gate does not authorize LowCmd output",
            )
        if not self._config.enabled:
            return OperationResult.failure(
                "GO2_LOW_LEVEL_DISABLED",
                "go2.low_level.enabled is false; observe-only mode cannot acquire LowCmd",
            )
        readiness = self.actuation_readiness()
        if not readiness.ok:
            return readiness
        sdk_bindings = self._sdk
        if sdk_bindings is None or not sdk_bindings.publisher_constructor_deferred_until_init:
            return OperationResult.failure(
                "GO2_PUBLISHER_CONSTRUCTION_CONTRACT_UNVERIFIED",
                "LowCmd publisher construction must be audited as side-effect-free until Init()",
            )
        now = self._clock.monotonic()
        acquire_timeout = self._required_float(self._config.acquire_timeout_s, "acquire_timeout_s")
        real_acquire_deadline = time.monotonic() + acquire_timeout
        mapping_version = self._required_str(self._config.mapping_version, "mapping_version")
        mapping_hash = self._required_str(self._config.mapping_hash, "mapping_hash")
        if not permit.authorizes(now, mapping_version=mapping_version, mapping_hash=mapping_hash):
            return OperationResult.failure(
                "GO2_OWNERSHIP_PERMIT_REJECTED",
                "Ownership transfer requires a fresh matching ground/disarmed/rotor-stop permit",
            )
        if (
            permit.valid_until_s - permit.timestamp_s
            > self._required_float(self._config.acquire_timeout_s, "acquire_timeout_s") + 1.0e-12
        ):
            return OperationResult.failure(
                "GO2_OWNERSHIP_PERMIT_TOO_LONG",
                "Acquisition permit duration exceeds acquire_timeout_s",
            )
        live_ground = self._verify_live_ground_transfer("acquire", permit)
        if not live_ground.ok:
            return live_ground
        with self._event_loop_guard("acquire-initial-state", real_acquire_deadline):
            if not self._connected:
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_DISCONNECTED", "Connect and verify LowState before acquiring"
                )
            if self._status.ownership_state is not LowCmdOwnershipState.OBSERVE_ONLY:
                return OperationResult.failure(
                    "GO2_INVALID_OWNERSHIP_STATE",
                    f"Cannot acquire from {self._status.ownership_state.value}",
                )
            if self._status.fault_reason is not None:
                return OperationResult.failure(
                    "GO2_LOW_STATE_UNSAFE",
                    "A valid LowState frame must clear the current feedback fault before acquisition: "
                    + self._status.fault_reason,
                )
            if not self._low_state_is_fresh_locked(now):
                return OperationResult.failure(
                    "GO2_LOW_STATE_STALE", "LowState is too old to capture a safe-hold pose"
                )
            feedback_failure = self._feedback_limit_failure_locked()
            if feedback_failure is not None:
                return OperationResult.failure("GO2_LOW_STATE_UNSAFE", feedback_failure)
            current_q = self._current_joint_positions_locked()
            if current_q is None:
                return OperationResult.failure(
                    "GO2_LOW_STATE_INCOMPLETE",
                    "All 12 mapped motors need valid q feedback before ownership transfer",
                )
            safe_hold = self._select_safe_hold_pose(current_q)
            if safe_hold is None:
                return OperationResult.failure(
                    "GO2_SAFE_HOLD_INVALID", "Configured safe-hold policy or pose is invalid"
                )
        verification_budget = min(
            max(0.0, real_acquire_deadline - time.monotonic()),
            max(0.0, permit.valid_until_s - self._clock.monotonic()),
        )
        if verification_budget <= 0.0:
            return OperationResult.failure(
                "GO2_OWNERSHIP_PERMIT_EXPIRED",
                "Ground permit expired before the runtime network audit began",
            )
        grant_task = asyncio.ensure_future(
            run_blocking(self._arbiter.acquire_low_level, verification_budget)
        )
        grant, grant_cancelled = await await_nonabandonable(grant_task)
        if grant_cancelled:
            # The worker may have acquired the inter-process lock.  Always
            # collect its result and release that exact epoch before allowing
            # cancellation to escape this pre-ReleaseMode phase.
            cancelled_epoch = grant.data.get("ownership_epoch")
            if (
                grant.ok
                and isinstance(cancelled_epoch, int)
                and not isinstance(cancelled_epoch, bool)
                and cancelled_epoch > 0
            ):
                self._release_uncommitted_arbiter_epoch(
                    cancelled_epoch,
                    "acquisition was cancelled after the arbiter grant",
                )
            raise asyncio.CancelledError
        if not grant.ok:
            return grant
        epoch_raw = grant.data.get("ownership_epoch")
        if isinstance(epoch_raw, bool) or not isinstance(epoch_raw, int) or epoch_raw <= 0:
            granted_epoch = self._arbiter.status().low_level_epoch
            if granted_epoch > 0:
                cleanup = self._release_uncommitted_arbiter_epoch(
                    granted_epoch,
                    "arbiter returned an invalid epoch payload",
                )
                if not cleanup.ok:
                    return cleanup
            return OperationResult.failure(
                "GO2_INVALID_OWNERSHIP_EPOCH", "Arbiter did not return a valid ownership epoch"
            )
        if grant.data.get("network_exclusivity_verified") is not True:
            cleanup = self._release_uncommitted_arbiter_epoch(
                epoch_raw,
                "arbiter grant lacked auditable network-exclusivity evidence",
            )
            if not cleanup.ok:
                return cleanup
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                "Arbiter granted no auditable runtime network-exclusivity proof",
            )
        epoch = epoch_raw
        with self._event_loop_guard("acquire-post-grant-state", real_acquire_deadline):
            self._release_rpc_attempted = False
            self._reset_handoff_tracking_locked()
        # The runtime verifier may take long enough for the one-shot permit or
        # LowState to expire.  Re-sample every local precondition immediately
        # before crossing the irreversible ReleaseMode boundary.
        post_grant_failure: Optional[OperationResult] = None
        with self._event_loop_guard("acquire-pre-release-revalidation", real_acquire_deadline):
            commit_now = self._clock.monotonic()
            if not permit.authorizes(
                commit_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                post_grant_failure = OperationResult.failure(
                    "GO2_OWNERSHIP_PERMIT_EXPIRED",
                    "Ground permit expired while network ownership was being verified",
                )
            elif (
                not self._connected
                or self._status.ownership_state is not LowCmdOwnershipState.OBSERVE_ONLY
                or self._status.fault_reason is not None
                or not self._low_state_is_fresh_locked(commit_now)
            ):
                post_grant_failure = OperationResult.failure(
                    "GO2_LOW_STATE_CHANGED_DURING_ACQUIRE",
                    "LowState or transport health changed before MotionSwitcher release",
                )
            else:
                feedback_failure = self._feedback_limit_failure_locked()
                current_q = self._current_joint_positions_locked()
                safe_hold = (
                    self._select_safe_hold_pose(current_q) if current_q is not None else None
                )
                if feedback_failure is not None or current_q is None or safe_hold is None:
                    post_grant_failure = OperationResult.failure(
                        "GO2_LOW_STATE_CHANGED_DURING_ACQUIRE",
                        feedback_failure
                        or "No complete, limit-safe pose remained available before MotionSwitcher release",
                    )
        if post_grant_failure is not None:
            cleanup = self._release_uncommitted_arbiter_epoch(
                epoch,
                "acquisition preconditions changed after the arbiter grant",
            )
            if not cleanup.ok:
                return cleanup
            return post_grant_failure
        assert current_q is not None
        assert safe_hold is not None
        # Create the DataWriter only after the network audit and host lock have
        # granted this epoch.  Observe-only connection must not itself appear
        # in the DDS graph as a competing LowCmd publisher.
        publisher: Optional[Any] = None
        try:
            sdk = self._sdk
            if sdk is None:
                raise RuntimeError("SDK bindings disappeared during acquisition")
            publisher = sdk.publisher_factory(
                self._required_str(self._config.low_command_topic, "low_command_topic"),
                type(sdk.low_cmd_factory()),
            )
            publisher.Init()
        except Exception as exc:
            return await self._abort_pre_release_acquisition(
                epoch,
                publisher,
                OperationResult.failure(
                    "GO2_LOW_CMD_PUBLISHER_INIT_FAILED",
                    f"LowCmd publisher could not be initialized: {exc}",
                ),
                timeout_s=max(0.0, real_acquire_deadline - time.monotonic()),
            )
        final_pre_release_failure: Optional[str] = None
        with self._event_loop_guard("acquire-final-pre-release", real_acquire_deadline):
            final_now = self._clock.monotonic()
            if time.monotonic() >= real_acquire_deadline or not permit.authorizes(
                final_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                final_pre_release_failure = "Ground permit or acquisition deadline expired while the DDS writer was initialized"
            elif (
                not self._connected
                or self._status.fault_reason is not None
                or not self._low_state_is_fresh_locked(final_now)
                or self._feedback_limit_failure_locked() is not None
            ):
                final_pre_release_failure = "LowState became stale, incomplete, or unsafe while the DDS writer was initialized"
            else:
                latest_q = self._current_joint_positions_locked()
                latest_safe_hold = (
                    self._select_safe_hold_pose(latest_q) if latest_q is not None else None
                )
                if latest_q is None or latest_safe_hold is None:
                    final_pre_release_failure = (
                        "No complete, limit-safe pose remained at the MotionSwitcher boundary"
                    )
                else:
                    current_q = latest_q
                    safe_hold = latest_safe_hold
        if final_pre_release_failure is not None:
            return await self._abort_pre_release_acquisition(
                epoch,
                publisher,
                OperationResult.failure(
                    "GO2_ACQUIRE_PRE_RELEASE_REVALIDATION_FAILED",
                    final_pre_release_failure,
                ),
                timeout_s=max(0.0, real_acquire_deadline - time.monotonic()),
            )
        with self._event_loop_guard("acquire-publisher-commit", real_acquire_deadline):
            self._publisher = publisher
            self._owner_epoch = epoch
            self._safe_hold_q = safe_hold
            self._last_commanded_q = current_q
            self._command_reference_generation = 0
            self._command_reference_q = None
            self._command_reference_write_s = None
            self._command_reference_ingress_cutoff = 0
            self._target = None
            self._last_accepted_sequence = None
            self._status = replace(
                self._status,
                timestamp=final_now,
                ownership_state=LowCmdOwnershipState.ACQUIRING,
                owner_epoch=epoch,
                healthy=False,
                target_sequence=None,
                target_age_s=None,
                target_deadline=None,
                mailbox_staged_target_sequence=None,
                writer_enqueued_target_sequence=None,
                actuator_applied_target_sequence=None,
                writer_enqueue_generation=0,
                writer_enqueued_q_rad=(),
                publisher_active=True,
                safe_hold_active=False,
                high_level_released=False,
                network_exclusivity_verified=True,
                continuous_owner_monitoring_active=False,
                independent_watchdog_active=False,
                writer_enqueue_ack_available=True,
                actuator_application_ack_available=False,
                watchdog_healthy=False,
                fault_reason=None,
            )
        cancelled = False
        release_task = asyncio.ensure_future(
            self._release_high_level_mode(
                max(0.001, real_acquire_deadline - time.monotonic()),
                permit,
            )
        )
        release_result, release_cancelled = await await_nonabandonable(release_task)
        # Complete the critical transfer. Returning from cancellation in the
        # gap after ReleaseMode but before the fixed-rate writer starts would
        # leave the robot with no controller.
        cancelled = cancelled or release_cancelled
        if not release_result.ok:
            release_attempted = release_result.data.get("release_rpc_attempted") is True
            if release_attempted:
                # Once ReleaseMode crossed the RPC boundary its outcome may be
                # ambiguous even without an ACK, so retain the epoch/host lock.
                # ReleaseMode ACK alone is not authority to publish.  Only a
                # subsequent CheckMode name=="" proves that the high-level
                # service is absent.  Keep the epoch/host lock, but send no
                # LowCmd while that fact is ambiguous.
                with self._event_loop_guard("acquire-release-fault", real_acquire_deadline):
                    self._status = replace(
                        self._status,
                        timestamp=self._clock.monotonic(),
                        ownership_state=LowCmdOwnershipState.FAULT,
                        owner_epoch=epoch,
                        healthy=False,
                        high_level_released=False,
                        network_exclusivity_verified=True,
                        safe_hold_active=False,
                        safe_hold_settled=False,
                        watchdog_healthy=False,
                        fault_reason=(
                            "MotionSwitcher release outcome is not fully verified: "
                            f"{release_result.message}"
                        ),
                    )
                failure = OperationResult.failure(
                    release_result.code,
                    release_result.message,
                    {
                        **dict(release_result.data),
                        "ownership_epoch": epoch,
                        "owner_lock_retained": True,
                        "fault_hold_started": False,
                    },
                )
            else:
                with self._event_loop_guard("acquire-abort-publisher", real_acquire_deadline):
                    publisher = self._publisher
                failure = await self._abort_pre_release_acquisition(
                    epoch,
                    publisher,
                    release_result,
                    timeout_s=max(0.0, real_acquire_deadline - time.monotonic()),
                )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        with self._event_loop_guard("acquire-start-writer", real_acquire_deadline):
            self._safe_hold_request_generation += 1
            self._safe_hold_command_reached_generation = 0
            self._safe_hold_command_reached_write_s = None
            self._safe_hold_feedback_sequence_required = 0
            self._safe_hold_feedback_ingress_token_required = 0
            self._status = replace(
                self._status,
                timestamp=self._clock.monotonic(),
                high_level_released=True,
                high_level_restore_form=self._high_level_restore_form,
                high_level_restore_mode=self._high_level_restore_mode,
                ownership_state=LowCmdOwnershipState.HOLDING,
                safe_hold_active=False,
                safe_hold_settled=False,
                safe_hold_request_generation=self._safe_hold_request_generation,
                watchdog_healthy=True,
            )
            self._start_writer_locked()
        first_write_task = asyncio.ensure_future(
            run_blocking(
                self._first_write.wait,
                max(0.0, real_acquire_deadline - time.monotonic()),
            )
        )
        first_write, first_write_cancelled = await await_nonabandonable(first_write_task)
        cancelled = cancelled or first_write_cancelled
        status = self.status()
        if (
            not first_write
            or status.fault_reason is not None
            or status.safe_hold_write_generation < status.safe_hold_request_generation
            or not status.safe_hold_active
        ):
            with self._event_loop_guard("acquire-first-write-fault", real_acquire_deadline):
                self._set_fault_locked("LowCmd writer did not establish a verified local enqueue")
            failure = OperationResult.failure(
                "GO2_LOW_CMD_START_FAILED",
                "LowCmd writer failed after high-level mode release; owner lock remains held",
                {"ownership_epoch": epoch},
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        settle_generation = status.safe_hold_request_generation
        settle_task = asyncio.ensure_future(
            run_blocking(
                self._wait_for_safe_hold_settled,
                epoch,
                settle_generation,
                min(
                    self._required_float(
                        self._config.safe_hold_ack_timeout_s,
                        "safe_hold_ack_timeout_s",
                    ),
                    max(0.0, real_acquire_deadline - time.monotonic()),
                ),
            )
        )
        settled, settle_cancelled = await await_nonabandonable(settle_task)
        cancelled = cancelled or settle_cancelled
        if not settled:
            with self._event_loop_guard("acquire-settle-fault", real_acquire_deadline):
                self._set_fault_locked(
                    "Initial safe-hold was enqueued but no causal in-tolerance LowState arrived"
                )
            failure = OperationResult.failure(
                "GO2_SAFE_HOLD_NOT_SETTLED",
                "Acquisition remains owned because initial safe-hold feedback did not settle",
                {"ownership_epoch": epoch, "safe_hold_generation": settle_generation},
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        if cancelled:
            raise asyncio.CancelledError
        return OperationResult.success(
            "LowCmd owner acquired and safe-hold stream started",
            {
                "ownership_epoch": epoch,
                "mapping_hash": mapping_hash,
                "local_single_instance_held": True,
                "network_exclusivity_verified": True,
                "network_verifier_name": grant.data.get("network_verifier_name"),
                "network_verification_timestamp_s": grant.data.get(
                    "network_verification_timestamp_s"
                ),
                "dds_matched_enqueue_ack_only": True,
                "actuator_ack_available": False,
                "continuous_owner_monitoring_active": False,
                "independent_watchdog_active": False,
            },
        )

    async def submit(
        self,
        command: Go2JointPositionCommand,
        *,
        ownership_epoch: int,
        mapping_hash: str,
    ) -> OperationResult:
        """Atomically replace the mailbox after validating epoch, hash, TTL, and limits."""

        if not isinstance(command, Go2JointPositionCommand):
            return OperationResult.failure(
                "GO2_INVALID_LOW_CMD_TARGET", "command must be a Go2JointPositionCommand"
            )
        now = self._clock.monotonic()
        logical_time_left_s = max(0.0, command.valid_until_s - now)
        real_deadline_s = time.monotonic() + min(
            self._maximum_owner_guard_wait_s(), logical_time_left_s
        )
        try:
            with self._event_loop_guard("submit", real_deadline_s):
                now = self._clock.monotonic()
                if self._status.ownership_state not in {
                    LowCmdOwnershipState.HOLDING,
                    LowCmdOwnershipState.MPC_ACTIVE,
                }:
                    return OperationResult.failure(
                        "GO2_LOW_LEVEL_NOT_ACTIVE",
                        "Acquire the LowCmd owner before submitting MPC",
                    )
                if (
                    type(ownership_epoch) is not int
                    or ownership_epoch <= 0
                    or ownership_epoch != self._owner_epoch
                ):
                    return OperationResult.failure(
                        "GO2_OWNERSHIP_EPOCH_MISMATCH",
                        "Target belongs to a stale or different LowCmd owner",
                        {"ownership_epoch": self._owner_epoch},
                    )
                expected_hash = self._required_str(self._config.mapping_hash, "mapping_hash")
                if mapping_hash != expected_hash:
                    return OperationResult.failure(
                        "GO2_MAPPING_HASH_MISMATCH",
                        "Target mapping hash does not match the commissioned actuator mapping",
                        {"expected_mapping_hash": expected_hash},
                    )
                if not command.is_fresh(now):
                    return OperationResult.failure(
                        "GO2_MPC_TARGET_STALE",
                        "Target is not inside its monotonic validity window",
                    )
                maximum_ttl = self._required_float(self._config.target_ttl_s, "target_ttl_s")
                if command.valid_until_s - command.timestamp_s > maximum_ttl + 1.0e-12:
                    return OperationResult.failure(
                        "GO2_MPC_TARGET_TTL_EXCEEDED",
                        "Target validity window exceeds the configured MPC TTL",
                    )
                if (
                    self._last_accepted_sequence is not None
                    and command.sequence <= self._last_accepted_sequence
                ):
                    return OperationResult.failure(
                        "GO2_MPC_TARGET_REPLAY",
                        "Target sequence must increase strictly within one ownership epoch",
                    )
                limit_failure = self._validate_algorithm_positions(command.joint_positions_rad)
                if limit_failure is not None:
                    return OperationResult.failure("GO2_JOINT_TARGET_LIMIT", limit_failure)
                if not self._low_state_is_fresh_locked(now):
                    return OperationResult.failure(
                        "GO2_LOW_STATE_STALE",
                        "Fresh motor feedback is required before target submit",
                    )
                self._target = command
                self._last_accepted_sequence = command.sequence
                self._status = replace(
                    self._status,
                    timestamp=now,
                    ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
                    target_sequence=command.sequence,
                    target_age_s=max(0.0, now - command.timestamp_s),
                    target_deadline=command.valid_until_s,
                    mailbox_staged_target_sequence=command.sequence,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    watchdog_healthy=True,
                    fault_reason=None,
                )
        except _OwnerGuardTimeout as exc:
            return self._owner_guard_timeout_failure(exc)
        return OperationResult.success(
            "Fresh joint target staged for the fixed-rate LowCmd owner",
            {
                "sequence": command.sequence,
                "ownership_epoch": ownership_epoch,
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                # submit() never waits for the writer thread or invents a
                # motor-side acknowledgement.
                "writer_enqueue_acknowledged": False,
                "writer_enqueued_target_sequence": None,
                "actuator_application_acknowledged": False,
                "actuator_applied_target_sequence": None,
            },
            code="GO2_LOW_CMD_TARGET_STAGED",
        )

    async def revoke(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        return await self._run_lifecycle_operation(
            "revoke",
            self._revoke_unlocked,
            reason,
            ownership_epoch=ownership_epoch,
        )

    async def _revoke_unlocked(
        self,
        reason: str,
        *,
        ownership_epoch: Optional[int] = None,
    ) -> OperationResult:
        """Drop the target lease and keep the same publisher in safe-hold."""

        if not isinstance(reason, str) or not reason.strip():
            return OperationResult.failure("GO2_REVOKE_REASON_REQUIRED", "reason cannot be empty")
        safe_hold_deadline = time.monotonic() + self._required_float(
            self._config.safe_hold_ack_timeout_s,
            "safe_hold_ack_timeout_s",
        )
        now = self._clock.monotonic()
        with self._event_loop_guard("revoke", safe_hold_deadline):
            if ownership_epoch is not None and (
                type(ownership_epoch) is not int
                or ownership_epoch <= 0
                or ownership_epoch != self._owner_epoch
            ):
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "Stale code cannot revoke a newer LowCmd ownership epoch",
                    {"ownership_epoch": self._owner_epoch},
                )
            if self._owner_epoch == 0:
                return OperationResult.success("No active LowCmd owner needed revocation")
            existing_fault = self._status.fault_reason
            if self._status.ownership_state is LowCmdOwnershipState.FAULT:
                generation = self._enter_safe_hold_locked(
                    now,
                    existing_fault or f"MPC target revoked during owner fault: {reason}",
                    fault=True,
                    force_new_generation=True,
                )
            else:
                generation = self._enter_safe_hold_locked(
                    now,
                    f"MPC target revoked: {reason}",
                    fault=False,
                    force_new_generation=True,
                )
            writer_alive = self._writer_thread is not None and self._writer_thread.is_alive()
            epoch = self._owner_epoch
        if not writer_alive:
            return OperationResult.failure(
                "GO2_SAFE_HOLD_WRITER_UNAVAILABLE",
                "MPC target was revoked, but the sole LowCmd writer is not alive",
                {"ownership_epoch": epoch, "safe_hold_generation": generation},
            )
        cancelled = False
        wait_task = asyncio.ensure_future(
            run_blocking(
                self._wait_for_safe_hold_write,
                epoch,
                generation,
                max(0.0, safe_hold_deadline - time.monotonic()),
            )
        )
        acknowledged, wait_cancelled = await await_nonabandonable(wait_task)
        # Revocation is a safety-critical commit. Let the local writer finish
        # (or time out) before propagating task cancellation.
        cancelled = cancelled or wait_cancelled
        if not acknowledged:
            with self._event_loop_guard("revoke-timeout-fault", safe_hold_deadline):
                self._set_fault_locked(
                    f"Safe-hold generation {generation} was not locally enqueued before timeout"
                )
            result = OperationResult.failure(
                "GO2_SAFE_HOLD_ACK_TIMEOUT",
                "The sole writer did not confirm a safe-hold enqueue before timeout",
                {"ownership_epoch": epoch, "safe_hold_generation": generation},
            )
            if cancelled:
                raise asyncio.CancelledError
            return result
        settle_task = asyncio.ensure_future(
            run_blocking(
                self._wait_for_safe_hold_settled,
                epoch,
                generation,
                max(0.0, safe_hold_deadline - time.monotonic()),
            )
        )
        settled, settle_cancelled = await await_nonabandonable(settle_task)
        cancelled = cancelled or settle_cancelled
        if not settled:
            with self._event_loop_guard("revoke-settle-fault", safe_hold_deadline):
                self._set_fault_locked(
                    f"Safe-hold generation {generation} was enqueued but did not settle"
                )
            result = OperationResult.failure(
                "GO2_SAFE_HOLD_NOT_SETTLED",
                "Post-write LowState did not satisfy the safe-hold position/velocity tolerances",
                {"ownership_epoch": epoch, "safe_hold_generation": generation},
            )
            if cancelled:
                raise asyncio.CancelledError
            return result
        if cancelled:
            raise asyncio.CancelledError
        return OperationResult.success(
            "MPC target revoked; the sole writer locally enqueued safe-hold",
            {
                "ownership_epoch": epoch,
                "safe_hold_generation": generation,
                "local_enqueue_acknowledged": True,
                "post_write_feedback_settled": True,
                "actuator_ack_available": False,
            },
        )

    async def release(
        self,
        permit: Go2OwnershipPermit,
        reason: str,
        *,
        ownership_epoch: int,
    ) -> OperationResult:
        return await self._run_lifecycle_operation(
            "release",
            self._release_unlocked,
            permit,
            reason,
            ownership_epoch=ownership_epoch,
        )

    async def _release_unlocked(
        self,
        permit: Go2OwnershipPermit,
        reason: str,
        *,
        ownership_epoch: int,
    ) -> OperationResult:
        """Stop the stream and local ownership only under a fresh ground permit."""

        if not isinstance(permit, Go2OwnershipPermit):
            return OperationResult.failure(
                "GO2_INVALID_OWNERSHIP_PERMIT", "A typed Go2OwnershipPermit is required"
            )
        if not isinstance(reason, str) or not reason.strip():
            return OperationResult.failure("GO2_RELEASE_REASON_REQUIRED", "reason cannot be empty")
        if type(ownership_epoch) is not int or ownership_epoch < 0:
            return OperationResult.failure(
                "GO2_OWNERSHIP_EPOCH_MISMATCH",
                "Ownership release requires an exact nonnegative integer epoch",
                {"ownership_epoch": self._owner_epoch},
            )
        with self._event_loop_guard("release-initial-state"):
            if not self._config.enabled and self._owner_epoch == 0:
                if ownership_epoch != 0:
                    return OperationResult.failure(
                        "GO2_OWNERSHIP_EPOCH_MISMATCH",
                        "Disabled LowCmd has no nonzero ownership epoch",
                    )
                return OperationResult.success("LowCmd is disabled and no owner was active")
            if type(ownership_epoch) is not int or ownership_epoch != self._owner_epoch:
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "Stale code cannot release the current LowCmd ownership epoch",
                    {"ownership_epoch": self._owner_epoch},
                )
        now = self._clock.monotonic()
        mapping_version = self._required_str(self._config.mapping_version, "mapping_version")
        mapping_hash = self._required_str(self._config.mapping_hash, "mapping_hash")
        if not permit.authorizes(
            now,
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
        ):
            return OperationResult.failure(
                "GO2_RELEASE_PERMIT_REJECTED",
                "Stopping LowCmd requires fresh support/disarm/rotor-stop evidence",
            )
        if (
            permit.valid_until_s - permit.timestamp_s
            > self._required_float(self._config.release_timeout_s, "release_timeout_s") + 1.0e-12
        ):
            return OperationResult.failure(
                "GO2_RELEASE_PERMIT_TOO_LONG",
                "Release permit duration exceeds release_timeout_s",
            )
        live_ground = self._verify_live_ground_transfer("release", permit)
        if not live_ground.ok:
            return live_ground
        release_timeout = self._required_float(self._config.release_timeout_s, "release_timeout_s")
        real_deadline = time.monotonic() + release_timeout
        with self._event_loop_guard("release-owner-snapshot", real_deadline):
            epoch = self._owner_epoch
            handoff_pending = self._status.high_level_released
            existing_thread = self._writer_thread
            writer_running = existing_thread is not None and existing_thread.is_alive()
            handback_only = epoch != 0 and self._publisher is None and not writer_running
            needs_new_hold = writer_running and (
                self._target is not None or not self._status.safe_hold_active
            )
        if handback_only:
            # A previous transaction has already obtained a non-ambiguous
            # local Close() result. There is no LowCmd endpoint left to hold or
            # restart, so requiring LowState/safe-hold convergence here could
            # permanently prevent the only safe recovery action: restore and
            # confirm the exact captured high-level service, then unlock.
            return await self._complete_closed_endpoint_handoff(
                permit,
                epoch=epoch,
                ownership_epoch=ownership_epoch,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                real_deadline=real_deadline,
            )
        # If a stream exists, first commit a new safe-hold generation through
        # that exact writer.  Merely clearing the MPC mailbox is not an ACK.
        if epoch != 0 and needs_new_hold:
            revoked = await self._revoke_unlocked(
                f"owner release requested: {reason}", ownership_epoch=epoch
            )
            if not revoked.ok:
                return revoked

        if epoch != 0 and writer_running:
            with self._event_loop_guard("release-hold-snapshot", real_deadline):
                hold_generation = self._safe_hold_request_generation
                hold_settled = self._status.safe_hold_settled
            if not hold_settled:
                settle_timeout = min(
                    self._required_float(
                        self._config.safe_hold_ack_timeout_s,
                        "safe_hold_ack_timeout_s",
                    ),
                    max(0.0, real_deadline - time.monotonic()),
                )
                cancelled = False
                settle_task = asyncio.ensure_future(
                    run_blocking(
                        self._wait_for_safe_hold_settled,
                        epoch,
                        hold_generation,
                        settle_timeout,
                    )
                )
                hold_settled, settle_cancelled = await await_nonabandonable(settle_task)
                cancelled = cancelled or settle_cancelled
                if not hold_settled:
                    with self._event_loop_guard("release-settle-fault", real_deadline):
                        self._set_fault_locked(
                            f"Safe-hold generation {hold_generation} did not settle before release timeout"
                        )
                    failure = OperationResult.failure(
                        "GO2_SAFE_HOLD_NOT_SETTLED",
                        "Writer remains active because post-write LowState did not satisfy safe-hold tolerances",
                        {
                            "ownership_epoch": epoch,
                            "safe_hold_generation": hold_generation,
                        },
                    )
                    if cancelled:
                        raise asyncio.CancelledError
                    return failure
                if cancelled:
                    raise asyncio.CancelledError

        # Revalidate independent live hardware evidence, the one-shot permit,
        # and all feedback immediately before stopping the sole writer.
        live_ground = self._verify_live_ground_transfer("release", permit)
        if not live_ground.ok:
            return live_ground
        # stopping the writer.  This closes the revoke/wait TOCTOU window.
        with self._event_loop_guard("release-writer-stop", real_deadline):
            commit_now = self._clock.monotonic()
            if self._owner_epoch != epoch or ownership_epoch != epoch:
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "Ownership changed during the release transaction",
                    {"ownership_epoch": self._owner_epoch},
                )
            if not permit.authorizes(
                commit_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                return OperationResult.failure(
                    "GO2_RELEASE_PERMIT_EXPIRED",
                    "Ground release permit expired before the writer-stop boundary",
                )
            if epoch != 0 and (
                not self._low_state_is_fresh_locked(commit_now)
                or self._feedback_limit_failure_locked() is not None
            ):
                return OperationResult.failure(
                    "GO2_RELEASE_LOW_STATE_UNSAFE",
                    "Fresh, complete, in-limit LowState is required at writer stop",
                )
            thread = self._writer_thread
            writer_running = thread is not None and thread.is_alive()
            if writer_running and (
                self._safe_hold_write_generation < self._safe_hold_request_generation
                or not self._status.safe_hold_active
                or not self._status.safe_hold_settled
            ):
                return OperationResult.failure(
                    "GO2_SAFE_HOLD_NOT_ACKNOWLEDGED",
                    "Writer stop is forbidden until the latest safe-hold generation is locally enqueued and causally settled",
                )
            if epoch != 0:
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.RELEASING,
                    healthy=False,
                )
                self._writer_stop.set()
        cancelled = False
        if epoch == 0:
            if not handoff_pending:
                return OperationResult.success("No LowCmd owner was active")
            with self._event_loop_guard("release-zero-epoch-publisher", real_deadline):
                publisher = self._publisher
            close_result = await self._close_owned_publisher_with_deadline(
                publisher,
                real_deadline_s=real_deadline,
            )
            if not close_result.ok:
                with self._event_loop_guard("release-zero-epoch-close-fault", real_deadline):
                    self._status = replace(
                        self._status,
                        timestamp=self._clock.monotonic(),
                        ownership_state=LowCmdOwnershipState.FAULT,
                        healthy=False,
                        fault_reason=close_result.message,
                    )
                self._allow_publisher_close_retry_after_failure(publisher, close_result)
                return close_result
            with self._event_loop_guard("release-zero-epoch-close-commit", real_deadline):
                if self._publisher is publisher:
                    self._publisher = None
                self._consume_publisher_close_result(publisher)
                self._status = replace(self._status, publisher_active=False)
            handoff_task = asyncio.ensure_future(
                self._restore_high_level_mode(max(0.0, real_deadline - time.monotonic()))
            )
            handoff, handoff_cancelled = await await_nonabandonable(handoff_task)
            cancelled = cancelled or handoff_cancelled
            if not handoff.ok:
                if cancelled:
                    raise asyncio.CancelledError
                return handoff
            with self._event_loop_guard("release-zero-epoch-handback-commit", real_deadline):
                self._high_level_restore_form = None
                self._high_level_restore_mode = None
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
                    healthy=self._low_state_is_fresh_locked(self._clock.monotonic()),
                    high_level_released=False,
                    high_level_restore_form=None,
                    high_level_restore_mode=None,
                    network_exclusivity_verified=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    fault_reason=None,
                )
            if cancelled:
                raise asyncio.CancelledError
            return OperationResult.success("High-level control handoff is now acknowledged")
        if thread is not None:
            remaining = max(0.0, real_deadline - time.monotonic())
            join_task = asyncio.ensure_future(
                run_blocking(
                    thread.join,
                    remaining,
                )
            )
            _, join_cancelled = await await_nonabandonable(join_task)
            cancelled = cancelled or join_cancelled
        if thread is not None and thread.is_alive():
            with self._event_loop_guard("release-stop-timeout-fault", real_deadline):
                self._set_fault_locked("LowCmd writer did not stop before release timeout")
            if cancelled:
                raise asyncio.CancelledError
            return OperationResult.failure(
                "GO2_LOW_CMD_STOP_TIMEOUT",
                "Writer may still be publishing; local owner lock remains held",
                {"ownership_epoch": epoch},
            )
        if thread is not None:
            # ``_writer_stop.set()`` is not itself a handoff fence: a LowState
            # callback can revoke/replace the safe-hold generation while the
            # thread is exiting.  Revalidate the exact settled generation
            # before destroying the publisher.  If the fence moved, restart
            # the same sole writer and retain the epoch; never SelectMode after
            # a post-stop feedback fault.
            post_stop_failure: Optional[str] = None
            with self._event_loop_guard("release-post-stop-fence", real_deadline):
                post_stop_now = self._clock.monotonic()
                if self._owner_epoch != epoch:
                    post_stop_failure = "Ownership epoch changed while the writer stopped"
                elif not self._low_state_is_fresh_locked(post_stop_now):
                    post_stop_failure = "LowState became stale while the writer stopped"
                else:
                    feedback_failure = self._feedback_limit_failure_locked()
                    if feedback_failure is not None:
                        post_stop_failure = feedback_failure
                    elif not (
                        self._safe_hold_request_generation
                        == self._safe_hold_write_generation
                        == self._safe_hold_command_reached_generation
                        and self._safe_hold_request_generation > 0
                        and self._status.safe_hold_active
                        and self._status.safe_hold_settled
                    ):
                        post_stop_failure = (
                            "The latest safe-hold generation changed or lost its "
                            "post-write settled feedback while the writer stopped"
                        )
                if post_stop_failure is not None:
                    self._enter_safe_hold_locked(
                        post_stop_now,
                        "Release post-stop fence failed: " + post_stop_failure,
                        fault=True,
                    )
                    self._writer_thread = None
                    self._start_writer_locked()
            if post_stop_failure is not None:
                if cancelled:
                    raise asyncio.CancelledError
                return OperationResult.failure(
                    "GO2_RELEASE_POST_STOP_FENCE_FAILED",
                    "Publisher remains active and the sole safe-hold writer was "
                    "restarted because release evidence changed after stop",
                    {
                        "ownership_epoch": epoch,
                        "owner_lock_retained": True,
                        "publisher_close_called": False,
                        "high_level_select_called": False,
                        "reason": post_stop_failure,
                    },
                )
        # Joining the writer and running the post-stop fence may consume the
        # entire authorization window. Re-read independent ground evidence
        # again at the actual Close boundary. If it changed, keep the endpoint
        # and restart the same writer rather than creating a control vacuum.
        final_live_ground = self._verify_live_ground_transfer("release", permit)
        if not final_live_ground.ok:
            with self._event_loop_guard("release-final-ground-failure", real_deadline):
                if thread is not None and self._publisher is not None:
                    restart_now = self._clock.monotonic()
                    self._reset_handoff_tracking_locked()
                    self._enter_safe_hold_locked(
                        restart_now,
                        "Release final live-ground check failed: " + final_live_ground.message,
                        fault=True,
                    )
                    self._writer_thread = None
                    self._start_writer_locked()
            failure = OperationResult.failure(
                final_live_ground.code,
                "Publisher was not closed because final live-ground evidence changed: "
                + final_live_ground.message,
                {
                    **dict(final_live_ground.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_close_called": False,
                    "high_level_select_called": False,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        # Final linearization fence.  It intentionally sits immediately before
        # Close and latches callbacks into handoff-fault reporting rather than
        # allowing them to create a generation after the writer has stopped.
        final_fence_failure: Optional[str] = None
        handoff_transaction_id = 0
        handoff_ingress_cutoff = 0
        with self._event_loop_guard("release-final-handoff-fence", real_deadline):
            final_fence_now = self._clock.monotonic()
            if self._owner_epoch != epoch:
                final_fence_failure = "Ownership epoch changed before publisher Close"
            elif time.monotonic() >= real_deadline or not permit.authorizes(
                final_fence_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                final_fence_failure = (
                    "Ground release permit or deadline expired before publisher Close"
                )
            elif epoch != 0 and self._publisher is None:
                final_fence_failure = (
                    "LowCmd publisher disappeared without a successful Close transaction"
                )
            elif not self._low_state_is_fresh_locked(final_fence_now):
                final_fence_failure = "LowState became stale before publisher Close"
            else:
                feedback_failure = self._feedback_limit_failure_locked()
                if feedback_failure is not None:
                    final_fence_failure = feedback_failure
                elif thread is not None and not (
                    self._safe_hold_request_generation
                    == self._safe_hold_write_generation
                    == self._safe_hold_command_reached_generation
                    and self._safe_hold_request_generation > 0
                    and self._status.safe_hold_active
                    and self._status.safe_hold_settled
                ):
                    final_fence_failure = (
                        "The safe-hold generation changed after the post-stop fence"
                    )
            if final_fence_failure is not None:
                if thread is not None:
                    self._reset_handoff_tracking_locked()
                    self._enter_safe_hold_locked(
                        final_fence_now,
                        "Release final handoff fence failed: " + final_fence_failure,
                        fault=True,
                    )
                    self._writer_thread = None
                    self._start_writer_locked()
            else:
                if self._handoff_committing and self._handoff_commit_epoch not in (
                    0,
                    epoch,
                ):
                    final_fence_failure = "A different epoch owns the handoff commit latch"
                else:
                    self._handoff_committing = True
                    self._handoff_commit_epoch = epoch
                    self._handoff_commit_generation = (
                        self._safe_hold_request_generation if thread is not None else 0
                    )
                    self._handoff_transaction_counter += 1
                    handoff_transaction_id = self._handoff_transaction_counter
                    self._handoff_active_transaction_id = handoff_transaction_id
                    with self._low_state_ingress_condition:
                        handoff_ingress_cutoff = self._low_state_ingress_sequence
                        self._handoff_ingress_cutoff = handoff_ingress_cutoff
                        self._handoff_ingress_open_transaction = handoff_transaction_id
                    # Each explicit, freshly ground-authorized retry starts a
                    # new handoff transaction.  A fault arriving after this
                    # exact fence becomes sticky until the transaction ends.
                    self._handoff_feedback_fault = None
                    if self._writer_thread is thread:
                        self._writer_thread = None
                    publisher = self._publisher
                    self._status = replace(
                        self._status,
                        timestamp=final_fence_now,
                        publisher_active=self._publisher is not None,
                        writer_alive=False,
                    )
        if final_fence_failure is not None:
            if cancelled:
                raise asyncio.CancelledError
            return OperationResult.failure(
                "GO2_RELEASE_FINAL_HANDOFF_FENCE_FAILED",
                "Publisher was not closed and the owner epoch remains held because "
                "feedback changed at the final handoff fence",
                {
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_close_called": False,
                    "high_level_select_called": False,
                    "reason": final_fence_failure,
                },
            )
        close_result = await self._close_owned_publisher_with_deadline(
            publisher,
            real_deadline_s=real_deadline,
        )
        if not close_result.ok:
            self._close_handoff_ingress_window(handoff_transaction_id)
            with self._event_loop_guard("release-close-fault", real_deadline):
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=True,
                    writer_alive=False,
                    watchdog_healthy=False,
                    fault_reason=close_result.message,
                )
            self._allow_publisher_close_retry_after_failure(publisher, close_result)
            if cancelled:
                raise asyncio.CancelledError
            return OperationResult.failure(
                close_result.code,
                close_result.message,
                {
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    **dict(close_result.data),
                },
            )
        with self._event_loop_guard("release-close-commit", real_deadline):
            if self._publisher is publisher:
                self._publisher = None
            self._consume_publisher_close_result(publisher)
            self._status = replace(
                self._status,
                timestamp=self._clock.monotonic(),
                publisher_active=False,
                safe_hold_active=False,
                safe_hold_settled=False,
            )
            # Close the registration window atomically with clearing the
            # publisher reference. A callback is therefore classified either
            # as crossing Close or as post-Close diagnostic telemetry; there
            # is no unclassified interval between those two facts.
            self._close_handoff_ingress_window(handoff_transaction_id)
        handoff_task = asyncio.ensure_future(
            self._restore_high_level_mode(max(0.0, real_deadline - time.monotonic()))
        )
        handoff, handoff_cancelled = await await_nonabandonable(handoff_task)
        cancelled = cancelled or handoff_cancelled
        drain_task = asyncio.ensure_future(
            run_blocking(
                self._wait_for_handoff_callbacks,
                handoff_transaction_id,
                handoff_ingress_cutoff,
                max(0.0, real_deadline - time.monotonic()),
            )
        )
        callbacks_drained, drain_cancelled = await await_nonabandonable(drain_task)
        cancelled = cancelled or drain_cancelled
        if not callbacks_drained:
            with self._event_loop_guard("release-callback-drain-fault", real_deadline):
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=not handoff.ok,
                    fault_reason=(
                        "LowCmd is closed, but LowState callbacks crossing the "
                        "Close boundary did not drain before timeout"
                    ),
                )
            failure = OperationResult.failure(
                "GO2_HANDOFF_CALLBACK_DRAIN_TIMEOUT",
                "LowCmd is closed, but pre-Close LowState callbacks remain in flight",
                {
                    **dict(handoff.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_closed": True,
                    "high_level_reactivation_acknowledged": handoff.ok,
                    "handoff_failure_code": None if handoff.ok else handoff.code,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        if not handoff.ok:
            with self._event_loop_guard("release-handback-fault", real_deadline):
                self._target = None
                self._last_accepted_sequence = None
                self._last_commanded_q = None
                self._command_reference_generation = 0
                self._command_reference_q = None
                self._command_reference_write_s = None
                self._command_reference_ingress_cutoff = 0
                self._safe_hold_q = None
                self._expected_deadline_s = None
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    target_sequence=None,
                    target_age_s=None,
                    target_deadline=None,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=True,
                    fault_reason=(
                        "LowCmd stopped on the verified ground, but high-level handoff "
                        "was not acknowledged; local owner lock is retained"
                    ),
                )
            failure = OperationResult.failure(
                handoff.code,
                "LowCmd is stopped on the ground, but exact high-level handback failed: "
                + handoff.message,
                {
                    **dict(handoff.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "high_level_reactivation_acknowledged": False,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        release_result: Optional[OperationResult] = None
        commit_failure: Optional[OperationResult] = None
        with self._event_loop_guard("release-epoch-commit", real_deadline):
            handoff_feedback_fault = self._handoff_feedback_fault
            commit_now = self._clock.monotonic()
            if handoff_feedback_fault is not None:
                # High-level control is now positively restored, so do not try
                # to resurrect a closed LowCmd writer. Retain the local epoch
                # and sticky fault for an explicit fresh-ground retry.
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=False,
                    network_exclusivity_verified=True,
                    fault_reason=(
                        "High-level control was restored, but LowState changed "
                        "after the final handoff fence: " + handoff_feedback_fault
                    ),
                )
            elif self._owner_epoch != epoch or self._publisher is not None:
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    healthy=False,
                    high_level_released=False,
                    fault_reason=(
                        "High-level mode was restored, but local owner state "
                        "changed before the epoch commit"
                    ),
                )
                commit_failure = OperationResult.failure(
                    "GO2_HANDOFF_COMMIT_STATE_CHANGED",
                    "Local owner or publisher state changed before the final epoch commit",
                    {
                        "ownership_epoch": self._owner_epoch,
                        "owner_lock_retained": True,
                    },
                )
            elif time.monotonic() >= real_deadline or not permit.authorizes(
                commit_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    healthy=False,
                    publisher_active=False,
                    high_level_released=False,
                    fault_reason=(
                        "Exact high-level mode was restored, but the release "
                        "permit expired before the local epoch commit"
                    ),
                )
                commit_failure = OperationResult.failure(
                    "GO2_RELEASE_PERMIT_EXPIRED",
                    "High-level mode is restored, but the expired transaction did not unlock the retained epoch",
                    {
                        "ownership_epoch": epoch,
                        "owner_lock_retained": True,
                        "publisher_closed": True,
                        "high_level_reactivation_acknowledged": True,
                    },
                )
            else:
                # Keep the bridge guard held across arbiter unlock and local
                # epoch clear. LowState callbacks use this same guard, which
                # makes this the final linearization point of the handoff.
                release_result = self._release_arbiter_epoch(epoch)
                if release_result.ok:
                    self._clear_released_owner_locked()
                else:
                    self._status = replace(
                        self._status,
                        timestamp=commit_now,
                        ownership_state=LowCmdOwnershipState.FAULT,
                        healthy=False,
                        publisher_active=False,
                        writer_alive=False,
                        safe_hold_active=False,
                        high_level_released=False,
                        network_exclusivity_verified=(
                            release_result.data.get("owner_lock_retained") is True
                            and release_result.data.get("local_single_instance_poisoned")
                            is not True
                            and release_result.data.get("ownership_exclusivity_lost") is not True
                        ),
                        fault_reason=release_result.message,
                    )
        if handoff_feedback_fault is not None:
            failure = OperationResult.failure(
                "GO2_HANDOFF_FEEDBACK_CHANGED",
                "Exact high-level mode was restored, but a late LowState fault "
                "prevents ownership epoch release",
                {
                    **dict(handoff.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_closed": True,
                    "high_level_reactivation_acknowledged": True,
                    "late_feedback_fault": handoff_feedback_fault,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        if commit_failure is not None:
            if cancelled:
                raise asyncio.CancelledError
            return commit_failure
        assert release_result is not None
        if not release_result.ok:
            failure = OperationResult.failure(
                release_result.code,
                release_result.message,
                {
                    **dict(release_result.data),
                    "ownership_epoch": epoch,
                    "owner_epoch_retained": True,
                    "publisher_closed": True,
                    "high_level_reactivation_acknowledged": True,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        result = OperationResult.success(
            "LowCmd publisher stopped on the ground and local ownership was released",
            {
                **dict(handoff.data),
                "high_level_service_detected": True,
                "high_level_reactivation_acknowledged": True,
            },
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _complete_closed_endpoint_handoff(
        self,
        permit: Go2OwnershipPermit,
        *,
        epoch: int,
        ownership_epoch: int,
        mapping_version: str,
        mapping_hash: str,
        real_deadline: float,
    ) -> OperationResult:
        """Recover a retained epoch after LowCmd ``Close`` already succeeded.

        This path deliberately does not depend on LowState. Once the publisher
        is proven locally closed, no safe-hold writer can or should be
        resurrected. Independent landed/disarmed/rotor-stop evidence and the
        one-shot operator permit still gate the exact MotionSwitcher handback.
        """

        live_ground = self._verify_live_ground_transfer("release", permit)
        if not live_ground.ok:
            return live_ground
        with self._event_loop_guard("closed-endpoint-handoff-initial", real_deadline):
            commit_now = self._clock.monotonic()
            writer = self._writer_thread
            pending_transaction_id = self._handoff_active_transaction_id
            pending_ingress_cutoff = self._handoff_ingress_cutoff
            if (
                self._owner_epoch != epoch
                or ownership_epoch != epoch
                or self._publisher is not None
                or (writer is not None and writer.is_alive())
            ):
                return OperationResult.failure(
                    "GO2_HANDOFF_RECOVERY_STATE_CHANGED",
                    "LowCmd endpoint state changed before high-level recovery",
                    {"ownership_epoch": self._owner_epoch},
                )
            if time.monotonic() >= real_deadline or not permit.authorizes(
                commit_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                return OperationResult.failure(
                    "GO2_RELEASE_PERMIT_EXPIRED",
                    "Ground release permit or deadline expired before high-level recovery",
                    {"ownership_epoch": epoch, "publisher_closed": True},
                )
            # The endpoint no longer exists, but an earlier callback may have
            # entered before Close and still be parsing. Preserve that
            # transaction until it drains; only later callbacks are purely
            # diagnostic.
            self._status = replace(
                self._status,
                timestamp=commit_now,
                ownership_state=LowCmdOwnershipState.RELEASING,
                owner_epoch=epoch,
                healthy=False,
                publisher_active=False,
                writer_alive=False,
                watchdog_healthy=False,
                safe_hold_active=False,
                safe_hold_settled=False,
            )

        cancelled = False
        handoff_task = asyncio.ensure_future(
            self._restore_high_level_mode(max(0.0, real_deadline - time.monotonic()))
        )
        handoff, handoff_cancelled = await await_nonabandonable(handoff_task)
        cancelled = cancelled or handoff_cancelled
        if pending_transaction_id != 0:
            drain_task = asyncio.ensure_future(
                run_blocking(
                    self._wait_for_handoff_callbacks,
                    pending_transaction_id,
                    pending_ingress_cutoff,
                    max(0.0, real_deadline - time.monotonic()),
                )
            )
            callbacks_drained, drain_cancelled = await await_nonabandonable(drain_task)
            cancelled = cancelled or drain_cancelled
        else:
            callbacks_drained = True
        if not callbacks_drained:
            with self._event_loop_guard("closed-endpoint-callback-fault", real_deadline):
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=not handoff.ok,
                    fault_reason=(
                        "Closed-endpoint recovery still has a pre-Close LowState callback in flight"
                    ),
                )
            failure = OperationResult.failure(
                "GO2_HANDOFF_CALLBACK_DRAIN_TIMEOUT",
                "High-level recovery cannot release the epoch while a pre-Close callback remains in flight",
                {
                    **dict(handoff.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_closed": True,
                    "high_level_reactivation_acknowledged": handoff.ok,
                    "handoff_failure_code": None if handoff.ok else handoff.code,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        with self._event_loop_guard("closed-endpoint-reset-handoff", real_deadline):
            self._reset_handoff_tracking_locked()
        if not handoff.ok:
            with self._event_loop_guard("closed-endpoint-handback-fault", real_deadline):
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    fault_reason=(
                        "LowCmd is already closed, but exact high-level recovery "
                        "was not acknowledged: " + handoff.message
                    ),
                )
            failure = OperationResult.failure(
                handoff.code,
                "LowCmd is already closed, but exact high-level recovery failed: "
                + handoff.message,
                {
                    **dict(handoff.data),
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_closed": True,
                    "high_level_reactivation_acknowledged": False,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure

        # The feedback callback uses the same bridge guard. Hold it across the
        # exact arbiter release and local epoch clear so there is no callback
        # window in which a new fault/generation can be silently discarded.
        release_result: OperationResult
        commit_failure: Optional[OperationResult] = None
        with self._event_loop_guard("closed-endpoint-commit", real_deadline):
            commit_now = self._clock.monotonic()
            writer = self._writer_thread
            if (
                self._owner_epoch != epoch
                or self._publisher is not None
                or (writer is not None and writer.is_alive())
            ):
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    healthy=False,
                    high_level_released=False,
                    fault_reason=(
                        "High-level mode was restored, but LowCmd endpoint state "
                        "changed before the recovery commit"
                    ),
                )
                commit_failure = OperationResult.failure(
                    "GO2_HANDOFF_RECOVERY_STATE_CHANGED",
                    "LowCmd endpoint state changed while high-level recovery was running",
                    {"ownership_epoch": self._owner_epoch},
                )
            elif time.monotonic() >= real_deadline or not permit.authorizes(
                commit_now,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
            ):
                self._status = replace(
                    self._status,
                    timestamp=commit_now,
                    ownership_state=LowCmdOwnershipState.FAULT,
                    healthy=False,
                    high_level_released=False,
                    fault_reason=(
                        "Exact high-level mode was restored, but the release "
                        "permit expired before the local epoch commit"
                    ),
                )
                commit_failure = OperationResult.failure(
                    "GO2_RELEASE_PERMIT_EXPIRED",
                    "High-level mode is restored, but the expired transaction did not unlock the retained epoch",
                    {
                        "ownership_epoch": epoch,
                        "owner_lock_retained": True,
                        "publisher_closed": True,
                        "high_level_reactivation_acknowledged": True,
                    },
                )
            if commit_failure is None:
                release_result = self._release_arbiter_epoch(epoch)
                if release_result.ok:
                    self._clear_released_owner_locked()
                else:
                    self._status = replace(
                        self._status,
                        timestamp=commit_now,
                        ownership_state=LowCmdOwnershipState.FAULT,
                        healthy=False,
                        publisher_active=False,
                        writer_alive=False,
                        high_level_released=False,
                        network_exclusivity_verified=(
                            release_result.data.get("owner_lock_retained") is True
                            and release_result.data.get("local_single_instance_poisoned")
                            is not True
                            and release_result.data.get("ownership_exclusivity_lost") is not True
                        ),
                        fault_reason=release_result.message,
                    )
            else:
                release_result = commit_failure
        if not release_result.ok:
            if cancelled:
                raise asyncio.CancelledError
            return release_result
        result = OperationResult.success(
            "Closed LowCmd endpoint recovered exact high-level control and released the local epoch",
            {
                **dict(handoff.data),
                "publisher_closed": True,
                "high_level_reactivation_acknowledged": True,
            },
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _clear_released_owner_locked(self) -> None:
        """Clear one released epoch atomically while ``self._guard`` is held."""

        now = self._clock.monotonic()
        feedback_fault = self._feedback_limit_failure_locked()
        feedback_fresh = self._low_state_is_fresh_locked(now)
        observe_healthy = self._connected and feedback_fresh and feedback_fault is None
        if observe_healthy:
            observe_fault: Optional[str] = None
        elif feedback_fault is not None:
            observe_fault = feedback_fault
        elif not self._connected:
            observe_fault = "LowState transport is disconnected after handback"
        else:
            observe_fault = "LowState is stale after handback"
        self._owner_epoch = 0
        self._publisher = None
        self._writer_thread = None
        self._target = None
        self._last_accepted_sequence = None
        self._last_commanded_q = None
        self._command_reference_generation = 0
        self._command_reference_q = None
        self._command_reference_write_s = None
        self._command_reference_ingress_cutoff = 0
        self._safe_hold_q = None
        self._expected_deadline_s = None
        self._last_successful_write_s = None
        with self._writer_health_guard:
            self._writer_started_s = None
            self._writer_write_started_s = None
            self._writer_heartbeat_s = None
        self._safe_hold_request_generation = 0
        self._safe_hold_write_generation = 0
        self._last_safe_hold_write_s = None
        self._safe_hold_command_reached_generation = 0
        self._safe_hold_command_reached_write_s = None
        self._safe_hold_feedback_sequence_required = 0
        self._safe_hold_feedback_ingress_token_required = 0
        self._high_level_restore_form = None
        self._high_level_restore_mode = None
        self._release_rpc_attempted = False
        self._reset_handoff_tracking_locked()
        self._status = replace(
            self._status,
            timestamp=now,
            ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
            owner_epoch=0,
            healthy=observe_healthy,
            target_sequence=None,
            target_age_s=None,
            target_deadline=None,
            mailbox_staged_target_sequence=None,
            writer_enqueued_target_sequence=None,
            actuator_applied_target_sequence=None,
            writer_enqueue_generation=0,
            writer_enqueued_q_rad=(),
            publisher_active=False,
            writer_alive=False,
            last_write_timestamp=None,
            watchdog_healthy=False,
            safe_hold_active=False,
            safe_hold_settled=False,
            safe_hold_request_generation=0,
            safe_hold_write_generation=0,
            last_safe_hold_write_timestamp=None,
            tracking_error_timestamp=0.0,
            tracking_reference_write_timestamp=0.0,
            tracking_reference_write_generation=0,
            tracking_reference_q_rad=(),
            position_error_rad=(),
            high_level_released=False,
            high_level_restore_form=None,
            high_level_restore_mode=None,
            network_exclusivity_verified=False,
            continuous_owner_monitoring_active=False,
            independent_watchdog_active=False,
            writer_enqueue_ack_available=False,
            actuator_application_ack_available=False,
            fault_reason=observe_fault,
        )

    async def disconnect(self) -> OperationResult:
        return await self._run_lifecycle_operation("disconnect", self._disconnect_unlocked)

    async def _disconnect_unlocked(self) -> OperationResult:
        """Close observe-only transport; active ownership must be released first."""

        with self._event_loop_guard("disconnect-owner-check"):
            if self._status.ownership_pending or (
                self._writer_thread is not None and self._writer_thread.is_alive()
            ):
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_OWNER_ACTIVE",
                    "Revoke and ground-release the LowCmd owner before disconnecting",
                )
            with self._low_state_ingress_condition:
                self._subscription_generation += 1
                self._low_state_ingress_condition.notify_all()
            subscriber = self._subscriber
            publisher = self._publisher
            self._subscriber = None
            self._publisher = None
            self._motion_switcher = None
            self._crc = None
            self._sdk = None
            self._connected = False
            with self._writer_health_guard:
                self._writer_started_s = None
                self._writer_write_started_s = None
                self._writer_heartbeat_s = None
            self._first_state.clear()
            self._valid_motors = ()
            self._envelope_motors = ()
            self._envelope_ingress_token = 0
            self._valid_low_state_sequence = 0
            self._valid_low_state_ingress_token = 0
            self._last_foot_force_source_tick = None
            self._safe_hold_feedback_sequence_required = 0
            self._safe_hold_feedback_ingress_token_required = 0
            self._high_level_restore_form = None
            self._high_level_restore_mode = None
            self._release_rpc_attempted = False
            self._reset_handoff_tracking_locked()
        self._close_transport(subscriber=subscriber, publisher=publisher)
        self._set_disconnected("low-level transport disconnected")
        return OperationResult.success("Go2 low-level observe transport disconnected")

    async def _release_high_level_mode(
        self, timeout_s: float, permit: Go2OwnershipPermit
    ) -> OperationResult:
        client = self._motion_switcher
        if client is None:
            return self._motion_release_failure(
                "GO2_MOTION_SWITCHER_MISSING",
                "MotionSwitcher was not initialized",
                attempted=False,
                acknowledged=False,
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        release_attempted = False
        release_acknowledged = False
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                return self._motion_release_failure(
                    "GO2_MOTION_RELEASE_TIMEOUT",
                    "MotionSwitcher release transaction exceeded its deadline",
                    attempted=release_attempted,
                    acknowledged=release_acknowledged,
                )
            checked = await self._check_mode_once(remaining)
            if not checked.ok:
                return self._motion_release_failure(
                    checked.code,
                    checked.message,
                    attempted=release_attempted,
                    acknowledged=release_acknowledged,
                    original_data=checked.data,
                )
            mode_name = checked.data.get("mode_name")
            mode_form = checked.data.get("mode_form")
            if not isinstance(mode_name, str) or not isinstance(mode_form, str):
                return self._motion_release_failure(
                    "GO2_MOTION_SWITCHER_INVALID_REPLY",
                    "CheckMode did not return string form/name values",
                    attempted=release_attempted,
                    acknowledged=release_acknowledged,
                )
            if not mode_name:
                if not release_attempted:
                    return self._motion_release_failure(
                        "GO2_HIGH_LEVEL_RESTORE_MODE_UNKNOWN",
                        "Cannot acquire LowCmd when CheckMode is already empty because no high-level restore mode can be captured",
                        attempted=False,
                        acknowledged=False,
                    )
                return OperationResult.success(
                    "MotionSwitcher confirmed no high-level service owns motion control",
                    {
                        "release_rpc_attempted": release_attempted,
                        "release_rpc_acknowledged": release_acknowledged,
                        "check_mode_acknowledged": True,
                        "high_level_restore_form": self._high_level_restore_form,
                        "high_level_restore_mode": self._high_level_restore_mode,
                    },
                )
            if not release_attempted:
                commissioned_form = self._required_str(
                    self._config.restore_mode_form, "restore_mode_form"
                )
                commissioned_name = self._required_str(
                    self._config.restore_mode_name, "restore_mode_name"
                )
                if (mode_form, mode_name) != (
                    commissioned_form,
                    commissioned_name,
                ):
                    return self._motion_release_failure(
                        "GO2_HIGH_LEVEL_RESTORE_MODE_MISMATCH",
                        "CheckMode did not match the commissioned pre-acquisition form/name",
                        attempted=False,
                        acknowledged=False,
                        original_data={
                            "expected_mode_form": commissioned_form,
                            "expected_mode_name": commissioned_name,
                            "observed_mode_form": mode_form,
                            "observed_mode_name": mode_name,
                        },
                    )
                with self._guard:
                    existing_restore_form = self._high_level_restore_form
                    existing_restore_mode = self._high_level_restore_mode
                    if existing_restore_form is not None or existing_restore_mode is not None:
                        if (existing_restore_form, existing_restore_mode) != (
                            mode_form,
                            mode_name,
                        ):
                            return self._motion_release_failure(
                                "GO2_HIGH_LEVEL_RESTORE_MODE_CHANGED",
                                "The high-level service changed while LowCmd acquisition was in progress",
                                attempted=False,
                                acknowledged=False,
                            )
                    self._high_level_restore_form = mode_form
                    self._high_level_restore_mode = mode_name
                    self._status = replace(
                        self._status,
                        timestamp=self._clock.monotonic(),
                        high_level_restore_form=mode_form,
                        high_level_restore_mode=mode_name,
                    )
                release_mode = getattr(client, "ReleaseMode", None)
                if not callable(release_mode):
                    return self._motion_release_failure(
                        "GO2_MOTION_SWITCHER_API_MISSING",
                        "MotionSwitcher does not provide ReleaseMode",
                        attempted=False,
                        acknowledged=False,
                    )
                remaining = deadline - loop.time()
                if remaining <= 0.0:
                    return self._motion_release_failure(
                        "GO2_MOTION_RELEASE_TIMEOUT",
                        "No acquisition time remained before ReleaseMode",
                        attempted=False,
                        acknowledged=False,
                    )
                live_ground = self._verify_live_ground_transfer("acquire", permit)
                if not live_ground.ok:
                    return self._motion_release_failure(
                        live_ground.code,
                        "Live ground evidence failed immediately before ReleaseMode: "
                        + live_ground.message,
                        attempted=False,
                        acknowledged=False,
                        original_data=live_ground.data,
                    )
                # CheckMode and the independent verifier can both consume the
                # short one-shot authorization window.  Re-read both clocks
                # and the mapping-bound permit immediately before crossing the
                # ReleaseMode RPC boundary; an earlier successful check is not
                # transferable authority.
                permit_now = self._clock.monotonic()
                remaining = deadline - loop.time()
                if remaining <= 0.0 or not permit.authorizes(
                    permit_now,
                    mapping_version=self._required_str(
                        self._config.mapping_version, "mapping_version"
                    ),
                    mapping_hash=self._required_str(self._config.mapping_hash, "mapping_hash"),
                ):
                    return self._motion_release_failure(
                        "GO2_OWNERSHIP_PERMIT_EXPIRED",
                        "Ground permit or acquire deadline expired immediately before ReleaseMode",
                        attempted=False,
                        acknowledged=False,
                    )
                local_precondition_failure: Optional[str] = None
                current_q: Optional[Tuple[float, ...]] = None
                latest_safe_hold: Optional[Tuple[float, ...]] = None
                with self._guard:
                    local_now = self._clock.monotonic()
                    if (
                        self._owner_epoch <= 0
                        or self._publisher is None
                        or self._status.ownership_state is not LowCmdOwnershipState.ACQUIRING
                    ):
                        local_precondition_failure = (
                            "LowCmd acquisition epoch, publisher, or state changed "
                            "before ReleaseMode"
                        )
                    elif not self._connected:
                        local_precondition_failure = (
                            "LowState transport disconnected before ReleaseMode"
                        )
                    elif self._status.fault_reason is not None:
                        local_precondition_failure = (
                            "LowState became unsafe before ReleaseMode: "
                            + self._status.fault_reason
                        )
                    elif not self._low_state_is_fresh_locked(local_now):
                        local_precondition_failure = "LowState became stale before ReleaseMode"
                    else:
                        feedback_failure = self._feedback_limit_failure_locked()
                        current_q = self._current_joint_positions_locked()
                        if feedback_failure is not None:
                            local_precondition_failure = feedback_failure
                        elif current_q is None:
                            local_precondition_failure = (
                                "All 12 mapped joints need valid feedback at ReleaseMode"
                            )
                        else:
                            latest_safe_hold = self._select_safe_hold_pose(current_q)
                            if latest_safe_hold is None:
                                local_precondition_failure = (
                                    "No limit-safe hold pose remained at ReleaseMode"
                                )
                    if local_precondition_failure is None:
                        assert current_q is not None
                        assert latest_safe_hold is not None
                        # Start the first LowCmd frame from the latest measured
                        # pose. capture_current freezes that same pose;
                        # configured_pose keeps its configured target but
                        # reaches it through the normal bounded slew.
                        self._last_commanded_q = current_q
                        self._command_reference_generation = 0
                        self._command_reference_q = None
                        self._command_reference_write_s = None
                        self._command_reference_ingress_cutoff = 0
                        self._safe_hold_q = latest_safe_hold
                        # This is the local linearization point. A callback
                        # fault after it is classified as post-boundary and the
                        # epoch is retained while the safe-hold writer starts.
                        self._release_rpc_attempted = True
                if local_precondition_failure is not None:
                    return self._motion_release_failure(
                        "GO2_ACQUIRE_PRE_RELEASE_LOCAL_STATE_FAILED",
                        local_precondition_failure,
                        attempted=False,
                        acknowledged=False,
                    )
                release_attempted = True
                self._set_motion_timeout(remaining)
                try:
                    code = await run_blocking(release_mode)
                except Exception as exc:
                    return self._motion_release_failure(
                        "GO2_MOTION_RELEASE_FAILED",
                        str(exc),
                        attempted=True,
                        acknowledged=False,
                    )
                if not self._rpc_succeeded(code):
                    return self._motion_release_failure(
                        "GO2_MOTION_RELEASE_REJECTED",
                        f"ReleaseMode returned {code!r}",
                        attempted=True,
                        acknowledged=False,
                    )
                release_acknowledged = True
            await asyncio.sleep(0.02)

    async def _restore_high_level_mode(self, timeout_s: float) -> OperationResult:
        """Select and confirm the exact service captured before ReleaseMode."""

        with self._guard:
            expected_form = self._high_level_restore_form
            expected_mode = self._high_level_restore_mode
        if (
            not isinstance(expected_form, str)
            or not expected_form
            or not isinstance(expected_mode, str)
            or not expected_mode
        ):
            return OperationResult.failure(
                "GO2_HIGH_LEVEL_RESTORE_MODE_UNKNOWN",
                "No commissioned pre-acquisition MotionSwitcher form/name was retained",
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            return OperationResult.failure(
                "GO2_HIGH_LEVEL_HANDOFF_TIMEOUT",
                "No release transaction time remains to restore high-level control",
                {
                    "expected_mode_form": expected_form,
                    "expected_mode_name": expected_mode,
                },
            )

        client = self._motion_switcher
        if client is None:
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_MISSING",
                "MotionSwitcher was not initialized for high-level handback",
                {
                    "expected_mode_form": expected_form,
                    "expected_mode_name": expected_mode,
                },
            )
        select_mode = getattr(client, "SelectMode", None)
        if not callable(select_mode):
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_API_MISSING",
                "MotionSwitcher does not provide SelectMode for high-level handback",
                {
                    "expected_mode_form": expected_form,
                    "expected_mode_name": expected_mode,
                },
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        selected = False
        last_check_error = ""
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                return OperationResult.failure(
                    "GO2_HIGH_LEVEL_HANDOFF_UNCONFIRMED",
                    "Captured high-level service was not confirmed before release timeout: "
                    + (last_check_error or "deadline expired before CheckMode"),
                    {
                        "expected_mode_form": expected_form,
                        "expected_mode_name": expected_mode,
                        "select_rpc_attempted": selected,
                        "select_rpc_acknowledged": selected,
                    },
                )
            checked = await self._check_mode_once(remaining)
            if checked.ok:
                current_mode = checked.data.get("mode_name")
                current_form = checked.data.get("mode_form")
                if not isinstance(current_mode, str) or not isinstance(current_form, str):
                    return OperationResult.failure(
                        "GO2_MOTION_SWITCHER_INVALID_REPLY",
                        "CheckMode did not return string form/name values during handback",
                        {
                            "expected_mode_form": expected_form,
                            "expected_mode_name": expected_mode,
                        },
                    )
                if (current_form, current_mode) == (expected_form, expected_mode):
                    return OperationResult.success(
                        "MotionSwitcher restored and confirmed the captured high-level service",
                        {
                            "expected_mode_form": expected_form,
                            "expected_mode_name": expected_mode,
                            "confirmed_mode_form": current_form,
                            "confirmed_mode_name": current_mode,
                            "select_rpc_attempted": selected,
                            "select_rpc_acknowledged": selected,
                        },
                        code="GO2_HIGH_LEVEL_HANDOFF_CONFIRMED",
                    )
                if current_mode:
                    return OperationResult.failure(
                        "GO2_HIGH_LEVEL_HANDOFF_CONFLICT",
                        "An unexpected high-level form/name became active during handback",
                        {
                            "expected_mode_form": expected_form,
                            "expected_mode_name": expected_mode,
                            "observed_mode_form": current_form,
                            "observed_mode_name": current_mode,
                            "select_rpc_attempted": selected,
                        },
                    )
                last_check_error = "CheckMode still reports no active high-level service"
                if not selected:
                    remaining = deadline - loop.time()
                    if remaining <= 0.0:
                        continue
                    self._set_motion_timeout(remaining)
                    try:
                        reply = await run_blocking(select_mode, expected_mode)
                    except Exception as exc:
                        return OperationResult.failure(
                            "GO2_HIGH_LEVEL_SELECT_FAILED",
                            f"SelectMode raised {type(exc).__name__}: {exc}",
                            {
                                "expected_mode_form": expected_form,
                                "expected_mode_name": expected_mode,
                            },
                        )
                    if not self._rpc_succeeded(reply):
                        return OperationResult.failure(
                            "GO2_HIGH_LEVEL_SELECT_REJECTED",
                            f"SelectMode({expected_mode!r}) returned {reply!r}",
                            {
                                "expected_mode_form": expected_form,
                                "expected_mode_name": expected_mode,
                            },
                        )
                    selected = True
            else:
                last_check_error = f"{checked.code}: {checked.message}"

            remaining = deadline - loop.time()
            if remaining <= 0.0:
                return OperationResult.failure(
                    "GO2_HIGH_LEVEL_HANDOFF_UNCONFIRMED",
                    "Captured high-level service was not confirmed before release timeout: "
                    + last_check_error,
                    {
                        "expected_mode_form": expected_form,
                        "expected_mode_name": expected_mode,
                        "select_rpc_attempted": selected,
                        "select_rpc_acknowledged": selected,
                    },
                )
            await asyncio.sleep(min(0.02, remaining))

    @staticmethod
    def _motion_release_failure(
        code: str,
        message: str,
        *,
        attempted: bool,
        acknowledged: bool,
        original_data: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult:
        data = dict(original_data) if original_data is not None else {}
        data.update(
            {
                "release_rpc_attempted": attempted,
                "release_rpc_acknowledged": acknowledged,
                "check_mode_acknowledged": False,
            }
        )
        return OperationResult.failure(code, message, data)

    async def _check_mode_once(self, timeout_s: Optional[float] = None) -> OperationResult:
        client = self._motion_switcher
        check_mode = getattr(client, "CheckMode", None)
        if not callable(check_mode):
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_API_MISSING", "MotionSwitcher does not provide CheckMode"
            )
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                return OperationResult.failure(
                    "GO2_MOTION_CHECK_TIMEOUT",
                    "No positive MotionSwitcher CheckMode timeout remains",
                )
            self._set_motion_timeout(timeout_s)
        try:
            reply = await run_blocking(check_mode)
        except Exception as exc:
            return OperationResult.failure("GO2_MOTION_CHECK_FAILED", str(exc))
        if not isinstance(reply, (tuple, list)) or len(reply) < 2:
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_INVALID_REPLY",
                f"Unexpected CheckMode reply: {reply!r}",
            )
        code = reply[0]
        if not self._rpc_succeeded(code):
            return OperationResult.failure(
                "GO2_MOTION_CHECK_REJECTED", f"CheckMode returned {code!r}"
            )
        # Current SDK2 Python returns ``(code, {"form": ..., "name": ...})``.
        # Retain support for the older/alternate three-item representation so
        # an SDK package change fails only on genuinely ambiguous data.
        if len(reply) == 2 and isinstance(reply[1], dict):
            form = reply[1].get("form", "")
            name = reply[1].get("name")
        elif len(reply) >= 3:
            form, name = reply[1], reply[2]
        else:
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_INVALID_REPLY",
                f"CheckMode payload is not a mapping: {reply[1]!r}",
            )
        if not isinstance(form, str) or not isinstance(name, str):
            return OperationResult.failure(
                "GO2_MOTION_SWITCHER_INVALID_REPLY",
                "CheckMode service form/name must both be strings",
            )
        return OperationResult.success(
            "MotionSwitcher mode checked", {"mode_form": form, "mode_name": name}
        )

    def _set_motion_timeout(self, timeout_s: float) -> None:
        client = self._motion_switcher
        setter = getattr(client, "SetTimeout", None)
        if callable(setter):
            try:
                setter(max(0.001, timeout_s))
            except Exception:
                # The subsequent RPC remains authoritative and will fail
                # closed if the client cannot honor its configured deadline.
                pass

    @staticmethod
    def _rpc_succeeded(value: Any) -> bool:
        # MotionSwitcher ReleaseMode returns ``(code, data)`` in the current
        # Python SDK, whereas some test/older adapters return the code alone.
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        return not isinstance(value, bool) and isinstance(value, int) and value == 0

    def _start_writer_locked(self) -> None:
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_stop.clear()
        self._first_write.clear()
        self._expected_deadline_s = None
        with self._writer_health_guard:
            self._writer_started_s = self._clock.monotonic()
            self._writer_write_started_s = None
            self._writer_heartbeat_s = None
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="aerogo2-go2-lowcmd-owner",
            # A daemon thread would disappear during ordinary interpreter
            # shutdown even when the ground-release gate rejected exit.
            # Keep it non-daemon so normal shutdown cannot silently turn an
            # airborne safe-hold stream into no command stream.
            daemon=False,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        period = self._required_float(self._config.send_period_s, "send_period_s")
        next_deadline = self._clock.monotonic()
        try:
            while not self._writer_stop.is_set():
                now = self._clock.monotonic()
                stream_was_established = self._first_write.is_set()
                self._writer_cycle(now, expected_deadline=next_deadline)
                after = self._clock.monotonic()
                # Never issue catch-up bursts after an overrun; one command per
                # measured period is safer than compressed writes with a large
                # q step.  Deadlines intentionally restart after an overrun.
                if after - now >= period and stream_was_established:
                    self._set_fault(f"LowCmd writer execution overran its {period:.6f}s period")
                    next_deadline = after + period
                elif after - now >= period:
                    # The first Write deliberately waits for a matched DDS
                    # reader while still on the verified ground.  Its bounded
                    # discovery delay is not steady-state scheduling jitter.
                    next_deadline = after + period
                else:
                    next_deadline = now + period
                self._expected_deadline_s = next_deadline
                self._writer_stop.wait(max(0.0, next_deadline - after))
        except BaseException as exc:
            # Do not release the owner lock: a dead writer is explicitly a
            # fault, never evidence that another publisher may safely start.
            self._set_fault(f"LowCmd writer terminated unexpectedly: {exc}")
        finally:
            with self._writer_health_guard:
                self._writer_write_started_s = None
            with self._guard:
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    writer_alive=False,
                )

    def _writer_cycle(self, now: float, *, expected_deadline: Optional[float]) -> None:
        """Run one complete owner cycle; kept deterministic for unit tests."""

        maximum_jitter = self._required_float(self._config.maximum_jitter_s, "maximum_jitter_s")
        with self._guard:
            if expected_deadline is not None:
                # The caller sampled before waiting for _guard. Re-sample at
                # the actual cycle boundary so lock contention cannot hide
                # stale LowState/targets or scheduling jitter.
                now = self._clock.monotonic()
            arbiter_status = self._arbiter.status()
            if (
                not arbiter_status.network_exclusivity_verified
                or arbiter_status.low_level_epoch != self._owner_epoch
                or not arbiter_status.local_single_instance_held
            ):
                self._enter_safe_hold_locked(
                    now,
                    "LowCmd ownership or cached runtime network proof was invalidated",
                    fault=True,
                )
            if expected_deadline is not None and abs(now - expected_deadline) > maximum_jitter:
                self._enter_safe_hold_locked(
                    now,
                    (
                        "LowCmd scheduling jitter exceeded limit: "
                        f"{abs(now - expected_deadline):.6f}s > {maximum_jitter:.6f}s"
                    ),
                    fault=True,
                )
            if not self._low_state_is_fresh_locked(now):
                self._enter_safe_hold_locked(now, "LowState freshness watchdog expired", fault=True)
            target = self._target
            if target is not None and not target.is_fresh(now):
                self._enter_safe_hold_locked(now, "MPC target TTL expired", fault=False)
                target = None
            desired = (
                target.joint_positions_rad
                if target is not None
                and self._status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE
                else self._safe_hold_q
            )
            previous = self._last_commanded_q
            if desired is None or previous is None:
                self._enter_safe_hold_locked(
                    now, "No verified pose is available for the LowCmd stream", fault=True
                )
                return
            period = self._required_float(self._config.send_period_s, "send_period_s")
            elapsed = (
                period
                if self._last_successful_write_s is None
                else now - self._last_successful_write_s
            )
            if not math.isfinite(elapsed) or elapsed < 0.0:
                self._enter_safe_hold_locked(
                    now, "LowCmd writer observed a non-monotonic send interval", fault=True
                )
                return
            # Never grant more than one nominal cycle of position change.  If
            # a cycle is late this is conservative; if it is early, the actual
            # elapsed time enforces dq_max and prevents a compressed burst.
            elapsed_for_slew = min(period, elapsed)
            commanded = self._slew_limit(previous, tuple(desired), elapsed_for_slew)
            try:
                (
                    message,
                    enqueued_q,
                    derated_joints,
                    feedback_degraded_joints,
                ) = self._make_low_command(
                    commanded,
                    previous_q=previous,
                    elapsed_s=elapsed_for_slew,
                    now_s=now,
                )
                if feedback_degraded_joints:
                    self._enter_safe_hold_locked(
                        now,
                        "LowState q/dq is unavailable, stale, or numerically unsafe; "
                        "using the commissioned feedback-loss gains protected by "
                        "the verified firmware torque clamp for joints "
                        + ",".join(str(item) for item in feedback_degraded_joints),
                        fault=True,
                    )
                    if self._safe_hold_q is None:
                        raise RuntimeError("feedback-loss fault hold has no verified safe pose")
                    target = None
                    commanded = self._slew_limit(
                        previous,
                        self._safe_hold_q,
                        elapsed_for_slew,
                    )
                    (
                        message,
                        enqueued_q,
                        _,
                        feedback_degraded_joints,
                    ) = self._make_low_command(
                        commanded,
                        previous_q=previous,
                        elapsed_s=elapsed_for_slew,
                        now_s=now,
                    )
                elif derated_joints:
                    self._enter_safe_hold_locked(
                        now,
                        "Total PD torque envelope required a gain-derated fault hold for joints "
                        + ",".join(str(item) for item in derated_joints),
                        fault=True,
                    )
                    if self._safe_hold_q is None:
                        raise RuntimeError("gain-derated fault hold has no verified safe pose")
                    target = None
                    commanded = self._slew_limit(
                        previous,
                        self._safe_hold_q,
                        elapsed_for_slew,
                    )
                    message, enqueued_q, _, feedback_degraded_joints = self._make_low_command(
                        commanded,
                        previous_q=previous,
                        elapsed_s=elapsed_for_slew,
                        now_s=now,
                    )
            except Exception as exc:
                self._enter_safe_hold_locked(
                    now, f"LowCmd construction or CRC failed: {exc}", fault=True
                )
                return

            # Command construction is non-trivial and may consume the rest of
            # a short target lease.  Re-sample immediately before touching the
            # DDS writer.  If the MPC lease expired, rebuild this cycle as a
            # conservative hold; never enqueue a target known to be stale.
            pre_write_time = self._clock.monotonic()
            if target is not None and not target.is_fresh(pre_write_time):
                self._enter_safe_hold_locked(
                    pre_write_time,
                    "MPC target TTL expired during LowCmd construction",
                    fault=False,
                )
                target = None
                if self._safe_hold_q is None:
                    self._enter_safe_hold_locked(
                        pre_write_time,
                        "Expired MPC target has no verified conservative hold pose",
                        fault=True,
                    )
                    return
                commanded = self._slew_limit(
                    previous,
                    self._safe_hold_q,
                    elapsed_for_slew,
                )
                try:
                    (
                        message,
                        enqueued_q,
                        hold_derated_joints,
                        hold_feedback_degraded_joints,
                    ) = self._make_low_command(
                        commanded,
                        previous_q=previous,
                        elapsed_s=elapsed_for_slew,
                        now_s=pre_write_time,
                    )
                except Exception as exc:
                    self._enter_safe_hold_locked(
                        pre_write_time,
                        f"TTL-expiry hold construction or CRC failed: {exc}",
                        fault=True,
                    )
                    return
                if hold_derated_joints or hold_feedback_degraded_joints:
                    affected = sorted(
                        set(hold_derated_joints) | set(hold_feedback_degraded_joints)
                    )
                    self._enter_safe_hold_locked(
                        pre_write_time,
                        "TTL-expiry hold required degraded protection for joints "
                        + ",".join(str(item) for item in affected),
                        fault=True,
                    )
            publisher = self._publisher
            if publisher is None:
                self._enter_safe_hold_locked(
                    now, "LowCmd publisher disappeared while ownership was active", fault=True
                )
                return
            safe_hold_generation = (
                self._safe_hold_request_generation
                if target is None
                or self._status.ownership_state
                in {
                    LowCmdOwnershipState.HOLDING,
                    LowCmdOwnershipState.SAFE_HOLD,
                    LowCmdOwnershipState.FAULT,
                }
                else 0
            )
            # Keep the owner guard held through Write.  Therefore revoke/fault
            # cannot commit a newer generation while an old MPC message is in
            # the construction-to-enqueue window.
            with self._writer_health_guard:
                self._writer_write_started_s = self._clock.monotonic()
            try:
                if self._first_write.is_set():
                    write_result = publisher.Write(message)
                else:
                    write_result = publisher.Write(
                        message,
                        self._required_float(self._config.acquire_timeout_s, "acquire_timeout_s"),
                    )
            except Exception as exc:
                with self._writer_health_guard:
                    self._writer_write_started_s = None
                self._enter_safe_hold_locked(now, f"LowCmd DDS enqueue raised: {exc}", fault=True)
                return
            with self._writer_health_guard:
                self._writer_write_started_s = None
            if write_result is not True:
                self._enter_safe_hold_locked(
                    now,
                    f"LowCmd DDS enqueue was not acknowledged: {write_result!r}",
                    fault=True,
                )
                return
            # Establish a causal ingress fence immediately after the accepted
            # Write while the owner guard is still held.  A callback that
            # entered before this point can finish parsing later, but must not
            # acknowledge the command just written.
            with self._low_state_ingress_condition:
                write_ingress_cutoff = self._low_state_ingress_sequence
            write_time = self._clock.monotonic()
            target_expired_during_write = target is not None and not target.is_fresh(write_time)
            if target_expired_during_write:
                # The local DDS call has already returned, so this host cannot
                # retract the stale frame.  Fail closed, do not attribute a
                # target ACK, and keep the sole publisher alive so the next
                # cycle replaces it with safe-hold.  A robot-side command
                # lease/watchdog remains a mandatory hardware release gate.
                self._enter_safe_hold_locked(
                    write_time,
                    "MPC target TTL expired while LowCmd DDS Write was in progress",
                    fault=True,
                )
            with self._writer_health_guard:
                self._writer_heartbeat_s = write_time
            self._last_successful_write_s = write_time
            self._last_commanded_q = enqueued_q
            if (
                target is not None
                and not target_expired_during_write
                and self._status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE
            ):
                # One MPC sequence denotes one physically bounded joint
                # reference.  Freeze that sequence at the exact q accepted by
                # its first DDS Write; otherwise the fixed-rate writer would
                # continue slewing toward the raw mailbox target after the
                # executor had already committed an older q to admittance
                # anti-windup.  Later cycles still repeat this frozen q at the
                # configured rate, and only a strictly newer sequence may
                # advance it again.
                self._target = replace(target, joint_positions_rad=enqueued_q)
            self._command_reference_generation += 1
            self._command_reference_q = enqueued_q
            self._command_reference_write_s = write_time
            self._command_reference_ingress_cutoff = write_ingress_cutoff
            status_changes: Dict[str, Any] = {
                "timestamp": write_time,
                "writer_alive": True,
                "last_write_timestamp": write_time,
                "writer_enqueue_generation": self._command_reference_generation,
                "writer_enqueued_q_rad": enqueued_q,
                # The q/generation always describe this latest Write.  Clear
                # any earlier target identity for hold/fault writes so the
                # fields cannot accidentally refer to different frames.
                "writer_enqueued_target_sequence": None,
            }
            if (
                target is not None
                and self._status.ownership_state is LowCmdOwnershipState.MPC_ACTIVE
            ):
                # This is a causal local DDS-enqueue watermark only.  Unitree's
                # current boundary exposes no motor-side command application
                # acknowledgement, so actuator_applied_target_sequence remains
                # None.
                status_changes["writer_enqueued_target_sequence"] = target.sequence
            if safe_hold_generation > 0:
                if safe_hold_generation > self._safe_hold_write_generation:
                    self._safe_hold_write_generation = safe_hold_generation
                    self._last_safe_hold_write_s = write_time
                position_tolerance = self._optional_float_tuple(
                    self._config.safe_hold_position_tolerance_rad
                )
                command_reached = (
                    self._safe_hold_q is not None
                    and position_tolerance is not None
                    and all(
                        abs(enqueued_q[index] - self._safe_hold_q[index])
                        <= position_tolerance[index]
                        for index in range(_JOINT_COUNT)
                    )
                )
                if (
                    command_reached
                    and self._safe_hold_command_reached_generation != safe_hold_generation
                ):
                    self._safe_hold_command_reached_generation = safe_hold_generation
                    self._safe_hold_command_reached_write_s = write_time
                    self._safe_hold_feedback_sequence_required = self._valid_low_state_sequence + 1
                    self._safe_hold_feedback_ingress_token_required = write_ingress_cutoff + 1
                elif (
                    not command_reached
                    and self._safe_hold_command_reached_generation == safe_hold_generation
                ):
                    self._safe_hold_command_reached_generation = 0
                    self._safe_hold_command_reached_write_s = None
                    self._safe_hold_feedback_sequence_required = 0
                    self._safe_hold_feedback_ingress_token_required = 0
                status_changes.update(
                    {
                        "safe_hold_active": (
                            self._safe_hold_write_generation >= self._safe_hold_request_generation
                        ),
                        "safe_hold_settled": self._safe_hold_is_settled_locked(),
                        "safe_hold_write_generation": self._safe_hold_write_generation,
                        "last_safe_hold_write_timestamp": self._last_safe_hold_write_s,
                    }
                )
            self._status = replace(
                self._status,
                **status_changes,
            )
            self._first_write.set()
            self._safe_hold_condition.notify_all()

    def _make_low_command(
        self,
        algorithm_q: Tuple[float, ...],
        *,
        previous_q: Tuple[float, ...],
        elapsed_s: float,
        now_s: float,
    ) -> Tuple[Any, Tuple[float, ...], Tuple[int, ...], Tuple[int, ...]]:
        sdk = self._sdk
        crc = self._crc
        if sdk is None or crc is None:
            raise RuntimeError("SDK/CRC is not initialized")
        message = sdk.low_cmd_factory()
        head = _read_value(message, "head", None)
        if head is None or len(head) < 2:
            raise RuntimeError("LowCmd.head is missing")
        head[0] = 0xFE
        head[1] = 0xEF
        message.level_flag = 255
        message.gpio = 0
        motor_commands = _read_value(message, "motor_cmd", None)
        if motor_commands is None or len(motor_commands) < _MOTOR_COMMAND_COUNT:
            raise RuntimeError("LowCmd.motor_cmd must expose at least 20 slots")
        for motor in motor_commands[:_MOTOR_COMMAND_COUNT]:
            motor.mode = 1
            motor.q = _POSITION_STOP_F
            motor.dq = _VELOCITY_STOP_F
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        motor_ids = self._required_int_tuple(self._config.motor_ids, "motor_ids")
        directions = self._required_int_tuple(self._config.directions, "directions")
        offsets = self._required_float_tuple(self._config.zero_offsets_rad, "zero_offsets_rad")
        kp = self._required_float_tuple(self._config.kp, "kp")
        kd = self._required_float_tuple(self._config.kd, "kd")
        tau_ff = self._required_float_tuple(self._config.tau_ff_nm, "tau_ff_nm")
        tau_limit = self._required_float_tuple(self._config.tau_limit_nm, "tau_limit_nm")
        degraded_kp = self._required_float_tuple(
            self._config.feedback_loss_degraded_kp,
            "feedback_loss_degraded_kp",
        )
        degraded_kd = self._required_float_tuple(
            self._config.feedback_loss_degraded_kd,
            "feedback_loss_degraded_kd",
        )
        degraded_tau_ff = self._required_float_tuple(
            self._config.feedback_loss_degraded_tau_ff_nm,
            "feedback_loss_degraded_tau_ff_nm",
        )
        firmware_tau_limit = self._required_float_tuple(
            self._config.firmware_torque_limit_nm,
            "firmware_torque_limit_nm",
        )
        if self._config.firmware_torque_clamp_verified is not True:
            raise RuntimeError("firmware torque clamp is not commissioned")
        q_min = self._required_float_tuple(self._config.q_min_rad, "q_min_rad")
        q_max = self._required_float_tuple(self._config.q_max_rad, "q_max_rad")
        dq_max = self._required_float_tuple(self._config.dq_max_rad_s, "dq_max_rad_s")
        delta_q = self._required_float_tuple(
            self._config.maximum_delta_q_rad, "maximum_delta_q_rad"
        )
        feedback = self._envelope_motors if self._feedback_envelope_is_fresh_locked(now_s) else ()
        encoded_q = []
        derated_joints = []
        feedback_degraded_joints = []
        for joint_index in range(_JOINT_COUNT):
            motor = motor_commands[motor_ids[joint_index]]
            direction = directions[joint_index]
            # Keep the software envelope inside the independently verified
            # firmware clamp as well as the configured software ceiling, so
            # normal control does not rely on unmodelled actuator saturation.
            limit = min(tau_limit[joint_index], firmware_tau_limit[joint_index])
            feed_forward = tau_ff[joint_index]
            stiffness = kp[joint_index]
            maximum_step = min(delta_q[joint_index], dq_max[joint_index] * elapsed_s)
            movement_low = max(q_min[joint_index], previous_q[joint_index] - maximum_step)
            movement_high = min(q_max[joint_index], previous_q[joint_index] + maximum_step)
            bounded_q = max(
                movement_low,
                min(movement_high, algorithm_q[joint_index]),
            )
            use_degraded = len(feedback) != _JOINT_COUNT
            measured_q: Optional[float] = None
            measured_dq: Optional[float] = None
            if not use_degraded:
                measured_q = feedback[joint_index].q_rad
                measured_dq = feedback[joint_index].dq_rad_s
                use_degraded = (
                    measured_q is None
                    or measured_dq is None
                    or not math.isfinite(measured_q)
                    or not math.isfinite(measured_dq)
                )

            standard_feasible = not use_degraded
            if standard_feasible:
                assert measured_q is not None
                assert measured_dq is not None
                damping_magnitude = abs(kd[joint_index] * measured_dq)
                position_budget = limit - damping_magnitude - abs(feed_forward)
                if not math.isfinite(damping_magnitude) or not math.isfinite(position_budget):
                    use_degraded = True
                    standard_feasible = False
                elif position_budget < 0.0:
                    standard_feasible = False
                elif stiffness > 0.0:
                    position_error_limit = position_budget / stiffness
                    feasible_low = max(movement_low, measured_q - position_error_limit)
                    feasible_high = min(movement_high, measured_q + position_error_limit)
                    if (
                        not math.isfinite(feasible_low)
                        or not math.isfinite(feasible_high)
                        or feasible_low > feasible_high + 1.0e-12
                    ):
                        standard_feasible = False
                    else:
                        bounded_q = max(
                            feasible_low,
                            min(feasible_high, algorithm_q[joint_index]),
                        )

            if standard_feasible:
                assert measured_q is not None
                assert measured_dq is not None
                effective_kp = stiffness
                effective_kd = kd[joint_index]
                effective_feed_forward = feed_forward
                conservative_envelope = (
                    abs(effective_kp * (bounded_q - measured_q))
                    + abs(effective_kd * measured_dq)
                    + abs(effective_feed_forward)
                )
                if (
                    not math.isfinite(conservative_envelope)
                    or conservative_envelope > limit + 1.0e-8
                ):
                    use_degraded = True
                    standard_feasible = False

            if not standard_feasible and not use_degraded:
                assert measured_q is not None
                assert measured_dq is not None
                # Never use cancellation between P, D and feed-forward terms
                # as a safety argument. Scale all three by one common factor
                # so their absolute-component sum is within the software
                # limit at the latest usable q/dq sample.
                components = (
                    abs(stiffness * (bounded_q - measured_q)),
                    abs(kd[joint_index] * measured_dq),
                    abs(feed_forward),
                )
                envelope = sum(components)
                if not math.isfinite(envelope):
                    use_degraded = True
                else:
                    scale = 1.0 if envelope <= limit else limit / envelope
                    effective_kp = stiffness * scale
                    effective_kd = kd[joint_index] * scale
                    effective_feed_forward = feed_forward * scale
                    derated_joints.append(joint_index)

            if use_degraded:
                # With lost/stale/non-finite q/dq, software cannot estimate PD
                # torque. Keep the fixed-rate stream alive using separately
                # commissioned reduced gains; the independently verified
                # firmware clamp is the actual torque bound in this branch.
                effective_kp = degraded_kp[joint_index]
                effective_kd = degraded_kd[joint_index]
                effective_feed_forward = degraded_tau_ff[joint_index]
                if abs(effective_feed_forward) > firmware_tau_limit[joint_index]:
                    raise RuntimeError(
                        "feedback-loss feed-forward exceeds the verified firmware clamp"
                    )
                feedback_degraded_joints.append(joint_index)
            encoded_q.append(bounded_q)
            motor.q = offsets[joint_index] + direction * bounded_q
            motor.dq = 0.0
            motor.kp = effective_kp
            motor.kd = effective_kd
            motor.tau = direction * effective_feed_forward
        crc_value = crc.Crc(message)
        if (
            isinstance(crc_value, bool)
            or not isinstance(crc_value, int)
            or crc_value < 0
            or crc_value > 0xFFFFFFFF
        ):
            raise RuntimeError("SDK CRC did not return an unsigned 32-bit integer")
        message.crc = crc_value
        return (
            message,
            tuple(encoded_q),
            tuple(derated_joints),
            tuple(feedback_degraded_joints),
        )

    def _slew_limit(
        self,
        previous: Tuple[float, ...],
        desired: Tuple[float, ...],
        elapsed_s: float,
    ) -> Tuple[float, ...]:
        dq_max = self._required_float_tuple(self._config.dq_max_rad_s, "dq_max_rad_s")
        delta_max = self._required_float_tuple(
            self._config.maximum_delta_q_rad, "maximum_delta_q_rad"
        )
        q_min = self._required_float_tuple(self._config.q_min_rad, "q_min_rad")
        q_max = self._required_float_tuple(self._config.q_max_rad, "q_max_rad")
        result = []
        for index in range(_JOINT_COUNT):
            maximum_step = min(delta_max[index], dq_max[index] * elapsed_s)
            error = desired[index] - previous[index]
            step = max(-maximum_step, min(maximum_step, error))
            result.append(max(q_min[index], min(q_max[index], previous[index] + step)))
        return tuple(result)

    def _begin_low_state_callback(
        self,
        subscription_generation: int,
    ) -> Optional[Tuple[int, int, float]]:
        """Register callback ingress before touching a potentially lazy message."""

        with self._low_state_ingress_condition:
            if subscription_generation != self._subscription_generation:
                return None
            receipt_time = self._clock.monotonic()
            self._low_state_ingress_sequence += 1
            token = self._low_state_ingress_sequence
            registered_transaction = self._handoff_ingress_open_transaction
            self._low_state_inflight[token] = registered_transaction
            return token, registered_transaction, receipt_time

    def _finish_low_state_callback(self, ingress_token: int) -> None:
        with self._low_state_ingress_condition:
            self._low_state_inflight.pop(ingress_token, None)
            self._low_state_ingress_condition.notify_all()

    def _wait_for_handoff_callbacks(
        self,
        transaction_id: int,
        cutoff: int,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._low_state_ingress_condition:
            while any(
                token <= cutoff or registered == transaction_id
                for token, registered in self._low_state_inflight.items()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._low_state_ingress_condition.wait(remaining)
            return True

    def _close_handoff_ingress_window(self, transaction_id: int) -> None:
        with self._low_state_ingress_condition:
            if self._handoff_ingress_open_transaction == transaction_id:
                self._handoff_ingress_open_transaction = 0
            self._low_state_ingress_condition.notify_all()

    def _reset_handoff_tracking_locked(self) -> None:
        """Cancel one handoff transaction while the bridge guard is held."""

        self._handoff_committing = False
        self._handoff_commit_epoch = 0
        self._handoff_commit_generation = 0
        self._handoff_feedback_fault = None
        self._handoff_active_transaction_id = 0
        self._handoff_ingress_cutoff = 0
        with self._low_state_ingress_condition:
            self._handoff_ingress_open_transaction = 0
            self._low_state_ingress_condition.notify_all()

    def _on_low_state(
        self,
        message: Any,
        *,
        subscription_generation: Optional[int] = None,
    ) -> None:
        generation = (
            self._subscription_generation
            if subscription_generation is None
            else subscription_generation
        )
        ingress = self._begin_low_state_callback(generation)
        if ingress is None:
            return
        ingress_token, registered_transaction, receipt_time = ingress
        try:
            self._process_low_state(
                message,
                subscription_generation=generation,
                ingress_token=ingress_token,
                registered_transaction=registered_transaction,
                receipt_time=receipt_time,
            )
        except Exception as exc:
            self._record_low_state_fault(
                receipt_time,
                f"LowState callback raised {type(exc).__name__}: {exc}",
                subscription_generation=generation,
                ingress_token=ingress_token,
                registered_transaction=registered_transaction,
            )
        finally:
            self._finish_low_state_callback(ingress_token)

    def _process_low_state(
        self,
        message: Any,
        *,
        subscription_generation: int,
        ingress_token: int,
        registered_transaction: int,
        receipt_time: float,
    ) -> None:
        now = receipt_time
        motor_states = _read_value(message, "motor_state", None)
        if motor_states is None:
            self._record_low_state_fault(
                now,
                "LowState.motor_state is missing",
                subscription_generation=subscription_generation,
                ingress_token=ingress_token,
                registered_transaction=registered_transaction,
            )
            return
        raw_foot_force, raw_foot_force_valid = _sdk_int16_quad(
            _read_value(message, "foot_force", None)
        )
        estimated_foot_force, estimated_foot_force_valid = _sdk_int16_quad(
            _read_value(message, "foot_force_est", None)
        )
        source_tick = _sdk_uint32(_read_value(message, "tick", None))
        motor_ids = self._optional_int_tuple(self._config.motor_ids)
        joint_names = self._optional_str_tuple(self._config.joint_names)
        directions = self._optional_int_tuple(self._config.directions)
        offsets = self._optional_float_tuple(self._config.zero_offsets_rad)
        if any(item is None for item in (motor_ids, joint_names, directions, offsets)):
            self._record_low_state_fault(
                now,
                "Actuator mapping is incomplete",
                subscription_generation=subscription_generation,
                ingress_token=ingress_token,
                registered_transaction=registered_transaction,
            )
            return
        assert motor_ids is not None
        assert joint_names is not None
        assert directions is not None
        assert offsets is not None
        feedback = []
        parse_fault: Optional[str] = None
        for joint_index, motor_id in enumerate(motor_ids):
            if motor_id < 0 or motor_id >= len(motor_states):
                feedback.append(
                    Go2MotorFeedback(
                        motor_id=motor_id,
                        joint_name=joint_names[joint_index],
                        lost=True,
                        timestamp=now,
                    )
                )
                parse_fault = f"LowState has no configured motor ID {motor_id}"
                continue
            raw = motor_states[motor_id]
            direction = directions[joint_index]
            q_sdk = _finite_optional(_read_value(raw, "q", None))
            dq_sdk = _finite_optional(_read_value(raw, "dq", None))
            tau_sdk = _finite_optional(_read_value(raw, "tau_est", None))
            temperature = _finite_optional(_read_value(raw, "temperature", None))
            raw_lost = _read_value(raw, "lost", True)
            if isinstance(raw_lost, (tuple, list)):
                reports_lost = any(bool(item) for item in raw_lost)
            else:
                reports_lost = bool(raw_lost)
            lost = reports_lost or q_sdk is None or dq_sdk is None
            q_algorithm = direction * (q_sdk - offsets[joint_index]) if q_sdk is not None else None
            dq_algorithm = direction * dq_sdk if dq_sdk is not None else None
            tau_algorithm = direction * tau_sdk if tau_sdk is not None else None
            feedback.append(
                Go2MotorFeedback(
                    motor_id=motor_id,
                    joint_name=joint_names[joint_index],
                    q_rad=q_algorithm,
                    dq_rad_s=dq_algorithm,
                    tau_est_nm=tau_algorithm,
                    temperature_c=temperature,
                    lost=lost,
                    timestamp=now,
                )
            )
            if lost and parse_fault is None:
                parse_fault = f"Mapped motor {motor_id} feedback is lost or non-finite"
        with self._guard:
            if subscription_generation != self._subscription_generation:
                return
            feedback_tuple = tuple(feedback)
            envelope_usable = len(feedback_tuple) == _JOINT_COUNT and all(
                not motor.lost
                and motor.q_rad is not None
                and motor.dq_rad_s is not None
                and math.isfinite(motor.q_rad)
                and math.isfinite(motor.dq_rad_s)
                for motor in feedback_tuple
            )
            is_latest_frame = ingress_token > self._envelope_ingress_token
            if is_latest_frame:
                source_tick_monotonic = False
                if source_tick is not None:
                    previous_tick = self._last_foot_force_source_tick
                    if previous_tick is None:
                        source_tick_monotonic = True
                    else:
                        # RFC-1982-style uint32 serial arithmetic accepts the
                        # natural wrap but rejects duplicates and backwards or
                        # ambiguous half-range jumps.  Do not infer a time in
                        # seconds from this counter: its period is not part of
                        # the public Go2 LowState contract.
                        tick_delta = (source_tick - previous_tick) & 0xFFFFFFFF
                        source_tick_monotonic = 0 < tick_delta < 0x80000000
                    if source_tick_monotonic:
                        self._last_foot_force_source_tick = source_tick
                foot_force_feedback = Go2FootForceFeedback(
                    receipt_timestamp_s=now,
                    receipt_sequence=ingress_token,
                    subscription_generation=subscription_generation,
                    source_tick=source_tick,
                    source_tick_valid=source_tick is not None,
                    source_tick_monotonic=source_tick_monotonic,
                    raw_sdk_int16=raw_foot_force,
                    estimated_sdk_int16=estimated_foot_force,
                    raw_valid=raw_foot_force_valid,
                    estimated_valid=estimated_foot_force_valid,
                )
                self._envelope_ingress_token = ingress_token
                self._envelope_motors = feedback_tuple if envelope_usable else ()
                self._status = replace(
                    self._status,
                    timestamp=now,
                    connected=True,
                    motors=feedback_tuple,
                    foot_force_feedback=foot_force_feedback,
                )
            # Evaluate the exact callback frame. An older callback that finishes
            # late must not roll latest status/cache backwards, but a fault in
            # that callback remains safety-significant during active ownership
            # and especially across the handoff fence.
            # Read-only commissioning must remain usable before q/dq/torque/
            # temperature limits are identified.  It validates the stream and
            # coordinate mapping only.  The complete limit envelope remains a
            # mandatory, separate gate for every LowCmd acquisition.
            if self._actuation_config_ready:
                limit_fault = self._feedback_limit_failure(feedback_tuple)
            else:
                limit_fault = self._observation_feedback_failure(feedback_tuple)
            fault = parse_fault or limit_fault
            active = self._owner_epoch != 0
            endpoint_active = active and self._publisher is not None
            handoff_relevant = self._callback_is_handoff_relevant_locked(
                ingress_token,
                registered_transaction,
            )
            if fault is not None:
                if handoff_relevant:
                    self._record_handoff_feedback_fault_locked(now, fault)
                elif endpoint_active:
                    self._enter_safe_hold_locked(now, fault, fault=True)
                elif is_latest_frame:
                    self._status = replace(
                        self._status,
                        healthy=False,
                        fault_reason=fault,
                    )
                return
            # Only a complete, finite, in-limit frame advances the effective
            # LowState timestamp.  A malformed/newer packet must never make
            # stale, previously valid joint feedback look fresh.
            if is_latest_frame:
                self._valid_motors = feedback_tuple
                self._valid_low_state_sequence += 1
                self._valid_low_state_ingress_token = ingress_token
                tracking_changes: Dict[str, Any] = {}
                reference_q = self._command_reference_q
                reference_write_s = self._command_reference_write_s
                if (
                    reference_q is not None
                    and reference_write_s is not None
                    and ingress_token > self._command_reference_ingress_cutoff
                    and now >= reference_write_s
                ):
                    tracking_changes = {
                        "tracking_error_timestamp": now,
                        "tracking_reference_write_timestamp": reference_write_s,
                        "tracking_reference_write_generation": (self._command_reference_generation),
                        "tracking_reference_q_rad": reference_q,
                        "position_error_rad": tuple(
                            cast(float, motor.q_rad) - reference_q[index]
                            for index, motor in enumerate(feedback_tuple)
                        ),
                    }
                self._status = replace(
                    self._status,
                    low_state_timestamp=now,
                    low_state_age_s=0.0,
                    **tracking_changes,
                )
            if handoff_relevant:
                # The writer/publisher handoff has crossed its linearization
                # fence. Re-evaluate the exact frozen safe-hold generation,
                # but never create a generation that no stopped writer can
                # honor. A valid-yet-moving frame is as significant here as a
                # malformed frame and therefore becomes a sticky handoff fault.
                if self._handoff_commit_generation > 0 and (
                    self._safe_hold_request_generation != self._handoff_commit_generation
                    or not self._feedback_frame_satisfies_safe_hold_locked(
                        feedback_tuple,
                        ingress_token,
                    )
                ):
                    self._record_handoff_feedback_fault_locked(
                        now,
                        "A post-fence LowState frame no longer satisfies the frozen safe-hold pose/velocity tolerances",
                    )
                else:
                    self._status = replace(
                        self._status,
                        safe_hold_settled=(
                            self._handoff_commit_generation > 0
                            and self._feedback_frame_satisfies_safe_hold_locked(
                                feedback_tuple,
                                ingress_token,
                            )
                        ),
                    )
                self._safe_hold_condition.notify_all()
            elif endpoint_active and is_latest_frame:
                self._status = replace(
                    self._status,
                    safe_hold_settled=self._safe_hold_is_settled_locked(),
                )
                self._safe_hold_condition.notify_all()
            elif active and is_latest_frame:
                # Close() has already proved the LowCmd endpoint locally
                # absent. LowState remains diagnostic, but it must not create
                # an impossible-to-service safe-hold generation or obstruct an
                # exact high-level recovery retry.
                self._status = replace(
                    self._status,
                    healthy=True,
                    fault_reason=None,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                )
            elif is_latest_frame and self._status.ownership_state in {
                LowCmdOwnershipState.DISCONNECTED,
                LowCmdOwnershipState.OBSERVE_ONLY,
            }:
                self._status = replace(self._status, healthy=True, fault_reason=None)
            if is_latest_frame:
                self._first_state.set()

    def _record_handoff_feedback_fault_locked(self, now: float, reason: str) -> None:
        self._handoff_feedback_fault = reason
        self._status = replace(
            self._status,
            timestamp=now,
            ownership_state=LowCmdOwnershipState.FAULT,
            healthy=False,
            safe_hold_active=False,
            safe_hold_settled=False,
            watchdog_healthy=False,
            fault_reason="LowState fault during handoff commit: " + reason,
        )
        self._safe_hold_condition.notify_all()

    def _callback_is_handoff_relevant_locked(
        self,
        ingress_token: int,
        registered_transaction: int,
    ) -> bool:
        return (
            self._owner_epoch != 0
            and self._handoff_committing
            and self._handoff_active_transaction_id != 0
            and self._handoff_commit_epoch == self._owner_epoch
            and (
                ingress_token <= self._handoff_ingress_cutoff
                or registered_transaction == self._handoff_active_transaction_id
                or self._publisher is not None
            )
        )

    def _record_low_state_fault(
        self,
        now: float,
        reason: str,
        *,
        subscription_generation: Optional[int] = None,
        ingress_token: Optional[int] = None,
        registered_transaction: int = 0,
    ) -> None:
        with self._guard:
            if (
                subscription_generation is not None
                and subscription_generation != self._subscription_generation
            ):
                return
            is_latest_frame = ingress_token is None or ingress_token > self._envelope_ingress_token
            if is_latest_frame:
                if ingress_token is not None:
                    self._envelope_ingress_token = ingress_token
                self._envelope_motors = ()
                invalid_motors = tuple(
                    replace(motor, lost=True, timestamp=now) for motor in self._status.motors
                )
                invalid_force_feedback = Go2FootForceFeedback(
                    receipt_timestamp_s=now,
                    receipt_sequence=(0 if ingress_token is None else ingress_token),
                    subscription_generation=(
                        self._subscription_generation
                        if subscription_generation is None
                        else subscription_generation
                    ),
                )
                self._status = replace(
                    self._status,
                    timestamp=now,
                    connected=True,
                    healthy=False,
                    motors=invalid_motors,
                    foot_force_feedback=invalid_force_feedback,
                    safe_hold_settled=False,
                    fault_reason=reason,
                )
            handoff_relevant = (
                ingress_token is not None
                and self._callback_is_handoff_relevant_locked(
                    ingress_token,
                    registered_transaction,
                )
            )
            if handoff_relevant:
                self._record_handoff_feedback_fault_locked(now, reason)
            elif self._owner_epoch != 0 and self._publisher is not None:
                self._enter_safe_hold_locked(now, reason, fault=True)

    def _feedback_limit_failure_locked(self) -> Optional[str]:
        return self._feedback_limit_failure(self._status.motors)

    @staticmethod
    def _observation_feedback_failure(
        motors: Tuple[Go2MotorFeedback, ...],
    ) -> Optional[str]:
        """Check stream integrity without pretending uncommissioned limits are safe."""

        if len(motors) != _JOINT_COUNT:
            return "LowState does not contain 12 mapped motor records"
        for motor in motors:
            if motor.lost or motor.q_rad is None or motor.dq_rad_s is None:
                return f"{motor.joint_name} motor q/dq feedback is lost or non-finite"
        return None

    def _feedback_limit_failure(self, motors: Tuple[Go2MotorFeedback, ...]) -> Optional[str]:
        if len(motors) != _JOINT_COUNT:
            return "LowState does not contain 12 mapped motor records"
        q_min = self._optional_float_tuple(self._config.q_min_rad)
        q_max = self._optional_float_tuple(self._config.q_max_rad)
        dq_max = self._optional_float_tuple(self._config.dq_max_rad_s)
        tau_limit = self._optional_float_tuple(self._config.tau_limit_nm)
        temperature_limit = self._optional_float_tuple(self._config.temperature_limit_c)
        if any(item is None for item in (q_min, q_max, dq_max, tau_limit, temperature_limit)):
            return "Joint safety limits are incomplete"
        assert q_min is not None
        assert q_max is not None
        assert dq_max is not None
        assert tau_limit is not None
        assert temperature_limit is not None
        for index, motor in enumerate(motors):
            if motor.lost:
                return f"{motor.joint_name} motor feedback reports lost"
            if motor.q_rad is None or not q_min[index] <= motor.q_rad <= q_max[index]:
                return f"{motor.joint_name} measured q is outside the commissioned hard range"
            if motor.dq_rad_s is None or abs(motor.dq_rad_s) > dq_max[index]:
                return f"{motor.joint_name} measured dq exceeds its software limit"
            if motor.tau_est_nm is None or abs(motor.tau_est_nm) > tau_limit[index]:
                return f"{motor.joint_name} estimated torque exceeds its software limit"
            if motor.temperature_c is None or motor.temperature_c >= temperature_limit[index]:
                return f"{motor.joint_name} temperature reached its configured fault threshold"
        return None

    def _verify_live_ground_transfer(
        self, transfer: str, permit: Go2OwnershipPermit
    ) -> OperationResult:
        """Re-read independent hardware evidence at each authority boundary."""

        verifier = self._ground_transfer_verifier
        if verifier is None:
            return OperationResult.failure(
                "GO2_LIVE_GROUND_VERIFIER_MISSING",
                "LowCmd ownership transfer requires an injected live Pixhawk/ESC/F446/Go2 ground verifier",
                {"transfer": transfer},
            )
        try:
            result = verifier(transfer, permit)
        except Exception as exc:
            return OperationResult.failure(
                "GO2_LIVE_GROUND_VERIFIER_FAILED",
                f"Live ground verifier raised {type(exc).__name__}: {exc}",
                {"transfer": transfer},
            )
        if not isinstance(result, OperationResult):
            return OperationResult.failure(
                "GO2_LIVE_GROUND_VERIFIER_PROTOCOL_ERROR",
                "Live ground verifier did not return an OperationResult",
                {"transfer": transfer},
            )
        if not result.ok:
            return OperationResult.failure(
                result.code,
                result.message,
                {"transfer": transfer, **dict(result.data)},
            )
        return OperationResult.success(
            "Live ground evidence revalidated",
            {"transfer": transfer, **dict(result.data)},
            code="GO2_LIVE_GROUND_VERIFIED",
        )

    def observation_readiness(self) -> OperationResult:
        """Validate only what is needed to interpret a read-only LowState stream."""

        if not self._config.observation_enabled:
            return OperationResult.failure(
                "GO2_LOW_STATE_OBSERVATION_DISABLED",
                "Neither go2.low_level.observe_only_enabled nor go2.low_level.enabled is true",
            )
        try:
            self._required_str(self._config.low_state_topic, "low_state_topic")
            maximum_age = self._required_float(
                self._config.low_state_max_age_s,
                "low_state_max_age_s",
            )
            if maximum_age <= 0.0:
                raise ValueError("low_state_max_age_s must be positive")
            mapping_version = self._required_str(
                self._config.mapping_version,
                "mapping_version",
            )
            configured_hash = self._required_str(
                self._config.mapping_hash,
                "mapping_hash",
            )
            joint_names = self._required_str_tuple(
                self._config.joint_names,
                "joint_names",
            )
            motor_ids = self._required_int_tuple(self._config.motor_ids, "motor_ids")
            directions = self._required_int_tuple(self._config.directions, "directions")
            offsets = self._required_float_tuple(
                self._config.zero_offsets_rad,
                "zero_offsets_rad",
            )
        except (TypeError, ValueError) as exc:
            return OperationResult.failure("GO2_LOW_STATE_CONFIG_INCOMPLETE", str(exc))
        expected_hash = compute_go2_mapping_hash(
            mapping_version,
            joint_names,
            motor_ids,
            directions,
            offsets,
        )
        if configured_hash != expected_hash:
            return OperationResult.failure(
                "GO2_MAPPING_HASH_UNVERIFIED",
                f"mapping_hash must equal the canonical commissioned value {expected_hash}",
            )
        if len(set(joint_names)) != _JOINT_COUNT or len(set(motor_ids)) != _JOINT_COUNT:
            return OperationResult.failure(
                "GO2_MAPPING_NOT_BIJECTIVE",
                "Joint names and motor IDs must each be unique",
            )
        if set(motor_ids) != set(range(_JOINT_COUNT)):
            return OperationResult.failure(
                "GO2_MOTOR_ID_RANGE",
                "SDK motor IDs must be a permutation of the 12 Go2 leg slots 0..11",
            )
        if any(direction not in (-1, 1) for direction in directions):
            return OperationResult.failure(
                "GO2_DIRECTION_INVALID",
                "Every joint direction must be exactly -1 or +1",
            )
        return OperationResult.success(
            "Go2 LowState observation configuration is complete and mapping-verified"
        )

    def actuation_readiness(self) -> OperationResult:
        """Validate the complete fail-closed LowCmd acquisition/write configuration."""

        if not self._config.enabled:
            return OperationResult.failure(
                "GO2_LOW_LEVEL_DISABLED", "go2.low_level.enabled is false"
            )
        observation = self.observation_readiness()
        if not observation.ok:
            return observation
        try:
            mapping_version = self._required_str(self._config.mapping_version, "mapping_version")
            self._required_str(self._config.restore_mode_form, "restore_mode_form")
            self._required_str(self._config.restore_mode_name, "restore_mode_name")
            configured_hash = self._required_str(self._config.mapping_hash, "mapping_hash")
            joint_names = self._required_str_tuple(self._config.joint_names, "joint_names")
            motor_ids = self._required_int_tuple(self._config.motor_ids, "motor_ids")
            directions = self._required_int_tuple(self._config.directions, "directions")
            offsets = self._required_float_tuple(self._config.zero_offsets_rad, "zero_offsets_rad")
            q_min = self._required_float_tuple(self._config.q_min_rad, "q_min_rad")
            q_max = self._required_float_tuple(self._config.q_max_rad, "q_max_rad")
            dq_max = self._required_float_tuple(self._config.dq_max_rad_s, "dq_max_rad_s")
            delta_q = self._required_float_tuple(
                self._config.maximum_delta_q_rad, "maximum_delta_q_rad"
            )
            kp = self._required_float_tuple(self._config.kp, "kp")
            kd = self._required_float_tuple(self._config.kd, "kd")
            tau_ff = self._required_float_tuple(self._config.tau_ff_nm, "tau_ff_nm")
            tau_limit = self._required_float_tuple(self._config.tau_limit_nm, "tau_limit_nm")
            degraded_kp = self._required_float_tuple(
                self._config.feedback_loss_degraded_kp,
                "feedback_loss_degraded_kp",
            )
            degraded_kd = self._required_float_tuple(
                self._config.feedback_loss_degraded_kd,
                "feedback_loss_degraded_kd",
            )
            degraded_tau_ff = self._required_float_tuple(
                self._config.feedback_loss_degraded_tau_ff_nm,
                "feedback_loss_degraded_tau_ff_nm",
            )
            firmware_tau_limit = self._required_float_tuple(
                self._config.firmware_torque_limit_nm,
                "firmware_torque_limit_nm",
            )
            if self._config.firmware_torque_clamp_verified is not True:
                raise ValueError(
                    "firmware_torque_clamp_verified must be exact True after robot-specific testing"
                )
            temperature = self._required_float_tuple(
                self._config.temperature_limit_c, "temperature_limit_c"
            )
            safe_hold = self._required_float_tuple(
                self._config.safe_hold_pose_rad, "safe_hold_pose_rad"
            )
            safe_hold_position_tolerance = self._required_float_tuple(
                self._config.safe_hold_position_tolerance_rad,
                "safe_hold_position_tolerance_rad",
            )
            safe_hold_velocity_tolerance = self._required_float_tuple(
                self._config.safe_hold_velocity_tolerance_rad_s,
                "safe_hold_velocity_tolerance_rad_s",
            )
            tracking_position_error_limit = self._required_float_tuple(
                self._config.tracking_position_error_limit_rad,
                "tracking_position_error_limit_rad",
            )
            for scalar_name in (
                "send_period_s",
                "low_state_max_age_s",
                "target_ttl_s",
                "acquire_timeout_s",
                "release_timeout_s",
                "safe_hold_ack_timeout_s",
            ):
                if self._required_float(getattr(self._config, scalar_name), scalar_name) <= 0.0:
                    raise ValueError(f"{scalar_name} must be positive")
            if self._required_float(self._config.maximum_jitter_s, "maximum_jitter_s") < 0.0:
                raise ValueError("maximum_jitter_s cannot be negative")
            period = self._required_float(self._config.send_period_s, "send_period_s")
            if self._required_float(self._config.maximum_jitter_s, "maximum_jitter_s") >= period:
                raise ValueError("maximum_jitter_s must be less than send_period_s")
            for scalar_name in (
                "low_state_max_age_s",
                "target_ttl_s",
                "safe_hold_ack_timeout_s",
            ):
                if self._required_float(getattr(self._config, scalar_name), scalar_name) < period:
                    raise ValueError(f"{scalar_name} must be at least send_period_s")
            if self._required_float(
                self._config.safe_hold_ack_timeout_s, "safe_hold_ack_timeout_s"
            ) > self._required_float(self._config.release_timeout_s, "release_timeout_s"):
                raise ValueError("safe_hold_ack_timeout_s must not exceed release_timeout_s")
            policy = self._required_str(self._config.safe_hold_policy, "safe_hold_policy")
        except (TypeError, ValueError) as exc:
            return OperationResult.failure("GO2_LOW_LEVEL_CONFIG_INCOMPLETE", str(exc))
        expected_hash = compute_go2_mapping_hash(
            mapping_version, joint_names, motor_ids, directions, offsets
        )
        if configured_hash != expected_hash:
            return OperationResult.failure(
                "GO2_MAPPING_HASH_UNVERIFIED",
                f"mapping_hash must equal the canonical commissioned value {expected_hash}",
            )
        if len(set(joint_names)) != _JOINT_COUNT or len(set(motor_ids)) != _JOINT_COUNT:
            return OperationResult.failure(
                "GO2_MAPPING_NOT_BIJECTIVE", "Joint names and motor IDs must each be unique"
            )
        if set(motor_ids) != set(range(_JOINT_COUNT)):
            return OperationResult.failure(
                "GO2_MOTOR_ID_RANGE",
                "SDK motor IDs must be a permutation of the 12 Go2 leg slots 0..11",
            )
        if any(direction not in (-1, 1) for direction in directions):
            return OperationResult.failure(
                "GO2_DIRECTION_INVALID", "Every joint direction must be exactly -1 or +1"
            )
        if policy not in {"capture_current", "configured_pose"}:
            return OperationResult.failure(
                "GO2_SAFE_HOLD_POLICY_INVALID",
                "safe_hold_policy must be capture_current or configured_pose",
            )
        for index in range(_JOINT_COUNT):
            if q_min[index] >= q_max[index]:
                return OperationResult.failure(
                    "GO2_JOINT_RANGE_INVALID", f"q_min_rad[{index}] must be below q_max_rad"
                )
            if not q_min[index] <= safe_hold[index] <= q_max[index]:
                return OperationResult.failure(
                    "GO2_SAFE_HOLD_LIMIT",
                    f"safe_hold_pose_rad[{index}] is outside its hard range",
                )
            if dq_max[index] <= 0.0 or delta_q[index] <= 0.0:
                return OperationResult.failure(
                    "GO2_RATE_LIMIT_INVALID", "dq and per-cycle delta limits must be positive"
                )
            if (
                tracking_position_error_limit[index] <= 0.0
                or tracking_position_error_limit[index] >= q_max[index] - q_min[index]
            ):
                return OperationResult.failure(
                    "GO2_TRACKING_ERROR_LIMIT_INVALID",
                    "tracking error limits must be positive and smaller than each q range",
                )
            if kp[index] < 0.0 or kd[index] < 0.0:
                return OperationResult.failure("GO2_GAIN_INVALID", "Kp and Kd cannot be negative")
            if (
                degraded_kp[index] < 0.0
                or degraded_kd[index] < 0.0
                or degraded_kp[index] > kp[index]
                or degraded_kd[index] > kd[index]
            ):
                return OperationResult.failure(
                    "GO2_DEGRADED_GAIN_INVALID",
                    "Feedback-loss Kp/Kd must be nonnegative and no larger than normal gains",
                )
            if tau_limit[index] <= 0.0 or abs(tau_ff[index]) > tau_limit[index]:
                return OperationResult.failure(
                    "GO2_TORQUE_LIMIT_INVALID",
                    "Feed-forward torque must lie inside a positive torque limit",
                )
            if (
                firmware_tau_limit[index] <= 0.0
                or firmware_tau_limit[index] > tau_limit[index]
                or abs(degraded_tau_ff[index]) > firmware_tau_limit[index]
            ):
                return OperationResult.failure(
                    "GO2_FIRMWARE_TORQUE_CLAMP_INVALID",
                    "Verified firmware torque limits must be positive, no larger than software limits, and bound degraded feed-forward",
                )
            if temperature[index] <= 0.0:
                return OperationResult.failure(
                    "GO2_TEMPERATURE_LIMIT_INVALID", "Temperature thresholds must be positive"
                )
            if temperature[index] > 150.0:
                return OperationResult.failure(
                    "GO2_TEMPERATURE_LIMIT_INVALID",
                    "Temperature thresholds cannot exceed 150 C",
                )
            if (
                safe_hold_position_tolerance[index] <= 0.0
                or safe_hold_position_tolerance[index] >= q_max[index] - q_min[index]
            ):
                return OperationResult.failure(
                    "GO2_SAFE_HOLD_TOLERANCE_INVALID",
                    "Safe-hold position tolerances must be positive and smaller than each q range",
                )
            if (
                safe_hold_velocity_tolerance[index] <= 0.0
                or safe_hold_velocity_tolerance[index] > dq_max[index]
            ):
                return OperationResult.failure(
                    "GO2_SAFE_HOLD_TOLERANCE_INVALID",
                    "Safe-hold velocity tolerances must be positive and no larger than dq_max",
                )
        return OperationResult.success("Go2 LowCmd configuration is complete and hash-verified")

    def _validate_algorithm_positions(self, values: Sequence[float]) -> Optional[str]:
        if len(values) != _JOINT_COUNT:
            return "Target must contain exactly 12 joint positions"
        q_min = self._required_float_tuple(self._config.q_min_rad, "q_min_rad")
        q_max = self._required_float_tuple(self._config.q_max_rad, "q_max_rad")
        for index, raw in enumerate(values):
            value = _finite_optional(raw)
            if value is None:
                return f"joint_positions_rad[{index}] must be finite"
            if not q_min[index] <= value <= q_max[index]:
                return f"joint_positions_rad[{index}] is outside its hard limit"
        return None

    def _enter_safe_hold_locked(
        self,
        now: float,
        reason: str,
        *,
        fault: bool,
        force_new_generation: bool = False,
    ) -> int:
        previous_state = self._status.ownership_state
        had_target = self._target is not None
        requested_state = LowCmdOwnershipState.FAULT if fault else LowCmdOwnershipState.SAFE_HOLD
        new_generation = (
            force_new_generation
            or had_target
            or self._safe_hold_request_generation == 0
            or previous_state is not requested_state
        )
        if new_generation and self._config.safe_hold_policy == "capture_current":
            # "Hold" means freeze near the posture at revocation, not return
            # toward the pose captured at acquisition. Use only the newest
            # envelope frame: a newer lost/non-finite frame explicitly
            # invalidates older nominal feedback, so it must fall back to the
            # last commanded pose rather than call stale q "current".
            current_q = self._current_envelope_joint_positions_locked(now)
            if current_q is not None:
                self._safe_hold_q = current_q
            elif self._last_commanded_q is not None:
                self._safe_hold_q = self._last_commanded_q
        self._target = None
        if new_generation:
            self._safe_hold_request_generation += 1
            self._safe_hold_command_reached_generation = 0
            self._safe_hold_command_reached_write_s = None
            self._safe_hold_feedback_sequence_required = 0
            self._safe_hold_feedback_ingress_token_required = 0
            safe_hold_active = False
            safe_hold_settled = False
        else:
            safe_hold_active = (
                self._safe_hold_write_generation >= self._safe_hold_request_generation
            )
            safe_hold_settled = self._status.safe_hold_settled
        self._status = replace(
            self._status,
            timestamp=now,
            ownership_state=requested_state,
            healthy=False if fault else self._status.healthy,
            target_sequence=None,
            target_age_s=None,
            target_deadline=None,
            mailbox_staged_target_sequence=None,
            watchdog_healthy=not fault,
            safe_hold_active=safe_hold_active,
            safe_hold_settled=safe_hold_settled,
            safe_hold_request_generation=self._safe_hold_request_generation,
            fault_reason=reason if fault else None,
        )
        self._safe_hold_condition.notify_all()
        return self._safe_hold_request_generation

    def _wait_for_safe_hold_write(
        self, ownership_epoch: int, generation: int, timeout_s: float
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            acquired = self._guard.acquire(
                timeout=min(self._maximum_owner_guard_wait_s(), remaining)
            )
            if acquired:
                try:
                    if self._owner_epoch != ownership_epoch:
                        return False
                    if self._safe_hold_write_generation >= generation:
                        return True
                    thread = self._writer_thread
                    if thread is None or not thread.is_alive() or self._owner_epoch == 0:
                        return False
                finally:
                    self._guard.release()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(0.001, remaining))

    def _wait_for_safe_hold_settled(
        self, ownership_epoch: int, generation: int, timeout_s: float
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            acquired = self._guard.acquire(
                timeout=min(self._maximum_owner_guard_wait_s(), remaining)
            )
            if acquired:
                try:
                    if (
                        self._owner_epoch == ownership_epoch
                        and ownership_epoch != 0
                        and self._safe_hold_request_generation == generation
                        and self._safe_hold_write_generation >= generation
                        and self._status.safe_hold_settled
                    ):
                        return True
                    if self._owner_epoch != ownership_epoch:
                        return False
                    thread = self._writer_thread
                    if thread is None or not thread.is_alive() or self._owner_epoch == 0:
                        return False
                finally:
                    self._guard.release()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(0.001, remaining))

    def _safe_hold_is_settled_locked(self) -> bool:
        if (
            self._safe_hold_q is None
            or self._safe_hold_write_generation < self._safe_hold_request_generation
            or self._safe_hold_command_reached_generation != self._safe_hold_request_generation
            or self._safe_hold_command_reached_write_s is None
            or self._safe_hold_feedback_sequence_required <= 0
            or self._valid_low_state_sequence < self._safe_hold_feedback_sequence_required
            or self._safe_hold_feedback_ingress_token_required <= 0
            or self._valid_low_state_ingress_token < self._safe_hold_feedback_ingress_token_required
            or len(self._valid_motors) != _JOINT_COUNT
        ):
            return False
        return self._feedback_frame_satisfies_safe_hold_locked(
            self._valid_motors,
            self._valid_low_state_ingress_token,
        )

    def _feedback_frame_satisfies_safe_hold_locked(
        self,
        motors: Tuple[Go2MotorFeedback, ...],
        ingress_token: int,
    ) -> bool:
        """Evaluate one exact frame without mutating the latest-frame cache."""

        if (
            self._safe_hold_q is None
            or self._safe_hold_command_reached_write_s is None
            or self._safe_hold_feedback_ingress_token_required <= 0
            or ingress_token < self._safe_hold_feedback_ingress_token_required
            or len(motors) != _JOINT_COUNT
        ):
            return False
        position_tolerance = self._optional_float_tuple(
            self._config.safe_hold_position_tolerance_rad
        )
        velocity_tolerance = self._optional_float_tuple(
            self._config.safe_hold_velocity_tolerance_rad_s
        )
        if position_tolerance is None or velocity_tolerance is None:
            return False
        for index, motor in enumerate(motors):
            if (
                motor.lost
                or motor.q_rad is None
                or motor.dq_rad_s is None
                or motor.timestamp <= self._safe_hold_command_reached_write_s
                or abs(motor.q_rad - self._safe_hold_q[index]) > position_tolerance[index]
                or abs(motor.dq_rad_s) > velocity_tolerance[index]
            ):
                return False
        return True

    def _set_fault(self, reason: str) -> None:
        now = self._clock.monotonic()
        with self._guard:
            self._set_fault_locked(reason, now=now)

    def _set_fault_locked(self, reason: str, *, now: Optional[float] = None) -> None:
        fault_time = self._clock.monotonic() if now is None else now
        if self._owner_epoch != 0:
            self._enter_safe_hold_locked(fault_time, reason, fault=True)
        else:
            self._status = replace(
                self._status,
                timestamp=fault_time,
                ownership_state=LowCmdOwnershipState.FAULT,
                healthy=False,
                fault_reason=reason,
            )

    def _set_disconnected(self, reason: str) -> None:
        with self._guard:
            self._connected = False
            self._last_foot_force_source_tick = None
            self._high_level_restore_form = None
            self._high_level_restore_mode = None
            self._status = replace(
                self._status,
                timestamp=self._clock.monotonic(),
                connected=False,
                ownership_state=(
                    LowCmdOwnershipState.DISCONNECTED
                    if self._config.observation_enabled
                    else LowCmdOwnershipState.DISABLED
                ),
                healthy=False,
                publisher_active=False,
                writer_alive=False,
                watchdog_healthy=False,
                safe_hold_active=False,
                safe_hold_settled=False,
                high_level_released=False,
                high_level_restore_form=None,
                high_level_restore_mode=None,
                network_exclusivity_verified=False,
                foot_force_feedback=Go2FootForceFeedback(),
                fault_reason=reason,
            )

    def _low_state_is_fresh_locked(self, now: float) -> bool:
        timestamp = self._status.low_state_timestamp
        maximum_age = self._config.low_state_max_age_s
        return (
            timestamp > 0.0
            and not isinstance(maximum_age, bool)
            and isinstance(maximum_age, Real)
            and math.isfinite(float(maximum_age))
            and float(maximum_age) > 0.0
            and 0.0 <= now - timestamp <= float(maximum_age)
        )

    def _feedback_envelope_is_fresh_locked(self, now: float) -> bool:
        """Whether the latest non-lost finite q/dq may bound this write.

        A frame may fail position, velocity, torque or temperature limits and
        still be the safest available q/dq for computing the *next* command's
        instantaneous envelope. Missing/lost/non-finite or stale q/dq never
        falls back to an older nominal frame.
        """

        if len(self._envelope_motors) != _JOINT_COUNT:
            return False
        maximum_age = self._config.low_state_max_age_s
        if (
            isinstance(maximum_age, bool)
            or not isinstance(maximum_age, Real)
            or not math.isfinite(float(maximum_age))
            or float(maximum_age) <= 0.0
        ):
            return False
        for motor in self._envelope_motors:
            if (
                motor.lost
                or motor.q_rad is None
                or motor.dq_rad_s is None
                or not math.isfinite(motor.q_rad)
                or not math.isfinite(motor.dq_rad_s)
                or motor.timestamp <= 0.0
                or not 0.0 <= now - motor.timestamp <= float(maximum_age)
            ):
                return False
        return True

    def _current_joint_positions_locked(self) -> Optional[Tuple[float, ...]]:
        if len(self._valid_motors) != _JOINT_COUNT:
            return None
        values = []
        for motor in self._valid_motors:
            if motor.lost or motor.q_rad is None or not math.isfinite(motor.q_rad):
                return None
            values.append(motor.q_rad)
        result = tuple(values)
        return result if self._validate_algorithm_positions(result) is None else None

    def _current_envelope_joint_positions_locked(self, now: float) -> Optional[Tuple[float, ...]]:
        if not self._feedback_envelope_is_fresh_locked(now):
            return None
        values = tuple(motor.q_rad for motor in self._envelope_motors)
        if any(value is None for value in values):
            return None
        finite_values = tuple(float(value) for value in values if value is not None)
        return finite_values if self._validate_algorithm_positions(finite_values) is None else None

    def _select_safe_hold_pose(self, current_q: Tuple[float, ...]) -> Optional[Tuple[float, ...]]:
        policy = self._config.safe_hold_policy
        selected: Optional[Tuple[float, ...]]
        if policy == "capture_current":
            selected = current_q
        elif policy == "configured_pose":
            selected = self._optional_float_tuple(self._config.safe_hold_pose_rad)
            if selected is None:
                return None
        else:
            return None
        return selected if self._validate_algorithm_positions(selected) is None else None

    @staticmethod
    def _close_transport(*, subscriber: Any, publisher: Any) -> None:
        for transport in (subscriber, publisher):
            close = getattr(transport, "Close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @staticmethod
    def _close_owned_publisher(publisher: Any) -> OperationResult:
        """Invoke the sole writer's local Close without hiding ambiguity."""

        if publisher is None:
            return OperationResult.success(
                "LowCmd publisher was already locally absent",
                {"publisher_close_called": False},
            )
        close = getattr(publisher, "Close", None)
        if not callable(close):
            return OperationResult.failure(
                "GO2_LOW_CMD_PUBLISHER_CLOSE_MISSING",
                "LowCmd publisher exposes no callable Close; high-level handback is forbidden",
                {"publisher_close_called": False},
            )
        try:
            close()
        except Exception as exc:
            return OperationResult.failure(
                "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED",
                f"LowCmd publisher Close raised {type(exc).__name__}: {exc}",
                {"publisher_close_called": True},
            )
        return OperationResult.success(
            "LowCmd publisher Close returned without exception",
            {
                "publisher_close_called": True,
                "dds_endpoint_absence_verified": False,
            },
        )

    async def _close_owned_publisher_with_deadline(
        self,
        publisher: Any,
        *,
        real_deadline_s: float,
    ) -> OperationResult:
        """Run a potentially blocking DDS ``Close`` outside the event loop.

        A timeout leaves the close worker, publisher reference, owner epoch and
        inter-process lock intact.  The caller must not SelectMode or claim the
        endpoint is closed.  A later ground-authorized retry observes the same
        in-flight/result task instead of issuing a second concurrent Close.
        """

        task = self._publisher_close_task
        retained_at_entry = task is not None
        if task is not None and self._publisher_close_target is not publisher:
            return OperationResult.failure(
                "GO2_LOW_CMD_PUBLISHER_CLOSE_CONFLICT",
                "A different LowCmd publisher Close transaction is still retained",
                {
                    "publisher_close_called": False,
                    "publisher_close_in_flight": not task.done(),
                },
            )
        remaining_s = max(0.0, real_deadline_s - time.monotonic())
        if task is None:
            if remaining_s <= 0.0:
                return OperationResult.failure(
                    "GO2_LOW_CMD_PUBLISHER_CLOSE_TIMEOUT",
                    "No transaction time remains to start LowCmd publisher Close",
                    {
                        "publisher_close_called": False,
                        "publisher_close_in_flight": False,
                        "owner_lock_retained": self._owner_epoch != 0,
                    },
                )
            task = asyncio.ensure_future(run_blocking(self._close_owned_publisher, publisher))
            self._publisher_close_task = task
            self._publisher_close_target = publisher

        remaining_s = max(0.0, real_deadline_s - time.monotonic())
        if remaining_s > 0.0 and not task.done():
            done, _ = await asyncio.wait((task,), timeout=remaining_s)
            if not done:
                return OperationResult.failure(
                    "GO2_LOW_CMD_PUBLISHER_CLOSE_TIMEOUT",
                    "LowCmd publisher Close did not return before the transaction deadline; high-level handback is forbidden",
                    {
                        "publisher_close_called": True,
                        "publisher_close_in_flight": True,
                        "owner_lock_retained": self._owner_epoch != 0,
                    },
                )
        if not task.done():
            return OperationResult.failure(
                "GO2_LOW_CMD_PUBLISHER_CLOSE_TIMEOUT",
                "No transaction time remains to confirm LowCmd publisher Close",
                {
                    "publisher_close_called": True,
                    "publisher_close_in_flight": True,
                    "owner_lock_retained": self._owner_epoch != 0,
                },
            )
        try:
            result = task.result()
        except Exception as exc:
            result = OperationResult.failure(
                "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED",
                f"LowCmd publisher Close worker raised {type(exc).__name__}: {exc}",
                {"publisher_close_called": True},
            )
        if retained_at_entry and result.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED":
            sdk = self._sdk
            if sdk is None or not sdk.publisher_close_retry_idempotency_verified:
                return OperationResult.failure(
                    "GO2_LOW_CMD_PUBLISHER_CLOSE_RETRY_UNVERIFIED",
                    "A previous Close raised, and idempotent retry is not verified for this SDK binding; retain the epoch and follow the commissioned recovery procedure",
                    {
                        **dict(result.data),
                        "owner_lock_retained": self._owner_epoch != 0,
                        "publisher_close_retry_called": False,
                        "process_restart_or_manual_recovery_required": True,
                    },
                )
            # This invocation is a new, serialized, ground-authorized release
            # transaction observing a completed exception from an earlier
            # attempt.  It may perform one non-concurrent retry.  The retry's
            # result remains subject to the same absolute deadline.
            self._publisher_close_task = None
            self._publisher_close_target = None
            return await self._close_owned_publisher_with_deadline(
                publisher,
                real_deadline_s=real_deadline_s,
            )
        return result

    def _consume_publisher_close_result(self, publisher: Any) -> None:
        """Consume a retained successful Close only at the guarded commit."""

        task = self._publisher_close_task
        if task is not None and task.done() and self._publisher_close_target is publisher:
            self._publisher_close_task = None
            self._publisher_close_target = None

    def _allow_publisher_close_retry_after_failure(
        self,
        publisher: Any,
        result: OperationResult,
    ) -> None:
        """Preserve legacy explicit retry only for a completed Close exception.

        Timeout/unknown outcomes remain latched and are never duplicated.  A
        fresh ground-authorized release may retry a completed exception; this
        behavior still requires SDK idempotency qualification before hardware
        release is enabled.
        """

        task = self._publisher_close_task
        sdk = self._sdk
        if (
            result.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
            and task is not None
            and task.done()
            and self._publisher_close_target is publisher
            and sdk is not None
            and sdk.publisher_close_retry_idempotency_verified
        ):
            self._publisher_close_task = None
            self._publisher_close_target = None

    def _release_arbiter_epoch(self, ownership_epoch: int) -> OperationResult:
        """Convert an unexpected arbiter/OS-lock exception into a failure result."""

        try:
            return self._arbiter.release_low_level(ownership_epoch)
        except Exception as exc:
            try:
                arbiter_status = self._arbiter.status()
                lock_held = arbiter_status.local_single_instance_held
                lock_poisoned = arbiter_status.local_single_instance_poisoned
            except Exception:
                lock_held = False
                lock_poisoned = True
            return OperationResult.failure(
                "GO2_ARBITER_RELEASE_FAILED",
                (
                    "Arbiter release became ambiguous; quarantine and restart "
                    "this process before any new LowCmd owner is allowed "
                    f"({type(exc).__name__}: {exc})"
                ),
                {
                    "ownership_epoch": ownership_epoch,
                    "owner_lock_retained": lock_held,
                    "local_single_instance_poisoned": lock_poisoned,
                    "lock_release_ambiguous": True,
                    "ownership_exclusivity_lost": not lock_held,
                    "process_restart_required": True,
                },
            )

    def _release_uncommitted_arbiter_epoch(
        self,
        ownership_epoch: int,
        context: str,
    ) -> OperationResult:
        """Release a pre-publisher grant or retain it as a visible recovery epoch."""

        released = self._release_arbiter_epoch(ownership_epoch)
        if released.ok:
            return released
        commissioned_form = self._required_str(self._config.restore_mode_form, "restore_mode_form")
        commissioned_mode = self._required_str(self._config.restore_mode_name, "restore_mode_name")
        with self._guard:
            self._owner_epoch = ownership_epoch
            self._publisher = None
            self._writer_thread = None
            self._target = None
            self._high_level_restore_form = commissioned_form
            self._high_level_restore_mode = commissioned_mode
            self._release_rpc_attempted = False
            self._reset_handoff_tracking_locked()
            self._status = replace(
                self._status,
                timestamp=self._clock.monotonic(),
                ownership_state=LowCmdOwnershipState.FAULT,
                owner_epoch=ownership_epoch,
                healthy=False,
                publisher_active=False,
                writer_alive=False,
                watchdog_healthy=False,
                safe_hold_active=False,
                safe_hold_settled=False,
                high_level_released=False,
                high_level_restore_form=commissioned_form,
                high_level_restore_mode=commissioned_mode,
                network_exclusivity_verified=(
                    released.data.get("owner_lock_retained") is True
                    and released.data.get("local_single_instance_poisoned") is not True
                    and released.data.get("ownership_exclusivity_lost") is not True
                ),
                fault_reason=context + "; " + released.message,
            )
        return OperationResult.failure(
            released.code,
            context + "; exact arbiter epoch cleanup failed: " + released.message,
            {
                **dict(released.data),
                "ownership_epoch": ownership_epoch,
                "owner_epoch_retained": True,
                "publisher_closed": True,
                "release_rpc_attempted": False,
            },
        )

    async def _abort_pre_release_acquisition(
        self,
        epoch: int,
        publisher: Any,
        original_failure: OperationResult,
        *,
        timeout_s: float,
    ) -> OperationResult:
        """Undo an acquire that provably never crossed ``ReleaseMode``.

        A DataWriter can exist even when ``Init`` raises.  Therefore the host
        lock is released only after ``Close`` returns without exception.  If
        endpoint destruction is ambiguous, a strong reference and the exact
        epoch remain owned so a second process/owner cannot be admitted.
        """

        commissioned_form = self._required_str(self._config.restore_mode_form, "restore_mode_form")
        commissioned_mode = self._required_str(self._config.restore_mode_name, "restore_mode_name")
        cleanup_deadline = time.monotonic() + max(0.0, timeout_s)
        with self._guard:
            # Retain a deterministic recovery target even when failure occurred
            # before the first CheckMode response.  This is a commissioned pair,
            # not a claim that the service is currently active; handback below
            # still requires exact CheckMode confirmation.
            if self._high_level_restore_form is None:
                self._high_level_restore_form = commissioned_form
            if self._high_level_restore_mode is None:
                self._high_level_restore_mode = commissioned_mode
            self._status = replace(
                self._status,
                high_level_restore_form=self._high_level_restore_form,
                high_level_restore_mode=self._high_level_restore_mode,
            )

        close_result = await self._close_owned_publisher_with_deadline(
            publisher,
            real_deadline_s=cleanup_deadline,
        )
        if not close_result.ok:
            with self._guard:
                self._publisher = publisher
                self._owner_epoch = epoch
                self._release_rpc_attempted = False
                self._target = None
                self._last_accepted_sequence = None
                self._safe_hold_q = None
                self._last_commanded_q = None
                self._command_reference_generation = 0
                self._command_reference_q = None
                self._command_reference_write_s = None
                self._command_reference_ingress_cutoff = 0
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=True,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=False,
                    network_exclusivity_verified=True,
                    fault_reason=(
                        f"{original_failure.code}: {original_failure.message}; "
                        f"publisher cleanup is ambiguous: {close_result.message}"
                    ),
                )
            self._allow_publisher_close_retry_after_failure(publisher, close_result)
            return OperationResult.failure(
                close_result.code,
                "LowCmd acquisition failed before ReleaseMode, but its publisher "
                "could not be proven closed; the exact owner epoch is retained",
                {
                    **dict(close_result.data),
                    "original_failure_code": original_failure.code,
                    "original_failure_message": original_failure.message,
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "release_rpc_attempted": False,
                },
            )

        # Closing an endpoint is not enough: while acquisition was in progress
        # CheckMode may already have been empty, or another fault may have
        # removed the high-level service.  Confirm (or explicitly Select and
        # confirm) the commissioned pair before unlocking the host owner.
        with self._guard:
            if self._publisher is publisher:
                self._publisher = None
            self._consume_publisher_close_result(publisher)
            self._owner_epoch = epoch
            self._release_rpc_attempted = False
            self._status = replace(
                self._status,
                timestamp=self._clock.monotonic(),
                ownership_state=LowCmdOwnershipState.FAULT,
                owner_epoch=epoch,
                healthy=False,
                publisher_active=False,
                writer_alive=False,
                watchdog_healthy=False,
                safe_hold_active=False,
                safe_hold_settled=False,
                high_level_released=False,
                network_exclusivity_verified=True,
                fault_reason=original_failure.message,
            )
        cancelled = False
        handoff_task = asyncio.ensure_future(self._restore_high_level_mode(timeout_s))
        handoff_result, handoff_cancelled = await await_nonabandonable(handoff_task)
        cancelled = cancelled or handoff_cancelled
        if not handoff_result.ok:
            with self._guard:
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    network_exclusivity_verified=True,
                    fault_reason=(
                        f"{original_failure.code}: {original_failure.message}; "
                        "exact high-level recovery was not confirmed: " + handoff_result.message
                    ),
                )
            failure = OperationResult.failure(
                handoff_result.code,
                "Pre-ReleaseMode acquisition cleanup closed LowCmd, but exact "
                "high-level recovery was not confirmed; the epoch remains held",
                {
                    **dict(handoff_result.data),
                    "original_failure_code": original_failure.code,
                    "original_failure_message": original_failure.message,
                    "ownership_epoch": epoch,
                    "owner_lock_retained": True,
                    "publisher_closed": True,
                    "release_rpc_attempted": False,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure

        # Keep callback exclusion across arbiter unlock and local epoch clear,
        # exactly as in the normal handoff commit path.
        with self._guard:
            if self._owner_epoch != epoch or self._publisher is not None:
                release_result = OperationResult.failure(
                    "GO2_PRE_RELEASE_CLEANUP_STATE_CHANGED",
                    "Publisher or ownership changed before cleanup epoch release",
                    {"ownership_epoch": self._owner_epoch},
                )
            else:
                release_result = self._release_arbiter_epoch(epoch)
            if not release_result.ok:
                if self._publisher is publisher:
                    self._publisher = None
                self._owner_epoch = epoch
                self._release_rpc_attempted = False
                self._status = replace(
                    self._status,
                    timestamp=self._clock.monotonic(),
                    ownership_state=LowCmdOwnershipState.FAULT,
                    owner_epoch=epoch,
                    healthy=False,
                    publisher_active=False,
                    writer_alive=False,
                    watchdog_healthy=False,
                    safe_hold_active=False,
                    safe_hold_settled=False,
                    high_level_released=False,
                    network_exclusivity_verified=(
                        release_result.data.get("owner_lock_retained") is True
                        and release_result.data.get("local_single_instance_poisoned") is not True
                        and release_result.data.get("ownership_exclusivity_lost") is not True
                    ),
                    fault_reason=release_result.message,
                )
            else:
                self._clear_released_owner_locked()
        if not release_result.ok:
            failure = OperationResult.failure(
                release_result.code,
                "Publisher closed, but the exact local ownership epoch could not "
                "be released: " + release_result.message,
                {
                    **dict(release_result.data),
                    "original_failure_code": original_failure.code,
                    "ownership_epoch": epoch,
                    "owner_epoch_retained": True,
                    "publisher_closed": True,
                    "release_rpc_attempted": False,
                },
            )
            if cancelled:
                raise asyncio.CancelledError
            return failure
        failure = OperationResult.failure(
            original_failure.code,
            original_failure.message,
            {
                **dict(original_failure.data),
                "publisher_closed": True,
                "owner_lock_released": True,
                "release_rpc_attempted": False,
            },
        )
        if cancelled:
            raise asyncio.CancelledError
        return failure

    @staticmethod
    def _required_str(value: Optional[str], name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value

    @staticmethod
    def _required_float(value: Optional[float], name: str) -> float:
        parsed = _finite_optional(value)
        if parsed is None:
            raise ValueError(f"{name} must be finite")
        return parsed

    @classmethod
    def _required_float_tuple(
        cls, value: Optional[Sequence[float]], name: str
    ) -> Tuple[float, ...]:
        result = cls._optional_float_tuple(value)
        if result is None:
            raise ValueError(f"{name} must contain exactly 12 finite numbers")
        return result

    @staticmethod
    def _optional_float_tuple(
        value: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, ...]]:
        if value is None or isinstance(value, (str, bytes)):
            return None
        try:
            raw = tuple(value)
        except TypeError:
            return None
        if len(raw) != _JOINT_COUNT:
            return None
        parsed = tuple(_finite_optional(item) for item in raw)
        if any(item is None for item in parsed):
            return None
        return cast(Tuple[float, ...], parsed)

    @classmethod
    def _required_int_tuple(cls, value: Optional[Sequence[int]], name: str) -> Tuple[int, ...]:
        result = cls._optional_int_tuple(value)
        if result is None:
            raise ValueError(f"{name} must contain exactly 12 integers")
        return result

    @staticmethod
    def _optional_int_tuple(
        value: Optional[Sequence[int]],
    ) -> Optional[Tuple[int, ...]]:
        if value is None or isinstance(value, (str, bytes)):
            return None
        try:
            raw = tuple(value)
        except TypeError:
            return None
        if len(raw) != _JOINT_COUNT or any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw
        ):
            return None
        return raw

    @classmethod
    def _required_str_tuple(cls, value: Optional[Sequence[str]], name: str) -> Tuple[str, ...]:
        result = cls._optional_str_tuple(value)
        if result is None:
            raise ValueError(f"{name} must contain exactly 12 nonempty strings")
        return result

    @staticmethod
    def _optional_str_tuple(
        value: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, ...]]:
        if value is None or isinstance(value, (str, bytes)):
            return None
        try:
            raw = tuple(value)
        except TypeError:
            return None
        if len(raw) != _JOINT_COUNT or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            return None
        return raw

    async def connect(self) -> OperationResult:
        return await self._run_lifecycle_operation("connect", self._connect_unlocked)

    def _abandon_subscription_attempt(
        self,
        subscription_generation: int,
        subscriber: Any,
        reason: str,
    ) -> None:
        """Invalidate and close one observe-only reader without touching a newer one."""

        with self._guard:
            if self._subscription_generation == subscription_generation:
                with self._low_state_ingress_condition:
                    self._subscription_generation += 1
                    self._low_state_ingress_condition.notify_all()
            if self._subscriber is subscriber:
                self._subscriber = None
                self._publisher = None
                self._motion_switcher = None
                self._crc = None
                self._sdk = None
            self._connected = False
            # Wake a worker blocked in Event.wait so cancellation cleanup does
            # not leave an executor job alive until the full connect timeout.
            self._first_state.set()
        self._close_transport(subscriber=subscriber, publisher=None)
        self._set_disconnected(reason)

    async def _connect_unlocked(self) -> OperationResult:
        """Initialize an observe-only LowState subscription; never publish."""

        with self._guard:
            if self._connected:
                now = self._clock.monotonic()
                writer_present = self._writer_thread is not None
                feedback_fault = (
                    self._feedback_limit_failure_locked()
                    if self._actuation_config_ready
                    else self._observation_feedback_failure(self._status.motors)
                )
                if (
                    self._status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
                    and self._publisher is None
                    and not writer_present
                    and self._low_state_is_fresh_locked(now)
                    and feedback_fault is None
                    and self._status.fault_reason is None
                ):
                    return OperationResult.success(
                        "Go2 LowState transport is already connected and healthy in observe-only mode",
                        {
                            "lowcmd_actuation_ready": self._actuation_config_ready,
                            "low_state_fresh": True,
                            "publisher_active": False,
                            "writer_present": False,
                        },
                    )
                return OperationResult.failure(
                    "GO2_LOW_STATE_EXISTING_CONNECTION_UNHEALTHY",
                    "Existing Go2 LowState connection is not a fresh, fault-free, read-only transport",
                    {
                        "low_state_fresh": self._low_state_is_fresh_locked(now),
                        "feedback_fault": feedback_fault or self._status.fault_reason,
                        "publisher_active": self._publisher is not None,
                        "writer_present": writer_present,
                    },
                )
            if not self._config.observation_enabled:
                return OperationResult.failure(
                    "GO2_LOW_STATE_OBSERVATION_DISABLED",
                    "Enable go2.low_level.observe_only_enabled (or the fully commissioned LowCmd path) before connecting",
                )
            with self._low_state_ingress_condition:
                self._subscription_generation += 1
                subscription_generation = self._subscription_generation
                self._low_state_ingress_condition.notify_all()
            self._first_state.clear()
            self._valid_motors = ()
            self._envelope_motors = ()
            self._envelope_ingress_token = 0
            self._valid_low_state_sequence = 0
            self._valid_low_state_ingress_token = 0
            self._last_foot_force_source_tick = None
            self._safe_hold_feedback_sequence_required = 0
            self._safe_hold_feedback_ingress_token_required = 0
            self._high_level_restore_form = None
            self._high_level_restore_mode = None
            self._status = replace(
                self._status,
                low_state_timestamp=0.0,
                low_state_age_s=math.inf,
                motors=(),
                foot_force_feedback=Go2FootForceFeedback(),
                high_level_restore_form=None,
                high_level_restore_mode=None,
            )
        readiness = self.observation_readiness()
        if not readiness.ok:
            self._set_fault(readiness.message)
            return readiness
        try:
            sdk = self._injected_sdk or Go2SdkBindings.load()
        except Exception as exc:
            self._set_disconnected(f"Unitree SDK2 unavailable: {exc}")
            return OperationResult.failure("GO2_SDK_UNAVAILABLE", str(exc))
        channel_result = self._arbiter.initialize_channel_factory(
            sdk.channel_factory_initialize,
            self._go2_config.domain_id,
            self._go2_config.network_interface,
        )
        if not channel_result.ok:
            self._set_disconnected(channel_result.message)
            return channel_result
        motion_switcher: Optional[Any] = None
        crc: Optional[Any] = None
        try:
            subscriber = sdk.subscriber_factory(
                self._required_str(self._config.low_state_topic, "low_state_topic"),
                sdk.low_state_type,
            )
            # A read-only connection must not initialize control-transfer or
            # command-integrity helpers.  They are prepared only when the full
            # actuation configuration has independently passed validation.
            if self._actuation_config_ready:
                motion_switcher = sdk.motion_switcher_factory()
                set_timeout = getattr(motion_switcher, "SetTimeout", None)
                if callable(set_timeout):
                    set_timeout(
                        self._required_float(
                            self._config.acquire_timeout_s,
                            "acquire_timeout_s",
                        )
                    )
                init_motion = getattr(motion_switcher, "Init", None)
                if callable(init_motion):
                    init_motion()
                crc = sdk.crc_factory()

            def receive_low_state(
                message: Any,
                generation: int = subscription_generation,
            ) -> None:
                self._on_low_state(
                    message,
                    subscription_generation=generation,
                )

            subscriber.Init(receive_low_state, 10)
        except Exception as exc:
            failed_subscriber = locals().get("subscriber")
            self._abandon_subscription_attempt(
                subscription_generation,
                failed_subscriber,
                f"Unitree low-level initialization failed: {exc}",
            )
            return OperationResult.failure("GO2_LOW_LEVEL_CONNECT_FAILED", str(exc))
        with self._guard:
            self._sdk = sdk
            self._subscriber = subscriber
            self._publisher = None
            self._motion_switcher = motion_switcher
            self._crc = crc
        wait_task = asyncio.ensure_future(
            run_blocking(
                self._first_state.wait,
                max(
                    self._go2_config.status_timeout_s,
                    self._required_float(
                        self._config.low_state_max_age_s,
                        "low_state_max_age_s",
                    )
                    * 5.0,
                ),
            )
        )
        try:
            received = await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            self._abandon_subscription_attempt(
                subscription_generation,
                subscriber,
                "LowState connection was cancelled before the first valid frame",
            )
            await await_nonabandonable(wait_task)
            raise
        except Exception as exc:
            self._abandon_subscription_attempt(
                subscription_generation,
                subscriber,
                f"LowState first-frame wait failed: {type(exc).__name__}: {exc}",
            )
            return OperationResult.failure(
                "GO2_LOW_STATE_WAIT_FAILED",
                f"Waiting for the first Go2 LowState failed: {exc}",
            )
        if not received:
            self._abandon_subscription_attempt(
                subscription_generation,
                subscriber,
                "No LowState was received before the connect timeout",
            )
            return OperationResult.failure(
                "GO2_LOW_STATE_TIMEOUT",
                "No Go2 LowState received; check interface, domain, topic, and robot power",
            )
        now = self._clock.monotonic()
        commit_failure: Optional[OperationResult] = None
        with self._guard:
            if (
                self._subscription_generation != subscription_generation
                or self._subscriber is not subscriber
            ):
                return OperationResult.failure(
                    "GO2_LOW_STATE_SUBSCRIPTION_REPLACED",
                    "LowState subscription changed before connect could commit",
                )
            feedback_fault = (
                self._feedback_limit_failure_locked()
                if self._actuation_config_ready
                else self._observation_feedback_failure(self._status.motors)
            )
            fresh = self._low_state_is_fresh_locked(now)
            writer_present = self._writer_thread is not None
            if self._publisher is not None or writer_present:
                commit_failure = OperationResult.failure(
                    "GO2_OBSERVE_ONLY_WRITE_ENDPOINT_PRESENT",
                    "Observe-only connect found an unexpected LowCmd publisher or writer",
                    {
                        "publisher_active": self._publisher is not None,
                        "writer_present": writer_present,
                    },
                )
            elif feedback_fault is not None or self._status.fault_reason is not None:
                commit_failure = OperationResult.failure(
                    "GO2_LOW_STATE_FEEDBACK_FAULT",
                    "Latest LowState failed observe-only feedback validation",
                    {
                        "feedback_fault": feedback_fault or self._status.fault_reason,
                        "low_state_fresh": fresh,
                        "publisher_active": False,
                        "writer_present": False,
                    },
                )
            elif not fresh:
                commit_failure = OperationResult.failure(
                    "GO2_LOW_STATE_STALE_AT_CONNECT",
                    "Latest valid LowState became stale before observe-only connect committed",
                    {
                        "low_state_age_s": (
                            max(0.0, now - self._status.low_state_timestamp)
                            if self._status.low_state_timestamp > 0.0
                            else math.inf
                        ),
                        "publisher_active": False,
                        "writer_present": False,
                    },
                )
            else:
                self._connected = True
                self._status = replace(
                    self._status,
                    timestamp=now,
                    connected=True,
                    ownership_state=LowCmdOwnershipState.OBSERVE_ONLY,
                    healthy=True,
                    publisher_active=False,
                    writer_alive=False,
                    mapping_hash_verified=True,
                    network_exclusivity_verified=False,
                    active_mapping_hash=self._required_str(
                        self._config.mapping_hash,
                        "mapping_hash",
                    ),
                    fault_reason=None,
                )
        if commit_failure is not None:
            self._abandon_subscription_attempt(
                subscription_generation,
                subscriber,
                commit_failure.message,
            )
            return commit_failure
        return OperationResult.success(
            "Go2 LowState is fresh; transport is observe-only and no LowCmd publisher exists",
            {
                "lowcmd_actuation_ready": self._actuation_config_ready,
                "low_state_fresh": True,
                "publisher_active": False,
                "writer_present": False,
            },
        )

    def status(self) -> Go2LowLevelStatus:
        """Return current owner health with age derived from monotonic time."""

        now = self._clock.monotonic()
        period = _finite_optional(self._config.send_period_s)
        jitter = _finite_optional(self._config.maximum_jitter_s)
        maximum_write_age = (
            period + jitter
            if period is not None and period > 0.0 and jitter is not None and jitter >= 0.0
            else 0.01
        )
        # Write is deliberately inside _guard to linearize revoke/fault against
        # the outgoing sample. Never let that make health reporting block
        # forever: a stuck DDS call must be externally observable as unhealthy.
        acquired = self._guard.acquire(timeout=max(0.001, maximum_write_age))
        if not acquired:
            status = self._status
            with self._writer_health_guard:
                write_started = self._writer_write_started_s
                heartbeat = self._writer_heartbeat_s
            writer_alive = self._writer_thread is not None and self._writer_thread.is_alive()
            arbiter_status = self._arbiter.status()
            low_age = (
                max(0.0, now - status.low_state_timestamp)
                if status.low_state_timestamp > 0.0
                else math.inf
            )
            detail = f"DDS Write/owner guard has not returned for at least {maximum_write_age:.6f}s"
            if write_started is not None:
                detail += f" (Write age {max(0.0, now - write_started):.6f}s)"
            return replace(
                status,
                timestamp=now,
                healthy=False,
                low_state_age_s=low_age,
                publisher_active=status.publisher_active,
                writer_alive=writer_alive,
                last_write_timestamp=heartbeat,
                watchdog_healthy=False,
                network_exclusivity_verified=(
                    arbiter_status.network_exclusivity_verified
                    and arbiter_status.local_single_instance_held
                    and arbiter_status.low_level_epoch == status.owner_epoch
                ),
                continuous_owner_monitoring_active=(
                    arbiter_status.continuous_network_monitoring_active
                ),
                independent_watchdog_active=False,
                actuator_application_ack_available=False,
                fault_reason=detail,
            )
        try:
            now = self._clock.monotonic()
            status = self._status
            low_age = (
                max(0.0, now - status.low_state_timestamp)
                if status.low_state_timestamp > 0.0
                else math.inf
            )
            target_age = (
                max(0.0, now - self._target.timestamp_s) if self._target is not None else None
            )
            state = status.ownership_state
            active = state in {
                LowCmdOwnershipState.HOLDING,
                LowCmdOwnershipState.MPC_ACTIVE,
                LowCmdOwnershipState.SAFE_HOLD,
            }
            writer_present = self._writer_thread is not None
            writer_alive = self._writer_thread is not None and self._writer_thread.is_alive()
            observe_only_has_no_writer = self._publisher is None and not writer_present
            with self._writer_health_guard:
                writer_started = self._writer_started_s
                write_started = self._writer_write_started_s
                heartbeat = self._writer_heartbeat_s
            arbiter_status = self._arbiter.status()
            maximum_age = _finite_optional(self._config.low_state_max_age_s)
            fresh = (
                maximum_age is not None
                and maximum_age > 0.0
                and self._low_state_is_fresh_locked(now)
            )
            owner_ok = (
                writer_alive
                and arbiter_status.local_single_instance_held
                and arbiter_status.low_level_epoch == self._owner_epoch
                and status.high_level_released
                and arbiter_status.network_exclusivity_verified
                and status.mapping_hash_verified
                and status.watchdog_healthy
            )
            write_reference = heartbeat if heartbeat is not None else writer_started
            write_fresh = (
                write_reference is not None
                and 0.0 <= now - write_reference <= maximum_write_age
                and (write_started is None or 0.0 <= now - write_started <= maximum_write_age)
            )
            if active:
                owner_ok = owner_ok and write_fresh
            transport_ok = self._connected and fresh and status.fault_reason is None
            if active:
                healthy = transport_ok and owner_ok
            elif state is LowCmdOwnershipState.OBSERVE_ONLY:
                healthy = (
                    transport_ok and status.mapping_hash_verified and observe_only_has_no_writer
                )
            else:
                healthy = False
            fault_reason = status.fault_reason
            if active and not write_fresh and fault_reason is None:
                fault_reason = "LowCmd writer heartbeat exceeded send_period_s + maximum_jitter_s"
            if (
                state is LowCmdOwnershipState.OBSERVE_ONLY
                and not observe_only_has_no_writer
                and fault_reason is None
            ):
                fault_reason = (
                    "Observe-only state unexpectedly retains a LowCmd publisher or writer"
                )
            publisher_active = self._publisher is not None
            return replace(
                status,
                timestamp=now,
                connected=self._connected,
                owner_epoch=self._owner_epoch,
                healthy=healthy,
                low_state_age_s=low_age,
                target_age_s=target_age,
                publisher_active=publisher_active,
                writer_enqueued_target_sequence=(
                    status.writer_enqueued_target_sequence if publisher_active else None
                ),
                actuator_applied_target_sequence=(
                    status.actuator_applied_target_sequence if publisher_active else None
                ),
                writer_enqueue_ack_available=(
                    status.writer_enqueue_ack_available and publisher_active
                ),
                writer_alive=writer_alive,
                last_write_timestamp=heartbeat,
                watchdog_healthy=(status.watchdog_healthy and (not active or write_fresh)),
                network_exclusivity_verified=(
                    arbiter_status.network_exclusivity_verified
                    if self._owner_epoch != 0
                    and arbiter_status.low_level_epoch == self._owner_epoch
                    else False
                ),
                continuous_owner_monitoring_active=(
                    arbiter_status.continuous_network_monitoring_active
                    if self._owner_epoch != 0
                    and arbiter_status.low_level_epoch == self._owner_epoch
                    else False
                ),
                independent_watchdog_active=False,
                actuator_application_ack_available=False,
                fault_reason=fault_reason,
            )
        finally:
            self._guard.release()


__all__ = [
    "Go2SdkBindings",
    "UnitreeGo2LowLevelSdkBridge",
    "compute_go2_mapping_hash",
]
