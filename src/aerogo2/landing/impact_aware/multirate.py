"""Fail-closed multi-rate execution seams for impact-aware landing.

The numerical :mod:`coordinator` remains a useful serial/offline reference,
but it must not be used as a hardware scheduler.  This module separates the
four timing domains used by the physical controller:

* an asynchronous MPC worker produces one short-lived force/residual policy;
* a high-rate leg loop owns contact detection and admittance/IK state;
* the flight controller owns its baseline-plus-residual fast loop; and
* an independent supervisor invalidates policies and requests both fallbacks.

Nothing in this module removes the production hardware gate.  In particular,
the current Go2 bridge acknowledges replacement of its target mailbox, not a
causal motor-side execution, and the repository still lacks a calibrated,
time-aligned LowState force/estimator frame and a verified cross-device commit.

中文说明：这里把“计算频率”和“执行频率”明确分离。异步低频 MPC 只发布带版本、
时间戳和 TTL 的最新策略；Go2 高频环在每个新鲜 LowState 上完成接触检测、导纳、
IK 和关节限幅；飞控在自己的高速环中保持姿态基线并只叠加尚未过期的残差；独立
监督环检查 ACK、数据年龄、owner 健康和人工撤权。共享数据通过不可变快照和 epoch
防止旧会话命令复活。此模块只定义调度和失效语义，不宣称 Python 线程是硬实时。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import time
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from aerogo2.common.async_utils import await_nonabandonable
from aerogo2.common.models import Go2LowLevelStatus, LowCmdOwnershipState
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.admittance import (
    LegAdmittanceController,
    LegAdmittanceOutput,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetection,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.coordinator import LandingInputFreshness
from aerogo2.landing.impact_aware.integration import (
    FlightControllerResidualSinkStatus,
    FlightControllerResidualState,
    FlightControllerRotorResidualCommand,
    Go2JointPositionCommand,
)
from aerogo2.landing.impact_aware.math_utils import require_rotation_matrix
from aerogo2.landing.impact_aware.nlp import (
    ImpactAwareMPCProblem,
    ImpactAwareNLP,
    MPCSolveResult,
    SLSQPSettings,
    reconstruct_transport_target,
)
from aerogo2.landing.impact_aware.normal_admittance import ForceObservationMode
from aerogo2.landing.impact_aware.rotor import (
    evaluate_rotor_constraints,
    first_order_thrust_rate,
)
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    validate_four_foot_leg_order,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
    ContactMask: TypeAlias = Tuple[bool, bool, bool, bool]
else:
    FloatArray = NDArray[np.float64]
    ContactMask = Tuple[bool, bool, bool, bool]
_SHA256_PREFIX = "sha256:"
_UINT32_MAX = 0xFFFFFFFF
_UINT32_SERIAL_HALF_RANGE = 0x80000000
_ACTIVE_LOW_CMD_STATES = frozenset({LowCmdOwnershipState.HOLDING, LowCmdOwnershipState.MPC_ACTIVE})
_HARDWARE_ACTUATION_BLOCKERS = (
    "cross_device_activation_transaction_unverified",
    "go2_motor_side_application_ack_unavailable",
    "continuous_dds_owner_monitor_unavailable",
    "independent_supervisor_watchdog_unverified",
    "production_atomic_force_sample_unavailable",
    "calibrated_normal_force_pipeline_unverified",
    "normal_force_tracking_error_unvalidated",
)


class MultiRateActuationMode(Enum):
    """Whether multi-rate outputs are diagnostic, simulated, or physical."""

    SHADOW = "shadow"
    SIMULATION = "simulation"
    HARDWARE = "hardware"


def _finite_real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _strict_integer(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _sha256_identity(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    digest = value[len(_SHA256_PREFIX) :] if value.startswith(_SHA256_PREFIX) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must use sha256:<64 lowercase hexadecimal digits>")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_tuple(name: str, value: object, length: int) -> Tuple[float, ...]:
    try:
        raw: Tuple[object, ...] = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(_finite_real(f"{name}[{index}]", item) for index, item in enumerate(raw))
    return result


def _fresh_past_time(value: object, *, now: float, maximum_age_s: float) -> bool:
    """Return true only for a finite monotonic timestamp inside a closed age bound."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    timestamp = float(value)
    return bool(math.isfinite(timestamp) and timestamp <= now and now - timestamp <= maximum_age_s)


def _nonnegative_tuple(name: str, value: object, length: int) -> Tuple[float, ...]:
    result = _finite_tuple(name, value, length)
    if any(item < 0.0 for item in result):
        raise ValueError(f"{name} cannot contain negative values")
    return result


async def _drain_task(task: asyncio.Future[Any]) -> None:
    """Observe a child task through its terminal state during safety cleanup."""

    try:
        await task
    except (Exception, asyncio.CancelledError):
        # The caller has already latched the fault.  Cleanup must observe the
        # child so an exception/pending coroutine cannot escape during teardown.
        return


def _boolean4(name: str, value: object) -> ContactMask:
    try:
        raw: Tuple[object, ...] = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw) != 4 or any(type(item) is not bool for item in raw):
        raise TypeError(f"{name} must contain exactly four booleans")
    return cast(ContactMask, raw)


def _readonly_array(name: str, value: object, shape: Tuple[int, ...]) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MultiRateExecutionConfig:
    """Required timing contract; every value must come from measurement.

    These are deliberately constructor-required fields.  They are not robot
    defaults and are not part of the current schema-v3 hardware profile.

    中文：这些量共同形成时间预算，不是彼此独立的经验参数。例如求解预算加提交
    余量必须小于策略 TTL；安全环最坏响应必须早于 LowCmd/策略过期；最大 source
    skew 禁止把不同物理时刻的姿态、足力和运动学拼成一个伪同步状态。
    """

    high_rate_period_s: float
    high_rate_max_jitter_s: float
    high_rate_max_gap_s: float
    lowcmd_target_ttl_s: float
    lowcmd_submit_reserve_s: float
    low_state_max_age_s: float
    contact_force_max_age_s: float
    state_estimate_max_age_s: float
    kinematics_max_age_s: float
    foot_plan_max_age_s: float
    fc_baseline_max_age_s: float
    maximum_source_skew_s: float
    mpc_release_period_s: float
    policy_ttl_s: float
    solver_budget_s: float
    solver_commit_reserve_s: float
    result_audit_budget_s: float
    worker_heartbeat_timeout_s: float
    high_rate_heartbeat_timeout_s: float
    safety_period_s: float
    safety_max_jitter_s: float
    contact_replan_deadline_s: float
    fc_status_max_age_s: float
    mpc_equality_tolerance: float
    mpc_inequality_tolerance: float
    force_zero_tolerance_n: float
    force_constraint_tolerance_n: float
    rotor_thrust_tolerance_n: float
    rotor_rate_tolerance_n_per_s: float
    initial_joint_alignment_tolerance_rad: float

    def __post_init__(self) -> None:
        positive_names = (
            "high_rate_period_s",
            "high_rate_max_gap_s",
            "lowcmd_target_ttl_s",
            "lowcmd_submit_reserve_s",
            "low_state_max_age_s",
            "contact_force_max_age_s",
            "state_estimate_max_age_s",
            "kinematics_max_age_s",
            "foot_plan_max_age_s",
            "fc_baseline_max_age_s",
            "maximum_source_skew_s",
            "mpc_release_period_s",
            "policy_ttl_s",
            "solver_budget_s",
            "solver_commit_reserve_s",
            "result_audit_budget_s",
            "worker_heartbeat_timeout_s",
            "high_rate_heartbeat_timeout_s",
            "safety_period_s",
            "contact_replan_deadline_s",
            "fc_status_max_age_s",
        )
        for name in positive_names:
            object.__setattr__(self, name, _finite_real(name, getattr(self, name), positive=True))
        for name in (
            "high_rate_max_jitter_s",
            "safety_max_jitter_s",
            "mpc_equality_tolerance",
            "mpc_inequality_tolerance",
            "force_zero_tolerance_n",
            "force_constraint_tolerance_n",
            "rotor_thrust_tolerance_n",
            "rotor_rate_tolerance_n_per_s",
            "initial_joint_alignment_tolerance_rad",
        ):
            value = _finite_real(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.high_rate_max_jitter_s >= self.high_rate_period_s:
            raise ValueError("high_rate_max_jitter_s must be below high_rate_period_s")
        if self.high_rate_max_gap_s < self.high_rate_period_s + self.high_rate_max_jitter_s:
            raise ValueError("high_rate_max_gap_s must cover the period plus allowed jitter")
        if self.high_rate_period_s >= self.mpc_release_period_s:
            raise ValueError("the leg loop must be faster than the MPC release period")
        if self.solver_budget_s + self.solver_commit_reserve_s >= self.policy_ttl_s:
            raise ValueError("solver budget plus commit reserve must be below policy_ttl_s")
        if self.result_audit_budget_s >= self.solver_commit_reserve_s:
            raise ValueError("result audit budget must be below the solver commit reserve")
        if self.contact_replan_deadline_s > self.lowcmd_target_ttl_s:
            raise ValueError("contact_replan_deadline_s cannot exceed the LowCmd target TTL")
        if self.lowcmd_submit_reserve_s >= min(
            self.lowcmd_target_ttl_s,
            self.contact_replan_deadline_s,
        ):
            raise ValueError("LowCmd submit reserve must fit inside both leg command leases")
        if self.safety_period_s + self.safety_max_jitter_s >= min(
            self.lowcmd_target_ttl_s,
            self.policy_ttl_s,
        ):
            raise ValueError("the safety response period must fit inside both command leases")


@dataclass(frozen=True)
class SolverQualification:
    """Target-specific evidence required before a backend may actuate hardware.

    ``maximum_observed_solve_time_s`` is intentionally named an observation,
    not a formal WCET proof.  Approval additionally requires a reviewed
    evidence digest tied to one target/build/problem envelope.
    """

    backend_name: str
    target_identity: str
    runtime_build_hash: str
    problem_envelope_hash: str
    evidence_hash: str
    maximum_observed_solve_time_s: Optional[float]
    maximum_observed_jitter_s: Optional[float]
    sample_count: int
    approved_for_hardware: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_name", _nonempty_string("backend_name", self.backend_name)
        )
        object.__setattr__(
            self,
            "target_identity",
            _nonempty_string("target_identity", self.target_identity),
        )
        for name in ("runtime_build_hash", "problem_envelope_hash", "evidence_hash"):
            object.__setattr__(self, name, _sha256_identity(name, getattr(self, name)))
        for name in ("maximum_observed_solve_time_s", "maximum_observed_jitter_s"):
            raw = getattr(self, name)
            if raw is not None:
                value = _finite_real(name, raw)
                if value < 0.0:
                    raise ValueError(f"{name} cannot be negative")
                object.__setattr__(self, name, value)
        _strict_integer("sample_count", self.sample_count)
        if type(self.approved_for_hardware) is not bool:
            raise TypeError("approved_for_hardware must be a bool")
        if self.approved_for_hardware and (
            self.maximum_observed_solve_time_s is None
            or self.maximum_observed_jitter_s is None
            or self.sample_count < 1
        ):
            raise ValueError("hardware approval requires timing observations and samples")

    def authorizes(
        self,
        *,
        target_identity: str,
        runtime_build_hash: str,
        problem_envelope_hash: str,
        evidence_hash: str,
        solver_budget_s: float,
        commit_reserve_s: float,
    ) -> bool:
        observed = self.maximum_observed_solve_time_s
        jitter = self.maximum_observed_jitter_s
        return bool(
            self.approved_for_hardware
            and self.target_identity == target_identity
            and self.runtime_build_hash == runtime_build_hash
            and self.problem_envelope_hash == problem_envelope_hash
            and self.evidence_hash == evidence_hash
            and observed is not None
            and jitter is not None
            and observed + jitter + commit_reserve_s < solver_budget_s
        )


class MPCSolverBackend(Protocol):
    """Synchronous, transport-free numerical backend used off the event loop."""

    @property
    def qualification(self) -> SolverQualification: ...

    def solve(
        self,
        problem: ImpactAwareMPCProblem,
        *,
        timeout_s: float,
    ) -> MPCSolveResult: ...


class SLSQPReferenceSolver:
    """SciPy SLSQP adapter that is permanently reference-only by default."""

    def __init__(self, settings: SLSQPSettings) -> None:
        if not isinstance(settings, SLSQPSettings):
            raise TypeError("settings must be SLSQPSettings")
        self._settings = settings
        digest = hashlib.sha256(repr(settings).encode("utf-8")).hexdigest()
        self._qualification = SolverQualification(
            backend_name="scipy-slsqp-reference",
            target_identity="REFERENCE_ONLY",
            runtime_build_hash=f"sha256:{digest}",
            problem_envelope_hash=f"sha256:{'0' * 64}",
            evidence_hash=f"sha256:{'0' * 64}",
            maximum_observed_solve_time_s=None,
            maximum_observed_jitter_s=None,
            sample_count=0,
            approved_for_hardware=False,
        )

    @property
    def qualification(self) -> SolverQualification:
        return self._qualification

    def solve(
        self,
        problem: ImpactAwareMPCProblem,
        *,
        timeout_s: float,
    ) -> MPCSolveResult:
        timeout = _finite_real("timeout_s", timeout_s, positive=True)
        settings = replace(self._settings, timeout_s=min(timeout, self._settings.timeout_s))
        return ImpactAwareNLP(problem).solve(settings)


@dataclass(frozen=True)
class PolicyDomain:
    """Identity fence shared by snapshots, policies and high-rate consumers.

    中文：会话 epoch、LowCmd ownership epoch、接触 epoch、配置/模型哈希任一变化，
    旧快照与旧策略都必须失效，防止异步求解的迟到结果跨会话重新生效。
    """

    landing_session_epoch: int
    ownership_epoch: int
    actuation_mode: MultiRateActuationMode
    contact_epoch: int
    invalidation_generation: int
    contacts: ContactMask
    leg_order: Tuple[str, str, str, str]
    configuration_hash: str
    model_hash: str

    def __post_init__(self) -> None:
        for name in (
            "landing_session_epoch",
            "ownership_epoch",
        ):
            object.__setattr__(
                self, name, _strict_integer(name, getattr(self, name), positive=True)
            )
        for name in ("contact_epoch", "invalidation_generation"):
            object.__setattr__(self, name, _strict_integer(name, getattr(self, name)))
        if not isinstance(self.actuation_mode, MultiRateActuationMode):
            raise TypeError("actuation_mode must be a MultiRateActuationMode")
        object.__setattr__(self, "contacts", _boolean4("contacts", self.contacts))
        leg_order = validate_four_foot_leg_order(self.leg_order, name="leg_order")
        if leg_order != GO2_SDK_LEG_ORDER:
            raise ValueError(
                "the current Go2 LowCmd mapping requires leg_order [FR, FL, RR, RL]"
            )
        object.__setattr__(self, "leg_order", leg_order)
        for name in ("configuration_hash", "model_hash"):
            object.__setattr__(self, name, _sha256_identity(name, getattr(self, name)))


@dataclass(frozen=True)
class MPCSnapshot:
    """Immutable state/problem snapshot released to the asynchronous worker."""

    domain: PolicyDomain
    snapshot_sequence: int
    timestamp_s: float
    valid_until_s: float
    freshness: LandingInputFreshness
    problem: ImpactAwareMPCProblem
    flight_controller_session_id: int
    flight_controller_target_tick: int
    flight_controller_baseline_version: int
    flight_controller_baseline_timestamp_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.domain, PolicyDomain):
            raise TypeError("domain must be a PolicyDomain")
        object.__setattr__(
            self,
            "snapshot_sequence",
            _strict_integer("snapshot_sequence", self.snapshot_sequence),
        )
        for name in ("timestamp_s", "valid_until_s", "flight_controller_baseline_timestamp_s"):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if self.timestamp_s < 0.0:
            raise ValueError("timestamp_s cannot be negative")
        if self.valid_until_s <= self.timestamp_s:
            raise ValueError("valid_until_s must be later than timestamp_s")
        for name in (
            "flight_controller_session_id",
            "flight_controller_target_tick",
            "flight_controller_baseline_version",
        ):
            object.__setattr__(
                self, name, _strict_integer(name, getattr(self, name), positive=True)
            )
        if not isinstance(self.freshness, LandingInputFreshness):
            raise TypeError("freshness must be LandingInputFreshness")
        freshness_error = self.freshness.failure_reason(self.timestamp_s)
        if freshness_error is not None:
            raise ValueError(f"snapshot source is unhealthy: {freshness_error}")
        oldest_source_timestamp = min(
            self.freshness.state_estimate_timestamp_s,
            self.freshness.contact_forces_timestamp_s,
            self.freshness.kinematics_timestamp_s,
            self.freshness.foot_plan_timestamp_s,
            self.freshness.flight_controller_baseline_timestamp_s,
        )
        source_deadline = oldest_source_timestamp + self.freshness.maximum_source_age_s
        if self.valid_until_s > source_deadline and not math.isclose(
            self.valid_until_s,
            source_deadline,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("valid_until_s cannot outlive the oldest source freshness deadline")
        if not isinstance(self.problem, ImpactAwareMPCProblem):
            raise TypeError("problem must be an ImpactAwareMPCProblem")
        planned_contacts = tuple(bool(value) for value in self.problem.contact_schedule[0])
        if planned_contacts != self.domain.contacts:
            raise ValueError("problem.contact_schedule[0] must match the contact generation")
        if self.problem.foot_leg_order != self.domain.leg_order:
            raise ValueError("problem.foot_leg_order must match the policy-domain leg order")
        plan = self.problem.rotor_execution_plan
        if plan is None:
            raise ValueError("multi-rate MPC requires an explicit rotor_execution_plan")
        if self.flight_controller_baseline_timestamp_s != (
            self.freshness.flight_controller_baseline_timestamp_s
        ):
            raise ValueError("FC baseline timestamps must agree")

    def is_fresh(self, now_s: float) -> bool:
        try:
            now = _finite_real("now_s", now_s)
        except (TypeError, ValueError):
            return False
        return self.timestamp_s <= now < self.valid_until_s and (
            self.freshness.failure_reason(now) is None
        )


@dataclass(frozen=True)
class MPCPolicy:
    """One activated FC residual and its matching desired foot-force lease."""

    domain: PolicyDomain
    policy_sequence: int
    solution_sequence: int
    source_timestamp_s: float
    solve_started_s: float
    solve_completed_s: float
    activated_s: float
    valid_until_s: float
    desired_contact_forces_world_n: Tuple[float, ...]
    rotor_command: FlightControllerRotorResidualCommand
    solver_status: str
    solver_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.domain, PolicyDomain):
            raise TypeError("domain must be a PolicyDomain")
        for name in ("policy_sequence", "solution_sequence"):
            object.__setattr__(self, name, _strict_integer(name, getattr(self, name)))
        for name in (
            "source_timestamp_s",
            "solve_started_s",
            "solve_completed_s",
            "activated_s",
            "valid_until_s",
            "solver_time_s",
        ):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if not (
            self.source_timestamp_s
            <= self.solve_started_s
            <= self.solve_completed_s
            <= self.activated_s
            < self.valid_until_s
        ):
            raise ValueError("policy timestamps must be monotonic and precede valid_until_s")
        if self.solver_time_s < 0.0:
            raise ValueError("solver_time_s cannot be negative")
        object.__setattr__(
            self,
            "desired_contact_forces_world_n",
            _finite_tuple(
                "desired_contact_forces_world_n",
                self.desired_contact_forces_world_n,
                12,
            ),
        )
        if not isinstance(self.rotor_command, FlightControllerRotorResidualCommand):
            raise TypeError("rotor_command must be a FlightControllerRotorResidualCommand")
        if self.rotor_command.sequence != self.policy_sequence:
            raise ValueError("rotor command and policy must share policy_sequence")
        if self.rotor_command.timestamp_s != self.source_timestamp_s:
            raise ValueError("rotor command timestamp must remain anchored to the snapshot")
        if self.rotor_command.valid_until_s != self.valid_until_s:
            raise ValueError("rotor command and policy must share the source-derived deadline")
        object.__setattr__(
            self, "solver_status", _nonempty_string("solver_status", self.solver_status)
        )

    def is_fresh(self, now_s: float) -> bool:
        try:
            now = _finite_real("now_s", now_s)
        except (TypeError, ValueError):
            return False
        return self.source_timestamp_s <= now < self.valid_until_s


@dataclass(frozen=True)
class PolicyMailboxStatus:
    timestamp_s: float
    domain: PolicyDomain
    active_policy_sequence: Optional[int]
    active_valid_until_s: Optional[float]
    invalidation_reason: str
    hardware_actuation_permitted: bool
    hardware_blockers: Tuple[str, ...]


class LatestPolicyMailbox:
    """Thread-safe, capacity-one policy register with generation fencing.

    中文：邮箱只保留最新策略，不排队重放历史控制量。发布时除 domain 与 TTL 外，
    还要求策略序号、源时间、飞控 tick 和基线版本严格前进。
    """

    def __init__(
        self,
        domain: PolicyDomain,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(domain, PolicyDomain):
            raise TypeError("domain must be a PolicyDomain")
        if domain.actuation_mode is MultiRateActuationMode.HARDWARE:
            raise ValueError(
                "physical multi-rate actuation is globally disabled; unresolved gates: "
                + ", ".join(_HARDWARE_ACTUATION_BLOCKERS)
            )
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._domain = domain
        self._clock = monotonic_clock
        self._policy: Optional[MPCPolicy] = None
        self._last_policy_sequence: Optional[int] = None
        self._last_policy_source_timestamp_s: Optional[float] = None
        self._fc_session_id: Optional[int] = None
        self._last_fc_target_tick: Optional[int] = None
        self._last_fc_baseline_version: Optional[int] = None
        self._last_fc_baseline_timestamp_s: Optional[float] = None
        self._invalidation_reason = "session initialized without an active policy"
        self._lock = threading.Lock()

    def domain(self) -> PolicyDomain:
        with self._lock:
            return self._domain

    def advance_contact(self, contacts: Sequence[bool], reason: str) -> PolicyDomain:
        contact_mask = _boolean4("contacts", contacts)
        detail = _nonempty_string("reason", reason)
        with self._lock:
            self._domain = replace(
                self._domain,
                contact_epoch=self._domain.contact_epoch + 1,
                invalidation_generation=self._domain.invalidation_generation + 1,
                contacts=contact_mask,
            )
            self._policy = None
            self._invalidation_reason = detail
            return self._domain

    def invalidate(self, reason: str) -> PolicyDomain:
        detail = _nonempty_string("reason", reason)
        with self._lock:
            self._domain = replace(
                self._domain,
                invalidation_generation=self._domain.invalidation_generation + 1,
            )
            self._policy = None
            self._invalidation_reason = detail
            return self._domain

    def publish(self, policy: MPCPolicy, *, now_s: float) -> bool:
        if not isinstance(policy, MPCPolicy):
            raise TypeError("policy must be an MPCPolicy")
        now = _finite_real("now_s", now_s)
        with self._lock:
            if policy.domain != self._domain or not policy.is_fresh(now):
                return False
            if (
                self._last_policy_sequence is not None
                and policy.policy_sequence <= self._last_policy_sequence
            ):
                return False
            command = policy.rotor_command
            if self._last_policy_source_timestamp_s is not None and (
                policy.source_timestamp_s <= self._last_policy_source_timestamp_s
                or command.fc_session_id != self._fc_session_id
                or self._last_fc_target_tick is None
                or command.target_fc_tick <= self._last_fc_target_tick
                or self._last_fc_baseline_version is None
                or command.baseline_version <= self._last_fc_baseline_version
                or self._last_fc_baseline_timestamp_s is None
                or command.baseline_timestamp_s <= self._last_fc_baseline_timestamp_s
            ):
                return False
            self._policy = policy
            self._last_policy_sequence = policy.policy_sequence
            self._last_policy_source_timestamp_s = policy.source_timestamp_s
            self._fc_session_id = command.fc_session_id
            self._last_fc_target_tick = command.target_fc_tick
            self._last_fc_baseline_version = command.baseline_version
            self._last_fc_baseline_timestamp_s = command.baseline_timestamp_s
            self._invalidation_reason = ""
            return True

    def latest(
        self,
        *,
        now_s: float,
        domain: Optional[PolicyDomain] = None,
    ) -> Optional[MPCPolicy]:
        now = _finite_real("now_s", now_s)
        with self._lock:
            policy = self._policy
            expected = self._domain if domain is None else domain
            if (
                policy is None
                or expected != self._domain
                or policy.domain != expected
                or not policy.is_fresh(now)
            ):
                return None
            return policy

    def status(self) -> PolicyMailboxStatus:
        now = _finite_real("monotonic clock", self._clock())
        with self._lock:
            return PolicyMailboxStatus(
                timestamp_s=now,
                domain=self._domain,
                active_policy_sequence=(
                    None if self._policy is None else self._policy.policy_sequence
                ),
                active_valid_until_s=(None if self._policy is None else self._policy.valid_until_s),
                invalidation_reason=self._invalidation_reason,
                hardware_actuation_permitted=False,
                hardware_blockers=_HARDWARE_ACTUATION_BLOCKERS,
            )


class RotorResidualSink(Protocol):
    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult: ...

    async def clear_rotor_residual(self, reason: str) -> OperationResult: ...

    def status(self) -> FlightControllerResidualSinkStatus: ...


class LowCmdLeaseSink(Protocol):
    async def submit(self, command: Go2JointPositionCommand) -> OperationResult: ...

    async def revoke_mpc_control(self, reason: str) -> OperationResult: ...


@dataclass(frozen=True)
class SafetySupervisorStatus:
    timestamp_s: float
    healthy: bool
    fault_latched: bool
    abort_generation: int
    residual_clear_pending: bool
    residual_clear_confirmed: bool
    last_error: str
    hardware_actuation_permitted: bool
    hardware_blockers: Tuple[str, ...]


class LandingSafetySupervisor:
    """Logically independent observer and non-abandonable dual-side fallback.

    This asyncio implementation is only a logical separation and never calls
    the solver.  Production still requires an OS-thread/process watchdog bridge
    plus the Go2-writer and FC-local watchdogs; the global hardware gate remains
    closed until that path has target-side timing evidence.

    中文：监督器不参与正常控制，只观察 worker、高频腿环、LowCmd owner 与飞控
    residual sink。trip 同时触发腿侧撤租约和旋翼侧清零；即使 asyncio 任务被取消，
    non-abandonable 安全动作也必须完成或明确返回失败。
    """

    def __init__(
        self,
        *,
        config: MultiRateExecutionConfig,
        mailbox: LatestPolicyMailbox,
        lowcmd_sink: LowCmdLeaseSink,
        residual_sink: RotorResidualSink,
        go2_status: Callable[[], Go2LowLevelStatus],
        expected_go2_mapping_hash: str,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, MultiRateExecutionConfig):
            raise TypeError("config must be MultiRateExecutionConfig")
        if not isinstance(mailbox, LatestPolicyMailbox):
            raise TypeError("mailbox must be LatestPolicyMailbox")
        for name, sink, methods in (
            ("lowcmd_sink", lowcmd_sink, ("submit", "revoke_mpc_control")),
            (
                "residual_sink",
                residual_sink,
                ("send_rotor_residual", "clear_rotor_residual", "status"),
            ),
        ):
            if any(not callable(getattr(sink, method, None)) for method in methods):
                raise TypeError(f"{name} does not implement its safety contract")
        if not callable(go2_status) or not callable(monotonic_clock):
            raise TypeError("status and clock callbacks must be callable")
        actuation_mode = mailbox.domain().actuation_mode
        if actuation_mode is MultiRateActuationMode.HARDWARE:
            raise ValueError(
                "physical multi-rate actuation is globally disabled; unresolved gates: "
                + ", ".join(_HARDWARE_ACTUATION_BLOCKERS)
            )
        if not all(
            getattr(sink, "simulation_only", False) is True for sink in (lowcmd_sink, residual_sink)
        ):
            raise ValueError(
                "SHADOW/SIMULATION sessions require explicitly simulation-only sinks; "
                "physical fallback writes are globally disabled"
            )
        self._config = config
        self._mailbox = mailbox
        self._lowcmd_sink = lowcmd_sink
        self._residual_sink = residual_sink
        self._go2_status = go2_status
        self._expected_go2_mapping_hash = _sha256_identity(
            "expected_go2_mapping_hash",
            expected_go2_mapping_hash,
        )
        self._clock = monotonic_clock
        self._actuation_mode = actuation_mode
        self._high_rate_status: Optional[Callable[[], HighRateLoopStatus]] = None
        self._worker_status: Optional[Callable[[], MPCWorkerStatus]] = None
        self._fault_latched = False
        self._abort_generation = 0
        self._last_error = ""
        self._abort_task: Optional[asyncio.Task[OperationResult]] = None
        self._residual_clear_task: Optional[asyncio.Task[OperationResult]] = None
        self._fc_activation_task: Optional[asyncio.Task[OperationResult]] = None
        self._residual_clear_confirmed = True

    @property
    def actuation_mode(self) -> MultiRateActuationMode:
        """Return the single mode bound to this mailbox and both exact sinks."""

        return self._actuation_mode

    def require_worker_binding(
        self,
        *,
        mailbox: LatestPolicyMailbox,
        residual_sink: RotorResidualSink,
    ) -> MultiRateActuationMode:
        """Reject a worker that is not in this exact safety/transport session."""

        if mailbox is not self._mailbox or residual_sink is not self._residual_sink:
            raise ValueError("MPC worker must share the safety session mailbox and FC sink")
        return self._actuation_mode

    def require_high_rate_binding(
        self,
        *,
        mailbox: LatestPolicyMailbox,
        lowcmd_sink: LowCmdLeaseSink,
    ) -> MultiRateActuationMode:
        """Reject a leg loop that is not in this exact safety/transport session."""

        if mailbox is not self._mailbox or lowcmd_sink is not self._lowcmd_sink:
            raise ValueError("high-rate loop must share the safety session mailbox and LowCmd sink")
        return self._actuation_mode

    def attach_monitors(
        self,
        *,
        high_rate_status: Callable[[], HighRateLoopStatus],
        worker_status: Callable[[], MPCWorkerStatus],
    ) -> None:
        if not callable(high_rate_status) or not callable(worker_status):
            raise TypeError("monitor callbacks must be callable")
        self._high_rate_status = high_rate_status
        self._worker_status = worker_status

    @property
    def abort_generation(self) -> int:
        return self._abort_generation

    @property
    def fault_latched(self) -> bool:
        return self._fault_latched

    def request_residual_clear(self, reason: str) -> None:
        """Start a non-blocking clear used while a contact replan is pending."""

        detail = _nonempty_string("reason", reason)
        if self._fault_latched:
            return
        task = self._residual_clear_task
        if task is None or task.done():
            self._residual_clear_confirmed = False
            self._residual_clear_task = asyncio.create_task(self._invoke_residual_clear(detail))

    def register_fc_activation(self, task: asyncio.Task[OperationResult]) -> None:
        """Expose the sole in-flight FC stage to the independent abort path."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("FC activation must be an asyncio Task")
        existing = self._fc_activation_task
        if existing is not None and not existing.done() and existing is not task:
            task.cancel()
            raise RuntimeError("another FC activation task is already in flight")
        self._fc_activation_task = task
        if self._fault_latched and not task.done():
            task.cancel()

    def unregister_fc_activation(self, task: asyncio.Task[OperationResult]) -> None:
        if self._fc_activation_task is task:
            self._fc_activation_task = None

    def residual_clear_result(self) -> Optional[OperationResult]:
        task = self._residual_clear_task
        if task is None or not task.done() or task.cancelled():
            return None
        try:
            return task.result()
        except BaseException as exc:
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )

    async def _invoke_residual_clear(self, reason: str) -> OperationResult:
        try:
            result = await self._residual_sink.clear_rotor_residual(reason)
        except (Exception, asyncio.CancelledError) as exc:
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, OperationResult):
            return OperationResult.failure(
                "FC_RESIDUAL_CLEAR_PROTOCOL_ERROR",
                "Residual sink returned an invalid clear result",
            )
        self._residual_clear_confirmed = result.ok
        return result

    async def reconfirm_residual_zero_after_race(self, reason: str) -> OperationResult:
        """Clear again after a cancelled FC stage has fully drained.

        A protocol-shaped sink need not cooperate with task cancellation: a
        stage operation may finish after the first abort CLEAR and make a late
        residual active.  This second barrier is ordered after stage-task
        termination, and its zero-state readback is part of confirmation.
        """

        detail = _nonempty_string("reason", reason)
        clear_result = await self._invoke_residual_clear(detail)
        if not clear_result.ok:
            self._residual_clear_confirmed = False
            self._last_error = f"{self._last_error}; post-race FC clear failed: {clear_result.code}"
            return clear_result
        try:
            status = self._residual_sink.status()
            now = _finite_real("monotonic clock", self._clock())
        except Exception as exc:
            self._residual_clear_confirmed = False
            self._last_error = (
                f"{self._last_error}; post-race FC status failed: {type(exc).__name__}: {exc}"
            )
            return OperationResult.failure(
                "FC_POST_RACE_CLEAR_STATUS_FAILED",
                self._last_error,
            )
        confirmed = bool(
            isinstance(status, FlightControllerResidualSinkStatus)
            and _fresh_past_time(
                status.timestamp_s,
                now=now,
                maximum_age_s=self._config.fc_status_max_age_s,
            )
            and status.residual_state is FlightControllerResidualState.CONFIRMED_ZERO
            and not status.residual_active
            and status.clear_confirmed
            and status.active_command_sequence is None
            and status.pending_command_sequence is None
            and status.active_valid_until_s is None
            and status.pending_started_s is None
            and status.pending_valid_until_s is None
        )
        self._residual_clear_confirmed = confirmed
        if not confirmed:
            self._last_error = f"{self._last_error}; post-race FC zero readback was not confirmed"
            return OperationResult.failure(
                "FC_POST_RACE_ZERO_UNCONFIRMED",
                self._last_error,
            )
        return OperationResult.success(
            "FC residual zero reconfirmed after the stage task drained",
            code="FC_POST_RACE_ZERO_CONFIRMED",
        )

    def begin_trip(self, reason: str) -> asyncio.Future[OperationResult]:
        """Synchronously latch a fault and start both fallback operations.

        This non-awaiting entry point is used before cancelling or draining an
        in-flight device transaction.  It prevents cleanup on one actuator from
        serially delaying the other actuator's fallback request.
        """

        detail = _nonempty_string("reason", reason)
        if not self._fault_latched:
            # This method contains no await: queued/in-flight solver results can
            # no longer publish before either actuator cleanup can block.
            self._fault_latched = True
            self._abort_generation += 1
            self._last_error = detail
            self._mailbox.invalidate(f"safety trip: {detail}")
        task = self._abort_task
        if task is not None and task.done() and self._abort_task_needs_retry(task):
            # A fallback request is idempotent.  Retaining a completed failure
            # here would make every later safety pass merely replay the old
            # result instead of asking the actuator boundaries to try again.
            task = None
            self._abort_task = None
        if task is None:
            task = asyncio.create_task(self._abort_both(detail))
            self._abort_task = task
        # Never expose the raw safety task: cancelling the returned shield must
        # not cancel the task before it has created both actuator children.
        return asyncio.shield(task)

    @staticmethod
    def _abort_task_needs_retry(task: asyncio.Task[OperationResult]) -> bool:
        try:
            result = task.result()
        except BaseException:
            return True
        return not isinstance(result, OperationResult) or not result.ok

    async def trip(self, reason: str) -> OperationResult:
        """Invalidate first, then clear FC and revoke Go2 without abandonment."""

        task = self.begin_trip(reason)
        result, cancellation_seen = await await_nonabandonable(task)
        if cancellation_seen:
            raise asyncio.CancelledError
        return result

    async def _abort_both(self, reason: str) -> OperationResult:
        # Both tasks are started before either is awaited.  Failure on one side
        # can therefore never suppress the other safety action.
        clear_task = asyncio.create_task(self._invoke_residual_clear(f"abort: {reason}"))
        revoke_task = asyncio.create_task(self._invoke_lowcmd_revoke(reason))
        activation_task = self._fc_activation_task
        if activation_task is not None and not activation_task.done():
            activation_task.cancel()
        post_race_result: Optional[OperationResult] = None
        if activation_task is not None:
            activation_drain = asyncio.create_task(_drain_task(activation_task))
            # FC finalization depends only on the first FC CLEAR and the
            # in-flight stage drain.  A stuck Go2 revoke must never delay the
            # second FC zero barrier.
            clear_result, _ = await asyncio.gather(clear_task, activation_drain)
            post_race_result = await self.reconfirm_residual_zero_after_race(
                "FC activation drained after abort; final zero barrier"
            )
            self.unregister_fc_activation(activation_task)
        else:
            clear_result = await clear_task
        revoke_result = await revoke_task
        if (
            clear_result.ok
            and revoke_result.ok
            and (post_race_result is None or post_race_result.ok)
        ):
            return OperationResult.success(
                "FC residual cleared and Go2 MPC lease revoked",
                {
                    "abort_generation": self._abort_generation,
                    "fc_code": clear_result.code,
                    "go2_code": revoke_result.code,
                    "post_race_fc_code": (
                        None if post_race_result is None else post_race_result.code
                    ),
                },
                code="MULTIRATE_FALLBACK_CONFIRMED",
            )
        return OperationResult.failure(
            "MULTIRATE_FALLBACK_UNCONFIRMED",
            "At least one fallback action was not confirmed",
            {
                "abort_generation": self._abort_generation,
                "fc_ok": clear_result.ok,
                "fc_code": clear_result.code,
                "go2_ok": revoke_result.ok,
                "go2_code": revoke_result.code,
                "post_race_fc_ok": (None if post_race_result is None else post_race_result.ok),
                "post_race_fc_code": (None if post_race_result is None else post_race_result.code),
            },
        )

    async def _invoke_lowcmd_revoke(self, reason: str) -> OperationResult:
        try:
            result = await self._lowcmd_sink.revoke_mpc_control(reason)
        except (Exception, asyncio.CancelledError) as exc:
            return OperationResult.failure(
                "LOWCMD_REVOKE_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, OperationResult):
            return OperationResult.failure(
                "LOWCMD_REVOKE_PROTOCOL_ERROR",
                "LowCmd sink returned an invalid revoke result",
            )
        return result

    async def run_once(
        self,
        *,
        manual_override: bool,
        require_active_policy: bool,
    ) -> OperationResult:
        """Perform one bounded observer pass; caller owns periodic scheduling."""

        if type(manual_override) is not bool or type(require_active_policy) is not bool:
            raise TypeError("manual_override and require_active_policy must be booleans")
        if manual_override:
            return await self._trip_from_monitor("manual override requested")
        try:
            now = _finite_real("monotonic clock", self._clock())
        except Exception as exc:
            return await self._trip_from_monitor(
                f"safety monotonic clock failed: {type(exc).__name__}: {exc}"
            )
        if self._fault_latched:
            assert self._abort_task is not None
            result, cancellation_seen = await await_nonabandonable(self._abort_task)
            if cancellation_seen:
                raise asyncio.CancelledError
            return self._tripped_result(self._last_error, result)

        clear_result = self.residual_clear_result()
        if clear_result is not None and not clear_result.ok:
            return await self._trip_from_monitor(
                f"contact-replan residual clear failed: {clear_result.code}"
            )

        try:
            failure = self._health_failure(now, require_active_policy=require_active_policy)
        except Exception as exc:
            return await self._trip_from_monitor(
                f"safety health evaluation raised {type(exc).__name__}: {exc}"
            )
        if failure is not None:
            return await self._trip_from_monitor(failure)
        return OperationResult.success(
            "Multi-rate safety inputs are healthy",
            {"abort_generation": self._abort_generation},
            code="MULTIRATE_SAFETY_HEALTHY",
        )

    async def _trip_from_monitor(self, reason: str) -> OperationResult:
        fallback = await self.trip(reason)
        return self._tripped_result(reason, fallback)

    def _tripped_result(self, reason: str, fallback: OperationResult) -> OperationResult:
        fallback_confirmed = bool(fallback.ok and self._residual_clear_confirmed)
        fallback_code = (
            fallback.code if self._residual_clear_confirmed else "FC_POST_RACE_ZERO_UNCONFIRMED"
        )
        return OperationResult.failure(
            "MULTIRATE_SAFETY_TRIPPED",
            reason,
            {
                "fallback_confirmed": fallback_confirmed,
                "fallback_code": fallback_code,
                "fallback_message": fallback.message,
                "abort_generation": self._abort_generation,
            },
        )

    def _health_failure(self, now: float, *, require_active_policy: bool) -> Optional[str]:
        domain = self._mailbox.domain()
        try:
            go2 = self._go2_status()
        except Exception as exc:
            return f"Go2 status raised {type(exc).__name__}: {exc}"
        now = self._observation_time_after(now)
        if not isinstance(go2, Go2LowLevelStatus):
            return "Go2 owner returned an invalid status"
        go2_bool_fields = (
            go2.connected,
            go2.healthy,
            go2.publisher_active,
            go2.writer_alive,
            go2.watchdog_healthy,
            go2.high_level_released,
            go2.network_exclusivity_verified,
            go2.mapping_hash_verified,
            go2.safe_hold_active,
            go2.safe_hold_settled,
        )
        status_timestamp_is_fresh = _fresh_past_time(
            go2.timestamp,
            now=now,
            maximum_age_s=self._config.low_state_max_age_s,
        )
        status_cache_age_s = now - float(go2.timestamp) if status_timestamp_is_fresh else math.inf
        reported_low_state_age_s = (
            float(go2.low_state_age_s)
            if not isinstance(go2.low_state_age_s, (bool, np.bool_))
            and isinstance(go2.low_state_age_s, Real)
            and math.isfinite(float(go2.low_state_age_s))
            and float(go2.low_state_age_s) >= 0.0
            else math.inf
        )
        low_state_timestamp_is_fresh = _fresh_past_time(
            go2.low_state_timestamp,
            now=now,
            maximum_age_s=self._config.low_state_max_age_s,
        )
        effective_low_state_age_s = max(
            (now - float(go2.low_state_timestamp) if low_state_timestamp_is_fresh else math.inf),
            reported_low_state_age_s + status_cache_age_s,
        )
        if (
            any(type(value) is not bool for value in go2_bool_fields)
            or not all(go2_bool_fields[:8])
            or isinstance(go2.owner_epoch, (bool, np.bool_))
            or not isinstance(go2.owner_epoch, int)
            or go2.owner_epoch != domain.ownership_epoch
            or not isinstance(go2.ownership_state, LowCmdOwnershipState)
            or go2.ownership_state not in _ACTIVE_LOW_CMD_STATES
            or not isinstance(go2.active_mapping_hash, str)
            or go2.active_mapping_hash != self._expected_go2_mapping_hash
            or go2.fault_reason is not None
            or effective_low_state_age_s > self._config.low_state_max_age_s
        ):
            return "Go2 LowCmd owner health/age/ownership check failed"

        try:
            fc = self._residual_sink.status()
        except Exception as exc:
            return f"FC residual status raised {type(exc).__name__}: {exc}"
        now = self._observation_time_after(now)
        if (
            not isinstance(fc, FlightControllerResidualSinkStatus)
            or fc.timestamp_s > now
            or now - fc.timestamp_s > self._config.fc_status_max_age_s
            or not fc.healthy
            or fc.fault_latched
        ):
            return "FC residual ACK/readback/status health check failed"

        policy = self._mailbox.latest(now_s=now, domain=domain)
        monitors_required = bool(
            require_active_policy
            or policy is not None
            or go2.ownership_state is LowCmdOwnershipState.MPC_ACTIVE
        )
        if monitors_required and (self._high_rate_status is None or self._worker_status is None):
            return "active multi-rate session is missing mandatory loop monitors"

        high_rate: Optional[HighRateLoopStatus] = None
        if self._high_rate_status is not None:
            try:
                candidate_high_rate = self._high_rate_status()
            except Exception as exc:
                return f"high-rate monitor raised {type(exc).__name__}: {exc}"
            now = self._observation_time_after(now)
            if not isinstance(candidate_high_rate, HighRateLoopStatus):
                return "high-rate monitor returned an invalid status"
            high_rate = candidate_high_rate
            if (
                high_rate.actuation_mode is not domain.actuation_mode
                or not high_rate.initialized
                or not _fresh_past_time(
                    high_rate.timestamp_s,
                    now=now,
                    maximum_age_s=self._config.high_rate_heartbeat_timeout_s,
                )
                or high_rate.last_progress_s is None
                or not _fresh_past_time(
                    high_rate.last_progress_s,
                    now=now,
                    maximum_age_s=self._config.high_rate_heartbeat_timeout_s,
                )
                or not high_rate.healthy
                or high_rate.fault_latched
                or high_rate.contact_epoch != domain.contact_epoch
                or high_rate.contacts != domain.contacts
            ):
                return "high-rate leg loop identity, heartbeat or health failed"

        worker: Optional[MPCWorkerStatus] = None
        if self._worker_status is not None:
            try:
                candidate_worker = self._worker_status()
            except Exception as exc:
                return f"MPC worker monitor raised {type(exc).__name__}: {exc}"
            now = self._observation_time_after(now)
            if not isinstance(candidate_worker, MPCWorkerStatus):
                return "MPC worker monitor returned an invalid status"
            worker = candidate_worker
            if (
                worker.actuation_mode is not domain.actuation_mode
                or not _fresh_past_time(
                    worker.timestamp_s,
                    now=now,
                    maximum_age_s=self._config.worker_heartbeat_timeout_s,
                )
                or not worker.running
                or not worker.healthy
            ):
                return "MPC worker identity, heartbeat or health failed"
            if (
                policy is not None
                and worker.last_published_policy_sequence != policy.policy_sequence
            ):
                return "MPC worker publication identity does not match the active policy"
            if worker.solve_started_s is None:
                if worker.last_progress_s is None or not _fresh_past_time(
                    worker.last_progress_s,
                    now=now,
                    maximum_age_s=self._config.worker_heartbeat_timeout_s,
                ):
                    return "MPC worker idle heartbeat failed"
            elif not _fresh_past_time(
                worker.solve_started_s,
                now=now,
                maximum_age_s=(self._config.solver_budget_s + self._config.solver_commit_reserve_s),
            ):
                return "MPC worker exceeded its external deadline"

        now = self._observation_time_after(now)
        if self._mailbox.domain() != domain:
            return "multi-rate policy domain changed during the safety observation"
        final_policy = self._mailbox.latest(now_s=now, domain=domain)
        if final_policy is not policy:
            return "active policy changed or expired during the safety observation"
        status_timestamp_is_fresh = _fresh_past_time(
            go2.timestamp,
            now=now,
            maximum_age_s=self._config.low_state_max_age_s,
        )
        status_cache_age_s = now - float(go2.timestamp) if status_timestamp_is_fresh else math.inf
        low_state_timestamp_is_fresh = _fresh_past_time(
            go2.low_state_timestamp,
            now=now,
            maximum_age_s=self._config.low_state_max_age_s,
        )
        effective_low_state_age_s = max(
            (now - float(go2.low_state_timestamp) if low_state_timestamp_is_fresh else math.inf),
            reported_low_state_age_s + status_cache_age_s,
        )
        if effective_low_state_age_s > self._config.low_state_max_age_s:
            return "Go2 LowCmd owner observation became stale during the safety pass"
        if not _fresh_past_time(
            fc.timestamp_s,
            now=now,
            maximum_age_s=self._config.fc_status_max_age_s,
        ):
            return "FC residual observation became stale during the safety pass"
        if high_rate is not None and (
            not _fresh_past_time(
                high_rate.timestamp_s,
                now=now,
                maximum_age_s=self._config.high_rate_heartbeat_timeout_s,
            )
            or high_rate.last_progress_s is None
            or not _fresh_past_time(
                high_rate.last_progress_s,
                now=now,
                maximum_age_s=self._config.high_rate_heartbeat_timeout_s,
            )
        ):
            return "high-rate observation became stale during the safety pass"
        if worker is not None and not _fresh_past_time(
            worker.timestamp_s,
            now=now,
            maximum_age_s=self._config.worker_heartbeat_timeout_s,
        ):
            return "MPC worker observation became stale during the safety pass"

        activation_pending = bool(
            worker is not None and worker.activating_policy_sequence is not None
        )
        if activation_pending:
            assert worker is not None
            if (
                worker.activating_domain != domain
                or worker.last_completed_snapshot_sequence != worker.activating_policy_sequence
                or worker.activation_started_s is None
                or worker.activation_valid_until_s is None
                or worker.activating_fc_session_id is None
                or not _fresh_past_time(
                    worker.activation_started_s,
                    now=now,
                    maximum_age_s=(
                        self._config.solver_commit_reserve_s - self._config.result_audit_budget_s
                    ),
                )
                or now >= worker.activation_valid_until_s
                or fc.fc_session_id != worker.activating_fc_session_id
            ):
                return "FC residual activation transaction identity/deadline failed"
            pending_sequence = worker.activating_policy_sequence
            if fc.residual_state is FlightControllerResidualState.STAGE_PENDING:
                if (
                    fc.last_sequence != pending_sequence
                    or fc.pending_command_sequence != pending_sequence
                    or fc.pending_valid_until_s != worker.activation_valid_until_s
                    or fc.pending_started_s is None
                    or not _fresh_past_time(
                        fc.pending_started_s,
                        now=now,
                        maximum_age_s=(
                            self._config.solver_commit_reserve_s
                            - self._config.result_audit_budget_s
                        ),
                    )
                ):
                    return "FC pending residual does not match the worker transaction"
                old_active_matches = bool(
                    (
                        policy is None
                        and fc.active_command_sequence is None
                        and fc.active_valid_until_s is None
                    )
                    or (
                        policy is not None
                        and fc.active_command_sequence == policy.policy_sequence
                        and fc.active_valid_until_s == policy.valid_until_s
                        and fc.fc_session_id == policy.rotor_command.fc_session_id
                    )
                )
                if not old_active_matches:
                    return "FC pending residual carries an unmatched old active residual"
            elif fc.residual_state is FlightControllerResidualState.ACTIVE:
                active_is_old_policy = bool(
                    policy is not None
                    # The sink consumes the candidate sequence immediately
                    # before crossing the asynchronous FC write boundary.  In
                    # that bounded interval the currently applied residual is
                    # still the old policy while ``last_sequence`` may already
                    # be the candidate watermark.  Accept only those two
                    # explained watermarks; any unrelated/higher sequence still
                    # fails closed.
                    and fc.last_sequence in {policy.policy_sequence, pending_sequence}
                    and fc.active_command_sequence == policy.policy_sequence
                    and fc.active_valid_until_s == policy.valid_until_s
                    and fc.fc_session_id == policy.rotor_command.fc_session_id
                )
                active_is_new_candidate = bool(
                    fc.last_sequence == pending_sequence
                    and fc.active_command_sequence == pending_sequence
                    and fc.active_valid_until_s == worker.activation_valid_until_s
                )
                if not (active_is_old_policy or active_is_new_candidate):
                    return "FC active residual is outside the bounded switch transaction"
            elif fc.residual_state is FlightControllerResidualState.CONFIRMED_ZERO:
                if policy is not None:
                    return "FC became zero while an old leg policy remained active"
            else:
                return "FC entered an unsafe state during residual activation"

        if not activation_pending:
            clear_grace = bool(
                fc.residual_state is FlightControllerResidualState.CLEAR_PENDING
                and self._residual_clear_task is not None
                and not self._residual_clear_task.done()
                and policy is None
                and high_rate is not None
                and high_rate.replan_pending
                and high_rate.replan_grace_authorized
                and high_rate.replan_deadline_s is not None
                and now < high_rate.replan_deadline_s
            )
            if policy is None:
                if (
                    fc.residual_state is not FlightControllerResidualState.CONFIRMED_ZERO
                    and not clear_grace
                ):
                    return "FC exposes an orphan residual without a policy or transaction"
            elif (
                fc.residual_state is not FlightControllerResidualState.ACTIVE
                or fc.fc_session_id != policy.rotor_command.fc_session_id
                or fc.last_sequence != policy.policy_sequence
                or fc.active_command_sequence != policy.policy_sequence
                or fc.active_valid_until_s != policy.valid_until_s
            ):
                return "FC executed identity does not match the active MPC policy"

        replan_grace = bool(
            policy is None
            and high_rate is not None
            and high_rate.replan_pending
            and high_rate.replan_grace_authorized
            and high_rate.replan_deadline_s is not None
            and now < high_rate.replan_deadline_s
        )
        go2_target_age_s = go2.target_age_s
        go2_target_deadline = go2.target_deadline
        if policy is not None or replan_grace:
            target_deadline_value = (
                float(cast(Real, go2_target_deadline))
                if not isinstance(go2_target_deadline, (bool, np.bool_))
                and isinstance(go2_target_deadline, Real)
                else math.nan
            )
            target_age_value = (
                float(cast(Real, go2_target_age_s))
                if not isinstance(go2_target_age_s, (bool, np.bool_))
                and isinstance(go2_target_age_s, Real)
                else math.nan
            )
            target_deadline_is_valid = bool(
                math.isfinite(target_deadline_value) and now < target_deadline_value
            )
            target_age_is_valid = bool(
                math.isfinite(target_age_value)
                and target_age_value >= 0.0
                and target_age_value + status_cache_age_s <= self._config.lowcmd_target_ttl_s
            )
            target_lease_is_valid = bool(
                target_deadline_is_valid
                and target_age_is_valid
                and target_deadline_value - now + target_age_value + status_cache_age_s
                <= self._config.lowcmd_target_ttl_s + 1.0e-12
            )
            if (
                high_rate is None
                or high_rate.last_staged_frame_sequence is None
                or high_rate.last_staged_frame_deadline_s is None
                or high_rate.last_actuator_applied_frame_sequence is None
                or (
                    policy is not None
                    and high_rate.last_actuator_applied_policy_sequence
                    != policy.policy_sequence
                )
                or (policy is not None and high_rate.last_policy_sequence != policy.policy_sequence)
                or go2.ownership_state is not LowCmdOwnershipState.MPC_ACTIVE
                or go2.safe_hold_active
                or go2.safe_hold_settled
                or isinstance(go2.mailbox_staged_target_sequence, (bool, np.bool_))
                or not isinstance(go2.mailbox_staged_target_sequence, int)
                or go2.mailbox_staged_target_sequence
                != high_rate.last_staged_frame_sequence
                or go2.writer_enqueued_target_sequence
                != high_rate.last_staged_frame_sequence
                or go2.actuator_applied_target_sequence
                != high_rate.last_actuator_applied_frame_sequence
                or not target_deadline_is_valid
                or not math.isclose(
                    target_deadline_value,
                    high_rate.last_staged_frame_deadline_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not target_lease_is_valid
            ):
                return (
                    "Go2 staged/enqueued/applied target identity or deadline does not "
                    "match the high-rate frame"
                )
        elif (
            go2.ownership_state is not LowCmdOwnershipState.HOLDING
            or not go2.safe_hold_active
            or not go2.safe_hold_settled
            or go2.target_sequence is not None
            or go2.mailbox_staged_target_sequence is not None
            or go2.target_age_s is not None
            or go2.target_deadline is not None
        ):
            return "Go2 must remain in confirmed owner safe-hold without an executable policy"

        if require_active_policy and policy is None:
            if not replan_grace and not activation_pending:
                return "no fresh activated MPC policy is available"
        return None

    def _observation_time_after(self, previous_s: float) -> float:
        current_s = _finite_real("monotonic clock", self._clock())
        if current_s < previous_s:
            raise ValueError("monotonic clock moved backwards during safety observation")
        return current_s

    def status(self) -> SafetySupervisorStatus:
        now = _finite_real("monotonic clock", self._clock())
        clear_task = self._residual_clear_task
        return SafetySupervisorStatus(
            timestamp_s=now,
            healthy=not self._fault_latched,
            fault_latched=self._fault_latched,
            abort_generation=self._abort_generation,
            residual_clear_pending=clear_task is not None and not clear_task.done(),
            residual_clear_confirmed=self._residual_clear_confirmed,
            last_error=self._last_error,
            hardware_actuation_permitted=False,
            hardware_blockers=_HARDWARE_ACTUATION_BLOCKERS,
        )


@dataclass(frozen=True)
class MPCWorkerStatus:
    timestamp_s: float
    running: bool
    healthy: bool
    reference_only: bool
    actuation_mode: MultiRateActuationMode
    pending_snapshot_sequence: Optional[int]
    solving_snapshot_sequence: Optional[int]
    last_completed_snapshot_sequence: Optional[int]
    last_published_policy_sequence: Optional[int]
    solve_started_s: Optional[float]
    activating_policy_sequence: Optional[int]
    activating_domain: Optional[PolicyDomain]
    activating_fc_session_id: Optional[int]
    activation_started_s: Optional[float]
    activation_valid_until_s: Optional[float]
    last_progress_s: Optional[float]
    late_result_count: int
    coalesced_snapshot_count: int
    last_error: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        for name in ("running", "healthy", "reference_only"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.actuation_mode, MultiRateActuationMode):
            raise TypeError("actuation_mode must be a MultiRateActuationMode")
        for name in (
            "pending_snapshot_sequence",
            "solving_snapshot_sequence",
            "last_completed_snapshot_sequence",
            "last_published_policy_sequence",
            "activating_policy_sequence",
            "activating_fc_session_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_integer(name, value, positive=name == "activating_fc_session_id")
        for name in (
            "solve_started_s",
            "activation_started_s",
            "activation_valid_until_s",
            "last_progress_s",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(name, value))
        for name in ("late_result_count", "coalesced_snapshot_count"):
            _strict_integer(name, getattr(self, name))
        if not isinstance(self.last_error, str):
            raise TypeError("last_error must be a string")
        activation_values = (
            self.activating_policy_sequence,
            self.activating_domain,
            self.activating_fc_session_id,
            self.activation_started_s,
            self.activation_valid_until_s,
        )
        if any(value is None for value in activation_values) != all(
            value is None for value in activation_values
        ):
            raise ValueError("FC activation identity fields must appear together")
        if self.activating_domain is not None and not isinstance(
            self.activating_domain, PolicyDomain
        ):
            raise TypeError("activating_domain must be a PolicyDomain")
        if (
            self.activation_started_s is not None
            and self.activation_valid_until_s is not None
            and self.activation_started_s >= self.activation_valid_until_s
        ):
            raise ValueError("FC activation deadline must follow its start time")


class AsyncLatestMPCWorker:
    """One-in-flight/one-pending asynchronous MPC and FC activation worker.

    The solver runs through an executor and never on the caller's event-loop
    thread.  A thread executor is useful for local tests but cannot kill a
    native solver that ignores cancellation; hardware qualification must also
    prove process/CPU isolation.  Generation fencing ensures such a late
    result can never publish or actuate a residual after invalidation.

    中文：worker 最多保留“一个求解中＋一个最新待求解”快照；中间快照直接合并，
    因为过期最优解没有执行价值。求解后再次审计源数据、domain、TTL 和首个控制量，
    先履行飞控 residual 激活契约，最后才向高频腿环发布策略。
    """

    def __init__(
        self,
        *,
        config: MultiRateExecutionConfig,
        solver: MPCSolverBackend,
        mailbox: LatestPolicyMailbox,
        residual_sink: RotorResidualSink,
        safety: LandingSafetySupervisor,
        executor: Optional[Executor] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        hardware_actuation_requested: bool = False,
        target_identity: str = "LOCAL_TEST_ONLY",
        runtime_build_hash: str = _SHA256_PREFIX + "0" * 64,
        problem_envelope_hash: str = _SHA256_PREFIX + "0" * 64,
        evidence_hash: str = _SHA256_PREFIX + "0" * 64,
        worker_isolation_verified: bool = False,
    ) -> None:
        if not isinstance(config, MultiRateExecutionConfig):
            raise TypeError("config must be MultiRateExecutionConfig")
        if not isinstance(mailbox, LatestPolicyMailbox):
            raise TypeError("mailbox must be LatestPolicyMailbox")
        if not isinstance(safety, LandingSafetySupervisor):
            raise TypeError("safety must be LandingSafetySupervisor")
        if not callable(getattr(solver, "solve", None)) or not isinstance(
            getattr(solver, "qualification", None), SolverQualification
        ):
            raise TypeError("solver must implement MPCSolverBackend")
        if any(
            not callable(getattr(residual_sink, method, None))
            for method in ("send_rotor_residual", "clear_rotor_residual", "status")
        ):
            raise TypeError("residual_sink does not implement the required contract")
        if type(hardware_actuation_requested) is not bool:
            raise TypeError("hardware_actuation_requested must be a bool")
        if type(worker_isolation_verified) is not bool:
            raise TypeError("worker_isolation_verified must be a bool")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        build_hash = _sha256_identity("runtime_build_hash", runtime_build_hash)
        envelope_hash = _sha256_identity("problem_envelope_hash", problem_envelope_hash)
        reviewed_evidence_hash = _sha256_identity("evidence_hash", evidence_hash)
        target = _nonempty_string("target_identity", target_identity)
        qualification_matches = worker_isolation_verified and solver.qualification.authorizes(
            target_identity=target,
            runtime_build_hash=build_hash,
            problem_envelope_hash=envelope_hash,
            evidence_hash=reviewed_evidence_hash,
            solver_budget_s=config.solver_budget_s,
            commit_reserve_s=config.solver_commit_reserve_s,
        )
        if hardware_actuation_requested:
            detail = (
                "the supplied solver qualification is also insufficient"
                if not qualification_matches
                else "solver timing evidence alone cannot open the system-level gate"
            )
            raise ValueError(
                "physical multi-rate actuation remains globally disabled: "
                + detail
                + "; unresolved gates: "
                + ", ".join(_HARDWARE_ACTUATION_BLOCKERS)
            )
        actuation_mode = safety.require_worker_binding(
            mailbox=mailbox,
            residual_sink=residual_sink,
        )
        self._config = config
        self._solver = solver
        self._mailbox = mailbox
        self._residual_sink = residual_sink
        self._safety = safety
        self._executor = executor
        self._clock = monotonic_clock
        self._actuation_mode = actuation_mode
        self._reference_only = True
        self._pending: Optional[MPCSnapshot] = None
        self._pending_lock = asyncio.Lock()
        self._wake_event = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_requested = False
        self._running = False
        self._healthy = True
        self._last_submitted_sequence: Optional[int] = None
        self._last_submitted_timestamp_s: Optional[float] = None
        self._last_submitted_source_timestamps: Optional[Tuple[float, ...]] = None
        self._fc_session_id: Optional[int] = None
        self._last_fc_target_tick: Optional[int] = None
        self._last_fc_baseline_version: Optional[int] = None
        self._last_fc_baseline_timestamp_s: Optional[float] = None
        self._latest_submitted_sequence: Optional[int] = None
        self._solving_sequence: Optional[int] = None
        self._last_completed_sequence: Optional[int] = None
        self._last_published_sequence: Optional[int] = None
        self._solve_started_s: Optional[float] = None
        self._activating_policy_sequence: Optional[int] = None
        self._activating_domain: Optional[PolicyDomain] = None
        self._activating_fc_session_id: Optional[int] = None
        self._activation_started_s: Optional[float] = None
        self._activation_valid_until_s: Optional[float] = None
        self._last_progress_s: Optional[float] = None
        self._late_result_count = 0
        self._coalesced_snapshot_count = 0
        self._solution_sequence = 0
        self._last_error = ""

    async def start(self) -> OperationResult:
        if self._running:
            return OperationResult.success(
                "MPC worker is already running", code="MPC_WORKER_RUNNING"
            )
        if self._stop_requested:
            return OperationResult.failure(
                "MPC_WORKER_SESSION_CLOSED",
                "A stopped worker is single-session; construct a new worker",
            )
        if self._safety.fault_latched:
            return OperationResult.failure(
                "MPC_WORKER_SAFETY_LATCHED",
                "A new landing session is required after a safety trip",
            )
        try:
            started_s = self._now()
        except Exception as exc:
            detail = f"MPC worker start clock failed: {type(exc).__name__}: {exc}"
            self._healthy = False
            self._last_error = detail
            fallback_task = self._safety.begin_trip(detail)
            fallback, cancellation_seen = await await_nonabandonable(fallback_task)
            if cancellation_seen:
                raise asyncio.CancelledError from exc
            return OperationResult.failure(
                "MPC_WORKER_START_CLOCK_FAILED",
                detail,
                {
                    "fallback_confirmed": fallback.ok,
                    "fallback_code": fallback.code,
                },
            )
        self._running = True
        self._stop_requested = False
        self._healthy = True
        self._last_progress_s = started_s
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(self._worker_task_done)
        return OperationResult.success("MPC worker started", code="MPC_WORKER_STARTED")

    async def submit_snapshot(self, snapshot: MPCSnapshot) -> OperationResult:
        if not isinstance(snapshot, MPCSnapshot):
            raise TypeError("snapshot must be an MPCSnapshot")
        now = self._now()
        if not self._running or self._task is None:
            return OperationResult.failure("MPC_WORKER_NOT_RUNNING", "Start the MPC worker first")
        if self._safety.fault_latched:
            return OperationResult.failure("MPC_WORKER_SAFETY_LATCHED", "Safety is fault-latched")
        if snapshot.domain != self._mailbox.domain():
            return OperationResult.failure(
                "MPC_SNAPSHOT_GENERATION_STALE",
                "Snapshot identity no longer matches the active policy generation",
            )
        source_failure = self._snapshot_source_failure(snapshot, now)
        if not (snapshot.timestamp_s <= now < snapshot.valid_until_s) or source_failure is not None:
            return OperationResult.failure(
                "MPC_SNAPSHOT_STALE",
                "Snapshot or one of its source samples is already stale",
                {"source_failure": source_failure or "snapshot lease is not current"},
            )
        source_timestamps = (
            snapshot.freshness.state_estimate_timestamp_s,
            snapshot.freshness.contact_forces_timestamp_s,
            snapshot.freshness.kinematics_timestamp_s,
            snapshot.freshness.foot_plan_timestamp_s,
            snapshot.freshness.flight_controller_baseline_timestamp_s,
        )
        if max(source_timestamps) - min(source_timestamps) > self._config.maximum_source_skew_s:
            return OperationResult.failure(
                "MPC_SNAPSHOT_SOURCE_SKEW",
                "Snapshot source timestamps exceed maximum_source_skew_s",
            )
        trusted_deadline = min(
            snapshot.timestamp_s + self._config.policy_ttl_s,
            self._snapshot_source_deadline(snapshot),
        )
        if snapshot.valid_until_s > trusted_deadline + 1e-12:
            return OperationResult.failure(
                "MPC_SNAPSHOT_TTL_INVALID",
                "Snapshot validity exceeds a trusted per-source age or policy TTL",
                {"trusted_deadline_s": trusted_deadline},
            )
        async with self._pending_lock:
            locked_now = self._now()
            locked_source_failure = self._snapshot_source_failure(snapshot, locked_now)
            if (
                not self._running
                or self._stop_requested
                or self._task is None
                or self._task.done()
                or self._safety.fault_latched
            ):
                return OperationResult.failure(
                    "MPC_WORKER_SESSION_CHANGED",
                    "Worker stop/fault won the snapshot submission race",
                )
            if snapshot.domain != self._mailbox.domain():
                return OperationResult.failure(
                    "MPC_SNAPSHOT_GENERATION_STALE",
                    "Snapshot generation changed while submission was pending",
                )
            if (
                not (snapshot.timestamp_s <= locked_now < snapshot.valid_until_s)
                or locked_source_failure is not None
            ):
                return OperationResult.failure(
                    "MPC_SNAPSHOT_STALE",
                    "Snapshot became stale while submission was pending",
                    {"source_failure": locked_source_failure or "snapshot lease is not current"},
                )
            if self._last_submitted_sequence is not None:
                assert self._last_submitted_timestamp_s is not None
                assert self._last_submitted_source_timestamps is not None
                if (
                    snapshot.snapshot_sequence <= self._last_submitted_sequence
                    or snapshot.timestamp_s <= self._last_submitted_timestamp_s
                    or snapshot.timestamp_s - self._last_submitted_timestamp_s
                    < self._config.mpc_release_period_s - 1e-12
                    or any(
                        current < previous
                        for current, previous in zip(
                            source_timestamps,
                            self._last_submitted_source_timestamps,
                        )
                    )
                    or snapshot.flight_controller_session_id != self._fc_session_id
                    or self._last_fc_target_tick is None
                    or snapshot.flight_controller_target_tick <= self._last_fc_target_tick
                    or self._last_fc_baseline_version is None
                    or snapshot.flight_controller_baseline_version <= self._last_fc_baseline_version
                    or self._last_fc_baseline_timestamp_s is None
                    or snapshot.flight_controller_baseline_timestamp_s
                    <= self._last_fc_baseline_timestamp_s
                ):
                    return OperationResult.failure(
                        "MPC_SNAPSHOT_OUT_OF_ORDER",
                        "Snapshot cadence/source/FC identities must advance monotonically",
                    )
            if self._pending is not None:
                self._coalesced_snapshot_count += 1
            self._pending = snapshot
            self._last_submitted_sequence = snapshot.snapshot_sequence
            self._last_submitted_timestamp_s = snapshot.timestamp_s
            self._last_submitted_source_timestamps = source_timestamps
            self._fc_session_id = snapshot.flight_controller_session_id
            self._last_fc_target_tick = snapshot.flight_controller_target_tick
            self._last_fc_baseline_version = snapshot.flight_controller_baseline_version
            self._last_fc_baseline_timestamp_s = snapshot.flight_controller_baseline_timestamp_s
            self._latest_submitted_sequence = snapshot.snapshot_sequence
            self._wake_event.set()
        return OperationResult.success(
            "Latest MPC snapshot accepted",
            {"snapshot_sequence": snapshot.snapshot_sequence},
            code="MPC_SNAPSHOT_ACCEPTED",
        )

    async def stop(self, reason: str = "MPC worker stopped") -> OperationResult:
        detail = _nonempty_string("reason", reason)
        self._stop_requested = True
        self._running = False
        self._mailbox.invalidate(detail)
        self._wake_event.set()
        # Latch and launch both actuator fallbacks before cancelling/draining
        # the worker.  A cancellation-resistant FC transaction must never
        # postpone the Go2 revoke request (or vice versa).
        fallback_task = self._safety.begin_trip(detail)
        cancellation_seen = False
        task = self._task
        if task is not None:
            if not task.done():
                task.cancel()
        async with self._pending_lock:
            self._pending = None
        if task is not None:
            drain_task = asyncio.create_task(_drain_task(task))
            _, drain_cancelled = await await_nonabandonable(drain_task)
            cancellation_seen = cancellation_seen or drain_cancelled
        self._task = None
        self._solving_sequence = None
        self._solve_started_s = None
        fallback, fallback_cancelled = await await_nonabandonable(fallback_task)
        self._last_progress_s = None
        cancellation_seen = cancellation_seen or fallback_cancelled
        if cancellation_seen:
            raise asyncio.CancelledError
        if not fallback.ok:
            return OperationResult.failure(
                "MPC_WORKER_STOP_FALLBACK_UNCONFIRMED",
                "Worker stopped, but the dual-side fallback was not confirmed",
                {"fallback_code": fallback.code},
            )
        return OperationResult.success(
            "MPC worker stopped; policy invalidated and dual-side fallback confirmed",
            {"fallback_code": fallback.code},
            code="MPC_WORKER_STOPPED",
        )

    async def _run(self) -> None:
        try:
            while self._running:
                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=0.5 * self._config.worker_heartbeat_timeout_s,
                    )
                except asyncio.TimeoutError:
                    # An idle worker still proves that its scheduling task is
                    # alive.  During a solve, solve_started_s owns the bound.
                    self._last_progress_s = self._now()
                    continue
                if not self._running:
                    break
                async with self._pending_lock:
                    snapshot = self._pending
                    self._pending = None
                    self._wake_event.clear()
                if snapshot is None:
                    continue
                await self._solve_activate_publish(snapshot)
        except asyncio.CancelledError:
            if not self._stop_requested:
                self._running = False
                self._healthy = False
                self._last_error = "MPC worker task was cancelled unexpectedly"
                self._mailbox.invalidate(self._last_error)
                fallback_task = self._safety.begin_trip(self._last_error)
                await await_nonabandonable(fallback_task)
            raise
        except Exception as exc:
            self._running = False
            self._healthy = False
            self._last_error = f"worker exception: {type(exc).__name__}: {exc}"
            if not self._safety.fault_latched:
                await self._safety.trip(self._last_error)

    def _worker_task_done(self, task: asyncio.Task[None]) -> None:
        """Fail closed even if cancellation wins before ``_run`` starts."""

        if task is not self._task or self._stop_requested or self._safety.fault_latched:
            return
        if task.cancelled():
            reason = "MPC worker task was cancelled before/without safe termination"
        else:
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
                reason = "MPC worker task was cancelled before/without safe termination"
            else:
                if error is None and not self._running:
                    return
                reason = (
                    "MPC worker task terminated unexpectedly"
                    if error is None
                    else f"MPC worker task escaped {type(error).__name__}: {error}"
                )
        self._running = False
        self._healthy = False
        self._last_error = reason
        self._mailbox.invalidate(reason)
        self._safety.begin_trip(reason)

    async def _solve_activate_publish(self, snapshot: MPCSnapshot) -> None:
        now = self._now()
        remaining = min(
            self._config.solver_budget_s,
            snapshot.valid_until_s - now - self._config.solver_commit_reserve_s,
        )
        if remaining <= 0.0:
            await self._fault_if_latest(snapshot, "MPC snapshot has no remaining solve budget")
            return
        self._solving_sequence = snapshot.snapshot_sequence
        self._solve_started_s = now
        self._last_progress_s = now
        loop = asyncio.get_running_loop()
        operation = partial(self._solver.solve, snapshot.problem, timeout_s=remaining)
        solve_future = loop.run_in_executor(self._executor, operation)
        try:
            result = await asyncio.wait_for(asyncio.shield(solve_future), timeout=remaining)
        except asyncio.CancelledError:
            if not solve_future.done():
                solve_future.add_done_callback(self._consume_late_future)
                self._late_result_count += 1
            raise
        except asyncio.TimeoutError:
            solve_future.add_done_callback(self._consume_late_future)
            self._late_result_count += 1
            # Even a superseded overrun is a session fault: its native work is
            # still consuming resources and no second solve may be launched.
            await self._fault_session("MPC backend exceeded its external deadline")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        except Exception as exc:
            await self._fault_if_latest(
                snapshot,
                f"MPC backend raised {type(exc).__name__}: {exc}",
            )
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        completed = self._now()
        self._last_completed_sequence = snapshot.snapshot_sequence
        self._last_progress_s = completed
        if not self._is_latest(snapshot):
            self._late_result_count += 1
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        if (
            completed + self._config.solver_commit_reserve_s >= snapshot.valid_until_s
            or self._snapshot_source_failure(snapshot, completed) is not None
        ):
            await self._fault_if_latest(
                snapshot, "MPC result completed after its source-age/deadline budget"
            )
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        if (
            not isinstance(result, MPCSolveResult)
            or not result.success
            or result.first_input is None
        ):
            status = result.status if isinstance(result, MPCSolveResult) else "invalid_result"
            await self._fault_if_latest(snapshot, f"MPC solve was not executable: {status}")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        audit_operation = partial(
            audit_first_mpc_input,
            snapshot.problem,
            result,
            self._config,
        )
        audit_future = loop.run_in_executor(self._executor, audit_operation)
        try:
            audit_error = await asyncio.wait_for(
                asyncio.shield(audit_future),
                timeout=self._config.result_audit_budget_s,
            )
        except asyncio.CancelledError:
            if not audit_future.done():
                audit_future.add_done_callback(self._consume_late_audit_future)
                self._late_result_count += 1
            raise
        except asyncio.TimeoutError:
            audit_future.add_done_callback(self._consume_late_audit_future)
            self._late_result_count += 1
            await self._fault_session("independent MPC result audit exceeded its deadline")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        except Exception as exc:
            await self._fault_if_latest(
                snapshot,
                f"independent MPC result audit raised {type(exc).__name__}: {exc}",
            )
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        audit_completed = self._now()
        self._last_progress_s = audit_completed
        if not self._is_latest(snapshot):
            self._late_result_count += 1
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        if self._snapshot_source_failure(snapshot, audit_completed) is not None:
            await self._fault_if_latest(
                snapshot,
                "MPC source became stale during independent result audit",
            )
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        activation_reserve_s = (
            self._config.solver_commit_reserve_s - self._config.result_audit_budget_s
        )
        if audit_completed + activation_reserve_s >= snapshot.valid_until_s:
            await self._fault_if_latest(
                snapshot,
                "independent MPC audit left insufficient FC activation reserve",
            )
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        if audit_error is not None:
            await self._fault_if_latest(snapshot, f"MPC first-input audit failed: {audit_error}")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        self._solution_sequence += 1
        try:
            candidate = _policy_from_result(
                snapshot,
                result,
                solution_sequence=self._solution_sequence,
                solve_started_s=now,
                solve_completed_s=completed,
                activated_s=completed,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            await self._fault_if_latest(snapshot, f"MPC result audit failed: {exc}")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        if self._actuation_mode is MultiRateActuationMode.SHADOW:
            # A real sink passed without explicit hardware authorization is
            # structurally diagnostic-only: no FC write and no executable
            # leg policy publication are possible.
            self._healthy = True
            self._last_error = "shadow result audited; actuation intentionally suppressed"
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        self._activating_policy_sequence = candidate.policy_sequence
        self._activating_domain = candidate.domain
        self._activating_fc_session_id = candidate.rotor_command.fc_session_id
        self._activation_started_s = self._now()
        self._activation_valid_until_s = candidate.valid_until_s
        self._last_progress_s = self._activation_started_s
        activation_budget_s = (
            self._config.solver_commit_reserve_s - self._config.result_audit_budget_s
        )
        activation_deadline_s = min(
            candidate.valid_until_s,
            self._activation_started_s + activation_budget_s,
        )
        activation_timeout_s = activation_deadline_s - self._now()
        if activation_timeout_s <= 0.0:
            await self._fault_session("FC residual activation has no remaining commit lease")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        activation_task = asyncio.create_task(
            self._residual_sink.send_rotor_residual(candidate.rotor_command)
        )
        self._safety.register_fc_activation(activation_task)
        try:
            activation = await asyncio.wait_for(
                asyncio.shield(activation_task),
                timeout=activation_timeout_s,
            )
        except asyncio.TimeoutError:
            fallback_task = self._begin_session_fault(
                "FC residual activation exceeded its commit reserve"
            )
            activation_task.cancel()
            await self._drain_activation_and_fallback(activation_task, fallback_task)
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        except asyncio.CancelledError:
            detail = (
                "MPC worker stopped during FC residual activation"
                if self._stop_requested
                else "MPC worker task was cancelled unexpectedly during FC residual activation"
            )
            fallback_task = self._begin_session_fault(detail)
            if not activation_task.done():
                activation_task.cancel()
            await self._drain_activation_and_fallback(activation_task, fallback_task)
            raise
        activated = self._now()
        self._safety.unregister_fc_activation(activation_task)
        if activated >= activation_deadline_s:
            await self._fault_session("FC residual activation missed its commit deadline")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return
        if not isinstance(activation, OperationResult) or not activation.ok:
            code = activation.code if isinstance(activation, OperationResult) else "invalid_result"
            await self._fault_session(f"FC residual activation failed: {code}")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        candidate = replace(candidate, activated_s=activated)
        if (
            not self._is_latest(snapshot)
            or not candidate.is_fresh(activated)
            or self._snapshot_source_failure(snapshot, activated) is not None
            or not self._mailbox.publish(candidate, now_s=activated)
        ):
            self._late_result_count += 1
            await self._fault_session("FC residual changed without a matching fresh leg policy")
            self._finish_solve_status(snapshot.snapshot_sequence)
            return

        self._last_published_sequence = candidate.policy_sequence
        self._clear_activation_status()
        self._healthy = True
        self._last_error = ""
        self._finish_solve_status(snapshot.snapshot_sequence)

    async def _fault_if_latest(self, snapshot: MPCSnapshot, reason: str) -> None:
        if not self._is_latest(snapshot):
            self._late_result_count += 1
            return
        self._healthy = False
        self._running = False
        self._last_error = reason
        await self._safety.trip(reason)

    async def _fault_session(self, reason: str) -> None:
        """Latch a session-wide fault once an FC transaction may have started."""

        fallback_task = self._begin_session_fault(reason)
        _, cancellation_seen = await await_nonabandonable(fallback_task)
        if cancellation_seen:
            raise asyncio.CancelledError

    def _begin_session_fault(self, reason: str) -> asyncio.Future[OperationResult]:
        """Latch a worker fault and launch both actuator fallbacks immediately."""

        self._healthy = False
        self._running = False
        self._last_error = reason
        self._mailbox.invalidate(reason)
        return self._safety.begin_trip(reason)

    async def _drain_activation_and_fallback(
        self,
        activation_task: asyncio.Task[OperationResult],
        fallback_task: asyncio.Future[OperationResult],
    ) -> None:
        """Join activation cleanup and dual fallback after both have started."""

        drain_task = asyncio.create_task(_drain_task(activation_task))
        _, activation_cancelled = await await_nonabandonable(drain_task)
        _, fallback_cancelled = await await_nonabandonable(fallback_task)
        self._safety.unregister_fc_activation(activation_task)
        if activation_cancelled or fallback_cancelled:
            raise asyncio.CancelledError

    def _clear_activation_status(self) -> None:
        self._activating_policy_sequence = None
        self._activating_domain = None
        self._activating_fc_session_id = None
        self._activation_started_s = None
        self._activation_valid_until_s = None

    def _snapshot_sources(
        self,
        snapshot: MPCSnapshot,
    ) -> Tuple[Tuple[str, float, float], ...]:
        """Return source timestamps with owner-configured (trusted) age limits."""

        freshness = snapshot.freshness
        return (
            (
                "state estimate",
                freshness.state_estimate_timestamp_s,
                self._config.state_estimate_max_age_s,
            ),
            (
                "contact forces",
                freshness.contact_forces_timestamp_s,
                self._config.contact_force_max_age_s,
            ),
            (
                "kinematics",
                freshness.kinematics_timestamp_s,
                self._config.kinematics_max_age_s,
            ),
            (
                "foot plan",
                freshness.foot_plan_timestamp_s,
                self._config.foot_plan_max_age_s,
            ),
            (
                "flight-controller baseline",
                freshness.flight_controller_baseline_timestamp_s,
                self._config.fc_baseline_max_age_s,
            ),
        )

    def _snapshot_source_deadline(self, snapshot: MPCSnapshot) -> float:
        return min(
            timestamp + maximum_age
            for _, timestamp, maximum_age in self._snapshot_sources(snapshot)
        )

    def _snapshot_source_failure(self, snapshot: MPCSnapshot, now_s: float) -> Optional[str]:
        """Enforce source health without trusting caller-supplied maximum age."""

        caller_failure = snapshot.freshness.failure_reason(now_s)
        if caller_failure is not None:
            return caller_failure
        for name, timestamp, maximum_age in self._snapshot_sources(snapshot):
            if timestamp > now_s:
                return f"{name} timestamp is in the future"
            if now_s - timestamp > maximum_age:
                return f"{name} exceeds its configured maximum age"
        return None

    def _is_latest(self, snapshot: MPCSnapshot) -> bool:
        return bool(
            self._running
            and snapshot.domain == self._mailbox.domain()
            and self._latest_submitted_sequence == snapshot.snapshot_sequence
            and not self._safety.fault_latched
        )

    def _finish_solve_status(self, sequence: int) -> None:
        if self._solving_sequence == sequence:
            self._solving_sequence = None
            self._solve_started_s = None
        self._last_progress_s = self._now()

    @staticmethod
    def _consume_late_future(future: asyncio.Future[MPCSolveResult]) -> None:
        try:
            future.exception()
        except (Exception, asyncio.CancelledError):
            pass

    @staticmethod
    def _consume_late_audit_future(future: asyncio.Future[Optional[str]]) -> None:
        try:
            future.exception()
        except (Exception, asyncio.CancelledError):
            pass

    def _now(self) -> float:
        return _finite_real("monotonic clock", self._clock())

    def status(self) -> MPCWorkerStatus:
        pending_sequence = None if self._pending is None else self._pending.snapshot_sequence
        return MPCWorkerStatus(
            timestamp_s=self._now(),
            running=self._running,
            healthy=self._healthy and not self._safety.fault_latched,
            reference_only=self._reference_only,
            actuation_mode=self._actuation_mode,
            pending_snapshot_sequence=pending_sequence,
            solving_snapshot_sequence=self._solving_sequence,
            last_completed_snapshot_sequence=self._last_completed_sequence,
            last_published_policy_sequence=self._last_published_sequence,
            solve_started_s=self._solve_started_s,
            activating_policy_sequence=self._activating_policy_sequence,
            activating_domain=self._activating_domain,
            activating_fc_session_id=self._activating_fc_session_id,
            activation_started_s=self._activation_started_s,
            activation_valid_until_s=self._activation_valid_until_s,
            last_progress_s=self._last_progress_s,
            late_result_count=self._late_result_count,
            coalesced_snapshot_count=self._coalesced_snapshot_count,
            last_error=self._last_error,
        )


def audit_first_mpc_input(
    problem: ImpactAwareMPCProblem,
    result: MPCSolveResult,
    config: MultiRateExecutionConfig,
) -> Optional[str]:
    """Independently reject an infeasible first input; never repair or clip it."""

    if not isinstance(problem, ImpactAwareMPCProblem):
        raise TypeError("problem must be an ImpactAwareMPCProblem")
    if not isinstance(result, MPCSolveResult):
        return "solver returned an invalid result type"
    if not isinstance(config, MultiRateExecutionConfig):
        raise TypeError("config must be MultiRateExecutionConfig")
    if type(result.success) is not bool or not result.success:
        return "solver success flag is not a strict true boolean"
    control = result.first_input
    if control is None:
        return "solver result has no executable first input"
    try:
        _finite_real("result.objective", result.objective)
        solve_time = _finite_real("result.solve_time_s", result.solve_time_s)
        equality_violation = _finite_real(
            "result.max_equality_violation",
            result.max_equality_violation,
        )
        inequality_residual = _finite_real(
            "result.min_inequality_residual",
            result.min_inequality_residual,
        )
    except (TypeError, ValueError) as exc:
        return f"solver diagnostics are invalid: {exc}"
    if solve_time < 0.0 or equality_violation < 0.0:
        return "solver time/equality diagnostics cannot be negative"
    if equality_violation > config.mpc_equality_tolerance:
        return "reported MPC equality violation exceeds the release tolerance"
    if inequality_residual < -config.mpc_inequality_tolerance:
        return "reported MPC inequality residual exceeds the release tolerance"

    try:
        nlp = ImpactAwareNLP(problem)
        decision = nlp.initial_guess(result)
        equality = nlp.equality_residual(decision)
        inequality = nlp.inequality_residual(decision)
        variable_bounds = nlp.variable_bound_residual(decision)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return f"complete MPC trajectory cannot be independently reconstructed: {exc}"
    recomputed_equality = float(np.max(np.abs(equality))) if equality.size else 0.0
    recomputed_inequality = float(np.min(inequality)) if inequality.size else math.inf
    recomputed_bound_margin = float(np.min(variable_bounds)) if variable_bounds.size else math.inf
    if recomputed_equality > config.mpc_equality_tolerance:
        return "recomputed MPC trajectory equality violation exceeds tolerance"
    if recomputed_inequality < -config.mpc_inequality_tolerance:
        return "recomputed MPC trajectory inequality residual exceeds tolerance"
    if recomputed_bound_margin < -config.mpc_inequality_tolerance:
        return "recomputed MPC decision-variable bound margin exceeds tolerance"

    forces = np.asarray(control.contact_forces_world_n, dtype=np.float64)
    schedule = np.asarray(problem.contact_schedule[0], dtype=np.int8)
    limits = problem.contact_limits
    geometry = problem.landing_contact_geometry
    ground_normal = (
        geometry.ground_normal_world
        if geometry is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    for leg_index in range(4):
        force = forces[leg_index]
        if schedule[leg_index] == 0:
            if float(np.max(np.abs(force))) > config.force_zero_tolerance_n:
                return f"non-contact leg {leg_index} has a nonzero desired force"
            continue
        normal = float(force @ ground_normal)
        tolerance = config.force_constraint_tolerance_n
        if normal < -tolerance:
            return f"contact leg {leg_index} has a negative normal force"
        if normal > float(limits.maximum_normal_force_n[leg_index]) + tolerance:
            return f"contact leg {leg_index} exceeds its normal-force limit"
        tangential_force = force - normal * ground_normal
        tangential = float(np.linalg.norm(tangential_force))
        friction_limit = float(limits.friction_coefficients[leg_index]) * max(normal, 0.0)
        if tangential > friction_limit + tolerance:
            return f"contact leg {leg_index} violates the friction cone"

    actuator = problem.rotor_actuator_config
    actual = problem.initial_state.rotor_thrusts_n
    commanded = control.rotor_thrust_commands_n
    try:
        rates = first_order_thrust_rate(actual, commanded, actuator)
        margins = evaluate_rotor_constraints(actual, rates, commanded, actuator)
    except (TypeError, ValueError) as exc:
        return f"rotor first-input audit failed: {exc}"
    thrust_margins = (
        margins.thrust_lower_margin_n,
        margins.thrust_upper_margin_n,
        margins.command_lower_margin_n,
        margins.command_upper_margin_n,
    )
    if any(bool(np.any(margin < -config.rotor_thrust_tolerance_n)) for margin in thrust_margins):
        return "rotor thrust/command magnitude violates the actuator envelope"
    rate_margins = (
        margins.thrust_rate_lower_margin_n_per_s,
        margins.thrust_rate_upper_margin_n_per_s,
    )
    if any(bool(np.any(margin < -config.rotor_rate_tolerance_n_per_s)) for margin in rate_margins):
        return "rotor thrust rate violates the actuator envelope"
    plan = problem.rotor_execution_plan
    if plan is None:
        return "multi-rate release requires a rotor execution plan"
    try:
        reconstruct_transport_target(plan, 0, commanded)
    except (TypeError, ValueError) as exc:
        return f"rotor command violates the gain-aware execution plan: {exc}"
    return None


def _policy_from_result(
    snapshot: MPCSnapshot,
    result: MPCSolveResult,
    *,
    solution_sequence: int,
    solve_started_s: float,
    solve_completed_s: float,
    activated_s: float,
) -> MPCPolicy:
    control = result.first_input
    if control is None:
        raise ValueError("solver result has no executable first input")
    plan = snapshot.problem.rotor_execution_plan
    if plan is None:
        raise ValueError("snapshot has no rotor execution plan")
    baseline = np.asarray(plan.baseline_thrusts_n[0], dtype=np.float64)
    applied = np.asarray(control.rotor_thrust_commands_n, dtype=np.float64)
    if baseline.shape != (4,) or applied.shape != (4,):
        raise ValueError("rotor values must contain four channels")
    residual = applied - baseline
    gain = float(plan.correction_gains[0])
    transport = reconstruct_transport_target(plan, 0, applied)
    if transport is None:
        raw: Optional[Tuple[float, ...]] = None
        semantics = "zero_gain_no_transport_target"
    else:
        raw_array = np.asarray(transport.target_thrusts_n, dtype=np.float64) - baseline
        raw = tuple(float(value) for value in raw_array)
        semantics = (
            "gain_limited_algebraic_reconstruction"
            if transport.is_gain_limited_reconstruction
            else "active_gain_one_transport_target"
        )
    command = FlightControllerRotorResidualCommand(
        sequence=snapshot.snapshot_sequence,
        timestamp_s=snapshot.timestamp_s,
        valid_until_s=snapshot.valid_until_s,
        fc_session_id=snapshot.flight_controller_session_id,
        target_fc_tick=snapshot.flight_controller_target_tick,
        baseline_version=snapshot.flight_controller_baseline_version,
        baseline_timestamp_s=snapshot.flight_controller_baseline_timestamp_s,
        baseline_thrusts_n=tuple(float(value) for value in baseline),
        transport_raw_residual_thrusts_n=raw,
        applied_residual_thrusts_n=tuple(float(value) for value in residual),
        applied_total_thrusts_n=tuple(float(value) for value in applied),
        correction_gain=gain,
        transport_target_semantics=semantics,
    )
    forces = np.asarray(control.contact_forces_world_n, dtype=np.float64)
    if forces.shape != (4, 3) or not np.all(np.isfinite(forces)):
        raise ValueError("first MPC contact-force input must have shape (4, 3)")
    return MPCPolicy(
        domain=snapshot.domain,
        policy_sequence=snapshot.snapshot_sequence,
        solution_sequence=solution_sequence,
        source_timestamp_s=snapshot.timestamp_s,
        solve_started_s=solve_started_s,
        solve_completed_s=solve_completed_s,
        activated_s=activated_s,
        valid_until_s=snapshot.valid_until_s,
        desired_contact_forces_world_n=tuple(float(value) for value in forces.reshape(-1)),
        rotor_command=command,
        solver_status=result.status,
        solver_time_s=result.solve_time_s,
    )


@dataclass(frozen=True)
class HighRateControlSample:
    """Atomic, calibrated sample required by the high-rate leg path.

    ``go2_body_origin_B_position_world_m`` is the position of the Go2
    kinematic body origin ``B``, not the total-system centre of mass ``C``.
    The high-rate admittance/IK path deliberately stays referenced to ``B``.
    ``rotation_body_to_world`` maps the corresponding Go2 body axes to world.

    The current production LowState bridge cannot yet construct this object;
    that absence is an intentional hardware release blocker.
    """

    landing_session_epoch: int
    ownership_epoch: int
    subscription_generation: int
    estimator_generation: int
    sample_sequence: int
    source_tick: int
    contact_force_sequence: int
    state_estimate_sequence: int
    kinematics_sequence: int
    sample_timestamp_s: float
    receipt_timestamp_s: float
    contact_force_timestamp_s: float
    state_estimate_timestamp_s: float
    kinematics_timestamp_s: float
    force_calibration_hash: str
    leg_order: Tuple[str, str, str, str]
    all_sources_healthy: bool
    force_observation_mode: ForceObservationMode
    ground_normal_world: object
    normal_forces_n: Tuple[float, ...]
    estimated_contact_forces_world_n: object
    rotation_body_to_world: object
    go2_body_origin_B_position_world_m: Tuple[float, ...]
    nominal_foot_positions_world_m: object
    joint_positions_rad: Tuple[float, ...]
    joint_velocities_rad_s: Tuple[float, ...]
    joint_torques_nm: Tuple[float, ...]
    motor_temperatures_c: Tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("landing_session_epoch", "ownership_epoch"):
            object.__setattr__(
                self, name, _strict_integer(name, getattr(self, name), positive=True)
            )
        for name in (
            "subscription_generation",
            "estimator_generation",
            "sample_sequence",
            "source_tick",
            "contact_force_sequence",
            "state_estimate_sequence",
            "kinematics_sequence",
        ):
            object.__setattr__(self, name, _strict_integer(name, getattr(self, name)))
        if self.source_tick > _UINT32_MAX:
            raise ValueError("source_tick must be a uint32")
        for name in (
            "sample_timestamp_s",
            "receipt_timestamp_s",
            "contact_force_timestamp_s",
            "state_estimate_timestamp_s",
            "kinematics_timestamp_s",
        ):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        object.__setattr__(
            self,
            "force_calibration_hash",
            _sha256_identity("force_calibration_hash", self.force_calibration_hash),
        )
        object.__setattr__(
            self,
            "leg_order",
            validate_four_foot_leg_order(self.leg_order, name="leg_order"),
        )
        if type(self.all_sources_healthy) is not bool:
            raise TypeError("all_sources_healthy must be a bool")
        if not isinstance(self.force_observation_mode, ForceObservationMode):
            raise TypeError("force_observation_mode must be a ForceObservationMode")
        if self.force_observation_mode is ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS:
            raise ValueError(
                "HighRateControlSample uses newton-valued force control; raw SDK counts "
                "must use the separate contact-event-only path"
            )
        normal_forces = _nonnegative_tuple("normal_forces_n", self.normal_forces_n, 4)
        estimated_forces = _readonly_array(
            "estimated_contact_forces_world_n",
            self.estimated_contact_forces_world_n,
            (4, 3),
        )
        ground_normal = _readonly_array(
            "ground_normal_world",
            self.ground_normal_world,
            (3,),
        )
        if not math.isclose(
            float(np.linalg.norm(ground_normal)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("ground_normal_world must be a unit vector")
        projected_normal_forces = estimated_forces @ ground_normal
        if not np.allclose(
            projected_normal_forces,
            np.asarray(normal_forces, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "normal_forces_n must equal the ground-normal projection of the "
                "world-frame force observation"
            )
        if self.force_observation_mode is ForceObservationMode.CALIBRATED_NORMAL_ONLY_N:
            uniquely_derived = np.outer(normal_forces, ground_normal)
            if not np.allclose(
                estimated_forces,
                uniquely_derived,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "calibrated normal-only forces must have exactly zero tangential component"
                )
        object.__setattr__(self, "normal_forces_n", normal_forces)
        object.__setattr__(self, "estimated_contact_forces_world_n", estimated_forces)
        object.__setattr__(self, "ground_normal_world", ground_normal)
        rotation = require_rotation_matrix(self.rotation_body_to_world)
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation_body_to_world", rotation)
        object.__setattr__(
            self,
            "go2_body_origin_B_position_world_m",
            _finite_tuple(
                "go2_body_origin_B_position_world_m",
                self.go2_body_origin_B_position_world_m,
                3,
            ),
        )
        object.__setattr__(
            self,
            "nominal_foot_positions_world_m",
            _readonly_array(
                "nominal_foot_positions_world_m",
                self.nominal_foot_positions_world_m,
                (4, 3),
            ),
        )
        for name in (
            "joint_positions_rad",
            "joint_velocities_rad_s",
            "joint_torques_nm",
            "motor_temperatures_c",
        ):
            object.__setattr__(self, name, _finite_tuple(name, getattr(self, name), 12))


def _high_rate_sample_digest(sample: HighRateControlSample) -> str:
    """Hash every immutable sample field so DDS duplicates must be byte-identical."""

    digest = hashlib.sha256()
    scalar_values = (
        sample.landing_session_epoch,
        sample.ownership_epoch,
        sample.subscription_generation,
        sample.estimator_generation,
        sample.sample_sequence,
        sample.source_tick,
        sample.contact_force_sequence,
        sample.state_estimate_sequence,
        sample.kinematics_sequence,
        sample.sample_timestamp_s,
        sample.receipt_timestamp_s,
        sample.contact_force_timestamp_s,
        sample.state_estimate_timestamp_s,
        sample.kinematics_timestamp_s,
        sample.force_calibration_hash,
        sample.leg_order,
        sample.force_observation_mode.value,
        sample.all_sources_healthy,
        sample.normal_forces_n,
        sample.go2_body_origin_B_position_world_m,
        sample.joint_positions_rad,
        sample.joint_velocities_rad_s,
        sample.joint_torques_nm,
        sample.motor_temperatures_c,
    )
    digest.update(repr(scalar_values).encode("utf-8"))
    for value in (
        sample.estimated_contact_forces_world_n,
        sample.ground_normal_world,
        sample.rotation_body_to_world,
        sample.nominal_foot_positions_world_m,
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(array.shape.__repr__().encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class HighRateStepResult:
    success: bool
    status: str
    message: str
    contact: Optional[ContactDetection]
    policy_sequence: Optional[int]
    command: Optional[Go2JointPositionCommand]
    leg_outputs: Tuple[LegAdmittanceOutput, ...]
    owner_result: Optional[OperationResult]
    replan_required: bool


@dataclass(frozen=True)
class HighRateLoopStatus:
    timestamp_s: float
    healthy: bool
    fault_latched: bool
    initialized: bool
    actuation_mode: MultiRateActuationMode
    contact_epoch: int
    contacts: ContactMask
    replan_pending: bool
    replan_grace_authorized: bool
    replan_deadline_s: Optional[float]
    last_sample_sequence: Optional[int]
    last_staged_frame_sequence: Optional[int]
    last_staged_frame_deadline_s: Optional[float]
    last_actuator_applied_frame_sequence: Optional[int]
    last_actuator_applied_policy_sequence: Optional[int]
    last_policy_sequence: Optional[int]
    last_progress_s: Optional[float]
    last_error: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_real("timestamp_s", self.timestamp_s))
        for name in (
            "healthy",
            "fault_latched",
            "initialized",
            "replan_pending",
            "replan_grace_authorized",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.actuation_mode, MultiRateActuationMode):
            raise TypeError("actuation_mode must be a MultiRateActuationMode")
        _strict_integer("contact_epoch", self.contact_epoch)
        object.__setattr__(self, "contacts", _boolean4("contacts", self.contacts))
        for name in (
            "last_sample_sequence",
            "last_staged_frame_sequence",
            "last_actuator_applied_frame_sequence",
            "last_actuator_applied_policy_sequence",
            "last_policy_sequence",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_integer(name, value)
        for name in (
            "replan_deadline_s",
            "last_staged_frame_deadline_s",
            "last_progress_s",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(name, value))
        if (self.last_staged_frame_sequence is None) != (
            self.last_staged_frame_deadline_s is None
        ):
            raise ValueError("last staged frame sequence and deadline must appear together")
        if self.replan_pending != (self.replan_deadline_s is not None):
            raise ValueError("replan pending and deadline must appear together")
        if self.replan_grace_authorized and not self.replan_pending:
            raise ValueError("replan grace requires a pending contact replan")
        if self.initialized and self.last_progress_s is None:
            raise ValueError("an initialized high-rate loop requires a progress timestamp")
        if not isinstance(self.last_error, str):
            raise TypeError("last_error must be a string")


class HighRateLegController:
    """High-rate LowState -> contact -> admittance/IK -> LowCmd path.

    The class has no solver dependency.  Calls must be strictly serialized so
    each LowState sequence advances contact/admittance exactly once.

    中文：每个原子样本绑定 LowState、法向力、状态估计、运动学和足端规划的序号、
    时间戳与 generation。新触地会提升 contact epoch 并要求 MPC 在截止时间内重规划；
    过渡宽限是短时且显式的，不能无限沿用空中策略。
    """

    def __init__(
        self,
        *,
        config: MultiRateExecutionConfig,
        mailbox: LatestPolicyMailbox,
        contact_detector: FootContactDetector,
        leg_controllers: Sequence[LegAdmittanceController],
        controller_leg_order: Tuple[str, str, str, str],
        lowcmd_sink: LowCmdLeaseSink,
        safety: LandingSafetySupervisor,
        force_calibration_hash: str,
        monotonic_clock: Callable[[], float] = time.monotonic,
        hardware_actuation_requested: bool = False,
    ) -> None:
        controllers = tuple(leg_controllers)
        if len(controllers) != 4 or not all(
            isinstance(controller, LegAdmittanceController) for controller in controllers
        ):
            raise TypeError("leg_controllers must contain four LegAdmittanceController values")
        if not isinstance(config, MultiRateExecutionConfig):
            raise TypeError("config must be MultiRateExecutionConfig")
        if not isinstance(mailbox, LatestPolicyMailbox):
            raise TypeError("mailbox must be LatestPolicyMailbox")
        if not isinstance(contact_detector, FootContactDetector):
            raise TypeError("contact_detector must be FootContactDetector")
        if not isinstance(safety, LandingSafetySupervisor):
            raise TypeError("safety must be LandingSafetySupervisor")
        if any(
            not callable(getattr(lowcmd_sink, method, None))
            for method in ("submit", "revoke_mpc_control")
        ):
            raise TypeError("lowcmd_sink does not implement the required contract")
        validated_controller_order = validate_four_foot_leg_order(
            controller_leg_order,
            name="controller_leg_order",
        )
        if validated_controller_order != mailbox.domain().leg_order:
            raise ValueError(
                "controller_leg_order must match the active policy-domain leg order"
            )
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if type(hardware_actuation_requested) is not bool:
            raise TypeError("hardware_actuation_requested must be a bool")
        if hardware_actuation_requested:
            raise ValueError(
                "physical multi-rate actuation remains globally disabled; unresolved gates: "
                + ", ".join(_HARDWARE_ACTUATION_BLOCKERS)
            )
        actuation_mode = safety.require_high_rate_binding(
            mailbox=mailbox,
            lowcmd_sink=lowcmd_sink,
        )
        self._config = config
        self._mailbox = mailbox
        self._contact_detector = contact_detector
        self._leg_controllers = controllers
        self._controller_leg_order = validated_controller_order
        self._lowcmd_sink = lowcmd_sink
        self._safety = safety
        self._force_calibration_hash = _sha256_identity(
            "force_calibration_hash",
            force_calibration_hash,
        )
        self._clock = monotonic_clock
        self._actuation_mode = actuation_mode
        self._last_sample_sequence: Optional[int] = None
        self._last_sample_timestamp_s: Optional[float] = None
        self._last_contact_force_timestamp_s: Optional[float] = None
        self._last_source_tick: Optional[int] = None
        self._last_contact_force_sequence: Optional[int] = None
        self._last_state_estimate_sequence: Optional[int] = None
        self._last_kinematics_sequence: Optional[int] = None
        self._last_state_estimate_timestamp_s: Optional[float] = None
        self._last_kinematics_timestamp_s: Optional[float] = None
        self._last_sample_digest: Optional[str] = None
        self._force_observation_mode: Optional[ForceObservationMode] = None
        self._ground_normal_world: Optional[FloatArray] = None
        self._last_staged_frame_sequence: Optional[int] = None
        self._last_staged_frame_deadline_s: Optional[float] = None
        self._last_actuator_applied_frame_sequence: Optional[int] = None
        self._last_actuator_applied_policy_sequence: Optional[int] = None
        self._last_policy_sequence: Optional[int] = None
        self._subscription_generation: Optional[int] = None
        self._estimator_generation: Optional[int] = None
        self._last_progress_s: Optional[float] = None
        self._initialized = False
        self._fault_latched = False
        self._last_error = ""
        self._contacts: ContactMask = mailbox.domain().contacts
        self._replan_pending = False
        self._replan_grace_authorized = False
        self._replan_deadline_s: Optional[float] = None
        self._lock = asyncio.Lock()

    async def process_sample(self, sample: HighRateControlSample) -> HighRateStepResult:
        """Run one serialized step with a fail-closed outer exception boundary."""

        if not isinstance(sample, HighRateControlSample):
            raise TypeError("sample must be a HighRateControlSample")
        try:
            return await self._process_sample_serialized(sample)
        except asyncio.CancelledError:
            detail = "high-rate leg step was cancelled outside the LowCmd submit boundary"
            self._fault_latched = True
            self._last_error = detail
            abort_task = self._safety.begin_trip(detail)
            await await_nonabandonable(abort_task)
            raise
        except Exception as exc:
            return await self._abort(f"high-rate leg step raised {type(exc).__name__}: {exc}")

    async def _process_sample_serialized(
        self,
        sample: HighRateControlSample,
    ) -> HighRateStepResult:
        async with self._lock:
            if self._actuation_mode is MultiRateActuationMode.SHADOW:
                return self._failure(
                    "high_rate_shadow_only",
                    "LowCmd actuation suppressed because no simulation marker or hardware authorization was supplied",
                )
            if self._fault_latched or self._safety.fault_latched:
                return self._failure("high_rate_fault_latched", "a new landing session is required")
            if self._last_sample_sequence is not None:
                if sample.sample_sequence == self._last_sample_sequence:
                    if _high_rate_sample_digest(sample) != self._last_sample_digest:
                        return await self._abort(
                            "duplicate LowState sequence carried a different atomic payload"
                        )
                    return HighRateStepResult(
                        success=True,
                        status="duplicate_sample_ignored",
                        message="duplicate LowState sample did not advance contact or admittance",
                        contact=None,
                        policy_sequence=self._last_policy_sequence,
                        command=None,
                        leg_outputs=(),
                        owner_result=None,
                        replan_required=self._replan_pending,
                    )
                if sample.sample_sequence < self._last_sample_sequence:
                    return await self._abort("LowState sample sequence moved backwards")
            validation_error = self._sample_error(sample)
            if validation_error is not None:
                return await self._abort(validation_error)

            dt_s = self._config.high_rate_period_s
            if self._last_sample_timestamp_s is not None:
                dt_s = sample.sample_timestamp_s - self._last_sample_timestamp_s
                minimum_dt = self._config.high_rate_period_s - self._config.high_rate_max_jitter_s
                if dt_s < minimum_dt or dt_s > self._config.high_rate_max_gap_s:
                    return await self._abort("high-rate sample interval violated jitter/gap bounds")

            try:
                contact = self._contact_detector.update(
                    sample.normal_forces_n,
                    sample.contact_force_timestamp_s,
                )
            except (TypeError, ValueError) as exc:
                return await self._abort(f"contact detector rejected LowState: {exc}")

            if not self._initialized:
                measured_q = np.asarray(sample.joint_positions_rad, dtype=np.float64).reshape(4, 3)
                initialization_errors = tuple(
                    float(np.max(np.abs(controller.previous_joint_command - measured_q[index])))
                    for index, controller in enumerate(self._leg_controllers)
                )
                if any(
                    error > self._config.initial_joint_alignment_tolerance_rad
                    for error in initialization_errors
                ):
                    return await self._abort(
                        "leg-controller initial joint reference does not match the first "
                        "atomic LowState sample"
                    )

            self._last_sample_sequence = sample.sample_sequence
            self._last_sample_timestamp_s = sample.sample_timestamp_s
            self._last_contact_force_timestamp_s = sample.contact_force_timestamp_s
            self._last_source_tick = sample.source_tick
            self._last_contact_force_sequence = sample.contact_force_sequence
            self._last_state_estimate_sequence = sample.state_estimate_sequence
            self._last_kinematics_sequence = sample.kinematics_sequence
            self._last_state_estimate_timestamp_s = sample.state_estimate_timestamp_s
            self._last_kinematics_timestamp_s = sample.kinematics_timestamp_s
            self._last_sample_digest = _high_rate_sample_digest(sample)
            if self._force_observation_mode is None:
                self._force_observation_mode = sample.force_observation_mode
                self._ground_normal_world = np.array(
                    sample.ground_normal_world,
                    dtype=np.float64,
                    copy=True,
                )
            self._last_progress_s = self._now()
            # A fresh, coherent source frame plus the explicit q comparison
            # above is the initialization barrier.  The controller cannot
            # silently start its rate limiter from a nominal pose that differs
            # from the first atomic LowState frame.
            self._initialized = True
            pre_event_now = self._now()
            pre_event_domain = self._mailbox.domain()
            pre_event_policy = self._mailbox.latest(
                now_s=pre_event_now,
                domain=pre_event_domain,
            )
            interrupted_applied_policy = bool(
                pre_event_policy is not None
                and self._last_staged_frame_sequence is not None
                and self._last_actuator_applied_frame_sequence
                == self._last_staged_frame_sequence
                and self._last_actuator_applied_policy_sequence
                == pre_event_policy.policy_sequence
            )
            event = any(contact.touchdown_events) or any(contact.release_events)
            if event:
                self._contacts = contact.contacts
                self._mailbox.advance_contact(
                    contact.contacts,
                    "measured contact mode changed; old policy invalidated",
                )
                self._safety.request_residual_clear(
                    "measured contact mode changed; clear old residual before replan"
                )
                self._replan_pending = True
                # Measured-force-only grace is legal only as a bounded
                # continuation of a policy with explicit actuator-application
                # evidence. A mailbox stage (or local DDS enqueue) alone is
                # insufficient; startup/ambiguous cases stay in owner hold.
                self._replan_grace_authorized = interrupted_applied_policy
                self._replan_deadline_s = (
                    contact.timestamp_s + self._config.contact_replan_deadline_s
                )
            if any(contact.release_events):
                return await self._abort("post-touchdown contact release was detected")

            now = self._now()
            clear_result = self._safety.residual_clear_result()
            if clear_result is not None and not clear_result.ok:
                return await self._abort(
                    f"contact-replan residual clear failed: {clear_result.code}"
                )
            domain = self._mailbox.domain()
            policy = self._mailbox.latest(now_s=now, domain=domain)
            if (
                self._replan_pending
                and policy is not None
                and (clear_result is None or not clear_result.ok)
            ):
                policy = None
            if policy is None:
                if not self._replan_pending:
                    return HighRateStepResult(
                        success=True,
                        status="initialized_waiting_for_policy",
                        message=(
                            "validated LowState/contact frame; owner safe-hold remains active "
                            "and no LowCmd target was emitted"
                        ),
                        contact=contact,
                        policy_sequence=None,
                        command=None,
                        leg_outputs=(),
                        owner_result=None,
                        replan_required=False,
                    )
                deadline = self._replan_deadline_s
                if deadline is None or now >= deadline:
                    return await self._abort("contact replan deadline expired without a policy")
                if not self._replan_grace_authorized:
                    return HighRateStepResult(
                        success=True,
                        status="contact_replan_waiting_without_grace",
                        message=(
                            "contact changed before any actuator-applied policy was confirmed; "
                            "owner safe-hold remains active and no LowCmd target was emitted"
                        ),
                        contact=contact,
                        policy_sequence=None,
                        command=None,
                        leg_outputs=(),
                        owner_result=None,
                        replan_required=True,
                    )
                desired_forces = np.zeros((4, 3), dtype=np.float64)
                policy_sequence: Optional[int] = None
                leg_source_deadline_s = deadline
            else:
                if policy.domain.contacts != contact.contacts:
                    return await self._abort("policy contact mask disagrees with measured contact")
                desired_forces = np.asarray(
                    policy.desired_contact_forces_world_n,
                    dtype=np.float64,
                ).reshape(4, 3)
                policy_sequence = policy.policy_sequence
                leg_source_deadline_s = policy.valid_until_s
                self._replan_pending = False
                self._replan_grace_authorized = False
                self._replan_deadline_s = None

            try:
                transitions = tuple(
                    controller.preview(
                        current_time_s=sample.sample_timestamp_s,
                        dt_s=dt_s,
                        measured_contact=contact.contacts[index],
                        touchdown_time_s=contact.touchdown_times_s[index],
                        rotation_body_to_world=cast(
                            FloatArray,
                            sample.rotation_body_to_world,
                        ),
                        body_position_world=sample.go2_body_origin_B_position_world_m,
                        nominal_foot_position_world=cast(
                            FloatArray,
                            sample.nominal_foot_positions_world_m,
                        )[index],
                        desired_force_world=desired_forces[index],
                        estimated_force_world=cast(
                            FloatArray,
                            sample.estimated_contact_forces_world_n,
                        )[index],
                    )
                    for index, controller in enumerate(self._leg_controllers)
                )
                for controller, transition in zip(self._leg_controllers, transitions):
                    controller.validate_transition(transition)
                outputs = tuple(transition.output for transition in transitions)
                q = tuple(
                    float(value) for output in outputs for value in output.joint_position_command
                )
                frame_sequence = (
                    0
                    if self._last_staged_frame_sequence is None
                    else self._last_staged_frame_sequence + 1
                )
                command = Go2JointPositionCommand(
                    sequence=frame_sequence,
                    timestamp_s=sample.sample_timestamp_s,
                    valid_until_s=min(
                        sample.sample_timestamp_s + self._config.lowcmd_target_ttl_s,
                        leg_source_deadline_s,
                    ),
                    joint_positions_rad=q,
                    desired_contact_forces_world_n=tuple(
                        float(value) for value in desired_forces.reshape(-1)
                    ),
                    source_policy_sequence=policy_sequence,
                    source_policy_generation=(
                        None if policy is None else policy.domain.invalidation_generation
                    ),
                    source_contact_epoch=domain.contact_epoch,
                )
            except Exception as exc:
                return await self._abort(f"admittance/IK/limit preview failed: {exc}")

            abort_generation = self._safety.abort_generation
            if self._now() + self._config.lowcmd_submit_reserve_s >= command.valid_until_s:
                return await self._abort(
                    "LowCmd source lease lacks the required submit/ACK reserve"
                )
            submit_task = asyncio.create_task(self._lowcmd_sink.submit(command))
            try:
                owner_result = await asyncio.shield(submit_task)
            except asyncio.CancelledError:
                detail = "high-rate LowCmd submission was cancelled"
                self._fault_latched = True
                self._last_error = detail
                # Start FC clear and Go2 revoke before cancelling the submit
                # child.  ImpactAwareLowCmdExecutor performs non-abandonable
                # owner cleanup, which must not serialize the FC fallback.
                abort_task = self._safety.begin_trip(detail)
                if not submit_task.done():
                    submit_task.cancel()
                drain_task = asyncio.create_task(_drain_task(submit_task))
                await await_nonabandonable(drain_task)
                await await_nonabandonable(abort_task)
                raise
            except Exception as exc:
                return await self._abort(
                    f"LowCmd target mailbox raised {type(exc).__name__}: {exc}"
                )
            if not isinstance(owner_result, OperationResult) or not owner_result.ok:
                code = (
                    owner_result.code
                    if isinstance(owner_result, OperationResult)
                    else "invalid_result"
                )
                return await self._abort(f"LowCmd target mailbox rejected high-rate frame: {code}")
            evidence = owner_result.data
            if (
                evidence.get("mailbox_stage_acknowledged") is not True
                or evidence.get("mailbox_staged_target_sequence") != frame_sequence
            ):
                return await self._abort(
                    "LowCmd success lacked an explicit matching mailbox-stage acknowledgement"
                )
            writer_ack = evidence.get("writer_enqueue_acknowledged")
            writer_sequence = evidence.get("writer_enqueued_target_sequence")
            writer_generation = evidence.get("writer_enqueue_generation")
            writer_q = evidence.get("writer_enqueued_q_rad")
            if not (
                (writer_ack is True and writer_sequence == frame_sequence)
                or (writer_ack is False and writer_sequence is None)
            ):
                return await self._abort("LowCmd writer-enqueue evidence was internally inconsistent")
            if writer_ack is True:
                if (
                    isinstance(writer_generation, bool)
                    or not isinstance(writer_generation, int)
                    or writer_generation <= 0
                ):
                    return await self._abort("LowCmd writer ACK lacked a positive generation")
                try:
                    committed_q = _finite_tuple("writer_enqueued_q_rad", writer_q, 12)
                except (TypeError, ValueError) as exc:
                    return await self._abort(f"LowCmd writer ACK lacked limited q: {exc}")
            else:
                if writer_generation is not None or writer_q is not None:
                    return await self._abort(
                        "LowCmd non-ACK carried ambiguous writer generation or limited q"
                    )
                return await self._abort(
                    "LowCmd target was staged but no matching writer-enqueue ACK was available"
                )
            actuator_ack = evidence.get("actuator_application_acknowledged")
            actuator_sequence = evidence.get("actuator_applied_target_sequence")
            if not (
                (actuator_ack is True and actuator_sequence == frame_sequence)
                or (actuator_ack is False and actuator_sequence is None)
            ):
                return await self._abort(
                    "LowCmd actuator-application evidence was internally inconsistent"
                )
            if actuator_ack is True and writer_ack is not True:
                return await self._abort(
                    "LowCmd actuator application cannot precede writer enqueue acknowledgement"
                )
            if (
                self._safety.fault_latched
                or self._safety.abort_generation != abort_generation
                or domain != self._mailbox.domain()
                or (policy is not None and not policy.is_fresh(self._now()))
                or self._now() >= command.valid_until_s
            ):
                return await self._abort(
                    "safety/domain/policy changed while the LowCmd frame was being staged"
                )

            # Commit the software transition against the explicitly staged q
            # for local controller continuity. This is not renamed or promoted
            # to physical application evidence; that identity is tracked
            # separately and hardware remains globally gated.
            try:
                LegAdmittanceController.commit_many(
                    self._leg_controllers,
                    transitions,
                    tuple(
                        committed_q[3 * index : 3 * index + 3]
                        for index in range(4)
                    ),
                )
            except Exception as exc:
                return await self._abort(f"admittance transaction commit failed: {exc}")

            self._last_staged_frame_sequence = frame_sequence
            self._last_staged_frame_deadline_s = command.valid_until_s
            self._last_policy_sequence = policy_sequence
            if actuator_ack is True:
                self._last_actuator_applied_frame_sequence = frame_sequence
                self._last_actuator_applied_policy_sequence = policy_sequence
            self._last_progress_s = self._now()
            return HighRateStepResult(
                success=True,
                status=("contact_replan_grace" if policy is None else "high_rate_frame_staged"),
                message=(
                    "measured-force-only admittance used while waiting for contact replan"
                    if policy is None
                    else "high-rate joint frame accepted by the sole-owner mailbox"
                ),
                contact=contact,
                policy_sequence=policy_sequence,
                command=command,
                leg_outputs=outputs,
                owner_result=owner_result,
                replan_required=self._replan_pending,
            )

    def _sample_error(self, sample: HighRateControlSample) -> Optional[str]:
        now = self._now()
        domain = self._mailbox.domain()
        if (
            sample.landing_session_epoch != domain.landing_session_epoch
            or sample.ownership_epoch != domain.ownership_epoch
        ):
            return "LowState session or ownership epoch mismatch"
        if sample.force_calibration_hash != self._force_calibration_hash:
            return "LowState force calibration hash mismatch"
        if sample.leg_order != domain.leg_order:
            return "LowState/admittance leg order does not match the active policy domain"
        if (
            self._force_observation_mode is not None
            and sample.force_observation_mode is not self._force_observation_mode
        ):
            return "force observation mode changed inside one landing session"
        if self._ground_normal_world is not None and not np.array_equal(
            sample.ground_normal_world,
            self._ground_normal_world,
        ):
            return "ground normal changed inside one landing session"
        if not sample.all_sources_healthy:
            return "one or more high-rate input sources reported unhealthy"
        timestamps = (
            ("sample", sample.sample_timestamp_s, self._config.low_state_max_age_s),
            (
                "contact force",
                sample.contact_force_timestamp_s,
                self._config.contact_force_max_age_s,
            ),
            (
                "state estimate",
                sample.state_estimate_timestamp_s,
                self._config.state_estimate_max_age_s,
            ),
            (
                "kinematics",
                sample.kinematics_timestamp_s,
                self._config.kinematics_max_age_s,
            ),
            ("receipt", sample.receipt_timestamp_s, self._config.low_state_max_age_s),
        )
        for name, timestamp, maximum_age in timestamps:
            if timestamp > now or now - timestamp > maximum_age:
                return f"{name} timestamp is future or stale"
        source_timestamps = tuple(timestamp for _, timestamp, _ in timestamps[:-1])
        if max(source_timestamps) - min(source_timestamps) > self._config.maximum_source_skew_s:
            return "high-rate source timestamp skew exceeds the configured bound"
        if now >= sample.sample_timestamp_s + self._config.lowcmd_target_ttl_s:
            return "LowState sample cannot produce a still-valid LowCmd target"
        if sample.receipt_timestamp_s < sample.sample_timestamp_s:
            return "LowState receipt timestamp precedes its source timestamp"
        if self._subscription_generation is None:
            self._subscription_generation = sample.subscription_generation
            self._estimator_generation = sample.estimator_generation
        elif (
            sample.subscription_generation != self._subscription_generation
            or sample.estimator_generation != self._estimator_generation
        ):
            return "LowState subscription or estimator generation changed"
        if self._last_source_tick is not None:
            # Match the LowState bridge's RFC-1982-style uint32 serial
            # arithmetic: a small positive modular delta is forward progress,
            # including 0xffffffff -> 0.  Zero, the ambiguous half-range, and
            # larger deltas are duplicates or backwards movement.
            tick_delta = (sample.source_tick - self._last_source_tick) & _UINT32_MAX
            if not 0 < tick_delta < _UINT32_SERIAL_HALF_RANGE:
                return "LowState source tick did not increase strictly"
        monotonic_sources = (
            (
                "contact force",
                sample.contact_force_sequence,
                self._last_contact_force_sequence,
                sample.contact_force_timestamp_s,
                self._last_contact_force_timestamp_s,
            ),
            (
                "state estimate",
                sample.state_estimate_sequence,
                self._last_state_estimate_sequence,
                sample.state_estimate_timestamp_s,
                self._last_state_estimate_timestamp_s,
            ),
            (
                "kinematics",
                sample.kinematics_sequence,
                self._last_kinematics_sequence,
                sample.kinematics_timestamp_s,
                self._last_kinematics_timestamp_s,
            ),
        )
        for name, sequence, previous_sequence, timestamp, previous_timestamp in monotonic_sources:
            if previous_sequence is not None and sequence <= previous_sequence:
                return f"{name} source sequence did not increase strictly"
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                return f"{name} source timestamp did not increase strictly"
        return None

    async def _abort(self, reason: str) -> HighRateStepResult:
        self._fault_latched = True
        self._last_error = reason
        result = await self._safety.trip(f"high-rate leg loop: {reason}")
        return HighRateStepResult(
            success=False,
            status="high_rate_fault",
            message=f"{reason}; fallback={result.code}",
            contact=None,
            policy_sequence=self._last_policy_sequence,
            command=None,
            leg_outputs=(),
            owner_result=result,
            replan_required=self._replan_pending,
        )

    def _failure(self, status: str, message: str) -> HighRateStepResult:
        return HighRateStepResult(
            success=False,
            status=status,
            message=message,
            contact=None,
            policy_sequence=self._last_policy_sequence,
            command=None,
            leg_outputs=(),
            owner_result=None,
            replan_required=self._replan_pending,
        )

    def _now(self) -> float:
        return _finite_real("monotonic clock", self._clock())

    def status(self) -> HighRateLoopStatus:
        domain = self._mailbox.domain()
        return HighRateLoopStatus(
            timestamp_s=self._now(),
            healthy=not self._fault_latched and not self._safety.fault_latched,
            fault_latched=self._fault_latched,
            initialized=self._initialized,
            actuation_mode=self._actuation_mode,
            contact_epoch=domain.contact_epoch,
            contacts=self._contacts,
            replan_pending=self._replan_pending,
            replan_grace_authorized=self._replan_grace_authorized,
            replan_deadline_s=self._replan_deadline_s,
            last_sample_sequence=self._last_sample_sequence,
            last_staged_frame_sequence=self._last_staged_frame_sequence,
            last_staged_frame_deadline_s=self._last_staged_frame_deadline_s,
            last_actuator_applied_frame_sequence=(
                self._last_actuator_applied_frame_sequence
            ),
            last_actuator_applied_policy_sequence=(
                self._last_actuator_applied_policy_sequence
            ),
            last_policy_sequence=self._last_policy_sequence,
            last_progress_s=self._last_progress_s,
            last_error=self._last_error,
        )


__all__ = [
    "AsyncLatestMPCWorker",
    "HighRateControlSample",
    "HighRateLegController",
    "HighRateLoopStatus",
    "HighRateStepResult",
    "LandingSafetySupervisor",
    "LatestPolicyMailbox",
    "MPCPolicy",
    "MPCSnapshot",
    "MPCSolverBackend",
    "MPCWorkerStatus",
    "MultiRateActuationMode",
    "MultiRateExecutionConfig",
    "PolicyDomain",
    "PolicyMailboxStatus",
    "SafetySupervisorStatus",
    "SLSQPReferenceSolver",
    "SolverQualification",
    "audit_first_mpc_input",
]
