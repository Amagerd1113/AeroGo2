from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pytest

from aerogo2.common.models import Go2LowLevelStatus, LowCmdOwnershipState
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.admittance import (
    AdmittanceConfig,
    AxisAlignedWorkspace,
    LegAdmittanceController,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetectorConfig,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.coordinator import LandingInputFreshness
from aerogo2.landing.impact_aware.integration import (
    FlightControllerResidualSinkStatus,
    FlightControllerResidualState,
    FlightControllerRotorResidualCommand,
    Go2JointPositionCommand,
)
from aerogo2.landing.impact_aware.multirate import (
    AsyncLatestMPCWorker,
    HighRateControlSample,
    HighRateLegController,
    HighRateLoopStatus,
    LandingSafetySupervisor,
    LatestPolicyMailbox,
    MPCPolicy,
    MPCSnapshot,
    MPCWorkerStatus,
    MultiRateActuationMode,
    MultiRateExecutionConfig,
    PolicyDomain,
    SLSQPReferenceSolver,
    SolverQualification,
    audit_first_mpc_input,
)
from aerogo2.landing.impact_aware.nlp import (
    CONTROL_DIM,
    STATE_DIM,
    ContactForceLimits,
    ImpactAwareMPCProblem,
    ImpactAwareNLP,
    LandingContactGeometry,
    MPCReferences,
    MPCSolveResult,
    MPCWeights,
    RotorExecutionPlan,
    SLSQPSettings,
    StateBounds,
)
from aerogo2.landing.impact_aware.normal_admittance import ForceObservationMode
from aerogo2.landing.impact_aware.types import (
    GO2_SDK_LEG_ORDER,
    FootLeverArmsFromComBodyHorizon,
    ReducedDynamicsConfig,
    ReducedInput,
    ReducedState,
    RotorActuatorConfig,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


class ManualClock:
    def __init__(self, value: float) -> None:
        self._value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value


class FakeLowCmdSink:
    simulation_only = True

    def __init__(
        self,
        *,
        revoke_started: Optional[asyncio.Event] = None,
        revoke_release: Optional[asyncio.Event] = None,
        revoke_finished: Optional[asyncio.Event] = None,
    ) -> None:
        self.commands: list[Go2JointPositionCommand] = []
        self.revoke_reasons: list[str] = []
        self._revoke_started = revoke_started
        self._revoke_release = revoke_release
        self._revoke_finished = revoke_finished

    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        return OperationResult.success(
            "staged and applied by the simulation sink",
            {
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                "writer_enqueue_acknowledged": True,
                "writer_enqueued_target_sequence": command.sequence,
                "writer_enqueue_generation": command.sequence + 1,
                "writer_enqueued_q_rad": command.joint_positions_rad,
                "actuator_application_acknowledged": True,
                "actuator_applied_target_sequence": command.sequence,
            },
            code="TEST_LOWCMD_STAGED",
        )

    async def revoke_mpc_control(self, reason: str) -> OperationResult:
        self.revoke_reasons.append(reason)
        if self._revoke_started is not None:
            self._revoke_started.set()
        if self._revoke_release is not None:
            await self._revoke_release.wait()
        if self._revoke_finished is not None:
            self._revoke_finished.set()
        return OperationResult.success("revoked", code="TEST_LOWCMD_REVOKED")


class LimitedWriterQLowCmdSink(FakeLowCmdSink):
    """Return the q vector accepted after the owner's safety limiting."""

    def __init__(self, limited_q: tuple[float, ...]) -> None:
        super().__init__()
        self._limited_q = limited_q

    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        return OperationResult.success(
            "writer accepted a safety-limited joint frame",
            {
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                "writer_enqueue_acknowledged": True,
                "writer_enqueued_target_sequence": command.sequence,
                "writer_enqueue_generation": command.sequence + 1,
                "writer_enqueued_q_rad": self._limited_q,
                "actuator_application_acknowledged": False,
                "actuator_applied_target_sequence": None,
            },
            code="TEST_LOWCMD_LIMITED_Q",
        )


class _TransientRevokeLowCmdSink(FakeLowCmdSink):
    async def revoke_mpc_control(self, reason: str) -> OperationResult:
        self.revoke_reasons.append(reason)
        if len(self.revoke_reasons) == 1:
            return OperationResult.failure("TEST_REVOKE_TRANSIENT", "retry required")
        return OperationResult.success("revoked on retry", code="TEST_LOWCMD_REVOKED")


class FakeResidualSink:
    simulation_only = True

    def __init__(
        self,
        clock: Callable[[], float],
        *,
        clear_started: Optional[asyncio.Event] = None,
        clear_release: Optional[asyncio.Event] = None,
        clear_finished: Optional[asyncio.Event] = None,
    ) -> None:
        self._clock = clock
        self.commands: list[FlightControllerRotorResidualCommand] = []
        self.clear_reasons: list[str] = []
        self.command_sent = asyncio.Event()
        self._clear_started = clear_started
        self._clear_release = clear_release
        self._clear_finished = clear_finished
        self.status_override: Optional[FlightControllerResidualSinkStatus] = None

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        self.commands.append(command)
        self.command_sent.set()
        return OperationResult.success("activated", code="TEST_FC_ACTIVATED")

    async def clear_rotor_residual(self, reason: str) -> OperationResult:
        self.clear_reasons.append(reason)
        if self._clear_started is not None:
            self._clear_started.set()
        if self._clear_release is not None:
            await self._clear_release.wait()
        if self._clear_finished is not None:
            self._clear_finished.set()
        return OperationResult.success("cleared", code="TEST_FC_CLEARED")

    def status(self) -> FlightControllerResidualSinkStatus:
        if self.status_override is not None:
            return self.status_override
        return FlightControllerResidualSinkStatus(
            timestamp_s=self._clock(),
            healthy=True,
            fault_latched=False,
            residual_state=FlightControllerResidualState.CONFIRMED_ZERO,
            residual_active=False,
            clear_confirmed=True,
            fc_session_id=1,
            last_sequence=None,
            active_valid_until_s=None,
            last_error="",
        )


class ControlledSolver:
    def __init__(
        self,
        result: MPCSolveResult,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        started: Optional[asyncio.Event] = None,
        returned: Optional[asyncio.Event] = None,
        release: Optional[threading.Event] = None,
        before_return: Optional[Callable[[], None]] = None,
    ) -> None:
        self._result = result
        self._loop = loop
        self._started = started
        self._returned = returned
        self._release = release
        self._before_return = before_return
        self.calls: list[float] = []
        self.thread_ids: list[int] = []
        self._qualification = SolverQualification(
            backend_name="test-solver",
            target_identity="TEST_ONLY",
            runtime_build_hash=HASH_A,
            problem_envelope_hash=HASH_B,
            evidence_hash=HASH_C,
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
        assert isinstance(problem, ImpactAwareMPCProblem)
        self.calls.append(timeout_s)
        self.thread_ids.append(threading.get_ident())
        if self._loop is not None and self._started is not None:
            self._loop.call_soon_threadsafe(self._started.set)
        if self._release is not None and not self._release.wait(timeout=1.0):
            raise RuntimeError("test did not release the controlled solver")
        if self._before_return is not None:
            self._before_return()
        if self._loop is not None and self._returned is not None:
            self._loop.call_soon_threadsafe(self._returned.set)
        return self._result


def _config() -> MultiRateExecutionConfig:
    return MultiRateExecutionConfig(
        high_rate_period_s=0.01,
        high_rate_max_jitter_s=0.002,
        high_rate_max_gap_s=0.02,
        lowcmd_target_ttl_s=0.05,
        lowcmd_submit_reserve_s=0.001,
        low_state_max_age_s=0.1,
        contact_force_max_age_s=0.5,
        state_estimate_max_age_s=0.5,
        kinematics_max_age_s=0.5,
        foot_plan_max_age_s=0.5,
        fc_baseline_max_age_s=0.5,
        maximum_source_skew_s=0.01,
        mpc_release_period_s=0.1,
        policy_ttl_s=1.0,
        solver_budget_s=0.5,
        solver_commit_reserve_s=0.05,
        result_audit_budget_s=0.04,
        worker_heartbeat_timeout_s=0.75,
        high_rate_heartbeat_timeout_s=0.05,
        safety_period_s=0.01,
        safety_max_jitter_s=0.002,
        contact_replan_deadline_s=0.04,
        fc_status_max_age_s=0.1,
        mpc_equality_tolerance=1.0e-6,
        mpc_inequality_tolerance=1.0e-6,
        force_zero_tolerance_n=1.0e-8,
        force_constraint_tolerance_n=1.0e-8,
        rotor_thrust_tolerance_n=1.0e-8,
        rotor_rate_tolerance_n_per_s=1.0e-8,
        initial_joint_alignment_tolerance_rad=1.0e-6,
    )


def _domain(*, contact_epoch: int = 0, generation: int = 0) -> PolicyDomain:
    return PolicyDomain(
        landing_session_epoch=1,
        ownership_epoch=2,
        actuation_mode=MultiRateActuationMode.SIMULATION,
        contact_epoch=contact_epoch,
        invalidation_generation=generation,
        contacts=(False, False, False, False),
        leg_order=GO2_SDK_LEG_ORDER,
        configuration_hash=HASH_A,
        model_hash=HASH_B,
    )


def _problem() -> ImpactAwareMPCProblem:
    horizon = 1
    thrust = 4.905
    state = ReducedState(
        position_world_m=np.zeros(3),
        linear_velocity_world_m_per_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        angular_velocity_body_rad_per_s=np.zeros(3),
        rotor_thrusts_n=np.full(4, thrust),
    )
    previous = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.full(4, thrust),
    )
    allocation = np.zeros((6, 4))
    allocation[2, :] = 1.0
    return ImpactAwareMPCProblem(
        initial_state=state,
        previous_input=previous,
        dt_s=0.1,
        contact_schedule=np.zeros((horizon + 1, 4), dtype=int),
        foot_leg_order=GO2_SDK_LEG_ORDER,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            np.zeros((horizon + 1, 4, 3)),
            GO2_SDK_LEG_ORDER,
        ),
        leg_jacobians_body=np.zeros((horizon + 1, 4, 3, 3)),
        joint_velocities_rad_per_s=np.zeros((horizon + 1, 4, 3)),
        references=MPCReferences(
            position_world_m=np.zeros((horizon + 1, 3)),
            linear_velocity_world_m_per_s=np.zeros((horizon + 1, 3)),
            rotation_body_to_world=np.repeat(np.eye(3)[None, :, :], horizon + 1, axis=0),
            angular_velocity_body_rad_per_s=np.zeros((horizon + 1, 3)),
            contact_forces_world_n=np.zeros((horizon, 4, 3)),
            rotor_thrust_commands_n=np.full((horizon, 4), thrust),
        ),
        state_bounds=StateBounds(
            lower=np.full((horizon + 1, STATE_DIM), -np.inf),
            upper=np.full((horizon + 1, STATE_DIM), np.inf),
            soft_mask=np.zeros((horizon, STATE_DIM), dtype=bool),
        ),
        contact_limits=ContactForceLimits(
            friction_coefficients=np.full(4, 0.5),
            maximum_normal_force_n=np.full(4, 100.0),
        ),
        impact_events=(),
        dynamics_config=ReducedDynamicsConfig(
            mass_kg=2.0,
            inertia_body_kg_m2=np.eye(3),
            gravity_world_m_per_s2=np.array([0.0, 0.0, -9.81]),
            rotor_allocation_body=allocation,
        ),
        rotor_actuator_config=RotorActuatorConfig(
            time_constants_s=np.full(4, 0.2),
            thrust_min_n=np.zeros(4),
            thrust_max_n=np.full(4, 20.0),
            thrust_rate_min_n_per_s=np.full(4, -100.0),
            thrust_rate_max_n_per_s=np.full(4, 100.0),
        ),
        weights=MPCWeights(
            tracking=np.eye(12),
            input=np.eye(CONTROL_DIM),
            input_rate=np.eye(CONTROL_DIM),
            slack=np.eye(STATE_DIM),
            terminal_tracking=np.eye(12),
            impulse=np.eye(3),
            touchdown_velocity=np.eye(3),
        ),
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([0.0, 0.0, 1.0]),
            ground_plane_offset_m=-1.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
        rotor_execution_plan=RotorExecutionPlan(
            baseline_thrusts_n=np.full((horizon, 4), thrust),
            correction_gains=np.full(horizon, 0.5),
            maximum_raw_correction_n=np.full(4, 2.0),
        ),
    )


def _solve_result() -> MPCSolveResult:
    nlp = ImpactAwareNLP(_problem())
    decision = nlp.initial_guess()
    trajectories = nlp.unpack(decision)
    equality = nlp.equality_residual(decision)
    inequality = nlp.inequality_residual(decision)
    return MPCSolveResult(
        success=True,
        status="test_optimal",
        message="synthetic feasible first input",
        objective=nlp.objective(decision),
        solve_time_s=0.001,
        iterations=1,
        max_equality_violation=float(np.max(np.abs(equality))),
        min_inequality_residual=float(np.min(inequality)),
        states=trajectories.states,
        pre_impact_states={},
        controls=trajectories.controls,
        slacks=trajectories.slacks,
        impulses_by_step=trajectories.impulses_by_step,
        raw_solver_status=0,
    )


def _freshness(timestamp_s: float, *, maximum_age_s: float = 1.0) -> LandingInputFreshness:
    return LandingInputFreshness(
        state_estimate_timestamp_s=timestamp_s,
        contact_forces_timestamp_s=timestamp_s,
        kinematics_timestamp_s=timestamp_s,
        foot_plan_timestamp_s=timestamp_s,
        flight_controller_baseline_timestamp_s=timestamp_s,
        maximum_source_age_s=maximum_age_s,
        all_sources_healthy=True,
    )


def _snapshot(
    domain: PolicyDomain,
    sequence: int,
    *,
    timestamp_s: float = 1.0,
    valid_until_s: Optional[float] = None,
    maximum_source_age_s: float = 1.0,
) -> MPCSnapshot:
    deadline = timestamp_s + 0.4 if valid_until_s is None else valid_until_s
    return MPCSnapshot(
        domain=domain,
        snapshot_sequence=sequence,
        timestamp_s=timestamp_s,
        valid_until_s=deadline,
        freshness=_freshness(timestamp_s, maximum_age_s=maximum_source_age_s),
        problem=_problem(),
        flight_controller_session_id=1,
        flight_controller_target_tick=sequence + 1,
        flight_controller_baseline_version=sequence,
        flight_controller_baseline_timestamp_s=timestamp_s,
    )


def _rotor_command(
    sequence: int, timestamp_s: float, valid_until_s: float
) -> FlightControllerRotorResidualCommand:
    return FlightControllerRotorResidualCommand(
        sequence=sequence,
        timestamp_s=timestamp_s,
        valid_until_s=valid_until_s,
        fc_session_id=1,
        target_fc_tick=sequence + 1,
        baseline_version=sequence,
        baseline_timestamp_s=timestamp_s,
        baseline_thrusts_n=(5.0, 5.0, 5.0, 5.0),
        transport_raw_residual_thrusts_n=(0.1, 0.1, 0.1, 0.1),
        applied_residual_thrusts_n=(0.1, 0.1, 0.1, 0.1),
        applied_total_thrusts_n=(5.1, 5.1, 5.1, 5.1),
        correction_gain=1.0,
        transport_target_semantics="active_gain_one_transport_target",
    )


def _policy(
    domain: PolicyDomain,
    sequence: int,
    *,
    timestamp_s: float = 0.99,
    valid_until_s: float = 1.5,
) -> MPCPolicy:
    return MPCPolicy(
        domain=domain,
        policy_sequence=sequence,
        solution_sequence=sequence,
        source_timestamp_s=timestamp_s,
        solve_started_s=timestamp_s + 0.001,
        solve_completed_s=timestamp_s + 0.002,
        activated_s=timestamp_s + 0.003,
        valid_until_s=valid_until_s,
        desired_contact_forces_world_n=(0.0,) * 12,
        rotor_command=_rotor_command(sequence, timestamp_s, valid_until_s),
        solver_status="test_optimal",
        solver_time_s=0.001,
    )


def _healthy_go2_status(clock: Callable[[], float]) -> Go2LowLevelStatus:
    now = clock()
    return Go2LowLevelStatus(
        timestamp=now,
        connected=True,
        ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
        owner_epoch=2,
        healthy=True,
        low_state_timestamp=now,
        low_state_age_s=0.0,
        target_sequence=1,
        target_age_s=0.0,
        target_deadline=now + 0.05,
        mailbox_staged_target_sequence=1,
        writer_enqueued_target_sequence=1,
        actuator_applied_target_sequence=1,
        writer_enqueue_generation=1,
        writer_enqueued_q_rad=(0.0,) * 12,
        writer_enqueue_ack_available=True,
        actuator_application_ack_available=True,
        publisher_active=True,
        writer_alive=True,
        watchdog_healthy=True,
        high_level_released=True,
        network_exclusivity_verified=True,
        mapping_hash_verified=True,
        active_mapping_hash=HASH_A,
        fault_reason=None,
    )


def _safety_runtime(
    clock: ManualClock,
    *,
    domain: Optional[PolicyDomain] = None,
    lowcmd: Optional[FakeLowCmdSink] = None,
    residual: Optional[FakeResidualSink] = None,
) -> tuple[LatestPolicyMailbox, FakeLowCmdSink, FakeResidualSink, LandingSafetySupervisor]:
    mailbox = LatestPolicyMailbox(
        _domain() if domain is None else domain,
        monotonic_clock=clock,
    )
    lowcmd_sink = FakeLowCmdSink() if lowcmd is None else lowcmd
    residual_sink = FakeResidualSink(clock) if residual is None else residual
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd_sink,
        residual_sink=residual_sink,
        go2_status=lambda: _healthy_go2_status(clock),
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    return mailbox, lowcmd_sink, residual_sink, safety


def _healthy_high_rate_status(
    clock: Callable[[], float],
    domain: PolicyDomain,
    *,
    policy_sequence: Optional[int] = None,
) -> HighRateLoopStatus:
    now = clock()
    return HighRateLoopStatus(
        timestamp_s=now,
        healthy=True,
        fault_latched=False,
        initialized=True,
        actuation_mode=domain.actuation_mode,
        contact_epoch=domain.contact_epoch,
        contacts=domain.contacts,
        replan_pending=False,
        replan_grace_authorized=False,
        replan_deadline_s=None,
        last_sample_sequence=1,
        last_staged_frame_sequence=1,
        last_staged_frame_deadline_s=now + 0.05,
        last_actuator_applied_frame_sequence=1,
        last_actuator_applied_policy_sequence=policy_sequence,
        last_policy_sequence=policy_sequence,
        last_progress_s=now,
        last_error="",
    )


def _healthy_worker_status(
    clock: Callable[[], float],
    domain: PolicyDomain,
    *,
    activating_sequence: Optional[int] = None,
    activation_valid_until_s: Optional[float] = None,
    published_sequence: Optional[int] = None,
) -> MPCWorkerStatus:
    now = clock()
    return MPCWorkerStatus(
        timestamp_s=now,
        running=True,
        healthy=True,
        reference_only=True,
        actuation_mode=domain.actuation_mode,
        pending_snapshot_sequence=None,
        solving_snapshot_sequence=None,
        last_completed_snapshot_sequence=activating_sequence,
        last_published_policy_sequence=published_sequence,
        solve_started_s=None,
        activating_policy_sequence=activating_sequence,
        activating_domain=(domain if activating_sequence is not None else None),
        activating_fc_session_id=(1 if activating_sequence is not None else None),
        activation_started_s=(now if activating_sequence is not None else None),
        activation_valid_until_s=activation_valid_until_s,
        last_progress_s=now,
        late_result_count=0,
        coalesced_snapshot_count=0,
        last_error="",
    )


def _leg_controller(
    inverse_kinematics: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    *,
    anti_windup_enabled: bool = False,
    forward_kinematics: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> LegAdmittanceController:
    return LegAdmittanceController(
        AdmittanceConfig(
            transition_duration_s=0.2,
            touchdown_inertia=np.ones(3),
            stance_inertia=np.ones(3),
            touchdown_damping=np.ones(3),
            stance_damping=np.ones(3),
            restoring_stiffness=np.ones(3),
            joint_lower=np.full(3, -2.0),
            joint_upper=np.full(3, 2.0),
            joint_rate_limit=np.full(3, 100.0),
            anti_windup_enabled=anti_windup_enabled,
        ),
        (lambda foot: foot) if inverse_kinematics is None else inverse_kinematics,
        AxisAlignedWorkspace(np.full(3, -1.0), np.full(3, 1.0)),
        np.zeros(3),
        forward_kinematics=forward_kinematics,
    )


def _contact_detector() -> FootContactDetector:
    return FootContactDetector(
        ContactDetectorConfig(
            contact_on_threshold_n=np.full(4, 20.0),
            contact_off_threshold_n=np.full(4, 10.0),
            filter_time_constant_s=0.01,
            contact_confirm_s=0.02,
            release_confirm_s=0.02,
        )
    )


def _high_rate_sample(
    *,
    sequence: int = 7,
    source_tick: int = 100,
    timestamp_s: float = 1.0,
    normal_forces_n: object = (0.0, 0.0, 0.0, 0.0),
) -> HighRateControlSample:
    normal_force_array = np.asarray(normal_forces_n, dtype=np.float64)
    return HighRateControlSample(
        landing_session_epoch=1,
        ownership_epoch=2,
        subscription_generation=3,
        estimator_generation=4,
        sample_sequence=sequence,
        source_tick=source_tick,
        contact_force_sequence=sequence,
        state_estimate_sequence=sequence,
        kinematics_sequence=sequence,
        sample_timestamp_s=timestamp_s,
        receipt_timestamp_s=timestamp_s + 0.001,
        contact_force_timestamp_s=timestamp_s - 0.001,
        state_estimate_timestamp_s=timestamp_s - 0.002,
        kinematics_timestamp_s=timestamp_s - 0.003,
        force_calibration_hash=HASH_C,
        leg_order=GO2_SDK_LEG_ORDER,
        all_sources_healthy=True,
        force_observation_mode=ForceObservationMode.CALIBRATED_NORMAL_ONLY_N,
        ground_normal_world=(0.0, 0.0, 1.0),
        normal_forces_n=normal_forces_n,
        estimated_contact_forces_world_n=np.outer(
            normal_force_array,
            np.array([0.0, 0.0, 1.0]),
        ),
        rotation_body_to_world=np.eye(3),
        go2_body_origin_B_position_world_m=(0.0, 0.0, 0.0),
        nominal_foot_positions_world_m=np.zeros((4, 3)),
        joint_positions_rad=(0.0,) * 12,
        joint_velocities_rad_s=(0.0,) * 12,
        joint_torques_nm=(0.0,) * 12,
        motor_temperatures_c=(30.0,) * 12,
    )


async def _eventually(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail(f"condition did not become true within {timeout_s:.3f} s")


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"high_rate_max_jitter_s": 0.01}, "below high_rate_period_s"),
        ({"high_rate_max_gap_s": 0.011}, "cover the period plus allowed jitter"),
        (
            {"high_rate_period_s": 0.1, "high_rate_max_gap_s": 0.103},
            "leg loop must be faster",
        ),
        (
            {"solver_budget_s": 0.95, "solver_commit_reserve_s": 0.05},
            "below policy_ttl_s",
        ),
        ({"result_audit_budget_s": 0.05}, "below the solver commit reserve"),
        ({"contact_replan_deadline_s": 0.051}, "cannot exceed"),
        ({"lowcmd_submit_reserve_s": 0.04}, "fit inside both leg command leases"),
        ({"safety_period_s": 0.049, "safety_max_jitter_s": 0.001}, "fit inside"),
    ),
)
def test_multirate_config_rejects_cross_domain_timing_contracts(
    changes: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **changes)


def test_policy_domain_rejects_a_noncanonical_go2_leg_order() -> None:
    with pytest.raises(ValueError, match="requires leg_order"):
        replace(_domain(), leg_order=("FL", "FR", "RR", "RL"))


def test_snapshot_rejects_problem_leg_order_that_differs_from_domain() -> None:
    domain = _domain()
    wrong_order = ("FL", "FR", "RR", "RL")
    base = _problem()
    mismatched_problem = replace(
        base,
        foot_leg_order=wrong_order,
        foot_lever_arms_from_com_body_m=FootLeverArmsFromComBodyHorizon(
            base.foot_lever_arms_from_com_body_m.values_m,
            wrong_order,
        ),
    )

    with pytest.raises(ValueError, match="foot_leg_order must match"):
        replace(_snapshot(domain, 1), problem=mismatched_problem)


def test_high_rate_controller_rejects_controller_order_that_differs_from_domain() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, _, safety = _safety_runtime(clock)

    with pytest.raises(ValueError, match="controller_leg_order must match"):
        HighRateLegController(
            config=_config(),
            mailbox=mailbox,
            contact_detector=_contact_detector(),
            leg_controllers=tuple(_leg_controller() for _ in range(4)),
            controller_leg_order=("FL", "FR", "RR", "RL"),
            lowcmd_sink=lowcmd,
            safety=safety,
            force_calibration_hash=HASH_C,
            monotonic_clock=clock,
        )


def test_status_exposes_unimplemented_hardware_guarantees() -> None:
    clock = ManualClock(1.0)
    mailbox, _, _, safety = _safety_runtime(clock)
    expected = {
        "cross_device_activation_transaction_unverified",
        "go2_motor_side_application_ack_unavailable",
        "continuous_dds_owner_monitor_unavailable",
        "independent_supervisor_watchdog_unverified",
        "production_atomic_force_sample_unavailable",
        "calibrated_normal_force_pipeline_unverified",
        "normal_force_tracking_error_unvalidated",
    }

    mailbox_status = mailbox.status()
    safety_status = safety.status()

    assert not mailbox_status.hardware_actuation_permitted
    assert set(mailbox_status.hardware_blockers) == expected
    assert not safety_status.hardware_actuation_permitted
    assert set(safety_status.hardware_blockers) == expected


def test_slsqp_reference_solver_can_never_claim_hardware_qualification() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    solver = SLSQPReferenceSolver(
        SLSQPSettings(
            max_iterations=10,
            ftol=1e-8,
            constraint_tolerance=1e-6,
            timeout_s=0.2,
        )
    )

    assert not solver.qualification.approved_for_hardware
    residual.simulation_only = False
    with pytest.raises(ValueError, match="globally disabled"):
        AsyncLatestMPCWorker(
            config=_config(),
            solver=solver,
            mailbox=mailbox,
            residual_sink=residual,
            safety=safety,
            monotonic_clock=clock,
            hardware_actuation_requested=True,
            target_identity="go2-aarch64-test",
            runtime_build_hash=solver.qualification.runtime_build_hash,
            problem_envelope_hash=HASH_B,
            evidence_hash=HASH_C,
            worker_isolation_verified=True,
        )


def test_mailbox_is_latest_only_and_invalidation_fences_old_generation() -> None:
    clock = ManualClock(1.0)
    first_domain = _domain()
    mailbox = LatestPolicyMailbox(first_domain, monotonic_clock=clock)
    first = _policy(first_domain, 10)

    assert mailbox.publish(first, now_s=clock())
    assert mailbox.latest(now_s=clock(), domain=first_domain) is first
    assert not mailbox.publish(_policy(first_domain, 9), now_s=clock())

    next_domain = mailbox.invalidate("operator changed the control generation")
    assert next_domain.invalidation_generation == 1
    assert mailbox.latest(now_s=clock(), domain=first_domain) is None
    assert not mailbox.publish(first, now_s=clock())
    assert not mailbox.publish(_policy(first_domain, 11), now_s=clock())

    current = _policy(next_domain, 11, timestamp_s=0.995)
    assert mailbox.publish(current, now_s=clock())
    assert mailbox.latest(now_s=clock(), domain=next_domain) is current
    clock.set(current.valid_until_s)
    assert mailbox.latest(now_s=clock()) is None


@pytest.mark.asyncio
async def test_worker_runs_off_loop_and_newer_snapshot_suppresses_old_fc_activation() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    loop = asyncio.get_running_loop()
    solver_started = asyncio.Event()
    solver_release = threading.Event()
    solver = ControlledSolver(
        _solve_result(),
        loop=loop,
        started=solver_started,
        release=solver_release,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )
    main_thread = threading.get_ident()

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 10))).ok
    await asyncio.wait_for(solver_started.wait(), timeout=0.5)

    loop_progress = asyncio.Event()
    loop.call_soon(loop_progress.set)
    await asyncio.wait_for(loop_progress.wait(), timeout=0.5)
    assert solver.thread_ids == [solver.thread_ids[0]]
    assert solver.thread_ids[0] != main_thread

    clock.set(1.1)
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 11, timestamp_s=1.1))).ok
    solver_release.set()
    await asyncio.wait_for(residual.command_sent.wait(), timeout=0.5)
    await _eventually(lambda: worker.status().last_published_policy_sequence == 11)

    assert [command.sequence for command in residual.commands] == [11]
    assert mailbox.latest(now_s=clock()).policy_sequence == 11  # type: ignore[union-attr]
    assert worker.status().late_result_count == 1
    assert len(solver.calls) == 2
    assert (await worker.stop("test complete")).ok


@pytest.mark.parametrize(
    ("valid_until_s", "maximum_source_age_s", "completion_s"),
    (
        (1.08, 1.0, 1.08),
        (1.05, 0.05, 1.05),
    ),
)
@pytest.mark.asyncio
async def test_worker_never_renews_source_deadline_or_age_and_trips_both_fallbacks(
    valid_until_s: float,
    maximum_source_age_s: float,
    completion_s: float,
) -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    solver = ControlledSolver(_solve_result(), before_return=lambda: clock.set(completion_s))
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    snapshot = _snapshot(
        mailbox.domain(),
        20,
        valid_until_s=valid_until_s,
        maximum_source_age_s=maximum_source_age_s,
    )
    assert (await worker.submit_snapshot(snapshot)).ok
    await _eventually(lambda: safety.fault_latched)
    await _eventually(lambda: bool(residual.clear_reasons) and bool(lowcmd.revoke_reasons))

    assert residual.commands == []
    assert mailbox.latest(now_s=clock()) is None
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1
    assert "deadline" in safety.status().last_error
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_contact_generation_change_discards_inflight_result_before_fc_send() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    loop = asyncio.get_running_loop()
    solver_started = asyncio.Event()
    solver_returned = asyncio.Event()
    solver_release = threading.Event()
    solver = ControlledSolver(
        _solve_result(),
        loop=loop,
        started=solver_started,
        returned=solver_returned,
        release=solver_release,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 30))).ok
    await asyncio.wait_for(solver_started.wait(), timeout=0.5)
    mailbox.advance_contact(
        (False, False, False, False),
        "new measured contact generation",
    )
    solver_release.set()
    await asyncio.wait_for(solver_returned.wait(), timeout=0.5)
    await _eventually(lambda: worker.status().last_completed_snapshot_sequence == 30)

    assert residual.commands == []
    assert mailbox.latest(now_s=clock()) is None
    assert worker.status().late_result_count == 1
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_contact_invalidation_during_result_audit_cannot_reach_fc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    loop = asyncio.get_running_loop()
    audit_started = asyncio.Event()
    audit_release = threading.Event()
    original_audit = audit_first_mpc_input

    def blocking_audit(
        problem: ImpactAwareMPCProblem,
        result: MPCSolveResult,
        config: MultiRateExecutionConfig,
    ) -> Optional[str]:
        loop.call_soon_threadsafe(audit_started.set)
        if not audit_release.wait(timeout=0.5):
            raise RuntimeError("test did not release the result audit")
        return original_audit(problem, result, config)

    monkeypatch.setattr(
        "aerogo2.landing.impact_aware.multirate.audit_first_mpc_input",
        blocking_audit,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 33))).ok
    await asyncio.wait_for(audit_started.wait(), timeout=0.5)
    mailbox.advance_contact(
        (False, False, False, False),
        "contact generation changed while result audit was running",
    )
    audit_release.set()
    await _eventually(lambda: worker.status().solving_snapshot_sequence is None)

    assert residual.commands == []
    assert mailbox.latest(now_s=clock()) is None
    assert worker.status().late_result_count == 1
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_worker_does_not_trust_caller_supplied_source_age() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    solver = ControlledSolver(_solve_result())
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )
    stale_sources = _freshness(0.49, maximum_age_s=100.0)
    snapshot = replace(
        _snapshot(mailbox.domain(), 31, valid_until_s=1.05, maximum_source_age_s=100.0),
        freshness=stale_sources,
        flight_controller_baseline_timestamp_s=0.49,
    )

    assert (await worker.start()).ok
    rejected = await worker.submit_snapshot(snapshot)

    assert not rejected.ok
    assert rejected.code == "MPC_SNAPSHOT_STALE"
    assert "configured maximum age" in str(rejected.data.get("source_failure"))
    assert solver.calls == []
    assert residual.commands == []
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_worker_caps_policy_lease_by_each_trusted_source_deadline() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    solver = ControlledSolver(_solve_result())
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )
    caller_claims_long_age = _freshness(0.99, maximum_age_s=100.0)
    snapshot = replace(
        _snapshot(mailbox.domain(), 32, valid_until_s=1.6, maximum_source_age_s=100.0),
        freshness=caller_claims_long_age,
        flight_controller_baseline_timestamp_s=0.99,
    )

    assert (await worker.start()).ok
    rejected = await worker.submit_snapshot(snapshot)

    assert not rejected.ok
    assert rejected.code == "MPC_SNAPSHOT_TTL_INVALID"
    assert rejected.data["trusted_deadline_s"] == pytest.approx(1.49)
    assert solver.calls == []
    assert residual.commands == []
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_high_rate_path_uses_force_timestamp_once_and_separates_policy_and_frame_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    policy = _policy(mailbox.domain(), 41)
    assert mailbox.publish(policy, now_s=clock())
    controllers = tuple(_leg_controller() for _ in range(4))

    def solver_must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("the high-rate path called the MPC solver")

    monkeypatch.setattr(ImpactAwareNLP, "solve", solver_must_not_run)
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=controllers,
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    sample = _high_rate_sample()

    first = await high_rate.process_sample(sample)
    generations_after_first = tuple(controller.generation for controller in controllers)
    duplicate = await high_rate.process_sample(sample)

    assert first.success
    assert first.contact is not None
    assert first.contact.timestamp_s == sample.contact_force_timestamp_s
    assert first.policy_sequence == 41
    assert first.command is not None
    assert first.command.sequence == 0
    assert first.command.source_policy_sequence == 41
    assert first.command.source_policy_generation == policy.domain.invalidation_generation
    assert first.command.source_contact_epoch == policy.domain.contact_epoch
    assert duplicate.success
    assert duplicate.status == "duplicate_sample_ignored"
    assert duplicate.command is None
    assert tuple(controller.generation for controller in controllers) == generations_after_first
    assert generations_after_first == (1, 1, 1, 1)
    assert len(lowcmd.commands) == 1
    assert residual.clear_reasons == []
    assert lowcmd.revoke_reasons == []


@pytest.mark.asyncio
async def test_writer_limited_q_is_committed_as_admittance_feedback() -> None:
    clock = ManualClock(1.002)
    limited_leg_q = (0.125, 0.0, 0.0)
    limited_q = limited_leg_q * 4
    lowcmd = LimitedWriterQLowCmdSink(limited_q)
    mailbox, _, residual, safety = _safety_runtime(clock, lowcmd=lowcmd)
    assert mailbox.publish(_policy(mailbox.domain(), 42), now_s=clock())
    controllers = tuple(
        _leg_controller(
            anti_windup_enabled=True,
            forward_kinematics=lambda joint: joint,
        )
        for _ in range(4)
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=controllers,
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    result = await high_rate.process_sample(_high_rate_sample())

    assert result.success
    assert result.owner_result is not None
    assert result.owner_result.data["writer_enqueued_q_rad"] == limited_q
    for controller in controllers:
        np.testing.assert_allclose(controller.previous_joint_command, limited_leg_q)
        np.testing.assert_allclose(
            controller.state.correction_position_body,
            limited_leg_q,
        )
        np.testing.assert_allclose(
            controller.state.correction_velocity_body,
            np.zeros(3),
        )
    assert residual.clear_reasons == []
    assert lowcmd.revoke_reasons == []


@pytest.mark.asyncio
async def test_high_rate_sample_leg_order_mismatch_fails_closed_before_lowcmd() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 43), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    mismatched = replace(
        _high_rate_sample(),
        leg_order=("FL", "FR", "RR", "RL"),
    )

    result = await high_rate.process_sample(mismatched)

    assert not result.success
    assert "leg order does not match" in result.message
    assert safety.fault_latched
    assert lowcmd.commands == []
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_high_rate_accepts_uint32_lowstate_tick_wrap() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 42), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    first = await high_rate.process_sample(_high_rate_sample(source_tick=0xFFFFFFFF))
    clock.set(1.012)
    wrapped = await high_rate.process_sample(
        _high_rate_sample(sequence=8, source_tick=0, timestamp_s=1.01)
    )

    assert first.success
    assert wrapped.success
    assert wrapped.status == "high_rate_frame_staged"
    assert len(lowcmd.commands) == 2
    assert residual.clear_reasons == []
    assert lowcmd.revoke_reasons == []


@pytest.mark.asyncio
@pytest.mark.parametrize("next_tick", [100, 99])
async def test_high_rate_rejects_duplicate_or_backwards_uint32_lowstate_tick(
    next_tick: int,
) -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 43), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    assert (await high_rate.process_sample(_high_rate_sample(source_tick=100))).success
    clock.set(1.012)

    rejected = await high_rate.process_sample(
        _high_rate_sample(sequence=8, source_tick=next_tick, timestamp_s=1.01)
    )

    assert not rejected.success
    assert "source tick did not increase strictly" in rejected.message
    assert safety.fault_latched
    assert len(lowcmd.commands) == 1
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1


def test_high_rate_sample_rejects_source_tick_outside_uint32() -> None:
    with pytest.raises(ValueError, match="source_tick must be a uint32"):
        _high_rate_sample(source_tick=0x1_0000_0000)


def test_high_rate_sample_rejects_contradictory_normal_and_world_force() -> None:
    with pytest.raises(ValueError, match="ground-normal projection"):
        replace(
            _high_rate_sample(),
            normal_forces_n=(100.0,) * 4,
            estimated_contact_forces_world_n=np.zeros((4, 3)),
        )


def test_high_rate_sample_rejects_raw_counts_in_newton_force_path() -> None:
    with pytest.raises(ValueError, match="contact-event-only path"):
        replace(
            _high_rate_sample(),
            force_observation_mode=ForceObservationMode.CONTACT_EVENT_ONLY_COUNTS,
        )


def test_independent_3d_force_requires_matching_normal_projection() -> None:
    forces = np.array(
        [
            [2.0, -3.0, 10.0],
            [0.0, 4.0, 20.0],
            [-1.0, 0.0, 30.0],
            [3.0, 2.0, 40.0],
        ]
    )
    sample = replace(
        _high_rate_sample(),
        force_observation_mode=ForceObservationMode.INDEPENDENT_3D_WORLD_N,
        normal_forces_n=(10.0, 20.0, 30.0, 40.0),
        estimated_contact_forces_world_n=forces,
    )

    np.testing.assert_allclose(sample.estimated_contact_forces_world_n, forces)
    assert sample.normal_forces_n == pytest.approx((10.0, 20.0, 30.0, 40.0))


@pytest.mark.asyncio
async def test_empty_mailbox_initializes_high_rate_heartbeat_without_emitting_lowcmd() -> None:
    clock = ManualClock(1.002)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    go2_holding = replace(
        _healthy_go2_status(clock),
        ownership_state=LowCmdOwnershipState.HOLDING,
        target_sequence=None,
        mailbox_staged_target_sequence=None,
        target_age_s=None,
        target_deadline=None,
        safe_hold_active=True,
        safe_hold_settled=True,
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: go2_holding,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    safety.attach_monitors(
        high_rate_status=high_rate.status,
        worker_status=worker.status,
    )

    assert (await worker.start()).ok
    initialized = await high_rate.process_sample(_high_rate_sample())
    safety_result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert initialized.success
    assert initialized.status == "initialized_waiting_for_policy"
    assert initialized.command is None
    assert initialized.contact is not None
    assert high_rate.status().initialized
    assert lowcmd.commands == []
    assert safety_result.ok
    assert safety_result.code == "MULTIRATE_SAFETY_HEALTHY"
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_first_lowstate_joint_reference_mismatch_fails_before_admittance() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    controllers = tuple(_leg_controller() for _ in range(4))
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=controllers,
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    mismatched = replace(
        _high_rate_sample(),
        joint_positions_rad=(0.01,) + (0.0,) * 11,
    )

    result = await high_rate.process_sample(mismatched)

    assert not result.success
    assert "initial joint reference" in result.message
    assert not high_rate.status().initialized
    assert tuple(controller.generation for controller in controllers) == (0, 0, 0, 0)
    assert lowcmd.commands == []
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1


@pytest.mark.parametrize(
    "changes",
    (
        {"force_observation_mode": ForceObservationMode.INDEPENDENT_3D_WORLD_N},
        {"ground_normal_world": (1.0, 0.0, 0.0)},
    ),
)
@pytest.mark.asyncio
async def test_force_semantics_cannot_change_inside_landing_session(
    changes: dict[str, object],
) -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    assert (await high_rate.process_sample(_high_rate_sample())).success
    clock.set(1.012)
    changed = replace(
        _high_rate_sample(sequence=8, source_tick=101, timestamp_s=1.01),
        **changes,
    )

    result = await high_rate.process_sample(changed)

    assert not result.success
    assert result.status == "high_rate_fault"
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1


@pytest.mark.asyncio
async def test_nth_leg_ik_failure_is_atomic_and_trips_fc_and_go2_fallbacks() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 50), now_s=clock())

    def failing_ik(_: np.ndarray) -> np.ndarray:
        raise RuntimeError("synthetic fourth-leg IK failure")

    controllers = (
        _leg_controller(),
        _leg_controller(),
        _leg_controller(),
        _leg_controller(failing_ik),
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=controllers,
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    result = await high_rate.process_sample(_high_rate_sample())

    assert not result.success
    assert result.status == "high_rate_fault"
    assert "fourth-leg IK failure" in result.message
    assert lowcmd.commands == []
    assert tuple(controller.generation for controller in controllers) == (0, 0, 0, 0)
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1
    assert mailbox.latest(now_s=clock()) is None


@pytest.mark.asyncio
async def test_manual_override_cancellation_cannot_abandon_either_fallback() -> None:
    clock = ManualClock(1.0)
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    clear_finished = asyncio.Event()
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    revoke_finished = asyncio.Event()
    lowcmd = FakeLowCmdSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
        revoke_finished=revoke_finished,
    )
    residual = FakeResidualSink(
        clock,
        clear_started=clear_started,
        clear_release=clear_release,
        clear_finished=clear_finished,
    )
    _, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)

    observer = asyncio.create_task(
        safety.run_once(manual_override=True, require_active_policy=False)
    )
    await asyncio.wait_for(
        asyncio.gather(clear_started.wait(), revoke_started.wait()),
        timeout=0.5,
    )
    assert safety.fault_latched
    assert safety.abort_generation == 1

    observer.cancel()
    await asyncio.sleep(0)
    assert not observer.done()
    assert not clear_finished.is_set()
    assert not revoke_finished.is_set()

    clear_release.set()
    revoke_release.set()
    with pytest.raises(asyncio.CancelledError):
        await observer

    assert clear_finished.is_set()
    assert revoke_finished.is_set()
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1


@pytest.mark.asyncio
async def test_cancelling_begin_trip_handle_cannot_cancel_dual_fallback() -> None:
    clock = ManualClock(1.0)
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    lowcmd = FakeLowCmdSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
    )
    residual = FakeResidualSink(
        clock,
        clear_started=clear_started,
        clear_release=clear_release,
    )
    _, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)

    public_handle = safety.begin_trip("synthetic immediate cancellation")
    public_handle.cancel()
    await asyncio.wait_for(
        asyncio.gather(clear_started.wait(), revoke_started.wait()),
        timeout=0.5,
    )

    clear_release.set()
    revoke_release.set()
    fallback = await safety.trip("join existing trip")
    assert public_handle.cancelled()
    assert fallback.ok
    assert len(residual.clear_reasons) == 1
    assert len(lowcmd.revoke_reasons) == 1


@pytest.mark.asyncio
async def test_failed_dual_fallback_can_be_retried_idempotently() -> None:
    clock = ManualClock(1.0)
    lowcmd = _TransientRevokeLowCmdSink()
    residual = FakeResidualSink(clock)
    _, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)

    first = await safety.trip("synthetic transient revoke failure")
    second = await safety.trip("retry the same latched safety trip")

    assert not first.ok
    assert first.data["go2_code"] == "TEST_REVOKE_TRANSIENT"
    assert second.ok
    assert second.code == "MULTIRATE_FALLBACK_CONFIRMED"
    assert safety.abort_generation == 1
    assert len(lowcmd.revoke_reasons) == 2
    assert len(residual.clear_reasons) == 2


def test_first_input_audit_rejects_solver_claims_that_violate_release_constraints() -> None:
    problem = _problem()
    result = _solve_result()
    noncontact_force = ReducedInput(
        contact_forces_world_n=np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        rotor_thrust_commands_n=result.controls[0].rotor_thrust_commands_n,
    )
    bad_force = replace(result, controls=(noncontact_force,))
    assert audit_first_mpc_input(problem, bad_force, _config()) is not None

    excessive_rotor = ReducedInput(
        contact_forces_world_n=np.zeros((4, 3)),
        rotor_thrust_commands_n=np.full(4, 25.0),
    )
    bad_rotor = replace(result, controls=(excessive_rotor,))
    assert audit_first_mpc_input(problem, bad_rotor, _config()) is not None

    bad_diagnostic = replace(result, max_equality_violation=float("nan"))
    assert "diagnostics" in str(audit_first_mpc_input(problem, bad_diagnostic, _config()))


def test_first_input_audit_uses_configured_ground_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release audit must not silently assume that world +Z is ground normal."""

    class _FeasibleAuditNLP:
        def __init__(self, problem: ImpactAwareMPCProblem) -> None:
            self.problem = problem

        def initial_guess(self, result: MPCSolveResult) -> np.ndarray:
            del result
            return np.zeros(1)

        def equality_residual(self, decision: np.ndarray) -> np.ndarray:
            del decision
            return np.zeros(1)

        def inequality_residual(self, decision: np.ndarray) -> np.ndarray:
            del decision
            return np.zeros(1)

        def variable_bound_residual(self, decision: np.ndarray) -> np.ndarray:
            del decision
            return np.zeros(1)

    monkeypatch.setattr(
        "aerogo2.landing.impact_aware.multirate.ImpactAwareNLP",
        _FeasibleAuditNLP,
    )
    body_z_aligned_with_world_x = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    base = _problem()
    problem = replace(
        base,
        initial_state=replace(
            base.initial_state,
            rotation_body_to_world=body_z_aligned_with_world_x,
        ),
        contact_schedule=np.ones_like(base.contact_schedule),
        references=replace(
            base.references,
            rotation_body_to_world=np.repeat(
                body_z_aligned_with_world_x[None, :, :],
                base.references.horizon + 1,
                axis=0,
            ),
        ),
        landing_contact_geometry=LandingContactGeometry(
            ground_normal_world=np.array([1.0, 0.0, 0.0]),
            ground_plane_offset_m=0.0,
            touchdown_position_tolerance_m=0.01,
            minimum_downward_speed_m_per_s=0.0,
            maximum_tilt_from_ground_normal_rad=np.deg2rad(30.0),
        ),
    )
    result = _solve_result()
    along_configured_normal = ReducedInput(
        contact_forces_world_n=np.array(
            [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        rotor_thrust_commands_n=result.controls[0].rotor_thrust_commands_n,
    )
    assert audit_first_mpc_input(
        problem,
        replace(result, controls=(along_configured_normal,)),
        _config(),
    ) is None

    world_z_is_tangential_here = replace(
        along_configured_normal,
        contact_forces_world_n=np.array(
            [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )
    rejection = audit_first_mpc_input(
        problem,
        replace(result, controls=(world_z_is_tangential_here,)),
        _config(),
    )
    assert rejection is not None
    assert "friction cone" in rejection


@pytest.mark.asyncio
async def test_bad_first_input_never_reaches_fc_and_trips_both_sides() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    bad_control = ReducedInput(
        contact_forces_world_n=np.array(
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        rotor_thrust_commands_n=_solve_result().controls[0].rotor_thrust_commands_n,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(replace(_solve_result(), controls=(bad_control,))),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 60))).ok
    await _eventually(lambda: safety.fault_latched)
    await _eventually(lambda: bool(lowcmd.revoke_reasons) and bool(residual.clear_reasons))

    assert residual.commands == []
    assert mailbox.latest(now_s=clock()) is None
    assert "first-input audit" in safety.status().last_error
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_active_session_without_monitors_fails_closed() -> None:
    clock = ManualClock(1.0)
    _, lowcmd, residual, safety = _safety_runtime(clock)

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert safety.fault_latched
    assert "missing mandatory loop monitors" in safety.status().last_error
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_monitor_exception_fails_closed_instead_of_killing_safety_task() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    domain = mailbox.domain()

    def broken_high_rate_status() -> HighRateLoopStatus:
        raise RuntimeError("synthetic monitor failure")

    safety.attach_monitors(
        high_rate_status=broken_high_rate_status,
        worker_status=lambda: _healthy_worker_status(clock, domain),
    )
    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert safety.fault_latched
    assert "monitor raised RuntimeError" in safety.status().last_error
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_malformed_typed_go2_status_fails_closed_instead_of_escaping() -> None:
    clock = ManualClock(1.0)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    malformed = replace(
        _healthy_go2_status(clock),
        timestamp="not-a-number",  # type: ignore[arg-type]
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: malformed,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert "owner health/age/ownership check failed" in safety.status().last_error
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_exact_fc_pending_transaction_is_healthy_but_identity_mismatch_trips() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    domain = mailbox.domain()
    old_policy = _policy(domain, 41, valid_until_s=1.5)
    assert mailbox.publish(old_policy, now_s=clock())
    activation_deadline = 1.6
    high_status = _healthy_high_rate_status(clock, domain, policy_sequence=41)
    worker_status = _healthy_worker_status(
        clock,
        domain,
        activating_sequence=42,
        activation_valid_until_s=activation_deadline,
        published_sequence=41,
    )
    safety.attach_monitors(
        high_rate_status=lambda: high_status,
        worker_status=lambda: worker_status,
    )
    residual.status_override = FlightControllerResidualSinkStatus(
        timestamp_s=clock(),
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.STAGE_PENDING,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=1,
        last_sequence=42,
        active_valid_until_s=old_policy.valid_until_s,
        last_error="",
        active_command_sequence=41,
        pending_command_sequence=42,
        pending_started_s=clock(),
        pending_valid_until_s=activation_deadline,
    )

    healthy = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )
    assert healthy.ok
    assert healthy.code == "MULTIRATE_SAFETY_HEALTHY"

    residual.status_override = replace(
        residual.status_override,
        last_sequence=43,
        pending_command_sequence=43,
    )
    tripped = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )
    assert not tripped.ok
    assert tripped.code == "MULTIRATE_SAFETY_TRIPPED"
    assert tripped.data["fallback_confirmed"] is True
    assert safety.fault_latched
    assert "pending residual" in safety.status().last_error
    assert len(lowcmd.revoke_reasons) == 1


@pytest.mark.asyncio
async def test_activation_accepts_explained_old_active_before_new_stage_is_consumed() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    domain = mailbox.domain()
    old_policy = _policy(domain, 41, valid_until_s=1.5)
    assert mailbox.publish(old_policy, now_s=clock())
    activation_deadline = 1.6
    safety.attach_monitors(
        high_rate_status=lambda: _healthy_high_rate_status(
            clock,
            domain,
            policy_sequence=41,
        ),
        worker_status=lambda: _healthy_worker_status(
            clock,
            domain,
            activating_sequence=42,
            activation_valid_until_s=activation_deadline,
            published_sequence=41,
        ),
    )
    residual.status_override = FlightControllerResidualSinkStatus(
        timestamp_s=clock(),
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.ACTIVE,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=1,
        last_sequence=41,
        active_valid_until_s=old_policy.valid_until_s,
        last_error="",
        active_command_sequence=41,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )

    assert result.ok
    assert result.code == "MULTIRATE_SAFETY_HEALTHY"


@pytest.mark.asyncio
async def test_activation_accepts_old_active_after_candidate_watermark_is_consumed() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    domain = mailbox.domain()
    old_policy = _policy(domain, 41, valid_until_s=1.5)
    assert mailbox.publish(old_policy, now_s=clock())
    activation_deadline = 1.6
    safety.attach_monitors(
        high_rate_status=lambda: _healthy_high_rate_status(
            clock,
            domain,
            policy_sequence=41,
        ),
        worker_status=lambda: _healthy_worker_status(
            clock,
            domain,
            activating_sequence=42,
            activation_valid_until_s=activation_deadline,
            published_sequence=41,
        ),
    )
    residual.status_override = FlightControllerResidualSinkStatus(
        timestamp_s=clock(),
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.ACTIVE,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=1,
        last_sequence=42,
        active_valid_until_s=old_policy.valid_until_s,
        last_error="",
        active_command_sequence=41,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )

    assert result.ok
    assert result.code == "MULTIRATE_SAFETY_HEALTHY"


def test_loop_status_models_reject_truthy_non_boolean_health_fields() -> None:
    clock = ManualClock(1.0)
    domain = _domain()
    with pytest.raises(TypeError, match="healthy must be a bool"):
        replace(
            _healthy_high_rate_status(clock, domain),
            healthy=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="running must be a bool"):
        replace(
            _healthy_worker_status(clock, domain),
            running=1,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_policy_deadline_caps_every_high_rate_lowcmd_lease() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, _, safety = _safety_runtime(clock)
    policy = _policy(mailbox.domain(), 70, valid_until_s=1.01)
    assert mailbox.publish(policy, now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    result = await high_rate.process_sample(_high_rate_sample())

    assert result.success
    assert result.command is not None
    assert result.command.valid_until_s == policy.valid_until_s


@pytest.mark.asyncio
async def test_contact_replan_deadline_caps_measured_force_only_lowcmd_lease() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, _, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 71), now_s=clock())
    detector = FootContactDetector(
        ContactDetectorConfig(
            contact_on_threshold_n=np.full(4, 20.0),
            contact_off_threshold_n=np.full(4, 10.0),
            filter_time_constant_s=0.001,
            contact_confirm_s=0.005,
            release_confirm_s=0.005,
        )
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=detector,
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    high_forces = (100.0, 100.0, 100.0, 100.0)
    first = await high_rate.process_sample(_high_rate_sample(normal_forces_n=high_forces))
    assert first.success

    clock.set(1.012)
    touchdown = await high_rate.process_sample(
        _high_rate_sample(
            sequence=8,
            source_tick=101,
            timestamp_s=1.01,
            normal_forces_n=high_forces,
        )
    )

    assert touchdown.success
    assert touchdown.status == "contact_replan_grace"
    assert touchdown.replan_required
    assert touchdown.contact is not None
    assert all(touchdown.contact.touchdown_events)
    assert touchdown.command is not None
    expected_deadline = 1.009 + _config().contact_replan_deadline_s
    assert touchdown.command.valid_until_s == pytest.approx(expected_deadline)
    assert touchdown.command.source_policy_sequence is None
    assert mailbox.latest(now_s=clock()) is None


@pytest.mark.asyncio
async def test_startup_contact_without_applied_policy_cannot_emit_grace_lowcmd() -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, _, safety = _safety_runtime(clock)
    detector = FootContactDetector(
        ContactDetectorConfig(
            contact_on_threshold_n=np.full(4, 20.0),
            contact_off_threshold_n=np.full(4, 10.0),
            filter_time_constant_s=0.001,
            contact_confirm_s=0.005,
            release_confirm_s=0.005,
        )
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=detector,
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    high_forces = (100.0, 100.0, 100.0, 100.0)
    initialized = await high_rate.process_sample(_high_rate_sample(normal_forces_n=high_forces))
    assert initialized.status == "initialized_waiting_for_policy"

    clock.set(1.012)
    contact_without_policy = await high_rate.process_sample(
        _high_rate_sample(
            sequence=8,
            source_tick=101,
            timestamp_s=1.01,
            normal_forces_n=high_forces,
        )
    )

    assert contact_without_policy.success
    assert contact_without_policy.status == "contact_replan_waiting_without_grace"
    assert contact_without_policy.replan_required
    assert contact_without_policy.command is None
    assert lowcmd.commands == []


class _MailboxOnlyLowCmdSink(FakeLowCmdSink):
    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        return OperationResult.success(
            "mailbox stage only",
            {
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                "writer_enqueue_acknowledged": False,
                "writer_enqueued_target_sequence": None,
                "writer_enqueue_generation": None,
                "writer_enqueued_q_rad": None,
                "actuator_application_acknowledged": False,
                "actuator_applied_target_sequence": None,
            },
            code="TEST_LOWCMD_MAILBOX_ONLY",
        )


class _ImpossibleApplicationLowCmdSink(FakeLowCmdSink):
    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        return OperationResult.success(
            "contradictory application evidence",
            {
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                "writer_enqueue_acknowledged": False,
                "writer_enqueued_target_sequence": None,
                "writer_enqueue_generation": None,
                "writer_enqueued_q_rad": None,
                "actuator_application_acknowledged": True,
                "actuator_applied_target_sequence": command.sequence,
            },
            code="TEST_LOWCMD_IMPOSSIBLE_APPLICATION",
        )


@pytest.mark.asyncio
async def test_mailbox_stage_is_not_treated_as_applied_policy_for_contact_grace() -> None:
    clock = ManualClock(1.002)
    lowcmd = _MailboxOnlyLowCmdSink()
    mailbox, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd)
    assert mailbox.publish(_policy(mailbox.domain(), 72), now_s=clock())
    detector = FootContactDetector(
        ContactDetectorConfig(
            contact_on_threshold_n=np.full(4, 20.0),
            contact_off_threshold_n=np.full(4, 10.0),
            filter_time_constant_s=0.001,
            contact_confirm_s=0.005,
            release_confirm_s=0.005,
        )
    )
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=detector,
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    high_forces = (100.0, 100.0, 100.0, 100.0)
    rejected = await high_rate.process_sample(
        _high_rate_sample(normal_forces_n=high_forces)
    )

    assert not rejected.success
    assert "no matching writer-enqueue ACK" in rejected.message
    assert safety.fault_latched
    assert len(lowcmd.commands) == 1


@pytest.mark.asyncio
async def test_actuator_application_ack_requires_writer_enqueue_ack() -> None:
    clock = ManualClock(1.002)
    lowcmd = _ImpossibleApplicationLowCmdSink()
    mailbox, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd)
    assert mailbox.publish(_policy(mailbox.domain(), 73), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    result = await high_rate.process_sample(_high_rate_sample())

    assert not result.success
    assert "no matching writer-enqueue ACK" in result.message
    assert safety.fault_latched


class _BlockingLowCmdSink(FakeLowCmdSink):
    def __init__(self) -> None:
        super().__init__()
        self.submit_started = asyncio.Event()
        self.submit_release = asyncio.Event()

    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        self.submit_started.set()
        await self.submit_release.wait()
        return OperationResult.success(
            "staged and applied by the simulation sink",
            {
                "mailbox_stage_acknowledged": True,
                "mailbox_staged_target_sequence": command.sequence,
                "writer_enqueue_acknowledged": True,
                "writer_enqueued_target_sequence": command.sequence,
                "writer_enqueue_generation": command.sequence + 1,
                "writer_enqueued_q_rad": command.joint_positions_rad,
                "actuator_application_acknowledged": True,
                "actuator_applied_target_sequence": command.sequence,
            },
            code="TEST_LOWCMD_STAGED",
        )


@pytest.mark.asyncio
async def test_concurrent_leg_reset_cannot_partially_commit_four_leg_transaction() -> None:
    clock = ManualClock(1.002)
    lowcmd = _BlockingLowCmdSink()
    mailbox, _, residual, safety = _safety_runtime(clock, lowcmd=lowcmd)
    assert mailbox.publish(_policy(mailbox.domain(), 80), now_s=clock())
    controllers = tuple(_leg_controller() for _ in range(4))
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=controllers,
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    operation = asyncio.create_task(high_rate.process_sample(_high_rate_sample()))
    await lowcmd.submit_started.wait()
    controllers[3].reset(np.zeros(3))
    lowcmd.submit_release.set()
    result = await operation

    assert not result.success
    assert "stale or already committed" in result.message
    assert tuple(controller.generation for controller in controllers) == (0, 0, 0, 1)
    assert len(lowcmd.commands) == 1
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_field",
    ("joint_positions_rad", "go2_body_origin_B_position_world_m", "leg_order"),
)
async def test_duplicate_sequence_requires_exact_atomic_sample_payload(
    changed_field: str,
) -> None:
    clock = ManualClock(1.002)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    assert mailbox.publish(_policy(mailbox.domain(), 90), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )
    sample = _high_rate_sample()
    assert (await high_rate.process_sample(sample)).success

    if changed_field == "joint_positions_rad":
        changed_duplicate = replace(
            sample,
            joint_positions_rad=(0.01,) + sample.joint_positions_rad[1:],
        )
    elif changed_field == "go2_body_origin_B_position_world_m":
        changed_duplicate = replace(
            sample,
            go2_body_origin_B_position_world_m=(0.01, 0.0, 0.0),
        )
    else:
        changed_duplicate = replace(
            sample,
            leg_order=("FL", "FR", "RR", "RL"),
        )
    result = await high_rate.process_sample(changed_duplicate)

    assert not result.success
    assert "different atomic payload" in result.message
    assert len(lowcmd.commands) == 1
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


def test_session_mode_and_exact_sink_binding_prevent_cross_wiring() -> None:
    clock = ManualClock(1.0)
    hardware_domain = replace(
        _domain(),
        actuation_mode=MultiRateActuationMode.HARDWARE,
    )
    with pytest.raises(ValueError, match="globally disabled"):
        LatestPolicyMailbox(hardware_domain, monotonic_clock=clock)

    mailbox, _, _, safety = _safety_runtime(clock)
    alien_residual = FakeResidualSink(clock)
    with pytest.raises(ValueError, match="share the safety session"):
        AsyncLatestMPCWorker(
            config=_config(),
            solver=ControlledSolver(_solve_result()),
            mailbox=mailbox,
            residual_sink=alien_residual,
            safety=safety,
            monotonic_clock=clock,
        )

    shadow_domain = replace(
        _domain(),
        actuation_mode=MultiRateActuationMode.SHADOW,
    )
    shadow_mailbox = LatestPolicyMailbox(shadow_domain, monotonic_clock=clock)
    unmarked_lowcmd = FakeLowCmdSink()
    unmarked_residual = FakeResidualSink(clock)
    unmarked_lowcmd.simulation_only = False
    unmarked_residual.simulation_only = False
    with pytest.raises(ValueError, match="physical fallback writes are globally disabled"):
        LandingSafetySupervisor(
            config=_config(),
            mailbox=shadow_mailbox,
            lowcmd_sink=unmarked_lowcmd,
            residual_sink=unmarked_residual,
            go2_status=lambda: _healthy_go2_status(clock),
            expected_go2_mapping_hash=HASH_A,
            monotonic_clock=clock,
        )


@pytest.mark.asyncio
async def test_stop_wins_snapshot_submission_toctou_inside_pending_lock() -> None:
    clock = ManualClock(1.0)
    mailbox, _, residual, safety = _safety_runtime(clock)
    solver = ControlledSolver(_solve_result())
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    await worker._pending_lock.acquire()
    submission = asyncio.create_task(worker.submit_snapshot(_snapshot(mailbox.domain(), 99)))
    await asyncio.sleep(0)
    stop_task = asyncio.create_task(worker.stop("stop won snapshot submission race"))
    await asyncio.sleep(0)
    worker._pending_lock.release()

    rejected = await asyncio.wait_for(submission, timeout=0.5)
    stopped = await asyncio.wait_for(stop_task, timeout=0.5)
    assert not rejected.ok
    assert rejected.code == "MPC_WORKER_SESSION_CHANGED"
    assert stopped.ok
    assert solver.calls == []
    assert worker.status().pending_snapshot_sequence is None


@pytest.mark.asyncio
async def test_manual_override_bypasses_a_broken_safety_clock_and_still_falls_back() -> None:
    clock = ManualClock(1.0)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: _healthy_go2_status(clock),
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=lambda: float("nan"),
    )

    result = await safety.run_once(
        manual_override=True,
        require_active_policy=True,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert safety.fault_latched
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


class _CancellingResidualSink(FakeResidualSink):
    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        del command
        raise asyncio.CancelledError


class _BlockingResidualSink(FakeResidualSink):
    def __init__(self, clock: Callable[[], float], **kwargs: object) -> None:
        super().__init__(clock, **kwargs)
        self.send_started = asyncio.Event()

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        del command
        self.send_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _CancellationResistantActivationSink(FakeResidualSink):
    def __init__(self, clock: Callable[[], float], **kwargs: object) -> None:
        super().__init__(clock, **kwargs)
        self.send_started = asyncio.Event()
        self.send_cleanup_started = asyncio.Event()
        self.send_cleanup_release = asyncio.Event()

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        del command
        self.send_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.send_cleanup_started.set()
            await self.send_cleanup_release.wait()
            raise


class _LateActivatingResidualSink(FakeResidualSink):
    """Models a transport that swallows cancellation and writes after CLEAR."""

    def __init__(self, clock: Callable[[], float]) -> None:
        super().__init__(clock)
        self.send_started = asyncio.Event()
        self.late_send_release = asyncio.Event()
        self._active_command: Optional[FlightControllerRotorResidualCommand] = None

    async def send_rotor_residual(
        self,
        command: FlightControllerRotorResidualCommand,
    ) -> OperationResult:
        self.commands.append(command)
        self.send_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await self.late_send_release.wait()
            self._active_command = command
            return OperationResult.success("late activation", code="TEST_FC_LATE_ACTIVE")

    async def clear_rotor_residual(self, reason: str) -> OperationResult:
        result = await super().clear_rotor_residual(reason)
        self._active_command = None
        return result

    def status(self) -> FlightControllerResidualSinkStatus:
        command = self._active_command
        if command is None:
            return super().status()
        return FlightControllerResidualSinkStatus(
            timestamp_s=self._clock(),
            healthy=True,
            fault_latched=False,
            residual_state=FlightControllerResidualState.ACTIVE,
            residual_active=True,
            clear_confirmed=False,
            fc_session_id=command.fc_session_id,
            last_sequence=command.sequence,
            active_valid_until_s=command.valid_until_s,
            last_error="",
            active_command_sequence=command.sequence,
        )


class _CancellationResistantSubmitSink(FakeLowCmdSink):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.submit_started = asyncio.Event()
        self.submit_cleanup_started = asyncio.Event()
        self.submit_cleanup_release = asyncio.Event()

    async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
        self.commands.append(command)
        self.submit_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.submit_cleanup_started.set()
            await self.submit_cleanup_release.wait()
            raise


@pytest.mark.asyncio
async def test_unexpected_worker_cancellation_latches_and_falls_back() -> None:
    clock = ManualClock(1.0)
    residual = _CancellingResidualSink(clock)
    mailbox, lowcmd, _, safety = _safety_runtime(clock, residual=residual)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 100))).ok
    await _eventually(lambda: safety.fault_latched)
    await _eventually(lambda: bool(lowcmd.revoke_reasons) and bool(residual.clear_reasons))

    assert not worker.status().healthy
    assert not worker.status().running
    assert "cancelled unexpectedly" in safety.status().last_error
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_worker_cancelled_before_first_run_still_trips_both_sides() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert worker._task is not None
    worker._task.cancel()
    await _eventually(lambda: safety.fault_latched)
    await _eventually(lambda: bool(lowcmd.revoke_reasons) and bool(residual.clear_reasons))

    assert not worker.status().running
    assert not worker.status().healthy
    assert "cancelled before/without safe termination" in safety.status().last_error
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_worker_stop_joins_fallback_even_if_monotonic_clock_breaks() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    clock.set(float("nan"))
    stopped = await worker.stop("clock failed during stop")

    assert stopped.ok
    assert safety.fault_latched
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_fc_activation_timeout_cannot_publish_or_launch_another_solve() -> None:
    clock = ManualClock(1.0)
    residual = _BlockingResidualSink(clock)
    mailbox, lowcmd, _, safety = _safety_runtime(clock, residual=residual)
    solver = ControlledSolver(_solve_result())
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=solver,
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 110))).ok
    await asyncio.wait_for(residual.send_started.wait(), timeout=0.5)
    await _eventually(lambda: safety.fault_latched)
    await _eventually(lambda: bool(lowcmd.revoke_reasons) and bool(residual.clear_reasons))

    assert mailbox.latest(now_s=clock()) is None
    assert len(solver.calls) == 1
    assert not worker.status().running
    assert "commit reserve" in safety.status().last_error
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_stop_starts_both_fallbacks_before_fc_activation_cleanup_finishes() -> None:
    clock = ManualClock(1.0)
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    lowcmd = FakeLowCmdSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
    )
    residual = _CancellationResistantActivationSink(
        clock,
        clear_started=clear_started,
        clear_release=clear_release,
    )
    mailbox, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 120))).ok
    await asyncio.wait_for(residual.send_started.wait(), timeout=0.5)
    stop_task = asyncio.create_task(worker.stop("operator stop during activation"))

    await asyncio.wait_for(
        asyncio.gather(
            residual.send_cleanup_started.wait(),
            clear_started.wait(),
            revoke_started.wait(),
        ),
        timeout=0.5,
    )
    assert not stop_task.done()

    residual.send_cleanup_release.set()
    clear_release.set()
    revoke_release.set()
    assert (await asyncio.wait_for(stop_task, timeout=0.5)).ok


@pytest.mark.asyncio
async def test_activation_timeout_starts_go2_revoke_before_fc_cleanup_finishes() -> None:
    clock = ManualClock(1.0)
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    lowcmd = FakeLowCmdSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
    )
    residual = _CancellationResistantActivationSink(
        clock,
        clear_started=clear_started,
        clear_release=clear_release,
    )
    mailbox, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )

    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 121))).ok
    await asyncio.wait_for(residual.send_started.wait(), timeout=0.5)
    await asyncio.wait_for(
        asyncio.gather(
            residual.send_cleanup_started.wait(),
            clear_started.wait(),
            revoke_started.wait(),
        ),
        timeout=0.5,
    )
    assert safety.fault_latched
    assert worker.status().running is False

    residual.send_cleanup_release.set()
    clear_release.set()
    revoke_release.set()
    await _eventually(lambda: worker.status().solving_snapshot_sequence is None)
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_cancelled_high_rate_submit_starts_fc_clear_before_owner_cleanup_finishes() -> None:
    clock = ManualClock(1.002)
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    lowcmd = _CancellationResistantSubmitSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
    )
    residual = FakeResidualSink(
        clock,
        clear_started=clear_started,
        clear_release=clear_release,
    )
    mailbox, _, _, safety = _safety_runtime(clock, lowcmd=lowcmd, residual=residual)
    assert mailbox.publish(_policy(mailbox.domain(), 130), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    operation = asyncio.create_task(high_rate.process_sample(_high_rate_sample()))
    await asyncio.wait_for(lowcmd.submit_started.wait(), timeout=0.5)
    operation.cancel()
    await asyncio.wait_for(
        asyncio.gather(
            lowcmd.submit_cleanup_started.wait(),
            clear_started.wait(),
            revoke_started.wait(),
        ),
        timeout=0.5,
    )
    assert not operation.done()

    lowcmd.submit_cleanup_release.set()
    clear_release.set()
    revoke_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=0.5)


@pytest.mark.asyncio
async def test_worker_start_clock_failure_never_leaves_a_running_worker() -> None:
    clock = ManualClock(1.0)
    mailbox, lowcmd, residual, safety = _safety_runtime(clock)
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=lambda: float("nan"),
    )

    started = await worker.start()

    assert not started.ok
    assert started.code == "MPC_WORKER_START_CLOCK_FAILED"
    assert worker._running is False
    assert worker._task is None
    assert safety.fault_latched
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_safety_accepts_statuses_sampled_after_observation_start_on_real_clock() -> None:
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=time.monotonic)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(time.monotonic)

    def holding_status() -> Go2LowLevelStatus:
        return replace(
            _healthy_go2_status(time.monotonic),
            ownership_state=LowCmdOwnershipState.HOLDING,
            target_sequence=None,
            mailbox_staged_target_sequence=None,
            target_age_s=None,
            target_deadline=None,
            safe_hold_active=True,
            safe_hold_settled=True,
        )

    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=holding_status,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=time.monotonic,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert result.ok
    assert result.code == "MULTIRATE_SAFETY_HEALTHY"
    assert not safety.fault_latched


@pytest.mark.parametrize(
    "changes",
    (
        {"timestamp": float("nan")},
        {"low_state_timestamp": float("nan")},
        {"low_state_age_s": -1.0},
        {"publisher_active": 1},
    ),
)
@pytest.mark.asyncio
async def test_go2_owner_status_requires_strict_finite_fresh_fields(
    changes: dict[str, object],
) -> None:
    clock = ManualClock(1.0)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    malformed = replace(_healthy_go2_status(clock), **changes)  # type: ignore[arg-type]
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: malformed,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert "owner health/age/ownership check failed" in safety.status().last_error
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_go2_active_mapping_hash_must_equal_supervisor_expected_hash() -> None:
    clock = ManualClock(1.0)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    wrong_mapping = replace(
        _healthy_go2_status(clock),
        active_mapping_hash=HASH_B,
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: wrong_mapping,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=False,
    )

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert "owner health/age/ownership check failed" in safety.status().last_error
    assert safety.fault_latched
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1


@pytest.mark.asyncio
async def test_policy_and_fc_active_cannot_hide_go2_safe_hold() -> None:
    clock = ManualClock(1.0)
    domain = _domain()
    mailbox = LatestPolicyMailbox(domain, monotonic_clock=clock)
    policy = _policy(domain, 41, valid_until_s=1.5)
    assert mailbox.publish(policy, now_s=clock())
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    residual.status_override = FlightControllerResidualSinkStatus(
        timestamp_s=clock(),
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.ACTIVE,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=1,
        last_sequence=41,
        active_valid_until_s=policy.valid_until_s,
        last_error="",
        active_command_sequence=41,
    )
    holding = replace(
        _healthy_go2_status(clock),
        ownership_state=LowCmdOwnershipState.HOLDING,
        target_sequence=None,
        mailbox_staged_target_sequence=None,
        target_age_s=None,
        target_deadline=None,
        safe_hold_active=True,
        safe_hold_settled=True,
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: holding,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    safety.attach_monitors(
        high_rate_status=lambda: _healthy_high_rate_status(
            clock,
            domain,
            policy_sequence=41,
        ),
        worker_status=lambda: _healthy_worker_status(
            clock,
            domain,
            published_sequence=41,
        ),
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )

    assert not result.ok
    assert "staged/enqueued/applied target identity" in safety.status().last_error


@pytest.mark.asyncio
async def test_safety_does_not_treat_unexecuted_startup_contact_as_replan_grace() -> None:
    clock = ManualClock(1.0)
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    holding = replace(
        _healthy_go2_status(clock),
        ownership_state=LowCmdOwnershipState.HOLDING,
        target_sequence=None,
        mailbox_staged_target_sequence=None,
        target_age_s=None,
        target_deadline=None,
        safe_hold_active=True,
        safe_hold_settled=True,
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: holding,
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    high_status = replace(
        _healthy_high_rate_status(clock, mailbox.domain()),
        replan_pending=True,
        replan_grace_authorized=False,
        replan_deadline_s=1.03,
        last_staged_frame_sequence=None,
        last_staged_frame_deadline_s=None,
        last_actuator_applied_frame_sequence=None,
        last_actuator_applied_policy_sequence=None,
        last_policy_sequence=None,
    )
    safety.attach_monitors(
        high_rate_status=lambda: high_status,
        worker_status=lambda: _healthy_worker_status(clock, mailbox.domain()),
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )

    assert not result.ok
    assert "no fresh activated MPC policy" in safety.status().last_error


@pytest.mark.asyncio
async def test_fc_consumed_watermark_cannot_exceed_explained_active_identity() -> None:
    clock = ManualClock(1.0)
    domain = _domain()
    mailbox = LatestPolicyMailbox(domain, monotonic_clock=clock)
    policy = _policy(domain, 41, valid_until_s=1.5)
    assert mailbox.publish(policy, now_s=clock())
    lowcmd = FakeLowCmdSink()
    residual = FakeResidualSink(clock)
    residual.status_override = FlightControllerResidualSinkStatus(
        timestamp_s=clock(),
        healthy=True,
        fault_latched=False,
        residual_state=FlightControllerResidualState.ACTIVE,
        residual_active=True,
        clear_confirmed=False,
        fc_session_id=1,
        last_sequence=999,
        active_valid_until_s=policy.valid_until_s,
        last_error="",
        active_command_sequence=41,
    )
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: _healthy_go2_status(clock),
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    safety.attach_monitors(
        high_rate_status=lambda: _healthy_high_rate_status(
            clock,
            domain,
            policy_sequence=41,
        ),
        worker_status=lambda: _healthy_worker_status(
            clock,
            domain,
            published_sequence=41,
        ),
    )

    result = await safety.run_once(
        manual_override=False,
        require_active_policy=True,
    )

    assert not result.ok
    assert "executed identity" in safety.status().last_error


@pytest.mark.asyncio
async def test_manual_trip_waits_for_late_fc_stage_then_clears_again() -> None:
    clock = ManualClock(1.0)
    residual = _LateActivatingResidualSink(clock)
    revoke_started = asyncio.Event()
    revoke_release = asyncio.Event()
    lowcmd = FakeLowCmdSink(
        revoke_started=revoke_started,
        revoke_release=revoke_release,
    )
    mailbox, _, _, safety = _safety_runtime(
        clock,
        lowcmd=lowcmd,
        residual=residual,
    )
    worker = AsyncLatestMPCWorker(
        config=_config(),
        solver=ControlledSolver(_solve_result()),
        mailbox=mailbox,
        residual_sink=residual,
        safety=safety,
        monotonic_clock=clock,
    )
    assert (await worker.start()).ok
    assert (await worker.submit_snapshot(_snapshot(mailbox.domain(), 140))).ok
    await asyncio.wait_for(residual.send_started.wait(), timeout=0.5)

    observer = asyncio.create_task(
        safety.run_once(manual_override=True, require_active_policy=True)
    )
    await asyncio.wait_for(revoke_started.wait(), timeout=0.5)
    await _eventually(lambda: len(residual.clear_reasons) == 1)
    assert not observer.done()
    residual.late_send_release.set()
    await _eventually(lambda: len(residual.clear_reasons) == 2)
    assert residual.status().residual_state is FlightControllerResidualState.CONFIRMED_ZERO
    assert not observer.done()
    revoke_release.set()
    result = await asyncio.wait_for(observer, timeout=0.5)

    assert not result.ok
    assert result.code == "MULTIRATE_SAFETY_TRIPPED"
    assert result.data["fallback_confirmed"] is True
    assert len(residual.clear_reasons) == 2
    assert residual.status().residual_state is FlightControllerResidualState.CONFIRMED_ZERO
    assert len(lowcmd.revoke_reasons) == 1
    assert (await worker.stop("test complete")).ok


@pytest.mark.asyncio
async def test_unexpected_clock_failure_after_lowcmd_ack_trips_both_sides() -> None:
    class BreakableClock:
        def __init__(self) -> None:
            self.broken = False

        def __call__(self) -> float:
            if self.broken:
                raise RuntimeError("synthetic clock loss")
            return 1.002

    class ClockBreakingLowCmdSink(FakeLowCmdSink):
        async def submit(self, command: Go2JointPositionCommand) -> OperationResult:
            result = await super().submit(command)
            clock.broken = True
            return result

    clock = BreakableClock()
    mailbox = LatestPolicyMailbox(_domain(), monotonic_clock=clock)
    lowcmd = ClockBreakingLowCmdSink()
    residual = FakeResidualSink(clock)
    safety = LandingSafetySupervisor(
        config=_config(),
        mailbox=mailbox,
        lowcmd_sink=lowcmd,
        residual_sink=residual,
        go2_status=lambda: _healthy_go2_status(clock),
        expected_go2_mapping_hash=HASH_A,
        monotonic_clock=clock,
    )
    assert mailbox.publish(_policy(mailbox.domain(), 141), now_s=clock())
    high_rate = HighRateLegController(
        config=_config(),
        mailbox=mailbox,
        contact_detector=_contact_detector(),
        leg_controllers=tuple(_leg_controller() for _ in range(4)),
        controller_leg_order=GO2_SDK_LEG_ORDER,
        lowcmd_sink=lowcmd,
        safety=safety,
        force_calibration_hash=HASH_C,
        monotonic_clock=clock,
    )

    result = await high_rate.process_sample(_high_rate_sample())

    assert not result.success
    assert "synthetic clock loss" in result.message
    assert safety.fault_latched
    assert len(lowcmd.commands) == 1
    assert len(lowcmd.revoke_reasons) == 1
    assert len(residual.clear_reasons) == 1
