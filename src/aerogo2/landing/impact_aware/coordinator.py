"""Pure reference coordinator for one impact-aware landing control cycle.

The coordinator connects the numerical NLP, filtered contact detection,
per-leg admittance execution, and flight-controller correction blender.  It
returns the typed integration seam from :mod:`integration`; it never imports a
Unitree SDK, MAVLink transport, or ESC interface and cannot write hardware.

中文说明：本文件是论文算法的“单周期参考编排器”。它按固定顺序完成输入时效
检查、触地检测、MPC 求解、腿端导纳/逆解、旋翼残差安全融合以及成对命令封装。
这里返回的对象只是经过校验的候选命令，不代表命令已经被 Go2 或飞控执行；真正
上机时应使用 ``multirate`` 中的多速率结构，并由独占 LowCmd owner 和飞控残差
协议分别确认执行。任一输入过期、接触状态矛盾、求解失败或约束越界都会整包撤销，
避免“腿执行新指令而旋翼仍执行旧指令”的半更新状态。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from numbers import Real
from typing import Callable, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from aerogo2.common.enums import SystemState
from aerogo2.landing.impact_aware.admittance import (
    AdmittanceState,
    LegAdmittanceController,
    LegAdmittanceOutput,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetection,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.integration import (
    CoordinatedLandingCommand,
    FlightControllerRotorResidualCommand,
    Go2JointPositionCommand,
    ImpactLandingPhase,
    phase_for_system_state,
)
from aerogo2.landing.impact_aware.nlp import (
    ImpactAwareMPCProblem,
    ImpactAwareNLP,
    MPCSolveResult,
    RotorExecutionPlan,
    SLSQPSettings,
    reconstruct_transport_target,
)
from aerogo2.landing.impact_aware.rotor import (
    evaluate_rotor_constraints,
    first_order_thrust_rate,
)
from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionOutput,
)


def _finite_array(name: str, value: object, shape: Tuple[int, ...]) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _finite_real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class LandingInputFreshness:
    """Source-time and health evidence for the aggregated controller inputs.

    中文：每类输入保留其采样时刻而不是只记录“组包时刻”。任一源报告不健康、
    时间在未来或 age 超限，整个控制周期失效；这可阻止新姿态配旧足力的混合输入。
    """

    state_estimate_timestamp_s: float
    contact_forces_timestamp_s: float
    kinematics_timestamp_s: float
    foot_plan_timestamp_s: float
    flight_controller_baseline_timestamp_s: float
    maximum_source_age_s: float
    all_sources_healthy: bool

    def __post_init__(self) -> None:
        for name in (
            "state_estimate_timestamp_s",
            "contact_forces_timestamp_s",
            "kinematics_timestamp_s",
            "foot_plan_timestamp_s",
            "flight_controller_baseline_timestamp_s",
            "maximum_source_age_s",
        ):
            value = _finite_real(
                name,
                getattr(self, name),
                positive=name == "maximum_source_age_s",
            )
            object.__setattr__(self, name, value)
        if type(self.all_sources_healthy) is not bool:
            raise TypeError("all_sources_healthy must be a bool")

    def failure_reason(self, cycle_timestamp_s: float) -> Optional[str]:
        if not self.all_sources_healthy:
            return "one or more controller input sources reported unhealthy"
        for name in (
            "state_estimate_timestamp_s",
            "contact_forces_timestamp_s",
            "kinematics_timestamp_s",
            "foot_plan_timestamp_s",
            "flight_controller_baseline_timestamp_s",
        ):
            timestamp = float(getattr(self, name))
            if timestamp > cycle_timestamp_s:
                return f"{name} is in the future relative to the control cycle"
            if cycle_timestamp_s - timestamp > self.maximum_source_age_s:
                return f"{name} is stale"
        return None


@dataclass(frozen=True)
class LandingCycleInput:
    """All measured/planned values needed to compute one atomic command bundle.

    中文：该对象把一次 MPC 周期所需测量、规划、飞控基线身份和有效期冻结在一起。
    当前基线必须等于预测序列第一行，且命令 TTL 不得超过一个 MPC 步长。
    """

    sequence: int
    timestamp_s: float
    dt_s: float
    command_ttl_s: float
    system_state: SystemState
    freshness: LandingInputFreshness
    problem: ImpactAwareMPCProblem
    flight_controller_session_id: int
    flight_controller_target_tick: int
    flight_controller_baseline_version: int
    flight_controller_baseline_thrusts_n: object
    flight_controller_baseline_prediction_thrusts_n: object
    measured_normal_forces_n: object
    estimated_contact_forces_world_n: object
    nominal_foot_positions_world_m: object

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        for name in (
            "flight_controller_session_id",
            "flight_controller_target_tick",
            "flight_controller_baseline_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("timestamp_s", "dt_s", "command_ttl_s"):
            value = _finite_real(
                name,
                getattr(self, name),
                positive=name != "timestamp_s",
            )
            object.__setattr__(self, name, value)
        if self.timestamp_s < 0.0:
            raise ValueError("timestamp_s cannot be negative")
        if self.command_ttl_s > self.dt_s:
            raise ValueError("command_ttl_s cannot exceed one MPC cycle dt_s")
        if not math.isfinite(self.timestamp_s + self.command_ttl_s):
            raise ValueError("timestamp_s + command_ttl_s must be finite")
        if not isinstance(self.system_state, SystemState):
            raise TypeError("system_state must be a SystemState")
        if not isinstance(self.freshness, LandingInputFreshness):
            raise TypeError("freshness must be LandingInputFreshness")
        if self.freshness.maximum_source_age_s > min(self.dt_s, self.command_ttl_s):
            raise ValueError("maximum_source_age_s cannot exceed the command TTL or one MPC cycle")
        if not isinstance(self.problem, ImpactAwareMPCProblem):
            raise TypeError("problem must be an ImpactAwareMPCProblem")
        if not math.isclose(self.dt_s, self.problem.dt_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("dt_s must equal problem.dt_s")
        current_baseline = _finite_array(
            "flight_controller_baseline_thrusts_n",
            self.flight_controller_baseline_thrusts_n,
            (4,),
        )
        object.__setattr__(
            self,
            "flight_controller_baseline_thrusts_n",
            current_baseline,
        )
        baseline_prediction = _finite_array(
            "flight_controller_baseline_prediction_thrusts_n",
            self.flight_controller_baseline_prediction_thrusts_n,
            (self.problem.horizon, 4),
        )
        if not np.allclose(
            baseline_prediction[0],
            current_baseline,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "the first baseline prediction must equal the current FC baseline sample"
            )
        object.__setattr__(
            self,
            "flight_controller_baseline_prediction_thrusts_n",
            baseline_prediction,
        )
        normal_forces = _finite_array(
            "measured_normal_forces_n",
            self.measured_normal_forces_n,
            (4,),
        )
        if np.any(normal_forces < 0.0):
            raise ValueError("measured_normal_forces_n cannot be negative")
        object.__setattr__(self, "measured_normal_forces_n", normal_forces)
        object.__setattr__(
            self,
            "estimated_contact_forces_world_n",
            _finite_array(
                "estimated_contact_forces_world_n",
                self.estimated_contact_forces_world_n,
                (4, 3),
            ),
        )
        nominal_foot_positions = _finite_array(
            "nominal_foot_positions_world_m",
            self.nominal_foot_positions_world_m,
            (4, 3),
        )
        object.__setattr__(
            self,
            "nominal_foot_positions_world_m",
            nominal_foot_positions,
        )


@dataclass(frozen=True)
class LandingCycleResult:
    """Auditable cycle result; ``command`` is absent on every unhealthy path."""

    success: bool
    status: str
    message: str
    phase: ImpactLandingPhase
    contact_detection: Optional[ContactDetection]
    solver_result: Optional[MPCSolveResult]
    leg_outputs: Tuple[LegAdmittanceOutput, ...]
    rotor_output: Optional[RotorCorrectionOutput]
    command: Optional[CoordinatedLandingCommand]


class ImpactAwareLandingCoordinator:
    """Assemble one paper-controller cycle without owning any hardware transport.

    中文：这是便于复现实验与单元测试的串行参考实现。构造时固定四条腿控制器、
    旋翼分配矩阵以及 ``C 相对 B`` 的偏移，以保证运行时不能偷偷改变几何模型。
    ``compute`` 具有全有或全无语义：只有腿部和旋翼两侧均完成检查才返回 command。
    """

    def __init__(
        self,
        *,
        solver_settings: SLSQPSettings,
        contact_detector: FootContactDetector,
        leg_controllers: Sequence[LegAdmittanceController],
        rotor_blender: RotorCorrectionBlender,
        fixed_rotor_allocation_body: object,
        total_com_C_from_go2_body_origin_B_body_m: object,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        controllers = tuple(leg_controllers)
        if len(controllers) != 4 or not all(
            isinstance(controller, LegAdmittanceController) for controller in controllers
        ):
            raise TypeError("leg_controllers must contain four LegAdmittanceController values")
        if not isinstance(solver_settings, SLSQPSettings):
            raise TypeError("solver_settings must be SLSQPSettings")
        if not isinstance(contact_detector, FootContactDetector):
            raise TypeError("contact_detector must be FootContactDetector")
        if not isinstance(rotor_blender, RotorCorrectionBlender):
            raise TypeError("rotor_blender must be RotorCorrectionBlender")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._settings = solver_settings
        self._contact_detector = contact_detector
        self._leg_controllers = controllers
        self._rotor_blender = rotor_blender
        self._fixed_rotor_allocation_body = _finite_array(
            "fixed_rotor_allocation_body",
            fixed_rotor_allocation_body,
            (6, 4),
        )
        self._total_com_C_from_go2_body_origin_B_body_m = _finite_array(
            "total_com_C_from_go2_body_origin_B_body_m",
            total_com_C_from_go2_body_origin_B_body_m,
            (3,),
        )
        self._monotonic_clock = monotonic_clock
        self._contact_loss_latched = False

    def reset(self) -> None:
        """Explicitly start a new landing session and clear latched faults."""

        self._reset_runtime(clear_session_faults=True)

    def _reset_runtime(self, *, clear_session_faults: bool) -> None:
        """Reset transient state without silently clearing a failed session."""

        self._contact_detector.reset()
        self._rotor_blender.reset(clear_fault_latch=clear_session_faults)
        if clear_session_faults:
            self._contact_loss_latched = False
        for controller in self._leg_controllers:
            controller.reset(controller.previous_joint_command)

    def compute(self, cycle: LandingCycleInput) -> LandingCycleResult:
        """Solve and assemble a cycle, withholding the whole bundle on any failure.

        中文执行顺序：FSM 授权与 TTL → 锁存故障 → 固定几何/数据新鲜度 → 接触
        状态 → 实际 κ 执行计划 → NLP → 截止时间复检 → 四腿导纳和旋翼融合 →
        最终约束及序列化。每个失败出口都会请求残差归零，且 ``command`` 为 None。
        """

        if not isinstance(cycle, LandingCycleInput):
            raise TypeError("cycle must be a LandingCycleInput")
        phase = phase_for_system_state(cycle.system_state)
        if phase is ImpactLandingPhase.INACTIVE:
            # 非着陆状态不得沿用上一次算法状态。这里只清瞬态；会话级故障必须由
            # 显式 reset 清除，防止 FSM 往返一次就绕过锁存保护。
            self._reset_runtime(clear_session_faults=False)
            fallback = self._remove_rotor_correction(cycle, latch_failure=False)
            return self._failure(
                phase=phase,
                status="inactive_state",
                message="impact-aware output is allowed only in active landing states",
                rotor_output=fallback,
            )

        if self._settings.timeout_s > cycle.command_ttl_s:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="timing_contract_invalid",
                message="solver timeout must not exceed the one-cycle command TTL",
                rotor_output=fallback,
            )
        valid_until = cycle.timestamp_s + cycle.command_ttl_s
        try:
            cycle_started_at = float(self._monotonic_clock())
        except (TypeError, ValueError, OverflowError):
            cycle_started_at = math.inf
        if (
            not math.isfinite(cycle_started_at)
            or cycle_started_at < cycle.timestamp_s
            or cycle_started_at >= valid_until
        ):
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="command_time_window_invalid",
                message="control cycle did not start inside its monotonic validity window",
                rotor_output=fallback,
            )

        if self._contact_loss_latched:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="contact_loss_latched",
                message="a post-touchdown contact release remains latched for this session",
                rotor_output=fallback,
            )
        if self._rotor_blender.fault_latched:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="rotor_correction_fault_latched",
                message="a prior unhealthy cycle inhibits rotor correction until explicit reset",
                rotor_output=fallback,
            )

        if not np.array_equal(
            cycle.problem.dynamics_config.rotor_allocation_body,
            self._fixed_rotor_allocation_body,
        ):
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="static_rotor_geometry_mismatch",
                message=(
                    "the MPC rotor allocation differs from the fixed, audited, "
                    "fully deployed allocation pinned when this coordinator was built"
                ),
                rotor_output=fallback,
            )

        freshness_failure = cycle.freshness.failure_reason(cycle.timestamp_s)
        if freshness_failure is not None:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="input_source_unhealthy",
                message=freshness_failure,
                rotor_output=fallback,
            )

        try:
            contact = self._contact_detector.update(
                cycle.measured_normal_forces_n,
                cycle.freshness.contact_forces_timestamp_s,
            )
        except (TypeError, ValueError) as exc:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="invalid_contact_measurement",
                message=str(exc),
                rotor_output=fallback,
            )

        if any(contact.release_events):
            # 着陆后失去任一已确认接触可能导致再次坠落，不能按普通测量抖动处理。
            self._contact_loss_latched = True
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=ImpactLandingPhase.POST_TOUCHDOWN_RECOVERY,
                status="contact_loss_latched",
                message=(
                    "an unplanned post-touchdown contact release was detected; "
                    "explicit session reset is required"
                ),
                contact_detection=contact,
                rotor_output=fallback,
            )

        if any(contact.touchdown_events):
            phase = ImpactLandingPhase.TOUCHDOWN
        elif any(contact.contacts):
            phase = ImpactLandingPhase.POST_TOUCHDOWN_RECOVERY
        planned_current_contacts = tuple(bool(value) for value in cycle.problem.contact_schedule[0])
        if planned_current_contacts != contact.contacts:
            # A newly confirmed touchdown is an expected hybrid replan edge.
            # Preserve detector state and keep the session eligible for the
            # next problem whose stage-zero schedule matches the measurement.
            fallback = self._remove_rotor_correction(cycle, latch_failure=False)
            return self._failure(
                phase=phase,
                status="contact_schedule_mismatch",
                message=(
                    "measured current contacts disagree with contact_schedule[0]; "
                    "rebuild the MPC problem before using ground-reaction forces"
                ),
                contact_detection=contact,
                rotor_output=fallback,
            )

        try:
            # MPC 必须预测融合器当前真实可达到的 κ 轨迹，而不是目标 κ；否则
            # 优化器会依赖执行端实际上不会施加的旋翼控制量。
            actuator = cycle.problem.rotor_actuator_config
            safety = self._rotor_blender.config
            safety_minimum = np.asarray(safety.thrust_min_n, dtype=float)
            safety_maximum = np.asarray(safety.thrust_max_n, dtype=float)
            if not np.allclose(
                safety_minimum,
                actuator.thrust_min_n,
                rtol=0.0,
                atol=1.0e-12,
            ) or not np.allclose(
                safety_maximum,
                actuator.thrust_max_n,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("rotor blender thrust limits disagree with the MPC actuator model")
            predicted_gains = np.asarray(
                self._rotor_blender.preview_gains(
                    cycle.problem.dt_s,
                    cycle.problem.horizon,
                ),
                dtype=float,
            )
            execution_plan = RotorExecutionPlan(
                baseline_thrusts_n=cast(
                    NDArray[np.float64],
                    cycle.flight_controller_baseline_prediction_thrusts_n,
                ),
                correction_gains=predicted_gains,
                maximum_raw_correction_n=cast(
                    NDArray[np.float64],
                    safety.maximum_correction_n,
                ),
            )
            supplied_plan = cycle.problem.rotor_execution_plan
            if supplied_plan is not None and not (
                np.array_equal(
                    supplied_plan.baseline_thrusts_n,
                    execution_plan.baseline_thrusts_n,
                )
                and np.array_equal(
                    supplied_plan.correction_gains,
                    execution_plan.correction_gains,
                )
                and np.array_equal(
                    supplied_plan.maximum_raw_correction_n,
                    execution_plan.maximum_raw_correction_n,
                )
            ):
                raise ValueError(
                    "caller-supplied rotor execution plan disagrees with the live safety state"
                )
            problem = replace(cycle.problem, rotor_execution_plan=execution_plan)
        except (TypeError, ValueError) as exc:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="rotor_execution_plan_invalid",
                message=str(exc),
                contact_detection=contact,
                rotor_output=fallback,
            )

        try:
            solver = ImpactAwareNLP(problem).solve(self._settings)
        except Exception as exc:  # SciPy is optional and numerical backends can fail.
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="solver_exception",
                message=f"MPC solver raised {type(exc).__name__}: {exc}",
                contact_detection=contact,
                rotor_output=fallback,
            )

        control = solver.first_input
        if not solver.success or control is None:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="solver_unhealthy",
                message=f"MPC output withheld: {solver.status}: {solver.message}",
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )

        try:
            completed_at = float(self._monotonic_clock())
        except (TypeError, ValueError, OverflowError):
            completed_at = math.inf
        if (
            not math.isfinite(completed_at)
            or completed_at < cycle.timestamp_s
            or completed_at >= valid_until
            or solver.solve_time_s >= cycle.command_ttl_s
        ):
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="command_deadline_missed",
                message="MPC result completed after the command validity deadline",
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )
        completion_freshness_failure = cycle.freshness.failure_reason(completed_at)
        if completion_freshness_failure is not None:
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="input_source_became_stale",
                message=(f"MPC result withheld after solve because {completion_freshness_failure}"),
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )

        snapshots = tuple(
            (controller.state, controller.previous_joint_command)
            for controller in self._leg_controllers
        )
        leg_outputs = []
        try:
            initial = problem.initial_state
            # 动力学状态的位置属于整机质心 C；腿部 FK/IK 使用 Go2 机身原点 B。
            go2_body_origin_B_position_world_m = (
                initial.position_world_m
                - initial.rotation_body_to_world @ self._total_com_C_from_go2_body_origin_B_body_m
            )
            nominal_feet = cast(
                NDArray[np.float64],
                cycle.nominal_foot_positions_world_m,
            )
            estimated_forces = cast(
                NDArray[np.float64],
                cycle.estimated_contact_forces_world_n,
            )
            for index, controller in enumerate(self._leg_controllers):
                leg_outputs.append(
                    controller.step(
                        current_time_s=cycle.timestamp_s,
                        dt_s=cycle.dt_s,
                        measured_contact=contact.contacts[index],
                        touchdown_time_s=contact.touchdown_times_s[index],
                        rotation_body_to_world=initial.rotation_body_to_world,
                        body_position_world=go2_body_origin_B_position_world_m,
                        nominal_foot_position_world=nominal_feet[index],
                        desired_force_world=control.contact_forces_world_n[index],
                        estimated_force_world=estimated_forces[index],
                    )
                )
            transport_target = reconstruct_transport_target(
                execution_plan,
                0,
                control.rotor_thrust_commands_n,
            )
            rotor = self._rotor_blender.blend_modeled_applied(
                cycle.flight_controller_baseline_thrusts_n,
                control.rotor_thrust_commands_n,
                cycle.dt_s,
                expected_gain=float(execution_plan.correction_gains[0]),
            )
            if transport_target is None:
                if (
                    rotor.transport_target_thrusts_n is not None
                    or rotor.transport_raw_correction_n is not None
                    or rotor.transport_target_semantics != "zero_gain_no_transport_target"
                ):
                    raise ValueError("zero-gain transport invented a transport target")
            else:
                if (
                    rotor.transport_target_thrusts_n is None
                    or rotor.transport_raw_correction_n is None
                ):
                    raise ValueError("positive gain did not produce a transport reconstruction")
                reconstructed = np.asarray(rotor.transport_target_thrusts_n, dtype=float)
                if not np.allclose(
                    reconstructed,
                    transport_target.target_thrusts_n,
                    rtol=1.0e-9,
                    atol=1.0e-9,
                ):
                    raise ValueError("transport target disagrees with the execution plan")
                expected_semantics = (
                    "gain_limited_algebraic_reconstruction"
                    if transport_target.is_gain_limited_reconstruction
                    else "active_gain_one_transport_target"
                )
                if rotor.transport_target_semantics != expected_semantics:
                    raise ValueError("transport target semantics disagree with the gain")
            final_command = np.asarray(rotor.applied_total_thrusts_n, dtype=float)
            thrust_rates = first_order_thrust_rate(
                initial.rotor_thrusts_n,
                final_command,
                problem.rotor_actuator_config,
            )
            rotor_margins = evaluate_rotor_constraints(
                initial.rotor_thrusts_n,
                thrust_rates,
                final_command,
                problem.rotor_actuator_config,
            )
            if not rotor_margins.is_feasible(atol=self._settings.constraint_tolerance):
                raise ValueError(
                    "blended FC command violates identified thrust or thrust-rate constraints"
                )
        except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # 四腿共用一个命令事务；任一腿或旋翼检查失败都回滚全部导纳积分状态。
            self._restore_legs(snapshots)
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="execution_boundary_invalid",
                message=f"command bundle withheld: {exc}",
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )

        try:
            assembled_at = float(self._monotonic_clock())
        except (TypeError, ValueError, OverflowError):
            assembled_at = math.inf
        if (
            not math.isfinite(assembled_at)
            or assembled_at < cycle.timestamp_s
            or assembled_at >= valid_until
        ):
            self._restore_legs(snapshots)
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="command_deadline_missed",
                message="leg/rotor execution processing crossed the command validity deadline",
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )

        try:
            joint_positions = tuple(
                float(value) for output in leg_outputs for value in output.joint_position_command
            )
            desired_forces = tuple(
                float(value) for value in control.contact_forces_world_n.reshape(12)
            )
            leg_command = Go2JointPositionCommand(
                sequence=cycle.sequence,
                timestamp_s=cycle.timestamp_s,
                valid_until_s=valid_until,
                joint_positions_rad=joint_positions,
                desired_contact_forces_world_n=desired_forces,
            )
            rotor_command = FlightControllerRotorResidualCommand(
                sequence=cycle.sequence,
                timestamp_s=cycle.timestamp_s,
                valid_until_s=valid_until,
                fc_session_id=cycle.flight_controller_session_id,
                target_fc_tick=cycle.flight_controller_target_tick,
                baseline_version=cycle.flight_controller_baseline_version,
                baseline_timestamp_s=cycle.freshness.flight_controller_baseline_timestamp_s,
                baseline_thrusts_n=rotor.baseline_thrusts_n,
                transport_raw_residual_thrusts_n=rotor.transport_raw_correction_n,
                applied_residual_thrusts_n=rotor.applied_residual_thrusts_n,
                applied_total_thrusts_n=rotor.applied_total_thrusts_n,
                correction_gain=rotor.applied_gain,
                transport_target_semantics=rotor.transport_target_semantics,
            )
            command = CoordinatedLandingCommand(
                phase=phase,
                leg=leg_command,
                rotor=rotor_command,
                solver_succeeded=True,
                solver_status=solver.status,
                solver_time_s=solver.solve_time_s,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._restore_legs(snapshots)
            fallback = self._remove_rotor_correction(cycle)
            return self._failure(
                phase=phase,
                status="execution_boundary_invalid",
                message=f"command serialization withheld: {exc}",
                contact_detection=contact,
                solver_result=solver,
                rotor_output=fallback,
            )
        return LandingCycleResult(
            success=True,
            status="success",
            message="paper-controller command bundle assembled; no hardware write performed",
            phase=phase,
            contact_detection=contact,
            solver_result=solver,
            leg_outputs=tuple(leg_outputs),
            rotor_output=rotor,
            command=command,
        )

    def _restore_legs(
        self,
        snapshots: Sequence[Tuple[AdmittanceState, NDArray[np.float64]]],
    ) -> None:
        for controller, (state, previous) in zip(self._leg_controllers, snapshots):
            controller.reset(
                previous,
                correction_position_body=state.correction_position_body,
                correction_velocity_body=state.correction_velocity_body,
                contact_seen=state.contact_seen,
            )

    def _remove_rotor_correction(
        self,
        cycle: LandingCycleInput,
        *,
        latch_failure: bool = True,
    ) -> Optional[RotorCorrectionOutput]:
        try:
            return self._rotor_blender.blend(
                cycle.flight_controller_baseline_thrusts_n,
                cycle.flight_controller_baseline_thrusts_n,
                cycle.dt_s,
                healthy=False,
                latch_failure=latch_failure,
            )
        except (TypeError, ValueError, RuntimeError):
            self._rotor_blender.inhibit()
            return None

    @staticmethod
    def _failure(
        *,
        phase: ImpactLandingPhase,
        status: str,
        message: str,
        contact_detection: Optional[ContactDetection] = None,
        solver_result: Optional[MPCSolveResult] = None,
        rotor_output: Optional[RotorCorrectionOutput] = None,
    ) -> LandingCycleResult:
        return LandingCycleResult(
            success=False,
            status=status,
            message=message,
            phase=phase,
            contact_detection=contact_detection,
            solver_result=solver_result,
            leg_outputs=(),
            rotor_output=rotor_output,
            command=None,
        )


__all__ = [
    "ImpactAwareLandingCoordinator",
    "LandingCycleInput",
    "LandingCycleResult",
    "LandingInputFreshness",
]
