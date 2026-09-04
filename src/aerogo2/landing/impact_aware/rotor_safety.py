"""Fail-closed blending of MPC rotor corrections with a flight-controller baseline.

The paper defines the MPC solution as a total four-rotor command and then
forms an additive correction relative to the onboard flight controller.  This
module implements the engineering safety extension requested for AeroGo2::

    u_command = u_fc + kappa * (u_mpc - u_fc),  0 <= kappa <= 1

Only the correction is scaled.  The flight-controller baseline is never
attenuated.  A single scalar is used for all four rotors so headroom limiting
does not rotate the requested correction wrench merely because one motor has
less remaining authority.

中文说明：安全融合器保存 κ 的实际爬升状态，并用四轴共同缩放保证任何一轴都不越过
推力 headroom 或最大残差。``raw correction``、``applied residual`` 和最终总推力是
不同物理量：只有已经乘 κ 且通过限幅的 residual 才能传给飞控。健康条件失效时 κ
向零退让；严重异常会锁存故障，必须由新着陆会话显式复位，不能自动恢复正残差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _vector4(name: str, values: object) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(vector, dtype=np.float64).copy()


def _real_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tuple4(values: NDArray[np.float64]) -> Tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


@dataclass(frozen=True)
class RotorCorrectionSafetyConfig:
    """Validated bounds for the four-rotor correction boundary.

    ``target_gain`` is an experimental setting, not a value supplied by the
    paper.  Zero selects the flight-controller baseline.  One removes only
    the gain attenuation; identified correction/headroom limits can still make
    the result differ from an unconstrained paper command.  All limits must be
    identified for the actual propulsion system.

    中文：``target_gain`` 只是调试目标，不是论文自动给出的常数。κ 从零按最大
    上升率缓慢增加；每轴绝对残差和总推力上下界仍同时有效。未做安装状态推力标定
    时应保持 target_gain=0。
    """

    target_gain: float
    thrust_min_n: object
    thrust_max_n: object
    maximum_correction_n: object
    maximum_gain_rise_per_s: float

    def __post_init__(self) -> None:
        try:
            target_gain = _real_scalar("target_gain", self.target_gain)
        except ValueError as exc:
            raise ValueError("target_gain must be finite and within [0, 1]") from exc
        rise = _real_scalar("maximum_gain_rise_per_s", self.maximum_gain_rise_per_s)
        if not 0.0 <= target_gain <= 1.0:
            raise ValueError("target_gain must be finite and within [0, 1]")
        if rise <= 0.0:
            raise ValueError("maximum_gain_rise_per_s must be finite and positive")
        minimum = _vector4("thrust_min_n", self.thrust_min_n)
        maximum = _vector4("thrust_max_n", self.thrust_max_n)
        correction = _vector4("maximum_correction_n", self.maximum_correction_n)
        if np.any(minimum < 0.0):
            raise ValueError("thrust_min_n cannot be negative")
        if np.any(maximum <= minimum):
            raise ValueError("every thrust_max_n must exceed thrust_min_n")
        if np.any(correction <= 0.0):
            raise ValueError("maximum_correction_n must be positive")
        minimum.setflags(write=False)
        maximum.setflags(write=False)
        correction.setflags(write=False)
        object.__setattr__(self, "thrust_min_n", minimum)
        object.__setattr__(self, "thrust_max_n", maximum)
        object.__setattr__(self, "maximum_correction_n", correction)
        object.__setattr__(self, "target_gain", target_gain)
        object.__setattr__(self, "maximum_gain_rise_per_s", rise)


@dataclass(frozen=True)
class RotorCorrectionOutput:
    """Auditable transport reconstruction and applied command for one cycle.

    Only ``applied_residual_thrusts_n`` is an FC payload.  The transport target
    and raw correction are algebraic provenance at positive gain; they are not
    a second optimizer solution and are undefined when the applied gain is
    exactly zero.
    """

    baseline_thrusts_n: Tuple[float, float, float, float]
    transport_target_thrusts_n: Optional[Tuple[float, float, float, float]]
    transport_raw_correction_n: Optional[Tuple[float, float, float, float]]
    applied_residual_thrusts_n: Tuple[float, float, float, float]
    applied_total_thrusts_n: Tuple[float, float, float, float]
    requested_gain: float
    applied_gain: float
    headroom_gain: float
    headroom_limited: bool
    valid: bool
    reason: str
    transport_target_semantics: str


class RotorCorrectionBlender:
    """Stateful common-gain limiter placed immediately before the FC adapter.

    The gain rises from zero at a configured rate.  A solver/telemetry failure
    sets ``healthy=False`` and removes the MPC residual immediately, leaving
    the independent flight-controller baseline unchanged.

    中文：四个旋翼共用一个最终 κ，避免某一轴先饱和后改变残差力矩方向。健康失败
    时残差立即归零而非缓慢下降，因为继续施加未知旧残差比保持飞控基线更危险。
    """

    def __init__(self, config: RotorCorrectionSafetyConfig) -> None:
        self._config = config
        self._applied_gain = 0.0
        self._fault_latched = False

    @property
    def applied_gain(self) -> float:
        return self._applied_gain

    @property
    def config(self) -> RotorCorrectionSafetyConfig:
        """Expose the immutable limits used by the optimizer/execution check."""

        return self._config

    @property
    def fault_latched(self) -> bool:
        """Whether a failed cycle inhibits residuals until explicit reset."""

        return self._fault_latched

    def preview_gains(self, dt_s: float, steps: int) -> Tuple[float, ...]:
        """Return the gain ramp for a horizon without mutating runtime state.

        Headroom is intentionally not guessed here.  The MPC execution plan
        constrains its applied command and reconstructed raw residual with the
        same limits, so any later headroom reduction is treated as a model /
        execution mismatch and fails closed.
        """

        dt_s = _real_scalar("dt_s", dt_s)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        gain = self._applied_gain
        if self._fault_latched:
            return tuple(0.0 for _ in range(steps))
        result = []
        for _ in range(steps):
            gain = min(
                self._config.target_gain,
                gain + self._config.maximum_gain_rise_per_s * dt_s,
            )
            result.append(gain)
        return tuple(result)

    def blend_modeled_applied(
        self,
        baseline_thrusts_n: object,
        modeled_applied_thrusts_n: object,
        dt_s: float,
        *,
        expected_gain: float,
    ) -> RotorCorrectionOutput:
        """Execute an MPC command already modeled after the safety gain.

        The optimization variable is the total command that Eq. (11) will
        actually see.  For a positive gain this method reconstructs raw
        residual provenance, applies the normal blender, and proves that the
        result equals the modeled value.  The FC payload is the already-scaled
        applied residual, never this provenance value.  At zero gain, only the
        baseline is admissible and the canonical transmitted residual is zero;
        no transport target is inferred.
        """

        baseline = _vector4("baseline_thrusts_n", baseline_thrusts_n)
        modeled = _vector4("modeled_applied_thrusts_n", modeled_applied_thrusts_n)
        expected_gain = _real_scalar("expected_gain", expected_gain)
        if not 0.0 <= expected_gain <= 1.0:
            raise ValueError("expected_gain must be finite and within [0, 1]")
        previewed = self.preview_gains(dt_s, 1)[0]
        tolerance = 1.0e-10
        if not math.isclose(previewed, expected_gain, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError("expected_gain disagrees with the blender's non-mutating preview")

        residual = modeled - baseline
        if expected_gain == 0.0:
            if not np.allclose(residual, 0.0, rtol=0.0, atol=tolerance):
                raise ValueError("zero correction gain requires modeled command equal baseline")
            target = baseline
            target_semantics = "zero_gain_no_transport_target"
        else:
            raw = residual / expected_gain
            maximum = np.asarray(self._config.maximum_correction_n, dtype=float)
            if np.any(np.abs(raw) > maximum + tolerance):
                raise ValueError("modeled command requires raw correction above identified limit")
            target = baseline + raw
            target_semantics = (
                "active_gain_one_transport_target"
                if expected_gain == 1.0
                else "gain_limited_algebraic_reconstruction"
            )

        output = self.blend(
            baseline,
            target,
            dt_s,
            healthy=True,
            transport_target_semantics=target_semantics,
        )
        if expected_gain == 0.0:
            output = replace(
                output,
                transport_target_thrusts_n=None,
                transport_raw_correction_n=None,
                transport_target_semantics="zero_gain_no_transport_target",
            )
        actual = np.asarray(output.applied_total_thrusts_n, dtype=float)
        if (
            not output.valid
            or output.headroom_limited
            or not math.isclose(
                output.applied_gain,
                expected_gain,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
            or not np.allclose(actual, modeled, rtol=tolerance, atol=tolerance)
        ):
            self.inhibit()
            raise ValueError("blender changed the MPC-modeled applied command; correction removed")
        return output

    def reset(self, *, clear_fault_latch: bool = True) -> None:
        """Reset gain state; explicit session reset may clear a prior fault."""

        if type(clear_fault_latch) is not bool:
            raise TypeError("clear_fault_latch must be a bool")
        self._applied_gain = 0.0
        if clear_fault_latch:
            self._fault_latched = False

    def inhibit(self) -> None:
        """Immediately remove and latch the residual until explicit reset."""

        self._applied_gain = 0.0
        self._fault_latched = True

    def blend(
        self,
        baseline_thrusts_n: object,
        transport_target_thrusts_n: object,
        dt_s: float,
        *,
        healthy: bool = True,
        transport_target_semantics: str = "caller_supplied_transport_target",
        latch_failure: bool = True,
    ) -> RotorCorrectionOutput:
        baseline = _vector4("baseline_thrusts_n", baseline_thrusts_n)
        transport_target = _vector4(
            "transport_target_thrusts_n",
            transport_target_thrusts_n,
        )
        dt_s = _real_scalar("dt_s", dt_s)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if (
            not isinstance(transport_target_semantics, str)
            or not transport_target_semantics.strip()
        ):
            raise ValueError("transport_target_semantics must be a nonempty string")
        if type(latch_failure) is not bool:
            raise TypeError("latch_failure must be a bool")

        minimum = np.asarray(self._config.thrust_min_n, dtype=float)
        maximum = np.asarray(self._config.thrust_max_n, dtype=float)
        if np.any(baseline < minimum) or np.any(baseline > maximum):
            raise ValueError("flight-controller baseline lies outside identified thrust limits")

        raw = transport_target - baseline
        if not healthy or self._fault_latched:
            if latch_failure or self._fault_latched:
                self.inhibit()
            else:
                self._applied_gain = 0.0
            zeros = np.zeros(4, dtype=float)
            return RotorCorrectionOutput(
                baseline_thrusts_n=_tuple4(baseline),
                transport_target_thrusts_n=None,
                transport_raw_correction_n=None,
                applied_residual_thrusts_n=_tuple4(zeros),
                applied_total_thrusts_n=_tuple4(baseline),
                requested_gain=self._config.target_gain,
                applied_gain=0.0,
                headroom_gain=0.0,
                headroom_limited=False,
                valid=False,
                reason=(
                    "MPC correction removed because the control cycle is unhealthy "
                    "or a prior failure remains latched"
                ),
                transport_target_semantics="correction_removed",
            )

        ramped_gain = self.preview_gains(dt_s, 1)[0]
        headroom_gain = self._headroom_gain(baseline, raw)
        applied_gain = min(ramped_gain, headroom_gain)
        scaled = applied_gain * raw
        commanded = baseline + scaled

        tolerance = 1e-10
        if np.any(commanded < minimum - tolerance) or np.any(commanded > maximum + tolerance):
            raise RuntimeError("common-gain headroom projection failed")
        commanded = np.minimum(np.maximum(commanded, minimum), maximum)
        self._applied_gain = applied_gain
        limited = headroom_gain + tolerance < ramped_gain
        if applied_gain == 0.0:
            transport_target_value = None
            transport_raw_value = None
            output_semantics = "zero_gain_no_transport_target"
        else:
            transport_target_value = _tuple4(transport_target)
            transport_raw_value = _tuple4(raw)
            output_semantics = transport_target_semantics
        return RotorCorrectionOutput(
            baseline_thrusts_n=_tuple4(baseline),
            transport_target_thrusts_n=transport_target_value,
            transport_raw_correction_n=transport_raw_value,
            applied_residual_thrusts_n=_tuple4(scaled),
            applied_total_thrusts_n=_tuple4(commanded),
            requested_gain=self._config.target_gain,
            applied_gain=applied_gain,
            headroom_gain=headroom_gain,
            headroom_limited=limited,
            valid=True,
            reason="common-gain MPC correction applied",
            transport_target_semantics=output_semantics,
        )

    def _headroom_gain(
        self,
        baseline: NDArray[np.float64],
        correction: NDArray[np.float64],
    ) -> float:
        minimum = np.asarray(self._config.thrust_min_n, dtype=float)
        maximum = np.asarray(self._config.thrust_max_n, dtype=float)
        correction_limit = np.asarray(self._config.maximum_correction_n, dtype=float)
        gain = 1.0
        for index, delta in enumerate(correction):
            magnitude = abs(float(delta))
            if magnitude <= 1e-12:
                continue
            gain = min(gain, float(correction_limit[index]) / magnitude)
            if delta > 0.0:
                gain = min(gain, float(maximum[index] - baseline[index]) / float(delta))
            else:
                gain = min(gain, float(minimum[index] - baseline[index]) / float(delta))
        return min(max(gain, 0.0), 1.0)


__all__ = [
    "RotorCorrectionBlender",
    "RotorCorrectionOutput",
    "RotorCorrectionSafetyConfig",
]
