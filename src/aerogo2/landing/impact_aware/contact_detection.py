"""Filtered, hysteretic per-foot contact detection for paper Eq. (53).

The paper specifies filtered contact detection with hysteresis but supplies no
thresholds or timing values.  This implementation therefore requires every
value from calibration and exposes the detected touchdown time used by the
admittance blend.  It has no dependency on the Go2 transport.

中文说明：检测器对四条腿分别进行一阶低通、双阈值滞回和驻留时间确认。
``contact_on_threshold_n`` 以上持续足够时间才确认触地；已触地后必须低于更小的
``contact_off_threshold_n`` 并持续足够时间才释放。这样可抑制冲击尖峰和 SDK
计数噪声造成的状态抖动。输入必须是已完成脚序、符号及单位校准的法向力；原始
``foot_force``/``foot_force_est`` 整数不能在未经标定时冒充牛顿值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _finite_vector4(name: str, values: object) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=float)
    if array.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float64).copy()


def _finite_real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class ContactDetectorConfig:
    """四足独立滞回检测参数。

    开阈值必须严格高于关阈值；滤波时间常数用于衰减噪声，on/off dwell 规定事件
    必须持续多久才确认。所有四元素参数都必须采用系统统一脚序。
    """

    contact_on_threshold_n: object
    contact_off_threshold_n: object
    filter_time_constant_s: float
    contact_confirm_s: float
    release_confirm_s: float

    def __post_init__(self) -> None:
        on = _finite_vector4("contact_on_threshold_n", self.contact_on_threshold_n)
        off = _finite_vector4("contact_off_threshold_n", self.contact_off_threshold_n)
        if np.any(on <= 0.0) or np.any(off < 0.0):
            raise ValueError("contact thresholds must be nonnegative and on must be positive")
        if np.any(off >= on):
            raise ValueError("each contact_off_threshold_n must be below its on threshold")
        for name in ("filter_time_constant_s", "contact_confirm_s", "release_confirm_s"):
            value = _finite_real(name, getattr(self, name), positive=True)
            object.__setattr__(self, name, value)
        on.setflags(write=False)
        off.setflags(write=False)
        object.__setattr__(self, "contact_on_threshold_n", on)
        object.__setattr__(self, "contact_off_threshold_n", off)


@dataclass(frozen=True)
class ContactDetection:
    """一次检测结果；事件标志只在接触状态翻转的采样点为真。

    ``touchdown_times_s`` 保存各脚最近一次确认触地的单调时钟时间，供导纳参数
    平滑切换；从未确认触地的脚为 ``None``。
    """

    timestamp_s: float
    filtered_normal_forces_n: Tuple[float, float, float, float]
    contacts: Tuple[bool, bool, bool, bool]
    touchdown_events: Tuple[bool, bool, bool, bool]
    release_events: Tuple[bool, bool, bool, bool]
    touchdown_times_s: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]


class FootContactDetector:
    """带记忆的四脚接触检测器；同一对象必须按严格递增时间顺序更新。"""

    """Four independent Schmitt triggers with time confirmation."""

    def __init__(self, config: ContactDetectorConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._filtered: Optional[NDArray[np.float64]] = None
        self._contacts = np.zeros(4, dtype=bool)
        self._on_since: list[Optional[float]] = [None] * 4
        self._off_since: list[Optional[float]] = [None] * 4
        self._touchdown_times: list[Optional[float]] = [None] * 4
        self._last_timestamp: Optional[float] = None

    def update(self, normal_forces_n: object, timestamp_s: float) -> ContactDetection:
        """用一帧已校准的四脚法向力推进低通、滞回和驻留状态。

        首次采样只初始化滤波器，不会因单个冲击尖峰立即确认触地。时间倒退、NaN
        或形状错误均直接拒绝，以免接触状态机在不可信输入上继续运行。
        """
        forces = _finite_vector4("normal_forces_n", normal_forces_n)
        if np.any(forces < 0.0):
            raise ValueError("normal_forces_n cannot be negative")
        timestamp = _finite_real("timestamp_s", timestamp_s)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamp_s must increase strictly")

        if self._filtered is None:
            filtered = forces.copy()
        else:
            assert self._last_timestamp is not None
            dt_s = timestamp - self._last_timestamp
            alpha = 1.0 - math.exp(-dt_s / self._config.filter_time_constant_s)
            filtered = self._filtered + alpha * (forces - self._filtered)

        touchdown_events = np.zeros(4, dtype=bool)
        release_events = np.zeros(4, dtype=bool)
        on_threshold = np.asarray(self._config.contact_on_threshold_n, dtype=float)
        off_threshold = np.asarray(self._config.contact_off_threshold_n, dtype=float)
        for index in range(4):
            if not self._contacts[index]:
                self._off_since[index] = None
                if filtered[index] >= on_threshold[index]:
                    if self._on_since[index] is None:
                        self._on_since[index] = timestamp
                    on_since = self._on_since[index]
                    assert on_since is not None
                    if timestamp - on_since >= self._config.contact_confirm_s:
                        self._contacts[index] = True
                        touchdown_events[index] = True
                        self._touchdown_times[index] = timestamp
                        self._on_since[index] = None
                else:
                    self._on_since[index] = None
            else:
                self._on_since[index] = None
                if filtered[index] <= off_threshold[index]:
                    if self._off_since[index] is None:
                        self._off_since[index] = timestamp
                    off_since = self._off_since[index]
                    assert off_since is not None
                    if timestamp - off_since >= self._config.release_confirm_s:
                        self._contacts[index] = False
                        release_events[index] = True
                        self._touchdown_times[index] = None
                        self._off_since[index] = None
                else:
                    self._off_since[index] = None

        self._filtered = filtered.copy()
        self._last_timestamp = timestamp
        filtered_tuple = (
            float(filtered[0]),
            float(filtered[1]),
            float(filtered[2]),
            float(filtered[3]),
        )
        contacts_tuple = tuple(bool(self._contacts[index]) for index in range(4))
        touchdown_tuple = tuple(bool(touchdown_events[index]) for index in range(4))
        release_tuple = tuple(bool(release_events[index]) for index in range(4))
        return ContactDetection(
            timestamp_s=timestamp,
            filtered_normal_forces_n=filtered_tuple,
            contacts=(contacts_tuple[0], contacts_tuple[1], contacts_tuple[2], contacts_tuple[3]),
            touchdown_events=(
                touchdown_tuple[0],
                touchdown_tuple[1],
                touchdown_tuple[2],
                touchdown_tuple[3],
            ),
            release_events=(
                release_tuple[0],
                release_tuple[1],
                release_tuple[2],
                release_tuple[3],
            ),
            touchdown_times_s=(
                self._touchdown_times[0],
                self._touchdown_times[1],
                self._touchdown_times[2],
                self._touchdown_times[3],
            ),
        )


__all__ = ["ContactDetection", "ContactDetectorConfig", "FootContactDetector"]
