"""Strict, opt-in calibration of Go2 LowState foot-force integers.

The public Go2 SDK fields are not a ready-made three-axis contact wrench.  This
module only turns one reviewed scalar channel per foot into a nonnegative
normal force.  It is intentionally not wired into the production multi-rate
loop: deployment must additionally bind the returned leg order to kinematics,
frames, timing and the estimator used to form a complete atomic sample.

中文说明：Go2 LowState 的两个四元素字段是每脚一个整数通道，并非三轴力传感器。
适配器要求明确选择 raw 或 estimated 来源、脚序、零偏、比例、方向和校准哈希，才将
计数换算为非负法向力。若没有逆行标定/加载标定，只能把原始计数用于接触事件和趋势，
不能输出牛顿值给三维动力学，也不能据此积分论文中的真实冲量。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Iterable, Tuple, cast

from aerogo2.common.models import Go2FootForceFeedback
from aerogo2.landing.impact_aware.types import validate_four_foot_leg_order

_SHA256_PREFIX = "sha256:"
_FOOT_COUNT = 4


class Go2FootForceAdapterError(ValueError):
    """Raised when untrusted feedback cannot produce a calibrated sample."""


class Go2FootForceSource(str, Enum):
    """The exact SDK integer array selected for robot-specific calibration."""

    RAW_INT16 = "LOWSTATE_FOOT_FORCE_RAW_INT16"
    ESTIMATED_INT16 = "LOWSTATE_FOOT_FORCE_ESTIMATED_INT16"


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _names4(name: str, value: object) -> Tuple[str, str, str, str]:
    return validate_four_foot_leg_order(value, name=name)


def _indices4(name: str, value: object) -> Tuple[int, int, int, int]:
    try:
        raw = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw) != _FOOT_COUNT:
        raise ValueError(f"{name} must contain four integer SDK indices")
    parsed = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} must contain four integer SDK indices")
        parsed.append(int(item))
    result = cast(Tuple[int, int, int, int], tuple(parsed))
    if set(result) != set(range(_FOOT_COUNT)):
        raise ValueError(f"{name} must be a permutation of SDK indices 0..3")
    return result


def _finite4(
    name: str,
    value: object,
    *,
    positive: bool = False,
) -> Tuple[float, float, float, float]:
    try:
        raw = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw) != _FOOT_COUNT:
        raise ValueError(f"{name} must contain four values")
    result = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name} must contain only real numbers")
        parsed = float(item)
        if not math.isfinite(parsed) or (positive and parsed <= 0.0):
            qualifier = "finite positive" if positive else "finite"
            raise ValueError(f"{name} must contain only {qualifier} values")
        result.append(parsed)
    return (result[0], result[1], result[2], result[3])


def _signs4(name: str, value: object) -> Tuple[int, int, int, int]:
    try:
        raw = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable") from exc
    if len(raw) != _FOOT_COUNT:
        raise ValueError(f"{name} must contain four signs selected from -1 and 1")
    parsed = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} must contain four signs selected from -1 and 1")
        integer = int(item)
        if integer not in (-1, 1):
            raise ValueError(f"{name} must contain four signs selected from -1 and 1")
        parsed.append(integer)
    return cast(Tuple[int, int, int, int], tuple(parsed))


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    digest = value[len(_SHA256_PREFIX) :] if value.startswith(_SHA256_PREFIX) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must use sha256:<64 lowercase hexadecimal digits>")
    return value


def compute_go2_foot_force_mapping_hash(
    mapping_version: str,
    algorithm_leg_order: object,
    sdk_indices_by_leg: object,
) -> str:
    """Hash the explicitly commissioned SDK-index-to-algorithm-foot mapping."""

    payload = {
        "algorithm_leg_order": list(_names4("algorithm_leg_order", algorithm_leg_order)),
        "mapping_version": _nonempty("mapping_version", mapping_version),
        "sdk_indices_by_leg": list(_indices4("sdk_indices_by_leg", sdk_indices_by_leg)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def compute_go2_foot_force_calibration_hash(
    *,
    mapping_hash: str,
    calibration_version: str,
    source: Go2FootForceSource,
    offsets_sdk_by_algorithm_leg: object,
    scales_n_per_sdk_unit_by_algorithm_leg: object,
    signs_by_algorithm_leg: object,
    maximum_valid_normal_force_n_by_algorithm_leg: object,
) -> str:
    """Hash the exact affine scalar-normal-force calibration contract."""

    if not isinstance(source, Go2FootForceSource):
        raise TypeError("source must be a Go2FootForceSource")
    payload = {
        "calibration_version": _nonempty("calibration_version", calibration_version),
        "mapping_hash": _sha256(mapping_hash, "mapping_hash"),
        "maximum_valid_normal_force_n_by_algorithm_leg": list(
            _finite4(
                "maximum_valid_normal_force_n_by_algorithm_leg",
                maximum_valid_normal_force_n_by_algorithm_leg,
                positive=True,
            )
        ),
        "offsets_sdk_by_algorithm_leg": list(
            _finite4("offsets_sdk_by_algorithm_leg", offsets_sdk_by_algorithm_leg)
        ),
        "scales_n_per_sdk_unit_by_algorithm_leg": list(
            _finite4(
                "scales_n_per_sdk_unit_by_algorithm_leg",
                scales_n_per_sdk_unit_by_algorithm_leg,
                positive=True,
            )
        ),
        "signs_by_algorithm_leg": list(_signs4("signs_by_algorithm_leg", signs_by_algorithm_leg)),
        "source": source.value,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Go2FootForceCalibration:
    """Reviewed mapping plus per-robot affine calibration; no field defaults.

    All calibration vectors are in ``algorithm_leg_order``.  Entry ``i`` uses
    ``sdk_indices_by_leg[i]`` to select its SDK channel, then applies entry
    ``i`` of offset, scale and sign.  No vector in this object is SDK order
    except ``sdk_indices_by_leg`` itself.
    """

    mapping_version: str
    mapping_hash: str
    calibration_version: str
    calibration_hash: str
    algorithm_leg_order: Tuple[str, str, str, str]
    sdk_indices_by_leg: Tuple[int, int, int, int]
    source: Go2FootForceSource
    offsets_sdk_by_algorithm_leg: Tuple[float, float, float, float]
    scales_n_per_sdk_unit_by_algorithm_leg: Tuple[float, float, float, float]
    signs_by_algorithm_leg: Tuple[int, int, int, int]
    maximum_valid_normal_force_n_by_algorithm_leg: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        mapping_version = _nonempty("mapping_version", self.mapping_version)
        calibration_version = _nonempty("calibration_version", self.calibration_version)
        leg_order = _names4("algorithm_leg_order", self.algorithm_leg_order)
        indices = _indices4("sdk_indices_by_leg", self.sdk_indices_by_leg)
        if not isinstance(self.source, Go2FootForceSource):
            raise TypeError("source must be a Go2FootForceSource")
        offsets = _finite4(
            "offsets_sdk_by_algorithm_leg",
            self.offsets_sdk_by_algorithm_leg,
        )
        scales = _finite4(
            "scales_n_per_sdk_unit_by_algorithm_leg",
            self.scales_n_per_sdk_unit_by_algorithm_leg,
            positive=True,
        )
        signs = _signs4("signs_by_algorithm_leg", self.signs_by_algorithm_leg)
        maximum = _finite4(
            "maximum_valid_normal_force_n_by_algorithm_leg",
            self.maximum_valid_normal_force_n_by_algorithm_leg,
            positive=True,
        )
        mapping_hash = _sha256(self.mapping_hash, "mapping_hash")
        expected_mapping = compute_go2_foot_force_mapping_hash(
            mapping_version,
            leg_order,
            indices,
        )
        if mapping_hash != expected_mapping:
            raise ValueError("mapping_hash does not match the canonical foot-index mapping")
        calibration_hash = _sha256(self.calibration_hash, "calibration_hash")
        expected_calibration = compute_go2_foot_force_calibration_hash(
            mapping_hash=mapping_hash,
            calibration_version=calibration_version,
            source=self.source,
            offsets_sdk_by_algorithm_leg=offsets,
            scales_n_per_sdk_unit_by_algorithm_leg=scales,
            signs_by_algorithm_leg=signs,
            maximum_valid_normal_force_n_by_algorithm_leg=maximum,
        )
        if calibration_hash != expected_calibration:
            raise ValueError("calibration_hash does not match the canonical force calibration")
        object.__setattr__(self, "mapping_version", mapping_version)
        object.__setattr__(self, "mapping_hash", mapping_hash)
        object.__setattr__(self, "calibration_version", calibration_version)
        object.__setattr__(self, "calibration_hash", calibration_hash)
        object.__setattr__(self, "algorithm_leg_order", leg_order)
        object.__setattr__(self, "sdk_indices_by_leg", indices)
        object.__setattr__(self, "offsets_sdk_by_algorithm_leg", offsets)
        object.__setattr__(self, "scales_n_per_sdk_unit_by_algorithm_leg", scales)
        object.__setattr__(self, "signs_by_algorithm_leg", signs)
        object.__setattr__(
            self,
            "maximum_valid_normal_force_n_by_algorithm_leg",
            maximum,
        )


@dataclass(frozen=True)
class CalibratedGo2NormalForceSample:
    """One valid four-foot scalar normal-force observation in algorithm order."""

    algorithm_leg_order: Tuple[str, str, str, str]
    normal_forces_n: Tuple[float, float, float, float]
    source: Go2FootForceSource
    source_tick: int
    receipt_timestamp_s: float
    receipt_sequence: int
    subscription_generation: int
    mapping_hash: str
    calibration_hash: str

    def __post_init__(self) -> None:
        leg_order = _names4("algorithm_leg_order", self.algorithm_leg_order)
        forces = _finite4("normal_forces_n", self.normal_forces_n)
        if any(force < 0.0 for force in forces):
            raise ValueError("normal_forces_n cannot be negative")
        if not isinstance(self.source, Go2FootForceSource):
            raise TypeError("source must be a Go2FootForceSource")
        if (
            isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or not 0 <= self.source_tick <= 0xFFFFFFFF
        ):
            raise ValueError("source_tick must be a uint32")
        timestamp = self.receipt_timestamp_s
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, Real)
            or not math.isfinite(float(timestamp))
            or float(timestamp) < 0.0
        ):
            raise ValueError("receipt_timestamp_s must be finite and nonnegative")
        for name in ("receipt_sequence", "subscription_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "algorithm_leg_order", leg_order)
        object.__setattr__(self, "normal_forces_n", forces)
        object.__setattr__(self, "receipt_timestamp_s", float(timestamp))
        object.__setattr__(self, "mapping_hash", _sha256(self.mapping_hash, "mapping_hash"))
        object.__setattr__(
            self,
            "calibration_hash",
            _sha256(self.calibration_hash, "calibration_hash"),
        )


def calibrate_go2_normal_forces(
    feedback: Go2FootForceFeedback,
    calibration: Go2FootForceCalibration,
) -> CalibratedGo2NormalForceSample:
    """Map and calibrate one frame, raising instead of returning unsafe zeros.

    For algorithm leg ``i`` the formula is
    ``F_i = sign_i * (sdk[index_i] - offset_i) * scale_i``.  Offset, scale,
    sign and maximum entries are all indexed by algorithm leg ``i``.  Every
    result must be finite, nonnegative and inside its commissioned envelope.
    This scalar result is suitable for contact detection only; it is not a
    three-axis world-frame contact-force estimate for MPC state construction.
    """

    if not isinstance(feedback, Go2FootForceFeedback):
        raise TypeError("feedback must be Go2FootForceFeedback")
    if not isinstance(calibration, Go2FootForceCalibration):
        raise TypeError("calibration must be Go2FootForceCalibration")
    if not feedback.source_identity_valid or feedback.source_tick is None:
        raise Go2FootForceAdapterError(
            "LowState foot-force source tick/generation/receipt identity is invalid"
        )
    if calibration.source is Go2FootForceSource.RAW_INT16:
        if not feedback.raw_valid:
            raise Go2FootForceAdapterError("LowState foot_force is missing or malformed")
        sdk_values = feedback.raw_sdk_int16
    else:
        if not feedback.estimated_valid:
            raise Go2FootForceAdapterError("LowState foot_force_est is missing or malformed")
        sdk_values = feedback.estimated_sdk_int16
    ordered_sdk = tuple(sdk_values[index] for index in calibration.sdk_indices_by_leg)
    if any(value in (-32768, 32767) for value in ordered_sdk):
        raise Go2FootForceAdapterError("selected SDK foot-force channel is saturated")
    forces = tuple(
        calibration.signs_by_algorithm_leg[index]
        * (float(ordered_sdk[index]) - calibration.offsets_sdk_by_algorithm_leg[index])
        * calibration.scales_n_per_sdk_unit_by_algorithm_leg[index]
        for index in range(_FOOT_COUNT)
    )
    for index, force in enumerate(forces):
        if (
            not math.isfinite(force)
            or force < 0.0
            or force > calibration.maximum_valid_normal_force_n_by_algorithm_leg[index]
        ):
            raise Go2FootForceAdapterError(
                f"calibrated normal force for {calibration.algorithm_leg_order[index]} "
                "is outside the commissioned nonnegative range"
            )
    return CalibratedGo2NormalForceSample(
        algorithm_leg_order=calibration.algorithm_leg_order,
        normal_forces_n=cast(Tuple[float, float, float, float], forces),
        source=calibration.source,
        source_tick=feedback.source_tick,
        receipt_timestamp_s=feedback.receipt_timestamp_s,
        receipt_sequence=feedback.receipt_sequence,
        subscription_generation=feedback.subscription_generation,
        mapping_hash=calibration.mapping_hash,
        calibration_hash=calibration.calibration_hash,
    )


__all__ = [
    "CalibratedGo2NormalForceSample",
    "Go2FootForceAdapterError",
    "Go2FootForceCalibration",
    "Go2FootForceSource",
    "calibrate_go2_normal_forces",
    "compute_go2_foot_force_calibration_hash",
    "compute_go2_foot_force_mapping_hash",
]
