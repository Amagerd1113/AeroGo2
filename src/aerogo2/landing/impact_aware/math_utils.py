"""Small, dependency-light helpers for vectors and :math:`SO(3)`.

Frame convention used throughout this package:

* ``rotation_body_to_world`` maps a column vector from body frame ``B`` to
  world frame ``G``.
* angular velocity is expressed in ``B``.
* linear position and velocity are expressed in ``G``.

Only NumPy is used so this mathematical core remains independent of hardware,
SDK, and nonlinear-programming libraries.

中文说明：此处统一约定旋转矩阵 ``R_GB`` 将机体系 B 中的列向量变换到世界系 G。
角速度用 B 系表达，质心位置/线速度用 G 系表达。SO(3) 指数映射、反对称矩阵和
B/C 原点换算均在此实现；调用方不得混用 PX4 常见的 NED/FRD 与本模块的坐标约定，
必须先在飞控桥中完成坐标系和四元数顺序转换。
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]


def _finite_scalar(
    value: object,
    name: str,
    *,
    minimum: Optional[float] = None,
    strictly_greater: bool = False,
) -> float:
    """Return a finite real scalar or raise a boundary-specific error."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strictly_greater and converted <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
        if not strictly_greater and converted < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _as_finite_array(value: object, shape: Tuple[int, ...], name: str) -> FloatArray:
    """Defensively copy a finite numeric array with exactly ``shape``."""

    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = cast(FloatArray, np.array(raw, dtype=float, copy=True))
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_binary_vector(value: object, length: int, name: str) -> FloatArray:
    """Defensively copy an exactly binary vector as read-only integers."""

    raw = _as_finite_array(value, (length,), name)
    if not np.all((raw == 0.0) | (raw == 1.0)):
        raise ValueError(f"{name} must contain only 0 or 1")
    result = cast(FloatArray, raw.astype(float))
    result.setflags(write=False)
    return result


def _readonly(array: FloatArray) -> FloatArray:
    """Return a read-only defensive float copy."""

    result = cast(FloatArray, np.array(array, dtype=float, copy=True))
    result.setflags(write=False)
    return result


def skew(vector: object) -> FloatArray:
    """Return the skew matrix ``S(v)`` such that ``S(v) @ x == v x x``."""

    x, y, z = _as_finite_array(vector, (3,), "vector")
    return cast(
        FloatArray,
        np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        ),
    )


def vee(matrix: object, *, atol: float = 1e-9) -> FloatArray:
    """Map a skew-symmetric 3x3 matrix to its vector representation."""

    tolerance = _finite_scalar(atol, "atol", minimum=0.0)
    value = _as_finite_array(matrix, (3, 3), "matrix")
    if not np.allclose(value + value.T, 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("matrix must be skew-symmetric")
    return cast(
        FloatArray,
        np.array([value[2, 1], value[0, 2], value[1, 0]], dtype=float),
    )


def is_rotation_matrix(rotation: object, *, atol: float = 1e-9) -> bool:
    """Return whether ``rotation`` is a finite, proper 3D rotation matrix."""

    try:
        tolerance = _finite_scalar(atol, "atol", minimum=0.0)
        value = _as_finite_array(rotation, (3, 3), "rotation")
    except (TypeError, ValueError):
        return False
    identity_error_ok = np.allclose(
        value.T @ value,
        np.eye(3),
        rtol=0.0,
        atol=tolerance,
    )
    determinant_ok = math.isclose(
        float(np.linalg.det(value)),
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    return bool(identity_error_ok and determinant_ok)


def require_rotation_matrix(
    rotation: object,
    *,
    name: str = "rotation",
    atol: float = 1e-9,
) -> FloatArray:
    """Validate and defensively copy a body-to-world rotation matrix."""

    tolerance = _finite_scalar(atol, "atol", minimum=0.0)
    value = _as_finite_array(rotation, (3, 3), name)
    if not is_rotation_matrix(value, atol=tolerance):
        raise ValueError(f"{name} must be a proper SO(3) rotation matrix")
    return value


def so3_exp(rotation_vector_rad: object) -> FloatArray:
    """Evaluate the exponential map from a rotation vector to ``SO(3)``."""

    rotation_vector = _as_finite_array(
        rotation_vector_rad,
        (3,),
        "rotation_vector_rad",
    )
    theta_squared = float(rotation_vector @ rotation_vector)
    theta = math.sqrt(theta_squared)
    generator = skew(rotation_vector)

    if theta < 1e-8:
        theta_fourth = theta_squared * theta_squared
        sine_coefficient = 1.0 - theta_squared / 6.0 + theta_fourth / 120.0
        cosine_coefficient = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0
    else:
        sine_coefficient = math.sin(theta) / theta
        cosine_coefficient = (1.0 - math.cos(theta)) / theta_squared

    result = np.eye(3) + sine_coefficient * generator + cosine_coefficient * (generator @ generator)
    return cast(FloatArray, result)


def integrate_body_rotation(
    rotation_body_to_world: object,
    angular_velocity_body_rad_per_s: object,
    dt_s: object,
) -> FloatArray:
    """Advance attitude for constant body angular velocity over ``dt_s``.

    This is the Lie-group counterpart of ``R_dot = R S(omega_B)``.  It keeps
    the result on ``SO(3)`` without normalizing or projecting caller data.
    """

    rotation = require_rotation_matrix(
        rotation_body_to_world,
        name="rotation_body_to_world",
    )
    angular_velocity = _as_finite_array(
        angular_velocity_body_rad_per_s,
        (3,),
        "angular_velocity_body_rad_per_s",
    )
    dt = _finite_scalar(dt_s, "dt_s", minimum=0.0)
    result = rotation @ so3_exp(angular_velocity * dt)
    return require_rotation_matrix(result, name="integrated_rotation", atol=1e-8)


def total_com_C_position_world_from_go2_body_origin_B(
    go2_body_origin_B_position_world_m: object,
    rotation_body_to_world: object,
    total_com_C_from_go2_body_origin_B_body_m: object,
) -> FloatArray:
    """Transform the Go2 body-origin position into total-system CoM position.

    With ``r_BC_B`` denoting the fixed vector from the Go2 body origin ``B``
    to the total-system centre of mass ``C``, expressed in the Go2 body axes,
    this evaluates

    ``p_C_W = p_B_W + R_WB @ r_BC_B``.

    The returned vector is a read-only defensive copy.  This helper performs
    only rigid-point kinematics; callers remain responsible for supplying a
    configuration-appropriate, identified ``B``-to-``C`` offset.

    中文：输入的 B→C 偏移用机体系表达，因此必须先乘 ``R_WB`` 再与世界系 B
    位置相加；直接把三个坐标分量相加只在机体姿态恒为单位阵时成立。
    """

    position_B_world = _as_finite_array(
        go2_body_origin_B_position_world_m,
        (3,),
        "go2_body_origin_B_position_world_m",
    )
    rotation = require_rotation_matrix(
        rotation_body_to_world,
        name="rotation_body_to_world",
    )
    offset_BC_body = _as_finite_array(
        total_com_C_from_go2_body_origin_B_body_m,
        (3,),
        "total_com_C_from_go2_body_origin_B_body_m",
    )
    return _readonly(position_B_world + rotation @ offset_BC_body)


def total_com_C_linear_velocity_world_from_go2_body_origin_B(
    go2_body_origin_B_linear_velocity_world_m_per_s: object,
    rotation_body_to_world: object,
    angular_velocity_body_rad_per_s: object,
    total_com_C_from_go2_body_origin_B_body_m: object,
) -> FloatArray:
    """Transform Go2 body-origin velocity into total-system CoM velocity.

    Angular velocity and the fixed ``B``-to-``C`` offset are expressed in the
    Go2 body axes, while both linear velocities are expressed in the world
    frame.  The rigid-point relation is

    ``v_C_W = v_B_W + R_WB @ (omega_B x r_BC_B)``.

    Omitting the cross-product term is only correct when the offset or angular
    velocity is zero.  The returned vector is a read-only defensive copy.

    中文：当 C 不与 B 重合且机体有角速度时，C 还包含 ``ω×r_BC`` 的刚体附加
    速度；遗漏该项会把旋转误差带入 MPC 初始线速度。
    """

    velocity_B_world = _as_finite_array(
        go2_body_origin_B_linear_velocity_world_m_per_s,
        (3,),
        "go2_body_origin_B_linear_velocity_world_m_per_s",
    )
    rotation = require_rotation_matrix(
        rotation_body_to_world,
        name="rotation_body_to_world",
    )
    angular_velocity_B = _as_finite_array(
        angular_velocity_body_rad_per_s,
        (3,),
        "angular_velocity_body_rad_per_s",
    )
    offset_BC_body = _as_finite_array(
        total_com_C_from_go2_body_origin_B_body_m,
        (3,),
        "total_com_C_from_go2_body_origin_B_body_m",
    )
    rotational_velocity_world = rotation @ np.cross(angular_velocity_B, offset_BC_body)
    return _readonly(velocity_B_world + rotational_velocity_world)


__all__ = [
    "integrate_body_rotation",
    "is_rotation_matrix",
    "require_rotation_matrix",
    "skew",
    "so3_exp",
    "total_com_C_linear_velocity_world_from_go2_body_origin_B",
    "total_com_C_position_world_from_go2_body_origin_B",
    "vee",
]
