"""Strict YAML assembly for the impact-aware landing controller.

The production template intentionally contains ``null`` calibration fields and
therefore cannot produce a runnable bundle until those values are identified.
The synthetic profile is numerical-test data only and requires an explicit
``allow_synthetic=True`` opt-in.  No configuration path in this module writes
to hardware or weakens the unavailable hardware adapters.

中文说明：加载器采用严格 schema、拒绝重复键和未知字段，并在构造阶段校验单位、
维度、范围以及相互依赖关系。生产配置中的 ``null`` 表示必须实机辨识，不能由加载器
悄悄填成论文示例值；离线混合配置即使数值完整也保持 ``hardware_output_permitted``
为假。增加参数时应同时更新 schema、数据类验证和测试，不能只在 YAML 中加字段。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from aerogo2.landing.impact_aware.admittance import (
    AdmittanceConfig,
    AxisAlignedWorkspace,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetectorConfig,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.go2_foot_force import (
    Go2FootForceCalibration,
    Go2FootForceSource,
)
from aerogo2.landing.impact_aware.nlp import (
    CONTROL_DIM,
    STATE_DIM,
    TRACKING_DIM,
    ContactForceLimits,
    MPCWeights,
    SLSQPSettings,
    StateBounds,
)
from aerogo2.landing.impact_aware.rotor import (
    build_fixed_deployed_allocation_matrix,
)
from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)
from aerogo2.landing.impact_aware.types import (
    FOOT_COUNT,
    FixedDeployedRotorGeometry,
    FloatArray,
    ImpactLimits,
    ReducedDynamicsConfig,
    RotorActuatorConfig,
    RotorAerodynamics,
    validate_four_foot_leg_order,
)

_T = TypeVar("_T")
_PathLike = Union[str, Path]
_EXPECTED_SCHEMA_VERSION = 3
_WORLD_FRAME = "ENU_Z_UP"
_BODY_FRAME = "X_FORWARD_Y_LEFT_Z_UP"
_ROTOR_ORDER = ("RR", "LF", "LR", "RF")
_ROTOR_CONFIGURATION = "FIXED_DEPLOYED_LOCKED"
_TOTAL_SYSTEM_COM_REFERENCE = "TOTAL_SYSTEM_COM_C"
_ROTOR_REFERENCE_ORIGIN = _TOTAL_SYSTEM_COM_REFERENCE
_MAX_CONFIG_BYTES = 1_000_000


class ImpactAwareConfigError(ValueError):
    """A fail-closed, path-qualified impact-aware configuration error."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> Dict[object, object]:
    loader.flatten_mapping(node)
    construct_object = cast(Any, loader.construct_object)
    result: Dict[object, object] = {}
    for key_node, value_node in node.value:
        key = construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ImpactAwareConfigError(f"{path} must be a mapping")
    result: Dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ImpactAwareConfigError(f"{path} keys must be strings")
        result[key] = item
    return result


def _check_keys(
    section: Mapping[str, object],
    path: str,
    required: FrozenSet[str],
    optional: FrozenSet[str] = frozenset(),
) -> None:
    actual = frozenset(section)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing:
        raise ImpactAwareConfigError(f"{path} is missing required keys: {missing}")
    if unknown:
        raise ImpactAwareConfigError(f"{path} contains unknown keys: {unknown}")


def _section(
    parent: Mapping[str, object],
    key: str,
    parent_path: str,
    required: FrozenSet[str],
    optional: FrozenSet[str] = frozenset(),
) -> Mapping[str, object]:
    value = parent.get(key)
    path = f"{parent_path}.{key}"
    result = _mapping(value, path)
    _check_keys(result, path, required, optional)
    return result


def _required(section: Mapping[str, object], key: str, path: str) -> Any:
    value = section.get(key)
    if value is None:
        raise ImpactAwareConfigError(
            f"{path}.{key} is required and cannot be null; identify it on the target system"
        )
    return value


def _bool(section: Mapping[str, object], key: str, path: str) -> bool:
    value = _required(section, key, path)
    if type(value) is not bool:
        raise ImpactAwareConfigError(f"{path}.{key} must be a boolean")
    return value


def _string(section: Mapping[str, object], key: str, path: str) -> str:
    value = _required(section, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ImpactAwareConfigError(f"{path}.{key} must be a nonempty string")
    return value


def _positive_integer(section: Mapping[str, object], key: str, path: str) -> int:
    value = _required(section, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImpactAwareConfigError(f"{path}.{key} must be an integer")
    if value < 1:
        raise ImpactAwareConfigError(f"{path}.{key} must be at least 1")
    return int(value)


def _positive_float(section: Mapping[str, object], key: str, path: str) -> float:
    value = _required(section, key, path)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ImpactAwareConfigError(f"{path}.{key} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ImpactAwareConfigError(f"{path}.{key} must be finite and positive")
    return result


def _name_tuple4(section: Mapping[str, object], key: str, path: str) -> Tuple[str, str, str, str]:
    value = _required(section, key, path)
    if not isinstance(value, list) or len(value) != 4:
        raise ImpactAwareConfigError(f"{path}.{key} must be a four-item list")
    try:
        return validate_four_foot_leg_order(value, name=f"{path}.{key}")
    except (TypeError, ValueError) as exc:
        raise ImpactAwareConfigError(str(exc)) from exc


def _numeric_vector(
    value: object,
    length: int,
    path: str,
    *,
    allow_infinity: bool = False,
) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ImpactAwareConfigError(f"{path} must be a flat numeric vector") from exc
    if raw.dtype.kind not in "fiu":
        raise ImpactAwareConfigError(f"{path} must contain real numeric values")
    try:
        result = cast(FloatArray, np.array(raw, dtype=float, copy=True))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImpactAwareConfigError(f"{path} must contain real numeric values") from exc
    if result.shape != (length,):
        raise ImpactAwareConfigError(f"{path} must have shape ({length},)")
    if np.any(np.isnan(result)) or (not allow_infinity and not np.all(np.isfinite(result))):
        qualifier = "non-NaN" if allow_infinity else "finite"
        raise ImpactAwareConfigError(f"{path} must contain only {qualifier} values")
    return result


def _boolean_vector(
    value: object,
    length: int,
    path: str,
) -> NDArray[np.bool_]:
    if not isinstance(value, list) or len(value) != length:
        raise ImpactAwareConfigError(f"{path} must be a {length}-item boolean list")
    if any(type(item) is not bool for item in value):
        raise ImpactAwareConfigError(f"{path} must contain only booleans")
    return cast(
        NDArray[np.bool_],
        np.array(value, dtype=bool),
    )


def _diagonal_weight(value: object, dimension: int, path: str) -> FloatArray:
    diagonal = _numeric_vector(value, dimension, path)
    if np.any(diagonal < 0.0):
        raise ImpactAwareConfigError(f"{path} diagonal weights cannot be negative")
    return cast(FloatArray, np.diag(diagonal))


def _build(path: str, factory: Callable[[], _T]) -> _T:
    try:
        return factory()
    except ImpactAwareConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ImpactAwareConfigError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class ImpactAwareConfigBundle:
    """Immutable, audited assembly of all static impact-aware parameters."""

    source_path: Path
    profile: str
    parameters_identified: bool
    allow_hardware_output: bool
    physical_use_prohibited: bool
    world_frame: str
    body_frame: str
    rotor_order: Tuple[str, str, str, str]
    rotor_configuration: str
    rotor_reference_origin: str
    dynamics_reference_point: str
    rotor_geometry: FixedDeployedRotorGeometry
    rotor_aerodynamics: RotorAerodynamics
    rotor_actuator: RotorActuatorConfig
    dynamics: ReducedDynamicsConfig
    impact_limits: ImpactLimits
    rotor_correction_safety: RotorCorrectionSafetyConfig
    contact_detector: ContactDetectorConfig
    # Optional scalar contact-detection source.  The paper's full 3-D contact
    # force estimate may instead come from an independently validated sensor
    # or estimator, so a Go2 LowState adapter is never implied to be unique.
    go2_foot_force_calibration: Optional[Go2FootForceCalibration]
    contact_force_limits: ContactForceLimits
    mpc_horizon_steps: int
    mpc_dt_s: float
    mpc_weights: MPCWeights
    mpc_state_bounds: StateBounds
    solver_settings: SLSQPSettings
    admittance_leg_order: Tuple[str, str, str, str]
    admittance_configs: Tuple[
        AdmittanceConfig,
        AdmittanceConfig,
        AdmittanceConfig,
        AdmittanceConfig,
    ]
    admittance_workspaces: Tuple[
        AxisAlignedWorkspace,
        AxisAlignedWorkspace,
        AxisAlignedWorkspace,
        AxisAlignedWorkspace,
    ]

    @property
    def is_synthetic(self) -> bool:
        return self.profile in {
            "synthetic_demo",
            "aerogo2_provisional_offline_hybrid",
        }

    @property
    def hardware_output_permitted(self) -> bool:
        """Single fail-closed authority predicate for every assembled bundle."""

        return (
            self.parameters_identified
            and self.allow_hardware_output
            and not self.physical_use_prohibited
            and not self.is_synthetic
        )

    @property
    def contact_detector_config(self) -> ContactDetectorConfig:
        """Compatibility alias making the contained object's role explicit."""

        return self.contact_detector

    @property
    def rotor_correction_safety_config(self) -> RotorCorrectionSafetyConfig:
        """Compatibility alias making the contained object's role explicit."""

        return self.rotor_correction_safety

    def new_contact_detector(self) -> FootContactDetector:
        """Create fresh stateful contact-detection runtime state."""

        return FootContactDetector(self.contact_detector)

    def new_rotor_correction_blender(self) -> RotorCorrectionBlender:
        """Create a fresh fail-closed gain-ramp runtime boundary."""

        return RotorCorrectionBlender(self.rotor_correction_safety)


def load_impact_aware_config(
    path: _PathLike,
    *,
    allow_synthetic: bool = False,
    for_hardware: bool = False,
) -> ImpactAwareConfigBundle:
    """Load and strictly assemble one impact-aware YAML configuration.

    ``allow_synthetic`` must be explicitly true for ``synthetic_demo``.  A
    synthetic profile is rejected for hardware under all circumstances.
    ``for_hardware`` currently always fails because matching FC firmware and a
    reviewed cross-device committer are not present for physical output.
    """

    if type(allow_synthetic) is not bool:
        raise TypeError("allow_synthetic must be a bool")
    if type(for_hardware) is not bool:
        raise TypeError("for_hardware must be a bool")
    try:
        source_path = Path(path).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ImpactAwareConfigError(f"configuration path is not a file: {source_path}")
        if source_path.stat().st_size > _MAX_CONFIG_BYTES:
            raise ImpactAwareConfigError("configuration file exceeds the 1 MB safety limit")
        text = source_path.read_text(encoding="utf-8")
    except ImpactAwareConfigError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ImpactAwareConfigError(f"cannot read impact-aware config {path!r}: {exc}") from exc

    try:
        raw_document = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ImpactAwareConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    document = _mapping(raw_document, "root")
    common_top_keys = frozenset(
        {
            "schema_version",
            "profile",
            "parameters_identified",
            "allow_hardware_output",
            "frames",
            "rotor",
            "dynamics",
            "contact",
            "impact",
            "mpc",
            "admittance",
        }
    )
    _check_keys(document, "root", common_top_keys, frozenset({"physical_use_prohibited"}))

    schema_version = _required(document, "schema_version", "root")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _EXPECTED_SCHEMA_VERSION
    ):
        raise ImpactAwareConfigError(
            f"root.schema_version must be integer {_EXPECTED_SCHEMA_VERSION}"
        )
    profile = _string(document, "profile", "root")
    if profile not in {"production", "synthetic_demo"}:
        raise ImpactAwareConfigError("root.profile must be 'production' or 'synthetic_demo'")
    parameters_identified = _bool(document, "parameters_identified", "root")
    hardware_allowed = _bool(document, "allow_hardware_output", "root")

    physical_prohibited = False
    if profile == "synthetic_demo":
        if not allow_synthetic:
            raise ImpactAwareConfigError(
                "synthetic_demo requires explicit allow_synthetic=True opt-in"
            )
        if for_hardware:
            raise ImpactAwareConfigError("synthetic_demo is prohibited for hardware use")
        if "physical_use_prohibited" not in document:
            raise ImpactAwareConfigError(
                "synthetic_demo must declare physical_use_prohibited: true"
            )
        physical_value = document["physical_use_prohibited"]
        if type(physical_value) is not bool or physical_value is not True:
            raise ImpactAwareConfigError(
                "synthetic_demo must declare physical_use_prohibited: true"
            )
        if hardware_allowed:
            raise ImpactAwareConfigError("synthetic_demo cannot allow hardware output")
        physical_prohibited = True
    elif "physical_use_prohibited" in document:
        raise ImpactAwareConfigError(
            "root.physical_use_prohibited is only valid for synthetic_demo"
        )

    if for_hardware:
        raise ImpactAwareConfigError(
            "impact-aware hardware use is not configured: the repository now has a "
            "fail-closed Go2 LowCmd owner and a host-side residual sink, but still "
            "has no matching flight-controller residual firmware/transport or "
            "cross-device atomic committer"
        )
    if hardware_allowed:
        raise ImpactAwareConfigError(
            "allow_hardware_output must remain false until the flight-controller "
            "rotor-residual firmware/transport and cross-device atomic committer are implemented "
            "and reviewed"
        )

    frames = _section(
        document,
        "frames",
        "root",
        frozenset({"world", "body", "rotor_order"}),
    )
    world_frame = _string(frames, "world", "root.frames")
    body_frame = _string(frames, "body", "root.frames")
    if world_frame != _WORLD_FRAME:
        raise ImpactAwareConfigError(f"root.frames.world must be {_WORLD_FRAME}")
    if body_frame != _BODY_FRAME:
        raise ImpactAwareConfigError(f"root.frames.body must be {_BODY_FRAME}")
    rotor_order = _name_tuple4(frames, "rotor_order", "root.frames")
    if rotor_order != _ROTOR_ORDER:
        raise ImpactAwareConfigError("root.frames.rotor_order must be exactly [RR, LF, LR, RF]")

    rotor = _section(
        document,
        "rotor",
        "root",
        frozenset({"geometry", "aerodynamics", "actuator", "correction_safety"}),
    )
    geometry_values = _section(
        rotor,
        "geometry",
        "root.rotor",
        frozenset(
            {
                "configuration",
                "reference_origin",
                "lever_arms_from_com_body_m",
                "thrust_directions_body",
            }
        ),
    )
    aerodynamics_values = _section(
        rotor,
        "aerodynamics",
        "root.rotor",
        frozenset(
            {
                "thrust_coefficient_n_per_rad_s_squared",
                "drag_torque_coefficient_nm_per_rad_s_squared",
                "spin_directions",
            }
        ),
    )
    actuator_values = _section(
        rotor,
        "actuator",
        "root.rotor",
        frozenset(
            {
                "time_constants_s",
                "thrust_min_n",
                "thrust_max_n",
                "thrust_rate_min_n_per_s",
                "thrust_rate_max_n_per_s",
            }
        ),
    )
    correction_values = _section(
        rotor,
        "correction_safety",
        "root.rotor",
        frozenset({"target_gain", "maximum_correction_n", "maximum_gain_rise_per_s"}),
    )

    rotor_configuration = _string(
        geometry_values,
        "configuration",
        "root.rotor.geometry",
    )
    if rotor_configuration != _ROTOR_CONFIGURATION:
        raise ImpactAwareConfigError(
            "root.rotor.geometry.configuration must be "
            f"{_ROTOR_CONFIGURATION}; folding geometry is outside this landing model"
        )
    rotor_reference_origin = _string(
        geometry_values,
        "reference_origin",
        "root.rotor.geometry",
    )
    if rotor_reference_origin != _ROTOR_REFERENCE_ORIGIN:
        raise ImpactAwareConfigError(
            "root.rotor.geometry.reference_origin must be "
            f"{_ROTOR_REFERENCE_ORIGIN}; a frame-centre arm is not valid for the "
            "CoM Newton-Euler model"
        )

    rotor_geometry = _build(
        "root.rotor.geometry",
        lambda: FixedDeployedRotorGeometry(
            lever_arms_from_com_body_m=_required(
                geometry_values,
                "lever_arms_from_com_body_m",
                "root.rotor.geometry",
            ),
            thrust_directions_body=_required(
                geometry_values,
                "thrust_directions_body",
                "root.rotor.geometry",
            ),
        ),
    )
    rotor_aerodynamics = _build(
        "root.rotor.aerodynamics",
        lambda: RotorAerodynamics(
            thrust_coefficient_n_per_rad_s_squared=_required(
                aerodynamics_values,
                "thrust_coefficient_n_per_rad_s_squared",
                "root.rotor.aerodynamics",
            ),
            drag_torque_coefficient_nm_per_rad_s_squared=_required(
                aerodynamics_values,
                "drag_torque_coefficient_nm_per_rad_s_squared",
                "root.rotor.aerodynamics",
            ),
            spin_directions=_required(
                aerodynamics_values,
                "spin_directions",
                "root.rotor.aerodynamics",
            ),
        ),
    )
    rotor_actuator = _build(
        "root.rotor.actuator",
        lambda: RotorActuatorConfig(
            time_constants_s=_required(
                actuator_values,
                "time_constants_s",
                "root.rotor.actuator",
            ),
            thrust_min_n=_required(
                actuator_values,
                "thrust_min_n",
                "root.rotor.actuator",
            ),
            thrust_max_n=_required(
                actuator_values,
                "thrust_max_n",
                "root.rotor.actuator",
            ),
            thrust_rate_min_n_per_s=_required(
                actuator_values,
                "thrust_rate_min_n_per_s",
                "root.rotor.actuator",
            ),
            thrust_rate_max_n_per_s=_required(
                actuator_values,
                "thrust_rate_max_n_per_s",
                "root.rotor.actuator",
            ),
        ),
    )

    dynamics_values = _section(
        document,
        "dynamics",
        "root",
        frozenset(
            {
                "mass_kg",
                "inertia_body_kg_m2",
                "gravity_world_m_per_s2",
                "reference_point",
            }
        ),
    )
    dynamics_reference_point = _string(
        dynamics_values,
        "reference_point",
        "root.dynamics",
    )
    if dynamics_reference_point != _TOTAL_SYSTEM_COM_REFERENCE:
        raise ImpactAwareConfigError(
            "root.dynamics.reference_point must be TOTAL_SYSTEM_COM_C for the reduced model"
        )
    dynamics = _build(
        "root.dynamics",
        lambda: ReducedDynamicsConfig(
            mass_kg=_required(dynamics_values, "mass_kg", "root.dynamics"),
            inertia_body_kg_m2=_required(
                dynamics_values,
                "inertia_body_kg_m2",
                "root.dynamics",
            ),
            gravity_world_m_per_s2=_required(
                dynamics_values,
                "gravity_world_m_per_s2",
                "root.dynamics",
            ),
            rotor_allocation_body=build_fixed_deployed_allocation_matrix(
                rotor_geometry,
                rotor_aerodynamics,
            ),
        ),
    )
    gravity = np.asarray(dynamics.gravity_world_m_per_s2)
    if not np.allclose(gravity[:2], 0.0, rtol=0.0, atol=1.0e-9) or gravity[2] >= 0.0:
        raise ImpactAwareConfigError(
            "root.dynamics.gravity_world_m_per_s2 must be [0, 0, negative] for "
            "the horizontal ENU impact model"
        )
    allocation = np.asarray(dynamics.rotor_allocation_body)
    if int(np.linalg.matrix_rank(allocation, tol=1.0e-10)) < 4:
        raise ImpactAwareConfigError(
            "root.rotor geometry/aerodynamics do not provide rank-4 thrust allocation"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        hover_wrench = np.concatenate((-dynamics.mass_kg * gravity, np.zeros(3)))
    if not np.all(np.isfinite(hover_wrench)):
        raise ImpactAwareConfigError(
            "root.dynamics produces a nonfinite derived level-hover wrench"
        )
    try:
        hover_thrusts, _, _, _ = np.linalg.lstsq(allocation, hover_wrench, rcond=None)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ImpactAwareConfigError(
            "root.rotor allocation level-hover solve failed"
        ) from exc
    with np.errstate(over="ignore", invalid="ignore"):
        hover_error = float(np.linalg.norm(allocation @ hover_thrusts - hover_wrench))
        hover_wrench_norm = float(np.linalg.norm(hover_wrench))
        hover_tolerance = 1.0e-7 * max(1.0, hover_wrench_norm)
    if (
        not np.all(np.isfinite(hover_thrusts))
        or not math.isfinite(hover_error)
        or not math.isfinite(hover_wrench_norm)
        or not math.isfinite(hover_tolerance)
    ):
        raise ImpactAwareConfigError(
            "root.rotor allocation produced nonfinite derived level-hover values"
        )
    if hover_error > hover_tolerance:
        raise ImpactAwareConfigError(
            "root.rotor allocation cannot produce a level zero-torque hover wrench"
        )
    if np.any(hover_thrusts < rotor_actuator.thrust_min_n) or np.any(
        hover_thrusts > rotor_actuator.thrust_max_n
    ):
        raise ImpactAwareConfigError(
            "root.rotor actuator bounds do not contain the level-hover thrust solution"
        )

    contact_values = _section(
        document,
        "contact",
        "root",
        frozenset(
            {
                "friction_coefficients",
                "normal_force_max_n",
                "contact_on_threshold_n",
                "contact_off_threshold_n",
                "filter_time_constant_s",
                "contact_confirm_s",
                "release_confirm_s",
            }
        ),
        frozenset({"go2_lowstate_force_adapter"}),
    )
    go2_foot_force_calibration: Optional[Go2FootForceCalibration] = None
    if "go2_lowstate_force_adapter" in contact_values:
        adapter_values = _section(
            contact_values,
            "go2_lowstate_force_adapter",
            "root.contact",
            frozenset(
                {
                    "mapping_version",
                    "mapping_hash",
                    "calibration_version",
                    "calibration_hash",
                    "algorithm_leg_order",
                    "sdk_indices_by_leg",
                    "source",
                    "offsets_sdk_by_algorithm_leg",
                    "scales_n_per_sdk_unit_by_algorithm_leg",
                    "signs_by_algorithm_leg",
                    "maximum_valid_normal_force_n_by_algorithm_leg",
                }
            ),
        )
        source_text = _string(
            adapter_values,
            "source",
            "root.contact.go2_lowstate_force_adapter",
        )
        try:
            force_source = Go2FootForceSource(source_text)
        except ValueError as exc:
            raise ImpactAwareConfigError(
                "root.contact.go2_lowstate_force_adapter.source is not a supported "
                "LowState integer field"
            ) from exc
        go2_foot_force_calibration = _build(
            "root.contact.go2_lowstate_force_adapter",
            lambda: Go2FootForceCalibration(
                mapping_version=_string(
                    adapter_values,
                    "mapping_version",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                mapping_hash=_string(
                    adapter_values,
                    "mapping_hash",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                calibration_version=_string(
                    adapter_values,
                    "calibration_version",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                calibration_hash=_string(
                    adapter_values,
                    "calibration_hash",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                algorithm_leg_order=_name_tuple4(
                    adapter_values,
                    "algorithm_leg_order",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                sdk_indices_by_leg=_required(
                    adapter_values,
                    "sdk_indices_by_leg",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                source=force_source,
                offsets_sdk_by_algorithm_leg=_required(
                    adapter_values,
                    "offsets_sdk_by_algorithm_leg",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                scales_n_per_sdk_unit_by_algorithm_leg=_required(
                    adapter_values,
                    "scales_n_per_sdk_unit_by_algorithm_leg",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                signs_by_algorithm_leg=_required(
                    adapter_values,
                    "signs_by_algorithm_leg",
                    "root.contact.go2_lowstate_force_adapter",
                ),
                maximum_valid_normal_force_n_by_algorithm_leg=_required(
                    adapter_values,
                    "maximum_valid_normal_force_n_by_algorithm_leg",
                    "root.contact.go2_lowstate_force_adapter",
                ),
            ),
        )
    friction_coefficients = _required(
        contact_values,
        "friction_coefficients",
        "root.contact",
    )
    contact_force_limits = _build(
        "root.contact",
        lambda: ContactForceLimits(
            friction_coefficients=friction_coefficients,
            maximum_normal_force_n=_required(
                contact_values,
                "normal_force_max_n",
                "root.contact",
            ),
        ),
    )
    contact_detector = _build(
        "root.contact",
        lambda: ContactDetectorConfig(
            contact_on_threshold_n=_required(
                contact_values,
                "contact_on_threshold_n",
                "root.contact",
            ),
            contact_off_threshold_n=_required(
                contact_values,
                "contact_off_threshold_n",
                "root.contact",
            ),
            filter_time_constant_s=_required(
                contact_values,
                "filter_time_constant_s",
                "root.contact",
            ),
            contact_confirm_s=_required(
                contact_values,
                "contact_confirm_s",
                "root.contact",
            ),
            release_confirm_s=_required(
                contact_values,
                "release_confirm_s",
                "root.contact",
            ),
        ),
    )
    if np.any(
        np.asarray(contact_detector.contact_on_threshold_n, dtype=float)
        > contact_force_limits.maximum_normal_force_n
    ):
        raise ImpactAwareConfigError(
            "root.contact.contact_on_threshold_n cannot exceed normal_force_max_n"
        )
    if go2_foot_force_calibration is not None and np.any(
        np.asarray(contact_detector.contact_on_threshold_n, dtype=float)
        > np.asarray(
            go2_foot_force_calibration.maximum_valid_normal_force_n_by_algorithm_leg,
            dtype=float,
        )
    ):
        raise ImpactAwareConfigError(
            "root.contact.contact_on_threshold_n cannot exceed the calibrated "
            "LowState force validation envelope"
        )

    impact_values = _section(
        document,
        "impact",
        "root",
        frozenset(
            {
                "maximum_normal_impulse_ns",
                "impact_duration_s",
                "maximum_average_normal_force_n",
            }
        ),
    )
    impact_limits = _build(
        "root.impact",
        lambda: ImpactLimits(
            friction_coefficients=friction_coefficients,
            maximum_normal_impulse_ns=_required(
                impact_values,
                "maximum_normal_impulse_ns",
                "root.impact",
            ),
            impact_duration_s=_required(
                impact_values,
                "impact_duration_s",
                "root.impact",
            ),
            maximum_average_normal_force_n=_required(
                impact_values,
                "maximum_average_normal_force_n",
                "root.impact",
            ),
        ),
    )

    rotor_correction_safety = _build(
        "root.rotor.correction_safety",
        lambda: RotorCorrectionSafetyConfig(
            target_gain=_required(
                correction_values,
                "target_gain",
                "root.rotor.correction_safety",
            ),
            thrust_min_n=rotor_actuator.thrust_min_n,
            thrust_max_n=rotor_actuator.thrust_max_n,
            maximum_correction_n=_required(
                correction_values,
                "maximum_correction_n",
                "root.rotor.correction_safety",
            ),
            maximum_gain_rise_per_s=_required(
                correction_values,
                "maximum_gain_rise_per_s",
                "root.rotor.correction_safety",
            ),
        ),
    )

    mpc_values = _section(
        document,
        "mpc",
        "root",
        frozenset({"horizon_steps", "dt_s", "solver", "weights", "state_bounds"}),
    )
    horizon_steps = _positive_integer(mpc_values, "horizon_steps", "root.mpc")
    dt_value = _positive_float(mpc_values, "dt_s", "root.mpc")
    solver_values = _section(
        mpc_values,
        "solver",
        "root.mpc",
        frozenset({"max_iterations", "ftol", "constraint_tolerance", "timeout_s", "display"}),
    )
    solver_settings = _build(
        "root.mpc.solver",
        lambda: SLSQPSettings(
            max_iterations=_positive_integer(
                solver_values,
                "max_iterations",
                "root.mpc.solver",
            ),
            ftol=_required(solver_values, "ftol", "root.mpc.solver"),
            constraint_tolerance=_required(
                solver_values,
                "constraint_tolerance",
                "root.mpc.solver",
            ),
            timeout_s=_required(solver_values, "timeout_s", "root.mpc.solver"),
            display=_bool(solver_values, "display", "root.mpc.solver"),
        ),
    )
    weights_values = _section(
        mpc_values,
        "weights",
        "root.mpc",
        frozenset(
            {
                "tracking",
                "input",
                "input_rate",
                "slack",
                "terminal_tracking",
                "impulse",
                "touchdown_velocity",
            }
        ),
    )
    mpc_weights = _build(
        "root.mpc.weights",
        lambda: MPCWeights(
            tracking=_diagonal_weight(
                _required(weights_values, "tracking", "root.mpc.weights"),
                TRACKING_DIM,
                "root.mpc.weights.tracking",
            ),
            input=_diagonal_weight(
                _required(weights_values, "input", "root.mpc.weights"),
                CONTROL_DIM,
                "root.mpc.weights.input",
            ),
            input_rate=_diagonal_weight(
                _required(weights_values, "input_rate", "root.mpc.weights"),
                CONTROL_DIM,
                "root.mpc.weights.input_rate",
            ),
            slack=_diagonal_weight(
                _required(weights_values, "slack", "root.mpc.weights"),
                STATE_DIM,
                "root.mpc.weights.slack",
            ),
            terminal_tracking=_diagonal_weight(
                _required(weights_values, "terminal_tracking", "root.mpc.weights"),
                TRACKING_DIM,
                "root.mpc.weights.terminal_tracking",
            ),
            impulse=_diagonal_weight(
                _required(weights_values, "impulse", "root.mpc.weights"),
                3,
                "root.mpc.weights.impulse",
            ),
            touchdown_velocity=_diagonal_weight(
                _required(weights_values, "touchdown_velocity", "root.mpc.weights"),
                3,
                "root.mpc.weights.touchdown_velocity",
            ),
        ),
    )
    bounds_values = _section(
        mpc_values,
        "state_bounds",
        "root.mpc",
        frozenset({"lower", "upper", "soft_mask"}),
    )
    lower_row = _numeric_vector(
        _required(bounds_values, "lower", "root.mpc.state_bounds"),
        STATE_DIM,
        "root.mpc.state_bounds.lower",
        allow_infinity=True,
    )
    upper_row = _numeric_vector(
        _required(bounds_values, "upper", "root.mpc.state_bounds"),
        STATE_DIM,
        "root.mpc.state_bounds.upper",
        allow_infinity=True,
    )
    soft_row = _boolean_vector(
        _required(bounds_values, "soft_mask", "root.mpc.state_bounds"),
        STATE_DIM,
        "root.mpc.state_bounds.soft_mask",
    )
    mpc_state_bounds = _build(
        "root.mpc.state_bounds",
        lambda: StateBounds(
            lower=np.tile(lower_row, (horizon_steps + 1, 1)),
            upper=np.tile(upper_row, (horizon_steps + 1, 1)),
            soft_mask=np.tile(soft_row, (horizon_steps, 1)),
        ),
    )

    admittance_values = _section(
        document,
        "admittance",
        "root",
        frozenset(
            {
                "transition_duration_s",
                "touchdown_inertia",
                "stance_inertia",
                "touchdown_damping",
                "stance_damping",
                "restoring_stiffness",
                "stance_stiffness",
                "force_error_deadband_n",
                "correction_position_limit_m",
                "correction_velocity_limit_m_per_s",
                "contact_release_policy",
                "anti_windup_enabled",
                "leg_order",
                "legs",
            }
        ),
    )
    leg_order = _name_tuple4(admittance_values, "leg_order", "root.admittance")
    if (
        go2_foot_force_calibration is not None
        and leg_order != go2_foot_force_calibration.algorithm_leg_order
    ):
        raise ImpactAwareConfigError(
            "root.admittance.leg_order must exactly match "
            "root.contact.go2_lowstate_force_adapter.algorithm_leg_order"
        )
    legs_value = _required(admittance_values, "legs", "root.admittance")
    if not isinstance(legs_value, list) or len(legs_value) != FOOT_COUNT:
        raise ImpactAwareConfigError("root.admittance.legs must be a four-item list")
    anti_windup = _bool(
        admittance_values,
        "anti_windup_enabled",
        "root.admittance",
    )
    common_admittance = {
        key: _required(admittance_values, key, "root.admittance")
        for key in (
            "transition_duration_s",
            "touchdown_inertia",
            "stance_inertia",
            "touchdown_damping",
            "stance_damping",
            "restoring_stiffness",
            "stance_stiffness",
            "force_error_deadband_n",
            "correction_position_limit_m",
            "correction_velocity_limit_m_per_s",
            "contact_release_policy",
        )
    }
    leg_configs: List[AdmittanceConfig] = []
    workspaces: List[AxisAlignedWorkspace] = []
    leg_required = frozenset(
        {
            "workspace_lower_m",
            "workspace_upper_m",
            "joint_lower_rad",
            "joint_upper_rad",
            "joint_rate_limit_rad_per_s",
        }
    )
    for index, raw_leg in enumerate(legs_value):
        leg_path = f"root.admittance.legs[{index}]"
        leg = _mapping(raw_leg, leg_path)
        _check_keys(leg, leg_path, leg_required)
        try:
            leg_config = AdmittanceConfig(
                transition_duration_s=common_admittance["transition_duration_s"],
                touchdown_inertia=common_admittance["touchdown_inertia"],
                stance_inertia=common_admittance["stance_inertia"],
                touchdown_damping=common_admittance["touchdown_damping"],
                stance_damping=common_admittance["stance_damping"],
                restoring_stiffness=common_admittance["restoring_stiffness"],
                stance_stiffness=common_admittance["stance_stiffness"],
                force_error_deadband_n=common_admittance["force_error_deadband_n"],
                correction_position_limit_m=common_admittance[
                    "correction_position_limit_m"
                ],
                correction_velocity_limit_m_per_s=common_admittance[
                    "correction_velocity_limit_m_per_s"
                ],
                contact_release_policy=common_admittance["contact_release_policy"],
                joint_lower=_required(leg, "joint_lower_rad", leg_path),
                joint_upper=_required(leg, "joint_upper_rad", leg_path),
                joint_rate_limit=_required(
                    leg,
                    "joint_rate_limit_rad_per_s",
                    leg_path,
                ),
                anti_windup_enabled=anti_windup,
            )
            workspace = AxisAlignedWorkspace(
                _required(leg, "workspace_lower_m", leg_path),
                _required(leg, "workspace_upper_m", leg_path),
            )
        except (TypeError, ValueError) as exc:
            raise ImpactAwareConfigError(f"{leg_path}: {exc}") from exc
        leg_configs.append(leg_config)
        workspaces.append(workspace)

    bundle = ImpactAwareConfigBundle(
        source_path=source_path,
        profile=profile,
        parameters_identified=parameters_identified,
        allow_hardware_output=hardware_allowed,
        physical_use_prohibited=physical_prohibited,
        world_frame=world_frame,
        body_frame=body_frame,
        rotor_order=rotor_order,
        rotor_configuration=rotor_configuration,
        rotor_reference_origin=rotor_reference_origin,
        dynamics_reference_point=dynamics_reference_point,
        rotor_geometry=rotor_geometry,
        rotor_aerodynamics=rotor_aerodynamics,
        rotor_actuator=rotor_actuator,
        dynamics=dynamics,
        impact_limits=impact_limits,
        rotor_correction_safety=rotor_correction_safety,
        contact_detector=contact_detector,
        go2_foot_force_calibration=go2_foot_force_calibration,
        contact_force_limits=contact_force_limits,
        mpc_horizon_steps=horizon_steps,
        mpc_dt_s=dt_value,
        mpc_weights=mpc_weights,
        mpc_state_bounds=mpc_state_bounds,
        solver_settings=solver_settings,
        admittance_leg_order=leg_order,
        admittance_configs=cast(
            Tuple[AdmittanceConfig, AdmittanceConfig, AdmittanceConfig, AdmittanceConfig],
            tuple(leg_configs),
        ),
        admittance_workspaces=cast(
            Tuple[
                AxisAlignedWorkspace,
                AxisAlignedWorkspace,
                AxisAlignedWorkspace,
                AxisAlignedWorkspace,
            ],
            tuple(workspaces),
        ),
    )
    return bundle


__all__ = [
    "ImpactAwareConfigBundle",
    "ImpactAwareConfigError",
    "load_impact_aware_config",
]
