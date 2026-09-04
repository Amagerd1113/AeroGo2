"""Fail-closed preliminary vertical-only landing model.

This module is deliberately separate from the paper-faithful six-degree-of-
freedom model.  It allows offline work with the few parameters currently
known for AeroGo2 without inventing a centre of mass, inertia tensor, force
calibration, or hardware authority.  A configuration loaded here can never be
used as a hardware-release attestation.

The simplified contact semantics are:

* only the world-vertical normal force/impulse exists;
* tangential force, friction cones, tangential impulse, and impact-induced
  angular-velocity reset are outside the model;
* uncalibrated Unitree ``foot_force``/``foot_force_est`` integers may only
  produce contact events.  They are never interpreted as newtons;
* calibrated per-foot normal-force scalars may be returned only from an
  explicitly calibrated
  :class:`~aerogo2.landing.impact_aware.go2_foot_force.CalibratedGo2NormalForceSample`;
  this API never promotes those scalars to three-dimensional vectors.

There is no transport or actuator output in this module.

中文说明：该文件保存“当前数据不足时仍可运行”的简化模型及其严格加载器。
它把官方 URDF 可得量、用户给定几何和暂定质量，与尚未辨识的真机参数明确分开；
所有推导字段都可由重算工具从少量可编辑源参数重新生成。无论配置填写得多完整，
本模块的 ``hardware_output_permitted`` 恒为假，避免估计值被误当成上机标定值。
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import yaml
from numpy.typing import NDArray
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from aerogo2.common.models import Go2FootForceFeedback
from aerogo2.landing.impact_aware.go2_foot_force import (
    CalibratedGo2NormalForceSample,
    compute_go2_foot_force_mapping_hash,
)
from aerogo2.landing.impact_aware.go2_urdf import (
    Go2UrdfError,
    Go2UrdfMassProperties,
    load_go2_urdf_mass_properties,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = NDArray[np.float64]
PathLike = Union[str, Path]
ROTOR_ORDER = ("RR", "LF", "LR", "RF")
GEOMETRIC_ROTOR_SEQUENCE = ("LF", "LR", "RR", "RF")
# Schema 3 carried an inertia interval that did not close the configured
# first mass moment.  It is intentionally retired instead of being silently
# accepted with different physics; schemas 1/2 remain readable for their
# limited legacy offline uses, while all current physical priors use schema 4.
PRELIMINARY_SCHEMA_VERSION = 4
PREVIOUS_PRELIMINARY_SCHEMA_VERSION = 2
LEGACY_PRELIMINARY_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 256_000
GRAM_FORCE_TO_NEWTON = 9.80665e-3

_X8_12S_THROTTLE_PERCENT = (
    35.0,
    37.0,
    39.0,
    42.0,
    45.0,
    48.0,
    51.0,
    54.0,
    57.0,
    60.0,
    63.0,
    66.0,
    69.0,
    72.0,
    75.0,
    78.0,
    81.0,
    84.0,
    87.0,
    90.0,
    100.0,
)
_X8_12S_THRUST_GRAM_FORCE = (
    2021.0,
    2472.0,
    2472.0,
    2968.0,
    3493.0,
    3999.0,
    4476.0,
    4994.0,
    5456.0,
    6043.0,
    6452.0,
    6977.0,
    7951.0,
    8500.0,
    9050.0,
    10034.0,
    10467.0,
    11007.0,
    11548.0,
    12542.0,
    15006.0,
)
_X8_14S_THROTTLE_PERCENT = (
    33.0,
    35.0,
    37.0,
    39.0,
    42.0,
    45.0,
    48.0,
    51.0,
    54.0,
    57.0,
    60.0,
    63.0,
    66.0,
    69.0,
    72.0,
    75.0,
    78.0,
    81.0,
    84.0,
    87.0,
    90.0,
    100.0,
)
_X8_14S_THRUST_GRAM_FORCE = (
    3836.0,
    4459.0,
    4965.0,
    4965.0,
    6007.0,
    6450.0,
    7031.0,
    8026.0,
    8552.0,
    9509.0,
    10495.0,
    11062.0,
    12045.0,
    12992.0,
    13506.0,
    14456.0,
    14984.0,
    15544.0,
    16017.0,
    16518.0,
    17005.0,
    17464.0,
)

_X8_CURVE_IDS = (
    "X8_G2_12S_46V_MFP_30X11S_25C_SEA_LEVEL",
    "X8_G2_14S_54V_MFP_30X11S_25C_SEA_LEVEL",
)


class PreliminaryModelError(ValueError):
    """Raised when provisional model data violate the offline-only contract."""


class ContactModel(str, Enum):
    """Contact fidelity supported by the preliminary model."""

    NORMAL_ONLY_VERTICAL = "NORMAL_ONLY_VERTICAL"


class FootForceSemantics(str, Enum):
    """Permitted meaning of the public Unitree scalar foot-force fields."""

    UNCALIBRATED_CONTACT_EVENT_ONLY = "UNCALIBRATED_CONTACT_EVENT_ONLY"
    CALIBRATED_NORMAL_FORCE_N = "CALIBRATED_NORMAL_FORCE_N"


def _finite_real(
    value: object,
    name: str,
    *,
    minimum: Optional[float] = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PreliminaryModelError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PreliminaryModelError(f"{name} must be finite")
    if positive and parsed <= 0.0:
        raise PreliminaryModelError(f"{name} must be strictly positive")
    if minimum is not None and parsed < minimum:
        raise PreliminaryModelError(f"{name} must be at least {minimum}")
    return parsed


def _optional_finite_real(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
) -> Optional[float]:
    if value is None:
        return None
    return _finite_real(value, name, minimum=minimum)


def _readonly_array(value: object, shape: Tuple[int, ...], name: str) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PreliminaryModelError(f"{name} must be a numeric array") from exc
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise PreliminaryModelError(f"{name} must have shape {shape} and numeric entries")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise PreliminaryModelError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return cast(FloatArray, result)


def _optional_matrix3(value: object, name: str) -> Optional[FloatArray]:
    if value is None:
        return None
    return _readonly_array(value, (3, 3), name)


def _optional_vector3(value: object, name: str) -> Optional[FloatArray]:
    if value is None:
        return None
    return _readonly_array(value, (3,), name)


def _optional_nonempty(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PreliminaryModelError(f"{name} must be null or a nonempty string")
    return value


def _nonempty(value: object, name: str) -> str:
    result = _optional_nonempty(value, name)
    if result is None:
        raise PreliminaryModelError(f"{name} must be a nonempty string")
    return result


def _sha256_identity(value: object, name: str) -> str:
    """Return one normalized ``sha256:<hex>`` identity."""

    digest = _nonempty(value, name).lower()
    digest_hex = digest[7:] if digest.startswith("sha256:") else digest
    if len(digest_hex) != 64 or any(
        character not in "0123456789abcdef" for character in digest_hex
    ):
        raise PreliminaryModelError(f"{name} must contain exactly 64 hexadecimal digits")
    return "sha256:" + digest_hex


def _tuple4_names(value: object, name: str) -> Tuple[str, str, str, str]:
    try:
        values = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise PreliminaryModelError(f"{name} must be a four-item list") from exc
    if len(values) != 4 or any(not isinstance(item, str) or not item.strip() for item in values):
        raise PreliminaryModelError(f"{name} must contain four nonempty strings")
    if len(set(values)) != 4:
        raise PreliminaryModelError(f"{name} entries must be unique")
    return cast(Tuple[str, str, str, str], values)


@dataclass(frozen=True)
class ProvisionalMassProperties:
    """Component-sum mass prior plus optional measured uncertainty."""

    go2_nominal_kg: float
    added_system_nominal_kg: float
    total_nominal_kg: float
    total_uncertainty_kg: Optional[float]

    def __post_init__(self) -> None:
        go2 = _finite_real(self.go2_nominal_kg, "go2_nominal_kg", positive=True)
        added = _finite_real(
            self.added_system_nominal_kg,
            "added_system_nominal_kg",
            minimum=0.0,
        )
        total = _finite_real(self.total_nominal_kg, "total_nominal_kg", positive=True)
        uncertainty = _optional_finite_real(
            self.total_uncertainty_kg,
            "total_uncertainty_kg",
        )
        if uncertainty is not None and uncertainty <= 0.0:
            raise PreliminaryModelError("total_uncertainty_kg must be strictly positive when set")
        expected = go2 + added
        if not math.isclose(total, expected, rel_tol=0.0, abs_tol=1.0e-9):
            raise PreliminaryModelError(
                "total_nominal_kg must equal go2_nominal_kg + added_system_nominal_kg"
            )
        object.__setattr__(self, "go2_nominal_kg", go2)
        object.__setattr__(self, "added_system_nominal_kg", added)
        object.__setattr__(self, "total_nominal_kg", total)
        object.__setattr__(self, "total_uncertainty_kg", uncertainty)

    @property
    def measured_with_uncertainty(self) -> bool:
        """Whether a finite uncertainty has replaced the component-sum prior."""

        return self.total_uncertainty_kg is not None


@dataclass(frozen=True)
class PinnedGo2UrdfPrior:
    """Pinned official Go2 model and its evaluated reference-pose properties.

    The bundled URDF is an offline engineering prior.  Its limits are not
    automatically promoted to LowCmd safety limits, and its 16.087 kg mass is
    not represented as a physical weighing result.
    """

    source_repository: str
    source_file_commit: str
    model_quality: str
    upstream_raw_sha256: str
    bundled_path: Path
    bundled_sha256: str
    license_path: Path
    root_link: str
    urdf_root_from_body_origin_B_m: FloatArray
    body_origin_B_alignment_identified: bool
    reference_pose: str
    sdk_joint_order: Tuple[str, ...]
    sdk_joint_positions_rad: Tuple[float, ...]
    mass_properties: Go2UrdfMassProperties

    def __post_init__(self) -> None:
        source_repository = _nonempty(self.source_repository, "go2_urdf.source_repository")
        source_commit = _nonempty(self.source_file_commit, "go2_urdf.source_file_commit")
        model_quality = _nonempty(self.model_quality, "go2_urdf.model_quality")
        if model_quality != "URDF_MODEL_ESTIMATE":
            raise PreliminaryModelError("go2_urdf.model_quality must remain URDF_MODEL_ESTIMATE")
        upstream_hash = _sha256_identity(
            self.upstream_raw_sha256,
            "go2_urdf.upstream_raw_sha256",
        )
        bundled_hash = _sha256_identity(
            self.bundled_sha256,
            "go2_urdf.bundled_sha256",
        )
        root_from_body = _readonly_array(
            self.urdf_root_from_body_origin_B_m,
            (3,),
            "go2_urdf.urdf_root_from_body_origin_B_m",
        )
        if type(self.body_origin_B_alignment_identified) is not bool:
            raise PreliminaryModelError(
                "go2_urdf.body_origin_B_alignment_identified must be a boolean"
            )
        joint_order = tuple(self.sdk_joint_order)
        joint_positions = tuple(
            _finite_real(value, "go2_urdf.sdk_joint_positions_rad")
            for value in self.sdk_joint_positions_rad
        )
        if len(joint_order) != 12 or len(set(joint_order)) != 12:
            raise PreliminaryModelError(
                "go2_urdf.sdk_joint_order must contain twelve unique joint names"
            )
        if len(joint_positions) != 12:
            raise PreliminaryModelError(
                "go2_urdf.sdk_joint_positions_rad must contain twelve values"
            )
        evaluated_order = tuple(name for name, _ in self.mass_properties.joint_positions_rad)
        if set(evaluated_order) != set(joint_order):
            raise PreliminaryModelError(
                "go2_urdf reference pose does not cover the pinned URDF movable joints"
            )
        if self.mass_properties.source_path != self.bundled_path:
            raise PreliminaryModelError("go2_urdf evaluated a different bundled path")
        if self.mass_properties.source_sha256 != bundled_hash:
            raise PreliminaryModelError("go2_urdf evaluated hash does not match bundled_sha256")
        if self.mass_properties.root_link != self.root_link:
            raise PreliminaryModelError("go2_urdf evaluated root link does not match configuration")
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(self, "source_file_commit", source_commit)
        object.__setattr__(self, "model_quality", model_quality)
        object.__setattr__(self, "upstream_raw_sha256", upstream_hash)
        object.__setattr__(self, "bundled_sha256", bundled_hash)
        object.__setattr__(self, "urdf_root_from_body_origin_B_m", root_from_body)
        object.__setattr__(self, "reference_pose", _nonempty(self.reference_pose, "reference_pose"))
        object.__setattr__(self, "sdk_joint_order", joint_order)
        object.__setattr__(self, "sdk_joint_positions_rad", joint_positions)

    @property
    def hardware_identified(self) -> bool:
        """The official simulation model never proves the assembled hardware."""

        return False


@dataclass(frozen=True)
class ProvisionalOfflineInertiaEstimate:
    """Broad, deterministic inertia prior used only for offline algorithms.

    The interval bounds cover only the stated simplified radial mass model;
    they are not confidence intervals.  Cross terms remain unidentified.
    """

    method: str
    quality: str
    reference_point: str
    x8_system_mass_each_kg: float
    x8_count: int
    remaining_added_mass_kg: float
    remaining_added_mass_effective_com_from_body_origin_B_m: FloatArray
    remaining_mass_distribution_interval: str
    nominal_body_kg_m2: FloatArray
    diagonal_lower_body_kg_m2: FloatArray
    diagonal_upper_body_kg_m2: FloatArray
    cross_terms_unbounded: bool
    hardware_use_prohibited: bool

    def __post_init__(self) -> None:
        method = _nonempty(self.method, "offline_inertia_estimate.method")
        quality = _nonempty(self.quality, "offline_inertia_estimate.quality")
        reference = _nonempty(
            self.reference_point,
            "offline_inertia_estimate.reference_point",
        )
        if method != "GO2_URDF_HOME_PLUS_X8_AND_BALANCED_REMAINDER_INTERVAL_V2":
            raise PreliminaryModelError("offline inertia estimate method is not reviewed")
        if quality != "PROVISIONAL_OFFLINE_ONLY":
            raise PreliminaryModelError("offline inertia estimate quality must remain provisional")
        if reference != "TOTAL_SYSTEM_COM_C":
            raise PreliminaryModelError("offline inertia estimate must be referenced to COM C")
        x8_mass = _finite_real(
            self.x8_system_mass_each_kg,
            "offline_inertia_estimate.x8_system_mass_each_kg",
            positive=True,
        )
        if isinstance(self.x8_count, bool) or not isinstance(self.x8_count, int):
            raise PreliminaryModelError("offline_inertia_estimate.x8_count must be an integer")
        if self.x8_count != 4:
            raise PreliminaryModelError("offline inertia estimate requires exactly four X8 systems")
        remaining_mass = _finite_real(
            self.remaining_added_mass_kg,
            "offline_inertia_estimate.remaining_added_mass_kg",
            minimum=0.0,
        )
        remaining_com = _readonly_array(
            self.remaining_added_mass_effective_com_from_body_origin_B_m,
            (3,),
            ("offline_inertia_estimate.remaining_added_mass_effective_com_from_body_origin_B_m"),
        )
        distribution = _nonempty(
            self.remaining_mass_distribution_interval,
            "offline_inertia_estimate.remaining_mass_distribution_interval",
        )
        if distribution != ("LOWER_AT_FIRST_MOMENT_BALANCE_COM_UPPER_PLANAR_RADIUS_ABOUT_SAME_COM"):
            raise PreliminaryModelError("offline remaining-mass interval is not reviewed")
        nominal = _readonly_array(
            self.nominal_body_kg_m2,
            (3, 3),
            "offline_inertia_estimate.nominal_body_kg_m2",
        )
        lower = _readonly_array(
            self.diagonal_lower_body_kg_m2,
            (3,),
            "offline_inertia_estimate.diagonal_lower_body_kg_m2",
        )
        upper = _readonly_array(
            self.diagonal_upper_body_kg_m2,
            (3,),
            "offline_inertia_estimate.diagonal_upper_body_kg_m2",
        )
        if not np.allclose(nominal, nominal.T, rtol=0.0, atol=1.0e-10):
            raise PreliminaryModelError("offline nominal inertia must be symmetric")
        if float(np.min(np.linalg.eigvalsh(nominal))) <= 0.0:
            raise PreliminaryModelError("offline nominal inertia must be positive definite")
        if np.any(lower <= 0.0) or np.any(upper < lower):
            raise PreliminaryModelError("offline inertia diagonal interval is invalid")
        if not np.all((np.diag(nominal) >= lower) & (np.diag(nominal) <= upper)):
            raise PreliminaryModelError("offline nominal inertia must lie inside diagonal bounds")
        if self.cross_terms_unbounded is not True or self.hardware_use_prohibited is not True:
            raise PreliminaryModelError(
                "offline inertia cross terms must remain unbounded and hardware use prohibited"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "reference_point", reference)
        object.__setattr__(self, "x8_system_mass_each_kg", x8_mass)
        object.__setattr__(self, "remaining_added_mass_kg", remaining_mass)
        object.__setattr__(
            self,
            "remaining_added_mass_effective_com_from_body_origin_B_m",
            remaining_com,
        )
        object.__setattr__(self, "remaining_mass_distribution_interval", distribution)
        object.__setattr__(self, "nominal_body_kg_m2", nominal)
        object.__setattr__(self, "diagonal_lower_body_kg_m2", lower)
        object.__setattr__(self, "diagonal_upper_body_kg_m2", upper)

    @property
    def usable_for_hardware(self) -> bool:
        return False


@dataclass(frozen=True)
class StepCadSourceProvenance:
    """Identity and capability record for a STEP file used as CAD evidence.

    STEP geometry is not automatically a mass model.  The four boolean fields
    deliberately separate file identity, topological validity, material/mass
    metadata, and verified assembly scope so a geometric import cannot silently
    become an identified inertia tensor.
    """

    file_name: str
    sha256: str
    file_size_bytes: int
    step_schema: str
    length_unit: str
    brep_validation_passed: bool
    material_density_properties_present: bool
    mass_inertia_properties_present: bool
    complete_system_scope_verified: bool

    def __post_init__(self) -> None:
        file_name = _nonempty(self.file_name, "cad_source.file_name")
        digest = _nonempty(self.sha256, "cad_source.sha256").lower()
        if digest.startswith("sha256:"):
            digest_hex = digest[7:]
        else:
            digest_hex = digest
        if len(digest_hex) != 64 or any(
            character not in "0123456789abcdef" for character in digest_hex
        ):
            raise PreliminaryModelError(
                "cad_source.sha256 must contain exactly 64 hexadecimal digits"
            )
        size = self.file_size_bytes
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PreliminaryModelError("cad_source.file_size_bytes must be a positive integer")
        schema = _nonempty(self.step_schema, "cad_source.step_schema")
        unit = _nonempty(self.length_unit, "cad_source.length_unit")
        for name in (
            "brep_validation_passed",
            "material_density_properties_present",
            "mass_inertia_properties_present",
            "complete_system_scope_verified",
        ):
            if type(getattr(self, name)) is not bool:
                raise PreliminaryModelError(f"cad_source.{name} must be a boolean")
        object.__setattr__(self, "file_name", file_name)
        object.__setattr__(self, "sha256", "sha256:" + digest_hex)
        object.__setattr__(self, "step_schema", schema)
        object.__setattr__(self, "length_unit", unit)

    @property
    def directly_supports_identified_inertia(self) -> bool:
        """Whether this file alone is credible identified mass-property evidence."""

        return bool(
            self.brep_validation_passed
            and self.complete_system_scope_verified
            and self.mass_inertia_properties_present
        )


@dataclass(frozen=True)
class CadBomInertiaProperties:
    """Interface for a future complete CAD/BOM composite inertia result.

    Nominal inertia and its elementwise absolute uncertainty must be supplied
    together.  Both may remain ``None`` while the vertical-only model is used.
    The full six-DoF model must call :meth:`require_identified`.
    """

    method: str
    nominal_body_kg_m2: Optional[FloatArray]
    uncertainty_body_kg_m2: Optional[FloatArray]
    cad_bom_revision: Optional[str]
    reference_pose: Optional[str]

    def __post_init__(self) -> None:
        method = _nonempty(self.method, "inertia.method")
        if method != "COMPLETE_CAD_BOM_PARALLEL_AXIS":
            raise PreliminaryModelError("inertia.method must be COMPLETE_CAD_BOM_PARALLEL_AXIS")
        nominal = _optional_matrix3(self.nominal_body_kg_m2, "nominal_body_kg_m2")
        uncertainty = _optional_matrix3(
            self.uncertainty_body_kg_m2,
            "uncertainty_body_kg_m2",
        )
        if (nominal is None) != (uncertainty is None):
            raise PreliminaryModelError(
                "nominal_body_kg_m2 and uncertainty_body_kg_m2 must be supplied together"
            )
        revision = _optional_nonempty(self.cad_bom_revision, "cad_bom_revision")
        pose = _optional_nonempty(self.reference_pose, "reference_pose")
        if nominal is not None and uncertainty is not None:
            if not np.allclose(nominal, nominal.T, rtol=0.0, atol=1.0e-10):
                raise PreliminaryModelError("nominal_body_kg_m2 must be symmetric")
            principal_moments = np.linalg.eigvalsh(nominal)
            if float(np.min(principal_moments)) <= 0.0:
                raise PreliminaryModelError("nominal_body_kg_m2 must be positive definite")
            triangle_tolerance = 1.0e-10 * max(1.0, float(principal_moments[-1]))
            if (
                float(principal_moments[-1])
                > float(principal_moments[0] + principal_moments[1]) + triangle_tolerance
            ):
                raise PreliminaryModelError(
                    "nominal principal moments must satisfy the physical triangle inequality"
                )
            if not np.allclose(uncertainty, uncertainty.T, rtol=0.0, atol=1.0e-10):
                raise PreliminaryModelError("uncertainty_body_kg_m2 must be symmetric")
            if np.any(uncertainty < 0.0):
                raise PreliminaryModelError(
                    "uncertainty_body_kg_m2 must contain absolute nonnegative bounds"
                )
            if not np.any(uncertainty > 0.0):
                raise PreliminaryModelError(
                    "uncertainty_body_kg_m2 cannot be all zero; use null until a "
                    "nonzero physical uncertainty bound has been established"
                )
            if revision is None or pose is None:
                raise PreliminaryModelError(
                    "identified inertia requires cad_bom_revision and reference_pose"
                )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "nominal_body_kg_m2", nominal)
        object.__setattr__(self, "uncertainty_body_kg_m2", uncertainty)
        object.__setattr__(self, "cad_bom_revision", revision)
        object.__setattr__(self, "reference_pose", pose)

    @property
    def identified(self) -> bool:
        return self.nominal_body_kg_m2 is not None

    def require_identified(self) -> Tuple[FloatArray, FloatArray]:
        """Return the pair needed by six-DoF code or fail explicitly."""

        if self.nominal_body_kg_m2 is None or self.uncertainty_body_kg_m2 is None:
            raise PreliminaryModelError(
                "six-DoF dynamics remain blocked until complete CAD/BOM inertia and "
                "uncertainty are supplied"
            )
        return self.nominal_body_kg_m2, self.uncertainty_body_kg_m2


@dataclass(frozen=True)
class FixedXGeometryPrior:
    """Fixed deployed X-frame vectors, never recomputed by the control loop."""

    total_com_from_frame_center_body_m: Optional[FloatArray]
    body_origin_B_definition: Optional[str]
    frame_center_O_from_body_origin_B_m: Optional[FloatArray]
    total_com_C_from_body_origin_B_m: Optional[FloatArray]
    rotor_frame_com_from_body_origin_B_m: Optional[FloatArray]
    rotor_plane_from_frame_center_O_m: Optional[FloatArray]
    frame_offsets_identified: bool
    horizontal_radius_from_frame_center_m: float
    rotor_plane_z_from_com_m: float
    horizontal_origin_assumption: str
    azimuth_assumption: str
    azimuth_reference: str
    geometric_sequence_start_azimuth_deg: float
    adjacent_arm_spacing_deg: float
    geometric_sequence_rotor_order: Tuple[str, str, str, str]
    first_rotor_positive_xy_label: Optional[str]
    lever_arms_from_com_body_m: FloatArray
    thrust_directions_body: FloatArray
    lever_arm_uncertainty_m: Optional[FloatArray]

    def __post_init__(self) -> None:
        com_offset = _optional_vector3(
            self.total_com_from_frame_center_body_m,
            "total_com_from_frame_center_body_m",
        )
        body_definition = _optional_nonempty(
            self.body_origin_B_definition,
            "body_origin_B_definition",
        )
        frame_from_body = _optional_vector3(
            self.frame_center_O_from_body_origin_B_m,
            "frame_center_O_from_body_origin_B_m",
        )
        total_com_from_body = _optional_vector3(
            self.total_com_C_from_body_origin_B_m,
            "total_com_C_from_body_origin_B_m",
        )
        frame_com_from_body = _optional_vector3(
            self.rotor_frame_com_from_body_origin_B_m,
            "rotor_frame_com_from_body_origin_B_m",
        )
        rotor_plane_from_frame = _optional_vector3(
            self.rotor_plane_from_frame_center_O_m,
            "rotor_plane_from_frame_center_O_m",
        )
        if type(self.frame_offsets_identified) is not bool:
            raise PreliminaryModelError("frame_offsets_identified must be a boolean")
        explicit_frames = (
            body_definition,
            frame_from_body,
            total_com_from_body,
            frame_com_from_body,
            rotor_plane_from_frame,
            self.first_rotor_positive_xy_label,
        )
        if any(value is None for value in explicit_frames) and not all(
            value is None for value in explicit_frames
        ):
            raise PreliminaryModelError("explicit B/O/C geometry fields must be supplied together")
        radius = _finite_real(
            self.horizontal_radius_from_frame_center_m,
            "horizontal_radius_from_frame_center_m",
            positive=True,
        )
        height = _finite_real(
            self.rotor_plane_z_from_com_m,
            "rotor_plane_z_from_com_m",
        )
        assumption = _nonempty(
            self.horizontal_origin_assumption,
            "horizontal_origin_assumption",
        )
        if assumption != "FRAME_CENTER_XY_EQUALS_TOTAL_COM_XY_PROVISIONAL":
            raise PreliminaryModelError(
                "horizontal_origin_assumption must explicitly remain provisional"
            )
        if com_offset is not None and not np.allclose(
            com_offset[:2],
            np.zeros(2),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise PreliminaryModelError(
                "total_com_from_frame_center_body_m x/y must remain zero while the "
                "horizontal-origin assumption is active"
            )
        azimuth_assumption = _nonempty(self.azimuth_assumption, "azimuth_assumption")
        if azimuth_assumption not in {
            "SYMMETRIC_X_45_DEG_PROVISIONAL",
            "SYMMETRIC_EQUAL_SPACING_PROVISIONAL",
        }:
            raise PreliminaryModelError(
                "azimuth_assumption must explicitly describe the provisional symmetric X frame"
            )
        azimuth_reference = _nonempty(self.azimuth_reference, "azimuth_reference")
        if azimuth_reference != "BODY_POSITIVE_X_CCW_ABOUT_BODY_POSITIVE_Z":
            raise PreliminaryModelError(
                "azimuth_reference must be BODY_POSITIVE_X_CCW_ABOUT_BODY_POSITIVE_Z"
            )
        start_azimuth = _finite_real(
            self.geometric_sequence_start_azimuth_deg,
            "geometric_sequence_start_azimuth_deg",
        )
        if not 0.0 <= start_azimuth < 360.0:
            raise PreliminaryModelError("geometric_sequence_start_azimuth_deg must be in [0, 360)")
        spacing = _finite_real(
            self.adjacent_arm_spacing_deg,
            "adjacent_arm_spacing_deg",
            positive=True,
        )
        if not math.isclose(4.0 * spacing, 360.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise PreliminaryModelError(
                "four-rotor adjacent_arm_spacing_deg must close exactly one 360 degree turn"
            )
        if azimuth_assumption == "SYMMETRIC_X_45_DEG_PROVISIONAL" and not math.isclose(
            start_azimuth, 45.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise PreliminaryModelError(
                "legacy SYMMETRIC_X_45_DEG_PROVISIONAL requires a 45 degree start"
            )
        geometric_sequence = tuple(self.geometric_sequence_rotor_order)
        if len(geometric_sequence) != 4 or set(geometric_sequence) != set(ROTOR_ORDER):
            raise PreliminaryModelError(
                "geometric_sequence_rotor_order must contain RR, LF, LR, RF exactly once"
            )
        arms = _readonly_array(
            self.lever_arms_from_com_body_m,
            (4, 3),
            "lever_arms_from_com_body_m",
        )
        directions = _readonly_array(
            self.thrust_directions_body,
            (4, 3),
            "thrust_directions_body",
        )
        uncertainty = (
            None
            if self.lever_arm_uncertainty_m is None
            else _readonly_array(
                self.lever_arm_uncertainty_m,
                (4, 3),
                "lever_arm_uncertainty_m",
            )
        )
        if uncertainty is not None and np.any(uncertainty < 0.0):
            raise PreliminaryModelError("lever_arm_uncertainty_m cannot be negative")
        if uncertainty is not None and not np.any(uncertainty > 0.0):
            raise PreliminaryModelError(
                "lever_arm_uncertainty_m cannot be all zero; use null while position "
                "uncertainty is intentionally omitted"
            )
        azimuth_by_label = {
            label: (start_azimuth + index * spacing) % 360.0
            for index, label in enumerate(geometric_sequence)
        }
        expected_arms = np.array(
            [
                [
                    radius * math.cos(math.radians(azimuth_by_label[label])),
                    radius * math.sin(math.radians(azimuth_by_label[label])),
                    height,
                ]
                for label in ROTOR_ORDER
            ],
            dtype=float,
        )
        if not np.allclose(arms, expected_arms, rtol=0.0, atol=1.0e-6):
            raise PreliminaryModelError(
                "lever_arms_from_com_body_m do not match the declared fixed X-frame "
                "radius/height and [RR, LF, LR, RF] order"
            )
        if body_definition is not None:
            if body_definition != "GO2_DORSAL_BODY_REFERENCE_B_PROVISIONAL":
                raise PreliminaryModelError(
                    "body_origin_B_definition must explicitly remain provisional"
                )
            assert frame_from_body is not None
            assert total_com_from_body is not None
            assert frame_com_from_body is not None
            assert rotor_plane_from_frame is not None
            if not np.allclose(
                frame_from_body[:2], np.zeros(2), rtol=0.0, atol=1.0e-12
            ) or not np.allclose(total_com_from_body[:2], np.zeros(2), rtol=0.0, atol=1.0e-12):
                raise PreliminaryModelError("B, O and C must share the declared z axis")
            if not np.allclose(frame_com_from_body, frame_from_body, rtol=0.0, atol=1.0e-12):
                raise PreliminaryModelError("the simplified rotor-frame CoM must equal O")
            if not np.allclose(rotor_plane_from_frame[:2], np.zeros(2), rtol=0.0, atol=1.0e-12):
                raise PreliminaryModelError(
                    "the rotor-plane reference and O must share the declared z axis"
                )
            expected_com_from_frame = total_com_from_body - frame_from_body
            if com_offset is None or not np.allclose(
                com_offset,
                expected_com_from_frame,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise PreliminaryModelError(
                    "total_com_from_frame_center_body_m must equal p_C^B - p_O^B"
                )
            expected_rotor_height = float(
                frame_from_body[2] + rotor_plane_from_frame[2] - total_com_from_body[2]
            )
            if not math.isclose(height, expected_rotor_height, rel_tol=0.0, abs_tol=1.0e-12):
                raise PreliminaryModelError(
                    "rotor_plane_z_from_com_m is inconsistent with B/O/C geometry"
                )
            first_label = _nonempty(
                self.first_rotor_positive_xy_label,
                "first_rotor_positive_xy_label",
            )
            if first_label not in ROTOR_ORDER:
                raise PreliminaryModelError("first_rotor_positive_xy_label is not a rotor label")
            first_index = ROTOR_ORDER.index(first_label)
            if not (arms[first_index, 0] > 0.0 and arms[first_index, 1] > 0.0):
                raise PreliminaryModelError(
                    "first_rotor_positive_xy_label must identify the x>0,y>0 rotor"
                )
        expected_directions = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
        if not np.allclose(directions, expected_directions, rtol=0.0, atol=1.0e-12):
            raise PreliminaryModelError(
                "preliminary thrust_directions_body must be four +Z unit vectors"
            )
        object.__setattr__(self, "total_com_from_frame_center_body_m", com_offset)
        object.__setattr__(self, "body_origin_B_definition", body_definition)
        object.__setattr__(self, "frame_center_O_from_body_origin_B_m", frame_from_body)
        object.__setattr__(self, "total_com_C_from_body_origin_B_m", total_com_from_body)
        object.__setattr__(self, "rotor_frame_com_from_body_origin_B_m", frame_com_from_body)
        object.__setattr__(self, "rotor_plane_from_frame_center_O_m", rotor_plane_from_frame)
        object.__setattr__(self, "horizontal_radius_from_frame_center_m", radius)
        object.__setattr__(self, "rotor_plane_z_from_com_m", height)
        object.__setattr__(self, "horizontal_origin_assumption", assumption)
        object.__setattr__(self, "azimuth_assumption", azimuth_assumption)
        object.__setattr__(self, "azimuth_reference", azimuth_reference)
        object.__setattr__(self, "geometric_sequence_start_azimuth_deg", start_azimuth)
        object.__setattr__(self, "adjacent_arm_spacing_deg", spacing)
        object.__setattr__(
            self,
            "geometric_sequence_rotor_order",
            cast(Tuple[str, str, str, str], geometric_sequence),
        )
        object.__setattr__(
            self,
            "first_rotor_positive_xy_label",
            (
                None
                if self.first_rotor_positive_xy_label is None
                else _nonempty(
                    self.first_rotor_positive_xy_label,
                    "first_rotor_positive_xy_label",
                )
            ),
        )
        object.__setattr__(self, "lever_arms_from_com_body_m", arms)
        object.__setattr__(self, "thrust_directions_body", directions)
        object.__setattr__(self, "lever_arm_uncertainty_m", uncertainty)

    @property
    def surveyed_with_uncertainty(self) -> bool:
        return self.frame_offsets_identified and self.lever_arm_uncertainty_m is not None

    @property
    def azimuths_by_rotor_order_deg(self) -> Tuple[float, float, float, float]:
        """Return fixed azimuth metadata in ``[RR, LF, LR, RF]`` order."""

        azimuth_by_label = {
            label: (
                self.geometric_sequence_start_azimuth_deg + index * self.adjacent_arm_spacing_deg
            )
            % 360.0
            for index, label in enumerate(self.geometric_sequence_rotor_order)
        }
        return cast(
            Tuple[float, float, float, float],
            tuple(azimuth_by_label[label] for label in ROTOR_ORDER),
        )


@dataclass(frozen=True)
class ManufacturerThrottleThrustCurve:
    """One built-in reviewed Hobbywing static-bench throttle/thrust table.

    ``throttle_percent`` is the percentage printed in Hobbywing's report.  It
    is not a Pixhawk normalized actuator value, PWM pulse width, or proof of
    the force executed by an installed rotor.  Interpolation is permitted only
    inside the measured table interval; extrapolation is rejected.
    """

    curve_id: str
    motor_system_model: str
    propeller_model: str
    battery_series_cells: int
    report_test_voltage_v: float
    laboratory_temperature_c: float
    test_environment: str
    throttle_percent: FloatArray
    thrust_n: FloatArray

    def __post_init__(self) -> None:
        curve_id = _nonempty(self.curve_id, "curve_id")
        motor = _nonempty(self.motor_system_model, "motor_system_model")
        propeller = _nonempty(self.propeller_model, "propeller_model")
        if motor != "X8 G2" or propeller != "MFP 30x11S":
            raise PreliminaryModelError(
                "manufacturer curves are valid only for X8 G2 with MFP 30x11S"
            )
        cells = self.battery_series_cells
        if isinstance(cells, bool) or not isinstance(cells, int) or cells not in (12, 14):
            raise PreliminaryModelError("battery_series_cells must be 12 or 14")
        voltage = _finite_real(
            self.report_test_voltage_v,
            "report_test_voltage_v",
            positive=True,
        )
        temperature = _finite_real(
            self.laboratory_temperature_c,
            "laboratory_temperature_c",
        )
        environment = _nonempty(self.test_environment, "test_environment")
        if temperature != 25.0 or environment != "SEA_LEVEL_STATIC_BENCH":
            raise PreliminaryModelError(
                "manufacturer curve conditions must remain 25 C and sea-level static bench"
            )
        if (cells == 12 and voltage != 46.0) or (cells == 14 and voltage != 54.0):
            raise PreliminaryModelError("manufacturer report test voltage does not match S count")
        throttle_raw = np.asarray(self.throttle_percent)
        thrust_raw = np.asarray(self.thrust_n)
        if (
            throttle_raw.ndim != 1
            or thrust_raw.ndim != 1
            or throttle_raw.shape != thrust_raw.shape
            or throttle_raw.size < 2
        ):
            raise PreliminaryModelError(
                "manufacturer throttle/thrust arrays must be equal 1-D data"
            )
        throttle = _readonly_array(
            throttle_raw,
            (int(throttle_raw.size),),
            "throttle_percent",
        )
        thrust = _readonly_array(thrust_raw, (int(thrust_raw.size),), "thrust_n")
        if np.any(np.diff(throttle) <= 0.0):
            raise PreliminaryModelError("manufacturer throttle_percent must increase strictly")
        if np.any(thrust <= 0.0) or np.any(np.diff(thrust) < 0.0):
            raise PreliminaryModelError("manufacturer thrust_n must be positive and nondecreasing")
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "motor_system_model", motor)
        object.__setattr__(self, "propeller_model", propeller)
        object.__setattr__(self, "report_test_voltage_v", voltage)
        object.__setattr__(self, "laboratory_temperature_c", temperature)
        object.__setattr__(self, "test_environment", environment)
        object.__setattr__(self, "throttle_percent", throttle)
        object.__setattr__(self, "thrust_n", thrust)

    @property
    def minimum_throttle_percent(self) -> float:
        return float(self.throttle_percent[0])

    @property
    def maximum_throttle_percent(self) -> float:
        return float(self.throttle_percent[-1])

    def interpolate_thrust_n(self, throttle_percent: object) -> float:
        """Piecewise-linearly interpolate one in-range manufacturer datum."""

        throttle = _finite_real(throttle_percent, "throttle_percent")
        if throttle < self.minimum_throttle_percent or throttle > self.maximum_throttle_percent:
            raise PreliminaryModelError(
                "throttle_percent is outside the manufacturer table; extrapolation is prohibited"
            )
        return float(np.interp(throttle, self.throttle_percent, self.thrust_n))


def _manufacturer_curve(curve_id: str) -> ManufacturerThrottleThrustCurve:
    throttle: Tuple[float, ...]
    thrust_gram_force: Tuple[float, ...]
    if curve_id == _X8_CURVE_IDS[0]:
        cells = 12
        voltage = 46.0
        throttle = _X8_12S_THROTTLE_PERCENT
        thrust_gram_force = _X8_12S_THRUST_GRAM_FORCE
    elif curve_id == _X8_CURVE_IDS[1]:
        cells = 14
        voltage = 54.0
        throttle = _X8_14S_THROTTLE_PERCENT
        thrust_gram_force = _X8_14S_THRUST_GRAM_FORCE
    else:
        raise PreliminaryModelError(f"unknown manufacturer curve id: {curve_id}")
    return ManufacturerThrottleThrustCurve(
        curve_id=curve_id,
        motor_system_model="X8 G2",
        propeller_model="MFP 30x11S",
        battery_series_cells=cells,
        report_test_voltage_v=voltage,
        laboratory_temperature_c=25.0,
        test_environment="SEA_LEVEL_STATIC_BENCH",
        throttle_percent=np.asarray(throttle, dtype=float),
        thrust_n=GRAM_FORCE_TO_NEWTON * np.asarray(thrust_gram_force, dtype=float),
    )


@dataclass(frozen=True)
class QuasiStaticRotorPrior:
    """Offline selection boundary around the two manufacturer curves."""

    motor_system_model: str
    battery_series_cells: Optional[int]
    installed_propeller_model: Optional[str]
    manufacturer_curve_ids: Tuple[str, str]
    manufacturer_input_semantics: str
    throttle_percent_is_pixhawk_normalized: bool
    throttle_percent_is_pwm: bool
    curve_output_is_installed_executed_thrust: bool
    curve_output_may_be_used_as_hardware_command: bool
    curves: Tuple[ManufacturerThrottleThrustCurve, ManufacturerThrottleThrustCurve]

    def __post_init__(self) -> None:
        model = _nonempty(self.motor_system_model, "rotor_prior.motor_system_model")
        if model != "X8 G2":
            raise PreliminaryModelError("rotor_prior.motor_system_model must be X8 G2")
        cells = self.battery_series_cells
        if cells is not None and (
            isinstance(cells, bool) or not isinstance(cells, int) or cells not in (12, 14)
        ):
            raise PreliminaryModelError("battery_series_cells must be null, 12, or 14")
        propeller = _optional_nonempty(
            self.installed_propeller_model,
            "installed_propeller_model",
        )
        if (cells is None) != (propeller is None):
            raise PreliminaryModelError(
                "battery_series_cells and installed_propeller_model must be null or confirmed together"
            )
        curve_ids = tuple(self.manufacturer_curve_ids)
        if curve_ids != _X8_CURVE_IDS:
            raise PreliminaryModelError("manufacturer_curve_ids must be the two reviewed X8 IDs")
        semantics = _nonempty(
            self.manufacturer_input_semantics,
            "manufacturer_input_semantics",
        )
        if semantics != "ESC_REPORT_THROTTLE_PERCENT_STATIC_BENCH_ONLY":
            raise PreliminaryModelError("manufacturer throttle semantics were changed")
        for name in (
            "throttle_percent_is_pixhawk_normalized",
            "throttle_percent_is_pwm",
            "curve_output_is_installed_executed_thrust",
            "curve_output_may_be_used_as_hardware_command",
        ):
            value = getattr(self, name)
            if type(value) is not bool or value:
                raise PreliminaryModelError(f"{name} must remain false")
        curves = tuple(self.curves)
        if len(curves) != 2 or tuple(curve.curve_id for curve in curves) != _X8_CURVE_IDS:
            raise PreliminaryModelError("curves must contain the reviewed 12S then 14S tables")
        object.__setattr__(self, "motor_system_model", model)
        object.__setattr__(self, "installed_propeller_model", propeller)
        object.__setattr__(self, "manufacturer_curve_ids", curve_ids)
        object.__setattr__(self, "manufacturer_input_semantics", semantics)
        object.__setattr__(
            self,
            "curves",
            cast(Tuple[ManufacturerThrottleThrustCurve, ManufacturerThrottleThrustCurve], curves),
        )

    @property
    def installed_configuration_complete(self) -> bool:
        return self.battery_series_cells is not None and self.installed_propeller_model is not None

    def select_installed_curve(self) -> ManufacturerThrottleThrustCurve:
        """Select only when both actual installation fields exactly match."""

        if not self.installed_configuration_complete:
            raise PreliminaryModelError(
                "battery_series_cells and installed_propeller_model must both be confirmed"
            )
        if self.installed_propeller_model != "MFP 30x11S":
            raise PreliminaryModelError(
                "manufacturer thrust curves cannot be selected for a different propeller"
            )
        for curve in self.curves:
            if curve.battery_series_cells == self.battery_series_cells:
                return curve
        raise PreliminaryModelError("no reviewed manufacturer curve matches the installed battery")


@dataclass(frozen=True)
class FlightControllerContractPrior:
    """Known Pixhawk hardware and explicitly unavailable residual semantics."""

    hardware: str
    firmware_stack: Optional[str]
    firmware_version: Optional[str]
    airframe_type: Optional[str]
    mount_orientation: Optional[str]
    output_coordinate_frame: Optional[str]
    quaternion_order: Optional[str]
    imu_from_total_com_body_m: Optional[FloatArray]
    per_sample_execution_ack: bool
    same_tick_baseline: bool
    residual_ttl: bool
    newton_interface: bool

    def __post_init__(self) -> None:
        hardware = _nonempty(self.hardware, "flight_controller.hardware")
        if hardware != "PIXHAWK_6X":
            raise PreliminaryModelError("flight_controller.hardware must be PIXHAWK_6X")
        firmware = _optional_nonempty(self.firmware_stack, "flight_controller.firmware_stack")
        firmware_version = _optional_nonempty(
            self.firmware_version,
            "flight_controller.firmware_version",
        )
        airframe_type = _optional_nonempty(
            self.airframe_type,
            "flight_controller.airframe_type",
        )
        mount_orientation = _optional_nonempty(
            self.mount_orientation,
            "flight_controller.mount_orientation",
        )
        output_frame = _optional_nonempty(
            self.output_coordinate_frame,
            "flight_controller.output_coordinate_frame",
        )
        quaternion_order = _optional_nonempty(
            self.quaternion_order,
            "flight_controller.quaternion_order",
        )
        imu_offset = _optional_vector3(
            self.imu_from_total_com_body_m,
            "flight_controller.imu_from_total_com_body_m",
        )
        for name in (
            "per_sample_execution_ack",
            "same_tick_baseline",
            "residual_ttl",
            "newton_interface",
        ):
            value = getattr(self, name)
            if type(value) is not bool or value:
                raise PreliminaryModelError(f"flight_controller.{name} must remain false")
        object.__setattr__(self, "hardware", hardware)
        object.__setattr__(self, "firmware_stack", firmware)
        object.__setattr__(self, "firmware_version", firmware_version)
        object.__setattr__(self, "airframe_type", airframe_type)
        object.__setattr__(self, "mount_orientation", mount_orientation)
        object.__setattr__(self, "output_coordinate_frame", output_frame)
        object.__setattr__(self, "quaternion_order", quaternion_order)
        object.__setattr__(self, "imu_from_total_com_body_m", imu_offset)

    @property
    def residual_contract_ready(self) -> bool:
        return False

    @property
    def installation_conventions_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.firmware_stack,
                self.firmware_version,
                self.airframe_type,
                self.mount_orientation,
                self.output_coordinate_frame,
                self.quaternion_order,
                self.imu_from_total_com_body_m,
            )
        )


def compute_sdk_count_contact_detector_hash(
    *,
    detector_config_version: str,
    lowstate_source: str,
    mapping_version: str,
    mapping_hash: str,
    algorithm_leg_order: object,
    sdk_indices_by_leg: object,
    signs_by_algorithm_leg: object,
    contact_on_threshold_sdk_counts: object,
    contact_off_threshold_sdk_counts: object,
    filter_time_constant_s: object,
    contact_confirm_s: object,
    release_confirm_s: object,
    maximum_sample_gap_s: object,
    maximum_feedback_age_s: object,
    minimum_consecutive_samples: object,
    maximum_source_tick_jump: object,
) -> str:
    """Hash every semantic, mapping, threshold, and timing input."""

    version = _nonempty(detector_config_version, "detector_config_version")
    source = _nonempty(lowstate_source, "lowstate_source")
    if source not in {"foot_force", "foot_force_est"}:
        raise PreliminaryModelError("lowstate_source must be foot_force or foot_force_est")
    mapping_version_value = _nonempty(mapping_version, "mapping_version")
    leg_order = _tuple4_names(algorithm_leg_order, "algorithm_leg_order")
    indices = _optional_int4(sdk_indices_by_leg, "sdk_indices_by_leg")
    if indices is None or set(indices) != {0, 1, 2, 3}:
        raise PreliminaryModelError("sdk_indices_by_leg must be a permutation of 0..3")
    expected_mapping_hash = compute_go2_foot_force_mapping_hash(
        mapping_version_value,
        leg_order,
        indices,
    )
    if mapping_hash != expected_mapping_hash:
        raise PreliminaryModelError("mapping_hash does not match the SDK count mapping")
    signs = _optional_int4(signs_by_algorithm_leg, "signs_by_algorithm_leg")
    if signs is None or any(value not in (-1, 1) for value in signs):
        raise PreliminaryModelError("signs_by_algorithm_leg must contain only -1 or +1")
    on = _optional_float4(
        contact_on_threshold_sdk_counts,
        "contact_on_threshold_sdk_counts",
    )
    off = _optional_float4(
        contact_off_threshold_sdk_counts,
        "contact_off_threshold_sdk_counts",
    )
    if (
        on is None
        or off is None
        or any(off_value >= on_value for on_value, off_value in zip(on, off))
    ):
        raise PreliminaryModelError("each count off-threshold must be below its on-threshold")
    timing = {
        "contact_confirm_s": _finite_real(contact_confirm_s, "contact_confirm_s", positive=True),
        "filter_time_constant_s": _finite_real(
            filter_time_constant_s,
            "filter_time_constant_s",
            positive=True,
        ),
        "maximum_feedback_age_s": _finite_real(
            maximum_feedback_age_s,
            "maximum_feedback_age_s",
            positive=True,
        ),
        "maximum_sample_gap_s": _finite_real(
            maximum_sample_gap_s,
            "maximum_sample_gap_s",
            positive=True,
        ),
        "release_confirm_s": _finite_real(release_confirm_s, "release_confirm_s", positive=True),
    }
    if (
        isinstance(minimum_consecutive_samples, bool)
        or not isinstance(minimum_consecutive_samples, int)
        or minimum_consecutive_samples < 1
    ):
        raise PreliminaryModelError("minimum_consecutive_samples must be a positive integer")
    if (
        isinstance(maximum_source_tick_jump, bool)
        or not isinstance(maximum_source_tick_jump, int)
        or not 1 <= maximum_source_tick_jump < 0x80000000
    ):
        raise PreliminaryModelError("maximum_source_tick_jump must be in the RFC1982 forward range")
    payload = {
        "algorithm_leg_order": list(leg_order),
        "contact_off_threshold_sdk_counts": list(off),
        "contact_on_threshold_sdk_counts": list(on),
        "detector_config_version": version,
        "lowstate_source": source,
        "mapping_hash": mapping_hash,
        "mapping_version": mapping_version_value,
        "maximum_source_tick_jump": maximum_source_tick_jump,
        "minimum_consecutive_samples": minimum_consecutive_samples,
        "sdk_indices_by_leg": list(indices),
        "signs_by_algorithm_leg": list(signs),
        **timing,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SdkCountContactDetectorConfig:
    """Robot-specific Schmitt detector configuration in signed SDK counts."""

    detector_config_version: str
    detector_config_hash: str
    lowstate_source: str
    mapping_version: str
    mapping_hash: str
    algorithm_leg_order: Tuple[str, str, str, str]
    sdk_indices_by_leg: Tuple[int, int, int, int]
    signs_by_algorithm_leg: Tuple[int, int, int, int]
    contact_on_threshold_sdk_counts: Tuple[float, float, float, float]
    contact_off_threshold_sdk_counts: Tuple[float, float, float, float]
    filter_time_constant_s: float
    contact_confirm_s: float
    release_confirm_s: float
    maximum_sample_gap_s: float
    maximum_feedback_age_s: float
    minimum_consecutive_samples: int
    maximum_source_tick_jump: int

    def __post_init__(self) -> None:
        expected = compute_sdk_count_contact_detector_hash(
            detector_config_version=self.detector_config_version,
            lowstate_source=self.lowstate_source,
            mapping_version=self.mapping_version,
            mapping_hash=self.mapping_hash,
            algorithm_leg_order=self.algorithm_leg_order,
            sdk_indices_by_leg=self.sdk_indices_by_leg,
            signs_by_algorithm_leg=self.signs_by_algorithm_leg,
            contact_on_threshold_sdk_counts=self.contact_on_threshold_sdk_counts,
            contact_off_threshold_sdk_counts=self.contact_off_threshold_sdk_counts,
            filter_time_constant_s=self.filter_time_constant_s,
            contact_confirm_s=self.contact_confirm_s,
            release_confirm_s=self.release_confirm_s,
            maximum_sample_gap_s=self.maximum_sample_gap_s,
            maximum_feedback_age_s=self.maximum_feedback_age_s,
            minimum_consecutive_samples=self.minimum_consecutive_samples,
            maximum_source_tick_jump=self.maximum_source_tick_jump,
        )
        if self.detector_config_hash != expected:
            raise PreliminaryModelError(
                "detector_config_hash does not match the complete count-detector contract"
            )


@dataclass(frozen=True)
class SdkCountContactDetection:
    """One contact decision; numeric observation fields remain SDK counts."""

    receipt_timestamp_s: float
    receipt_sequence: int
    subscription_generation: int
    source_tick: int
    lowstate_source: str
    algorithm_leg_order: Tuple[str, str, str, str]
    mapping_hash: str
    detector_config_hash: str
    ordered_raw_sdk_counts: Tuple[int, int, int, int]
    signed_sdk_counts: Tuple[int, int, int, int]
    filtered_signed_sdk_counts: Tuple[float, float, float, float]
    contacts: Tuple[bool, bool, bool, bool]
    contact_confirmed_events: Tuple[bool, bool, bool, bool]
    contact_released_events: Tuple[bool, bool, bool, bool]
    contact_on_threshold_first_crossing_s: Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]
    contact_confirmed_at_s: Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]
    contact_off_threshold_first_crossing_s: Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]
    contact_release_confirmed_at_s: Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]


class SdkCountContactDetector:
    """Fail-closed source-selecting Schmitt detector for LowState counts."""

    def __init__(self, config: SdkCountContactDetectorConfig) -> None:
        if not isinstance(config, SdkCountContactDetectorConfig):
            raise TypeError("config must be a SdkCountContactDetectorConfig")
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._filtered: Optional[FloatArray] = None
        self._contacts = np.zeros(4, dtype=bool)
        self._on_first_crossing: list[Optional[float]] = [None] * 4
        self._off_first_crossing: list[Optional[float]] = [None] * 4
        self._contact_confirmed_at: list[Optional[float]] = [None] * 4
        self._on_consecutive = np.zeros(4, dtype=np.int64)
        self._off_consecutive = np.zeros(4, dtype=np.int64)
        self._last_timestamp_s: Optional[float] = None
        self._last_receipt_sequence: Optional[int] = None
        self._last_source_tick: Optional[int] = None
        self._subscription_generation: Optional[int] = None
        self._reset_required_reason: Optional[str] = None

    @property
    def reset_required(self) -> bool:
        return self._reset_required_reason is not None

    def _trip(self, reason: str) -> NoReturn:
        self._contacts[:] = False
        self._reset_required_reason = reason
        raise PreliminaryModelError(f"{reason}; reset detector explicitly")

    def update(
        self,
        feedback: Go2FootForceFeedback,
        *,
        now_s: object,
    ) -> SdkCountContactDetection:
        if self._reset_required_reason is not None:
            raise PreliminaryModelError(
                f"detector reset required after: {self._reset_required_reason}"
            )
        if not isinstance(feedback, Go2FootForceFeedback):
            raise TypeError("feedback must be a Go2FootForceFeedback")
        now = _finite_real(now_s, "now_s")
        if now < 0.0:
            raise PreliminaryModelError("now_s must be nonnegative")
        if not feedback.source_identity_valid or feedback.source_tick is None:
            self._trip("LowState count sample identity is invalid")
        timestamp = feedback.receipt_timestamp_s
        age = now - timestamp
        if age < 0.0:
            self._trip("feedback receipt timestamp is in the future")
        if age > self._config.maximum_feedback_age_s:
            self._trip("feedback is older than maximum_feedback_age_s")
        if self._last_timestamp_s is not None:
            if timestamp <= self._last_timestamp_s:
                self._trip("receipt_timestamp_s must increase strictly")
            if timestamp - self._last_timestamp_s > self._config.maximum_sample_gap_s:
                self._trip("LowState sample gap exceeds maximum_sample_gap_s")
        if (
            self._last_receipt_sequence is not None
            and feedback.receipt_sequence <= self._last_receipt_sequence
        ):
            self._trip("receipt_sequence must increase strictly")
        if (
            self._subscription_generation is not None
            and feedback.subscription_generation != self._subscription_generation
        ):
            self._trip("subscription generation changed")
        if self._last_source_tick is not None:
            tick_delta = (feedback.source_tick - self._last_source_tick) & 0xFFFFFFFF
            if tick_delta == 0 or tick_delta >= 0x80000000:
                self._trip("source_tick is not forward under RFC1982 serial arithmetic")
            if tick_delta > self._config.maximum_source_tick_jump:
                self._trip("source_tick jump exceeds maximum_source_tick_jump")
        if self._config.lowstate_source == "foot_force":
            if not feedback.raw_valid:
                self._trip("configured LowState foot_force field is invalid")
            source_values = feedback.raw_sdk_int16
        else:
            if not feedback.estimated_valid:
                self._trip("configured LowState foot_force_est field is invalid")
            source_values = feedback.estimated_sdk_int16
        ordered = tuple(source_values[index] for index in self._config.sdk_indices_by_leg)
        if any(value in (-32768, 32767) for value in ordered):
            self._trip("selected SDK count channel is saturated")
        signed = tuple(
            self._config.signs_by_algorithm_leg[index] * ordered[index] for index in range(4)
        )
        counts = np.asarray(signed, dtype=np.float64)
        if self._filtered is None:
            filtered = counts.copy()
        else:
            assert self._last_timestamp_s is not None
            dt_s = timestamp - self._last_timestamp_s
            alpha = 1.0 - math.exp(-dt_s / self._config.filter_time_constant_s)
            filtered = self._filtered + alpha * (counts - self._filtered)

        contacts = self._contacts.copy()
        on_first = list(self._on_first_crossing)
        off_first = list(self._off_first_crossing)
        confirmed_at = list(self._contact_confirmed_at)
        on_consecutive = self._on_consecutive.copy()
        off_consecutive = self._off_consecutive.copy()
        confirmed_events = np.zeros(4, dtype=bool)
        released_events = np.zeros(4, dtype=bool)
        release_confirmed_at: list[Optional[float]] = [None] * 4
        for index in range(4):
            if not contacts[index]:
                off_first[index] = None
                off_consecutive[index] = 0
                if filtered[index] >= self._config.contact_on_threshold_sdk_counts[index]:
                    if on_first[index] is None:
                        on_first[index] = timestamp
                    on_consecutive[index] += 1
                    crossing = on_first[index]
                    assert crossing is not None
                    enough_samples = (
                        on_consecutive[index] >= self._config.minimum_consecutive_samples
                    )
                    if enough_samples and timestamp - crossing >= self._config.contact_confirm_s:
                        contacts[index] = True
                        confirmed_events[index] = True
                        confirmed_at[index] = timestamp
                        on_consecutive[index] = 0
                else:
                    on_first[index] = None
                    on_consecutive[index] = 0
            else:
                on_consecutive[index] = 0
                if filtered[index] <= self._config.contact_off_threshold_sdk_counts[index]:
                    if off_first[index] is None:
                        off_first[index] = timestamp
                    off_consecutive[index] += 1
                    crossing = off_first[index]
                    assert crossing is not None
                    enough_samples = (
                        off_consecutive[index] >= self._config.minimum_consecutive_samples
                    )
                    if enough_samples and timestamp - crossing >= self._config.release_confirm_s:
                        contacts[index] = False
                        released_events[index] = True
                        release_confirmed_at[index] = timestamp
                        confirmed_at[index] = None
                        on_first[index] = None
                        off_consecutive[index] = 0
                else:
                    off_first[index] = None
                    off_consecutive[index] = 0

        self._filtered = filtered.copy()
        self._contacts = contacts
        self._on_first_crossing = on_first
        self._off_first_crossing = off_first
        self._contact_confirmed_at = confirmed_at
        self._on_consecutive = on_consecutive
        self._off_consecutive = off_consecutive
        self._last_timestamp_s = timestamp
        self._last_receipt_sequence = feedback.receipt_sequence
        self._last_source_tick = feedback.source_tick
        self._subscription_generation = feedback.subscription_generation
        return SdkCountContactDetection(
            receipt_timestamp_s=timestamp,
            receipt_sequence=feedback.receipt_sequence,
            subscription_generation=feedback.subscription_generation,
            source_tick=feedback.source_tick,
            lowstate_source=self._config.lowstate_source,
            algorithm_leg_order=self._config.algorithm_leg_order,
            mapping_hash=self._config.mapping_hash,
            detector_config_hash=self._config.detector_config_hash,
            ordered_raw_sdk_counts=cast(Tuple[int, int, int, int], ordered),
            signed_sdk_counts=cast(Tuple[int, int, int, int], signed),
            filtered_signed_sdk_counts=cast(
                Tuple[float, float, float, float],
                tuple(float(value) for value in filtered),
            ),
            contacts=cast(Tuple[bool, bool, bool, bool], tuple(bool(value) for value in contacts)),
            contact_confirmed_events=cast(
                Tuple[bool, bool, bool, bool],
                tuple(bool(value) for value in confirmed_events),
            ),
            contact_released_events=cast(
                Tuple[bool, bool, bool, bool],
                tuple(bool(value) for value in released_events),
            ),
            contact_on_threshold_first_crossing_s=cast(
                Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
                tuple(on_first),
            ),
            contact_confirmed_at_s=cast(
                Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
                tuple(confirmed_at),
            ),
            contact_off_threshold_first_crossing_s=cast(
                Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
                tuple(off_first),
            ),
            contact_release_confirmed_at_s=cast(
                Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
                tuple(release_confirmed_at),
            ),
        )


@dataclass(frozen=True)
class ScalarContactPrior:
    """Explicitly non-newton Unitree scalar contact-event interface."""

    model: ContactModel
    measurement_semantics: FootForceSemantics
    lowstate_source: Optional[str]
    detector_config_version: Optional[str]
    detector_config_hash: Optional[str]
    mapping_version: Optional[str]
    mapping_hash: Optional[str]
    algorithm_leg_order: Optional[Tuple[str, str, str, str]]
    sdk_indices_by_leg: Optional[Tuple[int, int, int, int]]
    signs_by_algorithm_leg: Optional[Tuple[int, int, int, int]]
    contact_on_threshold_sdk_counts: Optional[Tuple[float, float, float, float]]
    contact_off_threshold_sdk_counts: Optional[Tuple[float, float, float, float]]
    filter_time_constant_s: Optional[float]
    contact_confirm_s: Optional[float]
    release_confirm_s: Optional[float]
    maximum_sample_gap_s: Optional[float]
    maximum_feedback_age_s: Optional[float]
    minimum_consecutive_samples: Optional[int]
    maximum_source_tick_jump: Optional[int]
    normal_force_input_available: bool
    tangential_force_input_available: bool
    sdk_values_may_be_used_as_newtons: bool

    def __post_init__(self) -> None:
        if self.model is not ContactModel.NORMAL_ONLY_VERTICAL:
            raise PreliminaryModelError("contact.model must be NORMAL_ONLY_VERTICAL")
        if self.measurement_semantics is not FootForceSemantics.UNCALIBRATED_CONTACT_EVENT_ONLY:
            raise PreliminaryModelError(
                "the provisional file must retain UNCALIBRATED_CONTACT_EVENT_ONLY semantics"
            )
        source = self.lowstate_source
        if source is not None and source not in {"foot_force", "foot_force_est"}:
            raise PreliminaryModelError(
                "lowstate_source must be null, foot_force, or foot_force_est"
            )
        for name in (
            "normal_force_input_available",
            "tangential_force_input_available",
            "sdk_values_may_be_used_as_newtons",
        ):
            if type(getattr(self, name)) is not bool:
                raise PreliminaryModelError(f"{name} must be a boolean")
        if (
            self.normal_force_input_available
            or self.tangential_force_input_available
            or self.sdk_values_may_be_used_as_newtons
        ):
            raise PreliminaryModelError(
                "uncalibrated SDK scalar fields cannot advertise physical force inputs"
            )
        optional_values = (
            source,
            self.detector_config_version,
            self.detector_config_hash,
            self.mapping_version,
            self.mapping_hash,
            self.algorithm_leg_order,
            self.sdk_indices_by_leg,
            self.signs_by_algorithm_leg,
            self.contact_on_threshold_sdk_counts,
            self.contact_off_threshold_sdk_counts,
            self.filter_time_constant_s,
            self.contact_confirm_s,
            self.release_confirm_s,
            self.maximum_sample_gap_s,
            self.maximum_feedback_age_s,
            self.minimum_consecutive_samples,
            self.maximum_source_tick_jump,
        )
        if any(value is not None for value in optional_values) and not all(
            value is not None for value in optional_values
        ):
            raise PreliminaryModelError(
                "SDK source, mapping, count thresholds, filter, and confirmation times "
                "plus continuity limits must be supplied together"
            )
        if source is not None:
            assert self.detector_config_version is not None
            assert self.detector_config_hash is not None
            assert self.mapping_version is not None
            assert self.mapping_hash is not None
            assert self.algorithm_leg_order is not None
            assert self.sdk_indices_by_leg is not None
            assert self.signs_by_algorithm_leg is not None
            assert self.contact_on_threshold_sdk_counts is not None
            assert self.contact_off_threshold_sdk_counts is not None
            assert self.filter_time_constant_s is not None
            assert self.contact_confirm_s is not None
            assert self.release_confirm_s is not None
            assert self.maximum_sample_gap_s is not None
            assert self.maximum_feedback_age_s is not None
            assert self.minimum_consecutive_samples is not None
            assert self.maximum_source_tick_jump is not None
            SdkCountContactDetectorConfig(
                detector_config_version=self.detector_config_version,
                detector_config_hash=self.detector_config_hash,
                lowstate_source=source,
                mapping_version=self.mapping_version,
                mapping_hash=self.mapping_hash,
                algorithm_leg_order=self.algorithm_leg_order,
                sdk_indices_by_leg=self.sdk_indices_by_leg,
                signs_by_algorithm_leg=self.signs_by_algorithm_leg,
                contact_on_threshold_sdk_counts=self.contact_on_threshold_sdk_counts,
                contact_off_threshold_sdk_counts=self.contact_off_threshold_sdk_counts,
                filter_time_constant_s=self.filter_time_constant_s,
                contact_confirm_s=self.contact_confirm_s,
                release_confirm_s=self.release_confirm_s,
                maximum_sample_gap_s=self.maximum_sample_gap_s,
                maximum_feedback_age_s=self.maximum_feedback_age_s,
                minimum_consecutive_samples=self.minimum_consecutive_samples,
                maximum_source_tick_jump=self.maximum_source_tick_jump,
            )

    @property
    def contact_event_detection_configured(self) -> bool:
        return self.lowstate_source is not None

    def new_sdk_count_contact_detector(self) -> SdkCountContactDetector:
        """Build fresh detector state, or reject the current incomplete prior."""

        if not self.contact_event_detection_configured:
            raise PreliminaryModelError(
                "SDK-count contact detector cannot be created until source, mapping, "
                "thresholds, filtering, and dwell times are supplied"
            )
        assert self.lowstate_source is not None
        assert self.detector_config_version is not None
        assert self.detector_config_hash is not None
        assert self.mapping_version is not None
        assert self.mapping_hash is not None
        assert self.algorithm_leg_order is not None
        assert self.sdk_indices_by_leg is not None
        assert self.signs_by_algorithm_leg is not None
        assert self.contact_on_threshold_sdk_counts is not None
        assert self.contact_off_threshold_sdk_counts is not None
        assert self.filter_time_constant_s is not None
        assert self.contact_confirm_s is not None
        assert self.release_confirm_s is not None
        assert self.maximum_sample_gap_s is not None
        assert self.maximum_feedback_age_s is not None
        assert self.minimum_consecutive_samples is not None
        assert self.maximum_source_tick_jump is not None
        return SdkCountContactDetector(
            SdkCountContactDetectorConfig(
                detector_config_version=self.detector_config_version,
                detector_config_hash=self.detector_config_hash,
                lowstate_source=self.lowstate_source,
                mapping_version=self.mapping_version,
                mapping_hash=self.mapping_hash,
                algorithm_leg_order=self.algorithm_leg_order,
                sdk_indices_by_leg=self.sdk_indices_by_leg,
                signs_by_algorithm_leg=self.signs_by_algorithm_leg,
                contact_on_threshold_sdk_counts=self.contact_on_threshold_sdk_counts,
                contact_off_threshold_sdk_counts=self.contact_off_threshold_sdk_counts,
                filter_time_constant_s=self.filter_time_constant_s,
                contact_confirm_s=self.contact_confirm_s,
                release_confirm_s=self.release_confirm_s,
                maximum_sample_gap_s=self.maximum_sample_gap_s,
                maximum_feedback_age_s=self.maximum_feedback_age_s,
                minimum_consecutive_samples=self.minimum_consecutive_samples,
                maximum_source_tick_jump=self.maximum_source_tick_jump,
            )
        )


@dataclass(frozen=True)
class PreliminaryLandingModelConfig:
    """Offline-only known/prior inputs for the first vertical landing model.

    中文：顶层对象同时保留参数来源和 readiness 证据。``full_six_dof_ready`` 仅用于
    报告缺项，即使变为 True 也不会打开硬件门禁；硬件授权必须由独立生产配置和
    集成验收流程完成。
    """

    source_path: Path
    schema_version: int
    profile: str
    physical_use_prohibited: bool
    allow_hardware_output: bool
    parameters_identified: bool
    world_frame: str
    body_frame: str
    rotor_order: Tuple[str, str, str, str]
    dynamics_reference_point: str
    leg_kinematics_reference_point: str
    gravity_m_per_s2: float
    cad_source: Optional[StepCadSourceProvenance]
    go2_urdf: Optional[PinnedGo2UrdfPrior]
    mass: ProvisionalMassProperties
    inertia: CadBomInertiaProperties
    offline_inertia_estimate: Optional[ProvisionalOfflineInertiaEstimate]
    geometry: FixedXGeometryPrior
    rotor_prior: QuasiStaticRotorPrior
    flight_controller: FlightControllerContractPrior
    contact: ScalarContactPrior

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_PRELIMINARY_SCHEMA_VERSION,
            PREVIOUS_PRELIMINARY_SCHEMA_VERSION,
            PRELIMINARY_SCHEMA_VERSION,
        }:
            raise PreliminaryModelError("unsupported preliminary schema_version")
        if self.profile != "provisional_offline":
            raise PreliminaryModelError("profile must be provisional_offline")
        if self.physical_use_prohibited is not True:
            raise PreliminaryModelError("physical_use_prohibited must be true")
        if self.allow_hardware_output is not False:
            raise PreliminaryModelError("allow_hardware_output must be false")
        if self.parameters_identified is not False:
            raise PreliminaryModelError("parameters_identified must remain false")
        if self.world_frame != "ENU_Z_UP" or self.body_frame != "X_FORWARD_Y_LEFT_Z_UP":
            raise PreliminaryModelError("preliminary model requires ENU/body-Z-up frames")
        if self.rotor_order != ROTOR_ORDER:
            raise PreliminaryModelError("rotor_order must be exactly [RR, LF, LR, RF]")
        if self.dynamics_reference_point != "TOTAL_SYSTEM_COM_C":
            raise PreliminaryModelError("dynamics_reference_point must be TOTAL_SYSTEM_COM_C")
        if self.leg_kinematics_reference_point != "GO2_BODY_ORIGIN_B":
            raise PreliminaryModelError("leg_kinematics_reference_point must be GO2_BODY_ORIGIN_B")
        gravity = _finite_real(self.gravity_m_per_s2, "gravity_m_per_s2", positive=True)
        if self.go2_urdf is not None and not math.isclose(
            self.go2_urdf.mass_properties.mass_kg,
            self.mass.go2_nominal_kg,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise PreliminaryModelError(
                "mass.go2_nominal_kg does not match the evaluated pinned Go2 URDF"
            )
        if self.offline_inertia_estimate is not None and not math.isclose(
            self.offline_inertia_estimate.x8_count
            * self.offline_inertia_estimate.x8_system_mass_each_kg
            + self.offline_inertia_estimate.remaining_added_mass_kg,
            self.mass.added_system_nominal_kg,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise PreliminaryModelError(
                "offline inertia component masses do not equal added_system_nominal_kg"
            )
        if self.offline_inertia_estimate is not None:
            if self.go2_urdf is None:
                raise PreliminaryModelError(
                    "offline inertia estimate requires evaluated pinned Go2 URDF properties"
                )
            estimate = self.offline_inertia_estimate
            go2 = self.go2_urdf
            if self.geometry.total_com_C_from_body_origin_B_m is None:
                raise PreliminaryModelError(
                    "offline inertia estimate requires explicit total COM C from body B"
                )
            p_com_from_body = self.geometry.total_com_C_from_body_origin_B_m
            p_go2_com_from_body = (
                go2.urdf_root_from_body_origin_B_m + go2.mass_properties.com_from_root_m
            )
            p_x8_from_body = (
                self.geometry.lever_arms_from_com_body_m + p_com_from_body[np.newaxis, :]
            )
            p_remaining_from_body = estimate.remaining_added_mass_effective_com_from_body_origin_B_m

            # The configured total C and component centroids must satisfy the
            # first mass moment.  In particular, putting all unmodelled added
            # mass at O is incompatible with the current C= B+[0,0,0.05] m.
            first_moment_about_c = (
                go2.mass_properties.mass_kg * (p_go2_com_from_body - p_com_from_body)
                + estimate.x8_system_mass_each_kg * np.sum(p_x8_from_body - p_com_from_body, axis=0)
                + estimate.remaining_added_mass_kg * (p_remaining_from_body - p_com_from_body)
            )
            if not np.allclose(first_moment_about_c, 0.0, rtol=0.0, atol=1.0e-9):
                raise PreliminaryModelError(
                    "offline component centroids do not reproduce total COM C; "
                    "update the remaining-added-mass balance centroid"
                )

            def point_mass_inertia(mass_kg: float, offset_from_c_m: FloatArray) -> FloatArray:
                squared_distance = float(offset_from_c_m @ offset_from_c_m)
                return cast(
                    FloatArray,
                    mass_kg
                    * (
                        squared_distance * np.eye(3, dtype=np.float64)
                        - np.outer(offset_from_c_m, offset_from_c_m)
                    ),
                )

            lower_inertia = np.array(
                go2.mass_properties.inertia_about_com_root_axes_kg_m2,
                dtype=np.float64,
                copy=True,
            )
            lower_inertia += point_mass_inertia(
                go2.mass_properties.mass_kg,
                cast(FloatArray, p_go2_com_from_body - p_com_from_body),
            )
            for x8_position in p_x8_from_body:
                lower_inertia += point_mass_inertia(
                    estimate.x8_system_mass_each_kg,
                    cast(FloatArray, x8_position - p_com_from_body),
                )
            lower_inertia += point_mass_inertia(
                estimate.remaining_added_mass_kg,
                cast(FloatArray, p_remaining_from_body - p_com_from_body),
            )

            # Lower: the 5.62 kg remainder is concentrated at its required
            # balance centroid.  Upper: it has the same centroid but is spread
            # symmetrically in the x-y plane at the rotor radius.  The nominal
            # prior is the arithmetic midpoint; none is a confidence interval.
            radius = self.geometry.horizontal_radius_from_frame_center_m
            radial_increment = estimate.remaining_added_mass_kg * np.diag(
                [0.5 * radius * radius, 0.5 * radius * radius, radius * radius]
            )
            upper_inertia = lower_inertia + radial_increment
            nominal_inertia = lower_inertia + 0.5 * radial_increment
            if not np.allclose(
                estimate.diagonal_lower_body_kg_m2,
                np.diag(lower_inertia),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise PreliminaryModelError(
                    "offline inertia lower diagonal does not match the reviewed mass model"
                )
            if not np.allclose(
                estimate.diagonal_upper_body_kg_m2,
                np.diag(upper_inertia),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise PreliminaryModelError(
                    "offline inertia upper diagonal does not match the reviewed mass model"
                )
            if not np.allclose(
                estimate.nominal_body_kg_m2,
                nominal_inertia,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise PreliminaryModelError(
                    "offline nominal inertia does not match the reviewed midpoint model"
                )
        object.__setattr__(self, "gravity_m_per_s2", gravity)

    @property
    def hardware_output_permitted(self) -> bool:
        """A constant-false property useful in audit/log output."""

        return False

    @property
    def full_six_dof_ready(self) -> bool:
        return (
            self.mass.measured_with_uncertainty
            and self.inertia.identified
            and self.geometry.surveyed_with_uncertainty
            and self.go2_urdf is not None
            and self.go2_urdf.body_origin_B_alignment_identified
            and self.contact.normal_force_input_available
            and self.flight_controller.installation_conventions_complete
            and self.flight_controller.residual_contract_ready
        )


@dataclass(frozen=True)
class NormalOnlyVerticalState:
    """Height and vertical speed in the declared ENU frame."""

    height_world_m: float
    vertical_velocity_world_m_per_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "height_world_m",
            _finite_real(self.height_world_m, "height_world_m"),
        )
        object.__setattr__(
            self,
            "vertical_velocity_world_m_per_s",
            _finite_real(
                self.vertical_velocity_world_m_per_s,
                "vertical_velocity_world_m_per_s",
            ),
        )


def calibrated_normal_force_scalars_n(
    sample: CalibratedGo2NormalForceSample,
) -> Tuple[float, float, float, float]:
    """Return calibrated per-foot normal scalars without inventing 3-D axes.

    Requiring the calibrated sample type prevents raw SDK integers from being
    passed accidentally as newtons.  Conversion into world vectors requires a
    separately identified ground-normal/frame contract and is intentionally
    outside this preliminary API.
    """

    if not isinstance(sample, CalibratedGo2NormalForceSample):
        raise TypeError("sample must be a CalibratedGo2NormalForceSample")
    return sample.normal_forces_n


def manufacturer_static_thrust_n_at_throttle(
    throttle_percent: object,
    config: PreliminaryLandingModelConfig,
) -> float:
    """Interpolate the matching manufacturer table for offline analysis only.

    This does not convert a Pixhawk command to thrust and must never be sent to
    hardware.  Selection fails until the installed battery S count and exact
    propeller model are confirmed in configuration.
    """

    if not isinstance(config, PreliminaryLandingModelConfig):
        raise TypeError("config must be a PreliminaryLandingModelConfig")
    return config.rotor_prior.select_installed_curve().interpolate_thrust_n(throttle_percent)


def ideal_level_hover_thrust_per_rotor_n(config: PreliminaryLandingModelConfig) -> float:
    """Return ``m*g/4`` as an ideal diagnostic, never an actuator limit."""

    if not isinstance(config, PreliminaryLandingModelConfig):
        raise TypeError("config must be a PreliminaryLandingModelConfig")
    return config.mass.total_nominal_kg * config.gravity_m_per_s2 / 4.0


def normal_only_vertical_acceleration_m_per_s2(
    *,
    total_rotor_vertical_force_world_n: object,
    total_contact_normal_force_n: object,
    config: PreliminaryLandingModelConfig,
) -> float:
    """Evaluate ``z_ddot = -g + (T_world,z + Fz) / m`` offline.

    ``total_rotor_vertical_force_world_n`` is the already projected world-Z
    component summed over the four rotors.  It is not total thrust along body
    rotor axes; callers must use an explicitly known attitude/frame transform
    before calling this function.  This preliminary model has no attitude
    state and therefore cannot perform that projection itself.  Both inputs
    are newton-valued; uncalibrated SDK values must never be supplied as
    ``total_contact_normal_force_n``.
    """

    if not isinstance(config, PreliminaryLandingModelConfig):
        raise TypeError("config must be a PreliminaryLandingModelConfig")
    rotor_vertical = _finite_real(
        total_rotor_vertical_force_world_n,
        "total_rotor_vertical_force_world_n",
        minimum=0.0,
    )
    contact = _finite_real(
        total_contact_normal_force_n,
        "total_contact_normal_force_n",
        minimum=0.0,
    )
    return -config.gravity_m_per_s2 + (rotor_vertical + contact) / config.mass.total_nominal_kg


def normal_only_vertical_step(
    state: NormalOnlyVerticalState,
    *,
    acceleration_m_per_s2: object,
    dt_s: object,
) -> NormalOnlyVerticalState:
    """Advance the vertical model with constant-acceleration kinematics."""

    if not isinstance(state, NormalOnlyVerticalState):
        raise TypeError("state must be a NormalOnlyVerticalState")
    acceleration = _finite_real(acceleration_m_per_s2, "acceleration_m_per_s2")
    dt = _finite_real(dt_s, "dt_s", positive=True)
    return NormalOnlyVerticalState(
        height_world_m=(
            state.height_world_m
            + state.vertical_velocity_world_m_per_s * dt
            + 0.5 * acceleration * dt * dt
        ),
        vertical_velocity_world_m_per_s=(state.vertical_velocity_world_m_per_s + acceleration * dt),
    )


def normal_only_vertical_impact_reset(
    state_before: NormalOnlyVerticalState,
    *,
    total_normal_impulse_ns: object,
    config: PreliminaryLandingModelConfig,
) -> NormalOnlyVerticalState:
    """Apply ``v_z+ = v_z- + Lambda_z / m`` without angular reset."""

    if not isinstance(state_before, NormalOnlyVerticalState):
        raise TypeError("state_before must be a NormalOnlyVerticalState")
    if not isinstance(config, PreliminaryLandingModelConfig):
        raise TypeError("config must be a PreliminaryLandingModelConfig")
    impulse = _finite_real(total_normal_impulse_ns, "total_normal_impulse_ns", minimum=0.0)
    return NormalOnlyVerticalState(
        height_world_m=state_before.height_world_m,
        vertical_velocity_world_m_per_s=(
            state_before.vertical_velocity_world_m_per_s + impulse / config.mass.total_nominal_kg
        ),
    )


def ideal_arresting_normal_impulse_ns(
    vertical_velocity_before_m_per_s: object,
    config: PreliminaryLandingModelConfig,
) -> float:
    """Return the ideal total impulse that removes downward CoM velocity.

    This is a prediction target, not a measured impact, a peak-force estimate,
    or a four-foot load distribution.
    """

    if not isinstance(config, PreliminaryLandingModelConfig):
        raise TypeError("config must be a PreliminaryLandingModelConfig")
    velocity = _finite_real(
        vertical_velocity_before_m_per_s,
        "vertical_velocity_before_m_per_s",
    )
    return config.mass.total_nominal_kg * max(0.0, -velocity)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> Dict[object, object]:
    loader.flatten_mapping(node)
    result: Dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
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
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PreliminaryModelError(f"{name} must be a mapping")
    result: Dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PreliminaryModelError(f"{name} keys must be strings")
        result[key] = item
    return result


def _exact_keys(value: Mapping[str, object], expected: Iterable[str], name: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing:
        raise PreliminaryModelError(f"{name} is missing required keys: {missing}")
    if unknown:
        raise PreliminaryModelError(f"{name} contains unknown keys: {unknown}")


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise PreliminaryModelError(f"{name} must be a boolean")
    return value


def _optional_float4(value: object, name: str) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    try:
        raw = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise PreliminaryModelError(f"{name} must be null or a four-item list") from exc
    if len(raw) != 4:
        raise PreliminaryModelError(f"{name} must contain four values")
    return cast(
        Tuple[float, float, float, float],
        tuple(_finite_real(item, name) for item in raw),
    )


def _optional_int4(value: object, name: str) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    try:
        raw = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise PreliminaryModelError(f"{name} must be null or a four-item list") from exc
    if len(raw) != 4 or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise PreliminaryModelError(f"{name} must contain four integers")
    return cast(Tuple[int, int, int, int], raw)


def _optional_int(value: object, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreliminaryModelError(f"{name} must be null or an integer")
    return value


def _optional_names4(value: object, name: str) -> Optional[Tuple[str, str, str, str]]:
    if value is None:
        return None
    return _tuple4_names(value, name)


def _config_local_file(source_path: Path, value: object, name: str) -> Path:
    """Resolve a config-relative asset without allowing directory escape."""

    relative = Path(_nonempty(value, name))
    if relative.is_absolute():
        raise PreliminaryModelError(f"{name} must be relative to the configuration directory")
    try:
        candidate = (source_path.parent / relative).resolve(strict=True)
        candidate.relative_to(source_path.parent)
    except (OSError, ValueError) as exc:
        raise PreliminaryModelError(
            f"{name} must resolve to an existing file inside the configuration directory"
        ) from exc
    if not candidate.is_file():
        raise PreliminaryModelError(f"{name} must resolve to a regular file")
    return candidate


def _read_unique_preliminary_document(path: PathLike) -> Tuple[Path, Dict[str, object]]:
    """Read one bounded UTF-8 YAML document with duplicate-key rejection."""

    try:
        source_path = Path(path).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise PreliminaryModelError(f"configuration path is not a file: {source_path}")
        if source_path.stat().st_size > MAX_CONFIG_BYTES:
            raise PreliminaryModelError("configuration file exceeds the 256 kB limit")
        text = source_path.read_text(encoding="utf-8")
    except PreliminaryModelError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PreliminaryModelError(f"cannot read preliminary config {path!r}: {exc}") from exc
    try:
        document = dict(_mapping(yaml.load(text, Loader=_UniqueKeyLoader), "root"))
    except yaml.YAMLError as exc:
        raise PreliminaryModelError(f"invalid YAML in {source_path}: {exc}") from exc
    return source_path, document


def recompute_preliminary_derived_document(path: PathLike) -> Dict[str, object]:
    """Return a schema-4 candidate with all redundant physical values recomputed.

    The source file is only read.  This function does not open any robot or
    flight-controller transport and cannot change a hardware gate.  Callers
    must serialize the returned candidate to a *different* file and validate
    that file with :func:`load_preliminary_landing_model` before using it.

    Editable physical inputs remain untouched: added mass, B/O/C locations,
    rotor-plane offset from O, rotor radius/azimuths, X8 mass/count, and the
    pinned URDF identity/pose.  Only redundant fields are replaced.

    中文：函数只返回内存中的候选文档，不覆盖源 YAML。它重新求整机质量、一阶质量
    矩约束、C 到各旋翼力臂和惯量先验上下界；输出仍需写到不同文件并重新通过严格
    加载器。这样修改 B/O/C 或质量后不会遗留彼此矛盾的手填派生量。
    """

    source_path, raw_document = _read_unique_preliminary_document(path)
    if raw_document.get("schema_version") != PRELIMINARY_SCHEMA_VERSION:
        raise PreliminaryModelError(
            f"derived-parameter recomputation requires schema_version {PRELIMINARY_SCHEMA_VERSION}"
        )
    if raw_document.get("parameters_identified") is not False:
        raise PreliminaryModelError("recomputation requires parameters_identified: false")
    if raw_document.get("physical_use_prohibited") is not True:
        raise PreliminaryModelError("recomputation requires physical_use_prohibited: true")
    if raw_document.get("allow_hardware_output") is not False:
        raise PreliminaryModelError("recomputation requires allow_hardware_output: false")

    document = cast(Dict[str, Any], deepcopy(raw_document))

    def mutable_section(name: str) -> Dict[str, Any]:
        section = document.get(name)
        _mapping(section, name)
        if not isinstance(section, dict):  # pragma: no cover - narrowed by _mapping
            raise PreliminaryModelError(f"{name} must be a mutable mapping")
        return cast(Dict[str, Any], section)

    frames = mutable_section("frames")
    mass = mutable_section("mass")
    urdf = mutable_section("go2_urdf")
    geometry = mutable_section("geometry")
    offline = mutable_section("offline_inertia_estimate")

    rotor_order = _tuple4_names(frames.get("rotor_order"), "frames.rotor_order")
    if rotor_order != ROTOR_ORDER:
        raise PreliminaryModelError("rotor_order must be exactly [RR, LF, LR, RF]")
    geometric_sequence = _tuple4_names(
        geometry.get("geometric_sequence_rotor_order"),
        "geometry.geometric_sequence_rotor_order",
    )
    if set(geometric_sequence) != set(ROTOR_ORDER):
        raise PreliminaryModelError(
            "geometry.geometric_sequence_rotor_order must contain every rotor exactly once"
        )
    start_azimuth_deg = _finite_real(
        geometry.get("geometric_sequence_start_azimuth_deg"),
        "geometry.geometric_sequence_start_azimuth_deg",
    )
    spacing_deg = _finite_real(
        geometry.get("adjacent_arm_spacing_deg"),
        "geometry.adjacent_arm_spacing_deg",
        positive=True,
    )
    radius_m = _finite_real(
        geometry.get("horizontal_radius_from_frame_center_m"),
        "geometry.horizontal_radius_from_frame_center_m",
        positive=True,
    )
    p_o_from_b = _readonly_array(
        geometry.get("frame_center_O_from_body_origin_B_m"),
        (3,),
        "geometry.frame_center_O_from_body_origin_B_m",
    )
    p_c_from_b = _readonly_array(
        geometry.get("total_com_C_from_body_origin_B_m"),
        (3,),
        "geometry.total_com_C_from_body_origin_B_m",
    )
    p_rotor_plane_from_o = _readonly_array(
        geometry.get("rotor_plane_from_frame_center_O_m"),
        (3,),
        "geometry.rotor_plane_from_frame_center_O_m",
    )
    if not np.allclose(p_o_from_b[:2], p_c_from_b[:2], rtol=0.0, atol=1.0e-12):
        raise PreliminaryModelError("B, O and C must retain the declared common z axis")
    if not np.allclose(p_rotor_plane_from_o[:2], 0.0, rtol=0.0, atol=1.0e-12):
        raise PreliminaryModelError("rotor-plane offset from O must have zero x/y components")

    joint_order_raw = urdf.get("sdk_joint_order")
    joint_positions_raw = urdf.get("sdk_joint_positions_rad")
    try:
        joint_order_values = tuple(cast(Iterable[object], joint_order_raw))
        joint_position_values = tuple(cast(Iterable[object], joint_positions_raw))
    except TypeError as exc:
        raise PreliminaryModelError("go2_urdf SDK joint order and pose must be lists") from exc
    if len(joint_order_values) != 12 or any(
        not isinstance(value, str) or not value.strip() for value in joint_order_values
    ):
        raise PreliminaryModelError("go2_urdf.sdk_joint_order must contain twelve joint names")
    if len(joint_position_values) != 12:
        raise PreliminaryModelError("go2_urdf.sdk_joint_positions_rad must contain twelve values")
    joint_order = cast(Tuple[str, ...], joint_order_values)
    joint_positions = tuple(
        _finite_real(value, "go2_urdf.sdk_joint_positions_rad") for value in joint_position_values
    )
    bundled_path = _config_local_file(
        source_path,
        urdf.get("bundled_path"),
        "go2_urdf.bundled_path",
    )
    try:
        go2_properties = load_go2_urdf_mass_properties(
            bundled_path,
            expected_sha256=_sha256_identity(
                urdf.get("bundled_sha256"),
                "go2_urdf.bundled_sha256",
            ),
            expected_robot_name=_nonempty(
                urdf.get("expected_robot_name"),
                "go2_urdf.expected_robot_name",
            ),
            root_link=_nonempty(urdf.get("root_link"), "go2_urdf.root_link"),
            joint_positions_rad=dict(zip(joint_order, joint_positions)),
        )
    except (Go2UrdfError, OSError) as exc:
        raise PreliminaryModelError(f"go2_urdf cannot be evaluated: {exc}") from exc
    configured_go2_mass = _finite_real(
        mass.get("go2_nominal_kg"),
        "mass.go2_nominal_kg",
        positive=True,
    )
    if not math.isclose(
        configured_go2_mass,
        go2_properties.mass_kg,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise PreliminaryModelError(
            "mass.go2_nominal_kg is an input identity check and must match the pinned URDF"
        )

    added_mass_kg = _finite_real(
        mass.get("added_system_nominal_kg"),
        "mass.added_system_nominal_kg",
        positive=True,
    )
    x8_mass_each_kg = _finite_real(
        offline.get("x8_system_mass_each_kg"),
        "offline_inertia_estimate.x8_system_mass_each_kg",
        positive=True,
    )
    x8_count_raw = offline.get("x8_count")
    if isinstance(x8_count_raw, bool) or not isinstance(x8_count_raw, int):
        raise PreliminaryModelError("offline_inertia_estimate.x8_count must be an integer")
    if x8_count_raw != 4:
        raise PreliminaryModelError("preliminary model requires exactly four X8 systems")
    remaining_mass_kg = added_mass_kg - x8_count_raw * x8_mass_each_kg
    if remaining_mass_kg <= 0.0:
        raise PreliminaryModelError("added_system_nominal_kg must exceed the combined four-X8 mass")
    total_mass_kg = go2_properties.mass_kg + added_mass_kg

    p_com_from_o = p_c_from_b - p_o_from_b
    p_rotor_plane_from_b = p_o_from_b + p_rotor_plane_from_o
    rotor_height_from_c_m = float(p_rotor_plane_from_b[2] - p_c_from_b[2])
    azimuth_by_label = {
        label: (start_azimuth_deg + index * spacing_deg) % 360.0
        for index, label in enumerate(geometric_sequence)
    }
    arms_from_c = np.array(
        [
            [
                radius_m * math.cos(math.radians(azimuth_by_label[label])),
                radius_m * math.sin(math.radians(azimuth_by_label[label])),
                rotor_height_from_c_m,
            ]
            for label in ROTOR_ORDER
        ],
        dtype=np.float64,
    )

    p_urdf_root_from_b = _readonly_array(
        urdf.get("urdf_root_from_body_origin_B_m"),
        (3,),
        "go2_urdf.urdf_root_from_body_origin_B_m",
    )
    p_go2_com_from_b = p_urdf_root_from_b + go2_properties.com_from_root_m
    p_x8_from_b = arms_from_c + p_c_from_b[np.newaxis, :]
    p_remaining_from_b = (
        total_mass_kg * p_c_from_b
        - go2_properties.mass_kg * p_go2_com_from_b
        - x8_mass_each_kg * np.sum(p_x8_from_b, axis=0)
    ) / remaining_mass_kg

    def point_mass_inertia(mass_kg: float, offset_from_c_m: FloatArray) -> FloatArray:
        squared_distance = float(offset_from_c_m @ offset_from_c_m)
        return cast(
            FloatArray,
            mass_kg
            * (
                squared_distance * np.eye(3, dtype=np.float64)
                - np.outer(offset_from_c_m, offset_from_c_m)
            ),
        )

    lower_inertia = np.array(
        go2_properties.inertia_about_com_root_axes_kg_m2,
        dtype=np.float64,
        copy=True,
    )
    lower_inertia += point_mass_inertia(
        go2_properties.mass_kg,
        cast(FloatArray, p_go2_com_from_b - p_c_from_b),
    )
    for arm_from_c in arms_from_c:
        lower_inertia += point_mass_inertia(
            x8_mass_each_kg,
            cast(FloatArray, arm_from_c),
        )
    lower_inertia += point_mass_inertia(
        remaining_mass_kg,
        cast(FloatArray, p_remaining_from_b - p_c_from_b),
    )
    radial_increment = remaining_mass_kg * np.diag(
        [0.5 * radius_m * radius_m, 0.5 * radius_m * radius_m, radius_m * radius_m]
    )
    upper_inertia = lower_inertia + radial_increment
    nominal_inertia = lower_inertia + 0.5 * radial_increment

    # Only redundant fields are assigned below.  Safety gates, measured/raw
    # inputs, uncertainty declarations and hardware contracts are untouched.
    mass["total_nominal_kg"] = float(total_mass_kg)
    geometry["total_com_from_frame_center_body_m"] = p_com_from_o.tolist()
    geometry["rotor_frame_com_from_body_origin_B_m"] = p_o_from_b.tolist()
    geometry["rotor_plane_z_from_com_m"] = rotor_height_from_c_m
    geometry["lever_arms_from_com_body_m"] = arms_from_c.tolist()
    offline["remaining_added_mass_kg"] = float(remaining_mass_kg)
    offline["remaining_added_mass_effective_com_from_body_origin_B_m"] = p_remaining_from_b.tolist()
    offline["nominal_body_kg_m2"] = nominal_inertia.tolist()
    offline["diagonal_lower_body_kg_m2"] = np.diag(lower_inertia).tolist()
    offline["diagonal_upper_body_kg_m2"] = np.diag(upper_inertia).tolist()
    return cast(Dict[str, object], document)


def load_preliminary_landing_model(
    path: PathLike,
    *,
    allow_provisional: bool = False,
    for_hardware: bool = False,
) -> PreliminaryLandingModelConfig:
    """Load the incomplete vertical model with an explicit offline opt-in.

    ``for_hardware=True`` always fails.  This remains true even after filling
    all optional values because real-output release belongs to the separate,
    reviewed production integration path.

    中文：调用方必须显式传 ``allow_provisional=True`` 承认这是估计配置；任何
    ``for_hardware=True`` 请求无条件拒绝。加载过程还拒绝重复 YAML 键、越界路径、
    不匹配的 URDF 哈希和不闭合的质量/惯量关系。
    """

    if type(allow_provisional) is not bool or type(for_hardware) is not bool:
        raise TypeError("allow_provisional and for_hardware must be booleans")
    if for_hardware:
        raise PreliminaryModelError("the preliminary vertical model is prohibited for hardware")
    if not allow_provisional:
        raise PreliminaryModelError("explicit allow_provisional=True is required")
    source_path, document = _read_unique_preliminary_document(path)
    schema = document.get("schema_version")
    if (
        isinstance(schema, bool)
        or not isinstance(schema, int)
        or schema
        not in {
            LEGACY_PRELIMINARY_SCHEMA_VERSION,
            PREVIOUS_PRELIMINARY_SCHEMA_VERSION,
            PRELIMINARY_SCHEMA_VERSION,
        }
    ):
        raise PreliminaryModelError(
            "schema_version must be one of "
            f"{LEGACY_PRELIMINARY_SCHEMA_VERSION}, "
            f"{PREVIOUS_PRELIMINARY_SCHEMA_VERSION}, or {PRELIMINARY_SCHEMA_VERSION}"
        )
    root_keys: Tuple[str, ...] = (
        "schema_version",
        "profile",
        "parameters_identified",
        "physical_use_prohibited",
        "allow_hardware_output",
        "frames",
        "model",
        "mass",
        "inertia",
        "geometry",
        "rotor_prior",
        "flight_controller",
        "contact",
    )
    if schema >= PREVIOUS_PRELIMINARY_SCHEMA_VERSION:
        root_keys += ("cad_source",)
    if schema >= PRELIMINARY_SCHEMA_VERSION:
        root_keys += ("go2_urdf", "offline_inertia_estimate")
    _exact_keys(document, root_keys, "root")

    frames = _mapping(document["frames"], "frames")
    _exact_keys(frames, ("world", "body", "rotor_order"), "frames")
    model = _mapping(document["model"], "model")
    model_keys: Tuple[str, ...] = ("contact_model", "gravity_m_per_s2")
    if schema >= PRELIMINARY_SCHEMA_VERSION:
        model_keys += ("dynamics_reference_point", "leg_kinematics_reference_point")
    _exact_keys(model, model_keys, "model")
    mass_data = _mapping(document["mass"], "mass")
    _exact_keys(
        mass_data,
        (
            "go2_nominal_kg",
            "added_system_nominal_kg",
            "total_nominal_kg",
            "total_uncertainty_kg",
        ),
        "mass",
    )
    inertia_data = _mapping(document["inertia"], "inertia")
    _exact_keys(
        inertia_data,
        (
            "method",
            "nominal_body_kg_m2",
            "uncertainty_body_kg_m2",
            "cad_bom_revision",
            "reference_pose",
        ),
        "inertia",
    )
    go2_urdf_data: Optional[Mapping[str, object]] = None
    offline_inertia_data: Optional[Mapping[str, object]] = None
    if schema >= PRELIMINARY_SCHEMA_VERSION:
        go2_urdf_data = _mapping(document["go2_urdf"], "go2_urdf")
        _exact_keys(
            go2_urdf_data,
            (
                "source_repository",
                "source_file_commit",
                "model_quality",
                "upstream_raw_sha256",
                "bundled_path",
                "bundled_sha256",
                "license_path",
                "expected_robot_name",
                "root_link",
                "urdf_root_from_body_origin_B_m",
                "body_origin_B_alignment_identified",
                "reference_pose",
                "sdk_joint_order",
                "sdk_joint_positions_rad",
            ),
            "go2_urdf",
        )
        offline_inertia_data = _mapping(
            document["offline_inertia_estimate"],
            "offline_inertia_estimate",
        )
        _exact_keys(
            offline_inertia_data,
            (
                "method",
                "quality",
                "reference_point",
                "x8_system_mass_each_kg",
                "x8_count",
                "remaining_added_mass_kg",
                "remaining_added_mass_effective_com_from_body_origin_B_m",
                "remaining_mass_distribution_interval",
                "nominal_body_kg_m2",
                "diagonal_lower_body_kg_m2",
                "diagonal_upper_body_kg_m2",
                "cross_terms_unbounded",
                "hardware_use_prohibited",
            ),
            "offline_inertia_estimate",
        )
    cad_source_data: Optional[Mapping[str, object]] = None
    if schema >= PREVIOUS_PRELIMINARY_SCHEMA_VERSION:
        cad_source_data = _mapping(document["cad_source"], "cad_source")
        _exact_keys(
            cad_source_data,
            (
                "file_name",
                "sha256",
                "file_size_bytes",
                "step_schema",
                "length_unit",
                "brep_validation_passed",
                "material_density_properties_present",
                "mass_inertia_properties_present",
                "complete_system_scope_verified",
            ),
            "cad_source",
        )
    geometry_data = _mapping(document["geometry"], "geometry")
    geometry_keys: Tuple[str, ...] = (
        "horizontal_radius_from_frame_center_m",
        "rotor_plane_z_from_com_m",
        "horizontal_origin_assumption",
        "azimuth_assumption",
        "lever_arms_from_com_body_m",
        "thrust_directions_body",
        "lever_arm_uncertainty_m",
    )
    if schema >= PREVIOUS_PRELIMINARY_SCHEMA_VERSION:
        geometry_keys += (
            "total_com_from_frame_center_body_m",
            "azimuth_reference",
            "geometric_sequence_start_azimuth_deg",
            "adjacent_arm_spacing_deg",
            "geometric_sequence_rotor_order",
        )
    if schema >= PRELIMINARY_SCHEMA_VERSION:
        geometry_keys += (
            "body_origin_B_definition",
            "frame_center_O_from_body_origin_B_m",
            "total_com_C_from_body_origin_B_m",
            "rotor_frame_com_from_body_origin_B_m",
            "rotor_plane_from_frame_center_O_m",
            "frame_offsets_identified",
            "first_rotor_positive_xy_label",
        )
    _exact_keys(geometry_data, geometry_keys, "geometry")
    rotor_data = _mapping(document["rotor_prior"], "rotor_prior")
    _exact_keys(
        rotor_data,
        (
            "motor_system_model",
            "battery_series_cells",
            "installed_propeller_model",
            "manufacturer_curve_ids",
            "manufacturer_input_semantics",
            "throttle_percent_is_pixhawk_normalized",
            "throttle_percent_is_pwm",
            "curve_output_is_installed_executed_thrust",
            "curve_output_may_be_used_as_hardware_command",
        ),
        "rotor_prior",
    )
    flight_controller_data = _mapping(
        document["flight_controller"],
        "flight_controller",
    )
    flight_controller_keys: Tuple[str, ...] = (
        "hardware",
        "firmware_stack",
        "per_sample_execution_ack",
        "same_tick_baseline",
        "residual_ttl",
        "newton_interface",
    )
    if schema >= PRELIMINARY_SCHEMA_VERSION:
        flight_controller_keys += (
            "firmware_version",
            "airframe_type",
            "mount_orientation",
            "output_coordinate_frame",
            "quaternion_order",
            "imu_from_total_com_body_m",
        )
    _exact_keys(flight_controller_data, flight_controller_keys, "flight_controller")
    contact_data = _mapping(document["contact"], "contact")
    _exact_keys(
        contact_data,
        (
            "measurement_semantics",
            "lowstate_source",
            "detector_config_version",
            "detector_config_hash",
            "mapping_version",
            "mapping_hash",
            "algorithm_leg_order",
            "sdk_indices_by_leg",
            "signs_by_algorithm_leg",
            "contact_on_threshold_sdk_counts",
            "contact_off_threshold_sdk_counts",
            "filter_time_constant_s",
            "contact_confirm_s",
            "release_confirm_s",
            "maximum_sample_gap_s",
            "maximum_feedback_age_s",
            "minimum_consecutive_samples",
            "maximum_source_tick_jump",
            "normal_force_input_available",
            "tangential_force_input_available",
            "sdk_values_may_be_used_as_newtons",
        ),
        "contact",
    )
    try:
        contact_model = ContactModel(_nonempty(model["contact_model"], "contact_model"))
        semantics = FootForceSemantics(
            _nonempty(contact_data["measurement_semantics"], "measurement_semantics")
        )
    except ValueError as exc:
        raise PreliminaryModelError(str(exc)) from exc

    curve_ids_raw = rotor_data["manufacturer_curve_ids"]
    try:
        curve_ids_values = tuple(cast(Iterable[object], curve_ids_raw))
    except TypeError as exc:
        raise PreliminaryModelError(
            "rotor_prior.manufacturer_curve_ids must be a two-item list"
        ) from exc
    if len(curve_ids_values) != 2 or any(
        not isinstance(value, str) or not value.strip() for value in curve_ids_values
    ):
        raise PreliminaryModelError(
            "rotor_prior.manufacturer_curve_ids must contain two nonempty strings"
        )
    curve_ids = cast(Tuple[str, str], curve_ids_values)
    battery_series_cells_value = rotor_data["battery_series_cells"]
    if battery_series_cells_value is not None and (
        isinstance(battery_series_cells_value, bool)
        or not isinstance(battery_series_cells_value, int)
    ):
        raise PreliminaryModelError("rotor_prior.battery_series_cells must be null or an integer")

    cad_source: Optional[StepCadSourceProvenance] = None
    if cad_source_data is not None:
        cad_source = StepCadSourceProvenance(
            file_name=_nonempty(cad_source_data["file_name"], "cad_source.file_name"),
            sha256=_nonempty(cad_source_data["sha256"], "cad_source.sha256"),
            file_size_bytes=cad_source_data["file_size_bytes"],  # type: ignore[arg-type]
            step_schema=_nonempty(
                cad_source_data["step_schema"],
                "cad_source.step_schema",
            ),
            length_unit=_nonempty(
                cad_source_data["length_unit"],
                "cad_source.length_unit",
            ),
            brep_validation_passed=_boolean(
                cad_source_data["brep_validation_passed"],
                "cad_source.brep_validation_passed",
            ),
            material_density_properties_present=_boolean(
                cad_source_data["material_density_properties_present"],
                "cad_source.material_density_properties_present",
            ),
            mass_inertia_properties_present=_boolean(
                cad_source_data["mass_inertia_properties_present"],
                "cad_source.mass_inertia_properties_present",
            ),
            complete_system_scope_verified=_boolean(
                cad_source_data["complete_system_scope_verified"],
                "cad_source.complete_system_scope_verified",
            ),
        )

    go2_urdf: Optional[PinnedGo2UrdfPrior] = None
    if go2_urdf_data is not None:
        try:
            joint_order_raw = tuple(cast(Iterable[object], go2_urdf_data["sdk_joint_order"]))
            joint_positions_raw = tuple(
                cast(Iterable[object], go2_urdf_data["sdk_joint_positions_rad"])
            )
        except TypeError as exc:
            raise PreliminaryModelError("go2_urdf SDK joint order and pose must be lists") from exc
        if len(joint_order_raw) != 12 or any(
            not isinstance(value, str) or not value.strip() for value in joint_order_raw
        ):
            raise PreliminaryModelError(
                "go2_urdf.sdk_joint_order must contain twelve nonempty strings"
            )
        if len(joint_positions_raw) != 12:
            raise PreliminaryModelError(
                "go2_urdf.sdk_joint_positions_rad must contain twelve values"
            )
        joint_order = cast(Tuple[str, ...], joint_order_raw)
        joint_positions = tuple(
            _finite_real(value, "go2_urdf.sdk_joint_positions_rad") for value in joint_positions_raw
        )
        bundled_path = _config_local_file(
            source_path,
            go2_urdf_data["bundled_path"],
            "go2_urdf.bundled_path",
        )
        license_path = _config_local_file(
            source_path,
            go2_urdf_data["license_path"],
            "go2_urdf.license_path",
        )
        bundled_hash = _sha256_identity(
            go2_urdf_data["bundled_sha256"],
            "go2_urdf.bundled_sha256",
        )
        root_link = _nonempty(go2_urdf_data["root_link"], "go2_urdf.root_link")
        try:
            mass_properties = load_go2_urdf_mass_properties(
                bundled_path,
                expected_sha256=bundled_hash,
                expected_robot_name=_nonempty(
                    go2_urdf_data["expected_robot_name"],
                    "go2_urdf.expected_robot_name",
                ),
                root_link=root_link,
                joint_positions_rad=dict(zip(joint_order, joint_positions)),
            )
        except (Go2UrdfError, OSError) as exc:
            raise PreliminaryModelError(f"go2_urdf cannot be evaluated: {exc}") from exc
        go2_urdf = PinnedGo2UrdfPrior(
            source_repository=_nonempty(
                go2_urdf_data["source_repository"],
                "go2_urdf.source_repository",
            ),
            source_file_commit=_nonempty(
                go2_urdf_data["source_file_commit"],
                "go2_urdf.source_file_commit",
            ),
            model_quality=_nonempty(
                go2_urdf_data["model_quality"],
                "go2_urdf.model_quality",
            ),
            upstream_raw_sha256=_sha256_identity(
                go2_urdf_data["upstream_raw_sha256"],
                "go2_urdf.upstream_raw_sha256",
            ),
            bundled_path=bundled_path,
            bundled_sha256=bundled_hash,
            license_path=license_path,
            root_link=root_link,
            urdf_root_from_body_origin_B_m=_readonly_array(
                go2_urdf_data["urdf_root_from_body_origin_B_m"],
                (3,),
                "go2_urdf.urdf_root_from_body_origin_B_m",
            ),
            body_origin_B_alignment_identified=_boolean(
                go2_urdf_data["body_origin_B_alignment_identified"],
                "go2_urdf.body_origin_B_alignment_identified",
            ),
            reference_pose=_nonempty(
                go2_urdf_data["reference_pose"],
                "go2_urdf.reference_pose",
            ),
            sdk_joint_order=joint_order,
            sdk_joint_positions_rad=joint_positions,
            mass_properties=mass_properties,
        )

    offline_inertia_estimate: Optional[ProvisionalOfflineInertiaEstimate] = None
    if offline_inertia_data is not None:
        offline_inertia_estimate = ProvisionalOfflineInertiaEstimate(
            method=_nonempty(
                offline_inertia_data["method"],
                "offline_inertia_estimate.method",
            ),
            quality=_nonempty(
                offline_inertia_data["quality"],
                "offline_inertia_estimate.quality",
            ),
            reference_point=_nonempty(
                offline_inertia_data["reference_point"],
                "offline_inertia_estimate.reference_point",
            ),
            x8_system_mass_each_kg=_finite_real(
                offline_inertia_data["x8_system_mass_each_kg"],
                "offline_inertia_estimate.x8_system_mass_each_kg",
                positive=True,
            ),
            x8_count=offline_inertia_data["x8_count"],  # type: ignore[arg-type]
            remaining_added_mass_kg=_finite_real(
                offline_inertia_data["remaining_added_mass_kg"],
                "offline_inertia_estimate.remaining_added_mass_kg",
                minimum=0.0,
            ),
            remaining_added_mass_effective_com_from_body_origin_B_m=_readonly_array(
                offline_inertia_data["remaining_added_mass_effective_com_from_body_origin_B_m"],
                (3,),
                (
                    "offline_inertia_estimate."
                    "remaining_added_mass_effective_com_from_body_origin_B_m"
                ),
            ),
            remaining_mass_distribution_interval=_nonempty(
                offline_inertia_data["remaining_mass_distribution_interval"],
                "offline_inertia_estimate.remaining_mass_distribution_interval",
            ),
            nominal_body_kg_m2=_readonly_array(
                offline_inertia_data["nominal_body_kg_m2"],
                (3, 3),
                "offline_inertia_estimate.nominal_body_kg_m2",
            ),
            diagonal_lower_body_kg_m2=_readonly_array(
                offline_inertia_data["diagonal_lower_body_kg_m2"],
                (3,),
                "offline_inertia_estimate.diagonal_lower_body_kg_m2",
            ),
            diagonal_upper_body_kg_m2=_readonly_array(
                offline_inertia_data["diagonal_upper_body_kg_m2"],
                (3,),
                "offline_inertia_estimate.diagonal_upper_body_kg_m2",
            ),
            cross_terms_unbounded=_boolean(
                offline_inertia_data["cross_terms_unbounded"],
                "offline_inertia_estimate.cross_terms_unbounded",
            ),
            hardware_use_prohibited=_boolean(
                offline_inertia_data["hardware_use_prohibited"],
                "offline_inertia_estimate.hardware_use_prohibited",
            ),
        )

    if schema == LEGACY_PRELIMINARY_SCHEMA_VERSION:
        com_from_frame_center = None
        azimuth_reference = "BODY_POSITIVE_X_CCW_ABOUT_BODY_POSITIVE_Z"
        geometric_start_azimuth_deg = 45.0
        adjacent_arm_spacing_deg = 90.0
        geometric_sequence_rotor_order = GEOMETRIC_ROTOR_SEQUENCE
    else:
        com_from_frame_center = _optional_vector3(
            geometry_data["total_com_from_frame_center_body_m"],
            "geometry.total_com_from_frame_center_body_m",
        )
        azimuth_reference = _nonempty(
            geometry_data["azimuth_reference"],
            "geometry.azimuth_reference",
        )
        geometric_start_azimuth_deg = _finite_real(
            geometry_data["geometric_sequence_start_azimuth_deg"],
            "geometry.geometric_sequence_start_azimuth_deg",
        )
        adjacent_arm_spacing_deg = _finite_real(
            geometry_data["adjacent_arm_spacing_deg"],
            "geometry.adjacent_arm_spacing_deg",
            positive=True,
        )
        geometric_sequence_rotor_order = _tuple4_names(
            geometry_data["geometric_sequence_rotor_order"],
            "geometry.geometric_sequence_rotor_order",
        )

    if schema >= PRELIMINARY_SCHEMA_VERSION:
        body_origin_B_definition = _nonempty(
            geometry_data["body_origin_B_definition"],
            "geometry.body_origin_B_definition",
        )
        frame_center_O_from_body_origin_B_m = _readonly_array(
            geometry_data["frame_center_O_from_body_origin_B_m"],
            (3,),
            "geometry.frame_center_O_from_body_origin_B_m",
        )
        total_com_C_from_body_origin_B_m = _readonly_array(
            geometry_data["total_com_C_from_body_origin_B_m"],
            (3,),
            "geometry.total_com_C_from_body_origin_B_m",
        )
        rotor_frame_com_from_body_origin_B_m = _readonly_array(
            geometry_data["rotor_frame_com_from_body_origin_B_m"],
            (3,),
            "geometry.rotor_frame_com_from_body_origin_B_m",
        )
        rotor_plane_from_frame_center_O_m = _readonly_array(
            geometry_data["rotor_plane_from_frame_center_O_m"],
            (3,),
            "geometry.rotor_plane_from_frame_center_O_m",
        )
        frame_offsets_identified = _boolean(
            geometry_data["frame_offsets_identified"],
            "geometry.frame_offsets_identified",
        )
        first_rotor_positive_xy_label = _nonempty(
            geometry_data["first_rotor_positive_xy_label"],
            "geometry.first_rotor_positive_xy_label",
        )
    else:
        body_origin_B_definition = None
        frame_center_O_from_body_origin_B_m = None
        total_com_C_from_body_origin_B_m = None
        rotor_frame_com_from_body_origin_B_m = None
        rotor_plane_from_frame_center_O_m = None
        frame_offsets_identified = False
        first_rotor_positive_xy_label = None

    config = PreliminaryLandingModelConfig(
        source_path=source_path,
        schema_version=schema,
        profile=_nonempty(document["profile"], "profile"),
        physical_use_prohibited=_boolean(
            document["physical_use_prohibited"],
            "physical_use_prohibited",
        ),
        allow_hardware_output=_boolean(
            document["allow_hardware_output"],
            "allow_hardware_output",
        ),
        parameters_identified=_boolean(
            document["parameters_identified"],
            "parameters_identified",
        ),
        world_frame=_nonempty(frames["world"], "frames.world"),
        body_frame=_nonempty(frames["body"], "frames.body"),
        rotor_order=_tuple4_names(frames["rotor_order"], "frames.rotor_order"),
        dynamics_reference_point=(
            _nonempty(
                model["dynamics_reference_point"],
                "model.dynamics_reference_point",
            )
            if schema >= PRELIMINARY_SCHEMA_VERSION
            else "TOTAL_SYSTEM_COM_C"
        ),
        leg_kinematics_reference_point=(
            _nonempty(
                model["leg_kinematics_reference_point"],
                "model.leg_kinematics_reference_point",
            )
            if schema >= PRELIMINARY_SCHEMA_VERSION
            else "GO2_BODY_ORIGIN_B"
        ),
        gravity_m_per_s2=_finite_real(
            model["gravity_m_per_s2"],
            "model.gravity_m_per_s2",
            positive=True,
        ),
        cad_source=cad_source,
        go2_urdf=go2_urdf,
        mass=ProvisionalMassProperties(
            go2_nominal_kg=mass_data["go2_nominal_kg"],  # type: ignore[arg-type]
            added_system_nominal_kg=mass_data["added_system_nominal_kg"],  # type: ignore[arg-type]
            total_nominal_kg=mass_data["total_nominal_kg"],  # type: ignore[arg-type]
            total_uncertainty_kg=_optional_finite_real(
                mass_data["total_uncertainty_kg"],
                "mass.total_uncertainty_kg",
            ),
        ),
        inertia=CadBomInertiaProperties(
            method=_nonempty(inertia_data["method"], "inertia.method"),
            nominal_body_kg_m2=_optional_matrix3(
                inertia_data["nominal_body_kg_m2"],
                "inertia.nominal_body_kg_m2",
            ),
            uncertainty_body_kg_m2=_optional_matrix3(
                inertia_data["uncertainty_body_kg_m2"],
                "inertia.uncertainty_body_kg_m2",
            ),
            cad_bom_revision=_optional_nonempty(
                inertia_data["cad_bom_revision"],
                "inertia.cad_bom_revision",
            ),
            reference_pose=_optional_nonempty(
                inertia_data["reference_pose"],
                "inertia.reference_pose",
            ),
        ),
        offline_inertia_estimate=offline_inertia_estimate,
        geometry=FixedXGeometryPrior(
            total_com_from_frame_center_body_m=com_from_frame_center,
            body_origin_B_definition=body_origin_B_definition,
            frame_center_O_from_body_origin_B_m=frame_center_O_from_body_origin_B_m,
            total_com_C_from_body_origin_B_m=total_com_C_from_body_origin_B_m,
            rotor_frame_com_from_body_origin_B_m=rotor_frame_com_from_body_origin_B_m,
            rotor_plane_from_frame_center_O_m=rotor_plane_from_frame_center_O_m,
            frame_offsets_identified=frame_offsets_identified,
            horizontal_radius_from_frame_center_m=geometry_data[
                "horizontal_radius_from_frame_center_m"
            ],  # type: ignore[arg-type]
            rotor_plane_z_from_com_m=geometry_data["rotor_plane_z_from_com_m"],  # type: ignore[arg-type]
            horizontal_origin_assumption=_nonempty(
                geometry_data["horizontal_origin_assumption"],
                "geometry.horizontal_origin_assumption",
            ),
            azimuth_assumption=_nonempty(
                geometry_data["azimuth_assumption"],
                "geometry.azimuth_assumption",
            ),
            azimuth_reference=azimuth_reference,
            geometric_sequence_start_azimuth_deg=geometric_start_azimuth_deg,
            adjacent_arm_spacing_deg=adjacent_arm_spacing_deg,
            geometric_sequence_rotor_order=geometric_sequence_rotor_order,
            first_rotor_positive_xy_label=first_rotor_positive_xy_label,
            lever_arms_from_com_body_m=_readonly_array(
                geometry_data["lever_arms_from_com_body_m"],
                (4, 3),
                "geometry.lever_arms_from_com_body_m",
            ),
            thrust_directions_body=_readonly_array(
                geometry_data["thrust_directions_body"],
                (4, 3),
                "geometry.thrust_directions_body",
            ),
            lever_arm_uncertainty_m=(
                None
                if geometry_data["lever_arm_uncertainty_m"] is None
                else _readonly_array(
                    geometry_data["lever_arm_uncertainty_m"],
                    (4, 3),
                    "geometry.lever_arm_uncertainty_m",
                )
            ),
        ),
        rotor_prior=QuasiStaticRotorPrior(
            motor_system_model=_nonempty(
                rotor_data["motor_system_model"],
                "rotor_prior.motor_system_model",
            ),
            battery_series_cells=battery_series_cells_value,
            installed_propeller_model=_optional_nonempty(
                rotor_data["installed_propeller_model"],
                "rotor_prior.installed_propeller_model",
            ),
            manufacturer_curve_ids=curve_ids,
            manufacturer_input_semantics=_nonempty(
                rotor_data["manufacturer_input_semantics"],
                "rotor_prior.manufacturer_input_semantics",
            ),
            throttle_percent_is_pixhawk_normalized=_boolean(
                rotor_data["throttle_percent_is_pixhawk_normalized"],
                "rotor_prior.throttle_percent_is_pixhawk_normalized",
            ),
            throttle_percent_is_pwm=_boolean(
                rotor_data["throttle_percent_is_pwm"],
                "rotor_prior.throttle_percent_is_pwm",
            ),
            curve_output_is_installed_executed_thrust=_boolean(
                rotor_data["curve_output_is_installed_executed_thrust"],
                "rotor_prior.curve_output_is_installed_executed_thrust",
            ),
            curve_output_may_be_used_as_hardware_command=_boolean(
                rotor_data["curve_output_may_be_used_as_hardware_command"],
                "rotor_prior.curve_output_may_be_used_as_hardware_command",
            ),
            curves=(_manufacturer_curve(curve_ids[0]), _manufacturer_curve(curve_ids[1])),
        ),
        flight_controller=FlightControllerContractPrior(
            hardware=_nonempty(
                flight_controller_data["hardware"],
                "flight_controller.hardware",
            ),
            firmware_stack=_optional_nonempty(
                flight_controller_data["firmware_stack"],
                "flight_controller.firmware_stack",
            ),
            firmware_version=(
                _optional_nonempty(
                    flight_controller_data["firmware_version"],
                    "flight_controller.firmware_version",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            airframe_type=(
                _optional_nonempty(
                    flight_controller_data["airframe_type"],
                    "flight_controller.airframe_type",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            mount_orientation=(
                _optional_nonempty(
                    flight_controller_data["mount_orientation"],
                    "flight_controller.mount_orientation",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            output_coordinate_frame=(
                _optional_nonempty(
                    flight_controller_data["output_coordinate_frame"],
                    "flight_controller.output_coordinate_frame",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            quaternion_order=(
                _optional_nonempty(
                    flight_controller_data["quaternion_order"],
                    "flight_controller.quaternion_order",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            imu_from_total_com_body_m=(
                _optional_vector3(
                    flight_controller_data["imu_from_total_com_body_m"],
                    "flight_controller.imu_from_total_com_body_m",
                )
                if schema >= PRELIMINARY_SCHEMA_VERSION
                else None
            ),
            per_sample_execution_ack=_boolean(
                flight_controller_data["per_sample_execution_ack"],
                "flight_controller.per_sample_execution_ack",
            ),
            same_tick_baseline=_boolean(
                flight_controller_data["same_tick_baseline"],
                "flight_controller.same_tick_baseline",
            ),
            residual_ttl=_boolean(
                flight_controller_data["residual_ttl"],
                "flight_controller.residual_ttl",
            ),
            newton_interface=_boolean(
                flight_controller_data["newton_interface"],
                "flight_controller.newton_interface",
            ),
        ),
        contact=ScalarContactPrior(
            model=contact_model,
            measurement_semantics=semantics,
            lowstate_source=_optional_nonempty(
                contact_data["lowstate_source"],
                "contact.lowstate_source",
            ),
            detector_config_version=_optional_nonempty(
                contact_data["detector_config_version"],
                "contact.detector_config_version",
            ),
            detector_config_hash=_optional_nonempty(
                contact_data["detector_config_hash"],
                "contact.detector_config_hash",
            ),
            mapping_version=_optional_nonempty(
                contact_data["mapping_version"],
                "contact.mapping_version",
            ),
            mapping_hash=_optional_nonempty(
                contact_data["mapping_hash"],
                "contact.mapping_hash",
            ),
            algorithm_leg_order=_optional_names4(
                contact_data["algorithm_leg_order"],
                "contact.algorithm_leg_order",
            ),
            sdk_indices_by_leg=_optional_int4(
                contact_data["sdk_indices_by_leg"],
                "contact.sdk_indices_by_leg",
            ),
            signs_by_algorithm_leg=_optional_int4(
                contact_data["signs_by_algorithm_leg"],
                "contact.signs_by_algorithm_leg",
            ),
            contact_on_threshold_sdk_counts=_optional_float4(
                contact_data["contact_on_threshold_sdk_counts"],
                "contact.contact_on_threshold_sdk_counts",
            ),
            contact_off_threshold_sdk_counts=_optional_float4(
                contact_data["contact_off_threshold_sdk_counts"],
                "contact.contact_off_threshold_sdk_counts",
            ),
            filter_time_constant_s=_optional_finite_real(
                contact_data["filter_time_constant_s"],
                "contact.filter_time_constant_s",
            ),
            contact_confirm_s=_optional_finite_real(
                contact_data["contact_confirm_s"],
                "contact.contact_confirm_s",
            ),
            release_confirm_s=_optional_finite_real(
                contact_data["release_confirm_s"],
                "contact.release_confirm_s",
            ),
            maximum_sample_gap_s=_optional_finite_real(
                contact_data["maximum_sample_gap_s"],
                "contact.maximum_sample_gap_s",
            ),
            maximum_feedback_age_s=_optional_finite_real(
                contact_data["maximum_feedback_age_s"],
                "contact.maximum_feedback_age_s",
            ),
            minimum_consecutive_samples=_optional_int(
                contact_data["minimum_consecutive_samples"],
                "contact.minimum_consecutive_samples",
            ),
            maximum_source_tick_jump=_optional_int(
                contact_data["maximum_source_tick_jump"],
                "contact.maximum_source_tick_jump",
            ),
            normal_force_input_available=_boolean(
                contact_data["normal_force_input_available"],
                "contact.normal_force_input_available",
            ),
            tangential_force_input_available=_boolean(
                contact_data["tangential_force_input_available"],
                "contact.tangential_force_input_available",
            ),
            sdk_values_may_be_used_as_newtons=_boolean(
                contact_data["sdk_values_may_be_used_as_newtons"],
                "contact.sdk_values_may_be_used_as_newtons",
            ),
        ),
    )
    if config.contact.model is not contact_model:
        raise PreliminaryModelError("contact model construction failed")
    return config


__all__ = [
    "CadBomInertiaProperties",
    "ContactModel",
    "FlightControllerContractPrior",
    "FixedXGeometryPrior",
    "FootForceSemantics",
    "GRAM_FORCE_TO_NEWTON",
    "ManufacturerThrottleThrustCurve",
    "NormalOnlyVerticalState",
    "PinnedGo2UrdfPrior",
    "PreliminaryLandingModelConfig",
    "PreliminaryModelError",
    "ProvisionalMassProperties",
    "ProvisionalOfflineInertiaEstimate",
    "QuasiStaticRotorPrior",
    "ScalarContactPrior",
    "SdkCountContactDetection",
    "SdkCountContactDetector",
    "SdkCountContactDetectorConfig",
    "StepCadSourceProvenance",
    "calibrated_normal_force_scalars_n",
    "compute_sdk_count_contact_detector_hash",
    "ideal_arresting_normal_impulse_ns",
    "ideal_level_hover_thrust_per_rotor_n",
    "load_preliminary_landing_model",
    "manufacturer_static_thrust_n_at_throttle",
    "normal_only_vertical_acceleration_m_per_s2",
    "normal_only_vertical_impact_reset",
    "normal_only_vertical_step",
    "recompute_preliminary_derived_document",
]
