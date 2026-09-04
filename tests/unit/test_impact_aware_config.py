"""Fail-closed assembly tests for impact-aware YAML profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from aerogo2.landing.impact_aware.admittance import (
    AdmittanceConfig,
    AxisAlignedWorkspace,
)
from aerogo2.landing.impact_aware.config import (
    ImpactAwareConfigError,
    load_impact_aware_config,
)
from aerogo2.landing.impact_aware.contact_detection import (
    ContactDetectorConfig,
    FootContactDetector,
)
from aerogo2.landing.impact_aware.go2_foot_force import Go2FootForceCalibration
from aerogo2.landing.impact_aware.nlp import (
    ContactForceLimits,
    MPCWeights,
    SLSQPSettings,
    StateBounds,
)
from aerogo2.landing.impact_aware.rotor_safety import (
    RotorCorrectionBlender,
    RotorCorrectionSafetyConfig,
)
from aerogo2.landing.impact_aware.types import (
    FixedDeployedRotorGeometry,
    ImpactLimits,
    ReducedDynamicsConfig,
    RotorActuatorConfig,
    RotorAerodynamics,
)


def _demo_path(project_root: Path) -> Path:
    return project_root / "configs" / "impact_aware_mpc_demo.yaml"


def _template_path(project_root: Path) -> Path:
    return project_root / "configs" / "impact_aware_mpc.template.yaml"


def _demo_document(project_root: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load(_demo_path(project_root).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_yaml(tmp_path: Path, document: Dict[str, Any], name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _production_document(project_root: Path) -> Dict[str, Any]:
    document = _demo_document(project_root)
    document["profile"] = "production"
    document["parameters_identified"] = True
    document["allow_hardware_output"] = False
    document.pop("physical_use_prohibited")
    return document


def test_synthetic_demo_requires_explicit_opt_in(project_root: Path) -> None:
    with pytest.raises(ImpactAwareConfigError, match="allow_synthetic=True"):
        load_impact_aware_config(_demo_path(project_root))


def test_ragged_numeric_vector_uses_public_config_error(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["mpc"]["state_bounds"]["lower"] = [[0.0], [1.0, 2.0]]
    malformed = _write_yaml(tmp_path, document, "ragged.yaml")

    with pytest.raises(ImpactAwareConfigError, match="state_bounds.lower"):
        load_impact_aware_config(malformed, allow_synthetic=True)


def test_contact_confirmation_threshold_must_be_reachable_in_force_model(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["contact"]["normal_force_max_n"][0] = 19.0
    document["contact"]["contact_on_threshold_n"][0] = 20.0
    malformed = _write_yaml(tmp_path, document, "unreachable_contact.yaml")

    with pytest.raises(ImpactAwareConfigError, match="cannot exceed"):
        load_impact_aware_config(malformed, allow_synthetic=True)


def test_synthetic_demo_builds_every_static_runtime_object(project_root: Path) -> None:
    bundle = load_impact_aware_config(
        _demo_path(project_root),
        allow_synthetic=True,
    )

    assert bundle.source_path == _demo_path(project_root).resolve()
    assert bundle.profile == "synthetic_demo"
    assert bundle.is_synthetic
    assert not bundle.hardware_output_permitted
    assert bundle.physical_use_prohibited
    assert not bundle.parameters_identified
    assert not bundle.allow_hardware_output
    assert bundle.world_frame == "ENU_Z_UP"
    assert bundle.body_frame == "X_FORWARD_Y_LEFT_Z_UP"
    assert bundle.rotor_order == ("RR", "LF", "LR", "RF")
    assert bundle.rotor_configuration == "FIXED_DEPLOYED_LOCKED"
    assert bundle.rotor_reference_origin == "TOTAL_SYSTEM_COM_C"
    assert bundle.dynamics_reference_point == "TOTAL_SYSTEM_COM_C"

    assert isinstance(bundle.rotor_geometry, FixedDeployedRotorGeometry)
    assert isinstance(bundle.rotor_aerodynamics, RotorAerodynamics)
    assert isinstance(bundle.rotor_actuator, RotorActuatorConfig)
    assert isinstance(bundle.dynamics, ReducedDynamicsConfig)
    assert isinstance(bundle.impact_limits, ImpactLimits)
    assert isinstance(bundle.rotor_correction_safety, RotorCorrectionSafetyConfig)
    assert isinstance(bundle.contact_detector, ContactDetectorConfig)
    assert isinstance(bundle.go2_foot_force_calibration, Go2FootForceCalibration)
    assert bundle.go2_foot_force_calibration.algorithm_leg_order == (
        "SYNTH_0",
        "SYNTH_1",
        "SYNTH_2",
        "SYNTH_3",
    )
    assert isinstance(bundle.contact_force_limits, ContactForceLimits)
    assert isinstance(bundle.mpc_weights, MPCWeights)
    assert isinstance(bundle.mpc_state_bounds, StateBounds)
    assert isinstance(bundle.solver_settings, SLSQPSettings)
    assert all(isinstance(value, AdmittanceConfig) for value in bundle.admittance_configs)
    assert all(isinstance(value, AxisAlignedWorkspace) for value in bundle.admittance_workspaces)
    first_admittance = bundle.admittance_configs[0]
    np.testing.assert_allclose(np.diag(first_admittance.stance_stiffness), [120.0, 120.0, 180.0])
    np.testing.assert_allclose(first_admittance.force_error_deadband_n, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(first_admittance.correction_position_limit_m, [0.08, 0.08, 0.12])
    np.testing.assert_allclose(
        first_admittance.correction_velocity_limit_m_per_s,
        [0.5, 0.5, 0.8],
    )
    assert first_admittance.contact_release_policy == "reset"

    assert bundle.mpc_horizon_steps == 4
    assert bundle.mpc_dt_s == pytest.approx(0.05)
    assert bundle.mpc_state_bounds.lower.shape == (5, 16)
    assert bundle.mpc_state_bounds.upper.shape == (5, 16)
    assert bundle.mpc_state_bounds.soft_mask.shape == (4, 16)
    assert bundle.mpc_weights.tracking.shape == (12, 12)
    assert np.array_equal(
        np.diag(bundle.mpc_weights.tracking),
        [20, 20, 40, 10, 10, 30, 15, 15, 15, 5, 5, 5],
    )
    assert bundle.admittance_leg_order == (
        "SYNTH_0",
        "SYNTH_1",
        "SYNTH_2",
        "SYNTH_3",
    )
    assert bundle.contact_detector_config is bundle.contact_detector
    assert bundle.rotor_correction_safety_config is bundle.rotor_correction_safety

    first_detector = bundle.new_contact_detector()
    second_detector = bundle.new_contact_detector()
    first_blender = bundle.new_rotor_correction_blender()
    second_blender = bundle.new_rotor_correction_blender()
    assert isinstance(first_detector, FootContactDetector)
    assert isinstance(first_blender, RotorCorrectionBlender)
    assert first_detector is not second_detector
    assert first_blender is not second_blender

    with pytest.raises(ValueError):
        bundle.rotor_actuator.thrust_max_n[0] = 999.0
    with pytest.raises(ValueError):
        bundle.rotor_geometry.lever_arms_from_com_body_m[0, 0] = 999.0
    with pytest.raises(ValueError):
        bundle.rotor_correction_safety.thrust_max_n[0] = 999.0
    with pytest.raises(ValueError):
        bundle.contact_detector.contact_on_threshold_n[0] = 0.0


def test_synthetic_profile_is_never_accepted_for_hardware(project_root: Path) -> None:
    with pytest.raises(ImpactAwareConfigError, match="prohibited for hardware"):
        load_impact_aware_config(
            _demo_path(project_root),
            allow_synthetic=True,
            for_hardware=True,
        )


def test_production_template_fails_closed_until_identified(project_root: Path) -> None:
    with pytest.raises(ImpactAwareConfigError, match="hardware use is not configured"):
        load_impact_aware_config(
            _template_path(project_root),
            for_hardware=True,
        )

    with pytest.raises(
        ImpactAwareConfigError,
        match="lever_arms_from_com_body_m.*cannot be null",
    ):
        load_impact_aware_config(_template_path(project_root))


def test_complete_production_profile_can_be_assembled_for_offline_validation(
    project_root: Path,
    tmp_path: Path,
) -> None:
    path = _write_yaml(tmp_path, _production_document(project_root), "production.yaml")
    bundle = load_impact_aware_config(path)

    assert bundle.source_path == path.resolve()
    assert bundle.profile == "production"
    assert not bundle.is_synthetic
    assert bundle.parameters_identified
    assert not bundle.allow_hardware_output
    assert not bundle.physical_use_prohibited


def test_hardware_output_is_unavailable_even_with_edited_attestations(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _production_document(project_root)
    document["allow_hardware_output"] = True
    path = _write_yaml(tmp_path, document, "edited-attestations.yaml")

    with pytest.raises(ImpactAwareConfigError, match="hardware use is not configured"):
        load_impact_aware_config(path, for_hardware=True)
    with pytest.raises(ImpactAwareConfigError, match="must remain false"):
        load_impact_aware_config(path)


def test_hardware_attestations_do_not_hide_missing_calibration(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _production_document(project_root)
    document["dynamics"]["mass_kg"] = None
    path = _write_yaml(tmp_path, document)

    with pytest.raises(ImpactAwareConfigError, match="dynamics.mass_kg.*cannot be null"):
        load_impact_aware_config(path)


def test_unknown_nested_keys_and_wrong_boolean_types_are_rejected(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["rotor"]["actuator"]["unreviewed_limit"] = 1.0
    unknown_path = _write_yaml(tmp_path, document, "unknown.yaml")
    with pytest.raises(ImpactAwareConfigError, match="unknown keys.*unreviewed_limit"):
        load_impact_aware_config(unknown_path, allow_synthetic=True)

    document = _demo_document(project_root)
    document["mpc"]["state_bounds"]["soft_mask"][0] = 0
    boolean_path = _write_yaml(tmp_path, document, "boolean.yaml")
    with pytest.raises(ImpactAwareConfigError, match="soft_mask.*only booleans"):
        load_impact_aware_config(boolean_path, allow_synthetic=True)


def test_numeric_configuration_rejects_boolean_and_string_coercion(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["contact"]["contact_confirm_s"] = True
    boolean_path = _write_yaml(tmp_path, document, "numeric-bool.yaml")
    with pytest.raises(ImpactAwareConfigError, match="contact_confirm_s.*real number"):
        load_impact_aware_config(boolean_path, allow_synthetic=True)

    document = _demo_document(project_root)
    document["rotor"]["correction_safety"]["maximum_correction_n"] = ["10"] * 4
    string_path = _write_yaml(tmp_path, document, "numeric-string.yaml")
    with pytest.raises(ImpactAwareConfigError, match="maximum_correction_n.*real numeric"):
        load_impact_aware_config(string_path, allow_synthetic=True)


def test_rotor_order_is_the_fixed_aerogo2_fc_mapping(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["frames"]["rotor_order"] = ["LF", "RR", "LR", "RF"]
    path = _write_yaml(tmp_path, document, "wrong-order.yaml")

    with pytest.raises(ImpactAwareConfigError, match=r"exactly \[RR, LF, LR, RF\]"):
        load_impact_aware_config(path, allow_synthetic=True)


def test_go2_scalar_force_adapter_is_optional_but_hashed_when_present(
    project_root: Path,
    tmp_path: Path,
) -> None:
    external = _demo_document(project_root)
    external["contact"].pop("go2_lowstate_force_adapter")
    external_path = _write_yaml(tmp_path, external, "external-force-source.yaml")
    bundle = load_impact_aware_config(external_path, allow_synthetic=True)
    assert bundle.go2_foot_force_calibration is None

    modified_mapping = _demo_document(project_root)
    modified_mapping["contact"]["go2_lowstate_force_adapter"]["sdk_indices_by_leg"] = [1, 0, 2, 3]
    mapping_path = _write_yaml(tmp_path, modified_mapping, "wrong-foot-map-hash.yaml")
    with pytest.raises(ImpactAwareConfigError, match="mapping_hash"):
        load_impact_aware_config(mapping_path, allow_synthetic=True)

    wrong_order = _demo_document(project_root)
    wrong_order["admittance"]["leg_order"] = [
        "SYNTH_1",
        "SYNTH_0",
        "SYNTH_2",
        "SYNTH_3",
    ]
    order_path = _write_yaml(tmp_path, wrong_order, "wrong-foot-leg-order.yaml")
    with pytest.raises(ImpactAwareConfigError, match="leg_order must exactly match"):
        load_impact_aware_config(order_path, allow_synthetic=True)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("configuration", "FOLDING", "FIXED_DEPLOYED_LOCKED"),
        ("reference_origin", "FRAME_CENTRE_O", "TOTAL_SYSTEM_COM_C"),
    ],
)
def test_only_fixed_deployed_com_referenced_rotor_geometry_is_accepted(
    project_root: Path,
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    document = _demo_document(project_root)
    document["rotor"]["geometry"][key] = value
    path = _write_yaml(tmp_path, document, f"bad-{key}.yaml")

    with pytest.raises(ImpactAwareConfigError, match=message):
        load_impact_aware_config(path, allow_synthetic=True)


@pytest.mark.parametrize(
    "dynamic_key",
    ["fold_angle_rad", "hinge_length_m", "positions_body_m"],
)
def test_folding_and_legacy_geometry_inputs_are_rejected(
    project_root: Path,
    tmp_path: Path,
    dynamic_key: str,
) -> None:
    document = _demo_document(project_root)
    document["rotor"]["geometry"][dynamic_key] = 0.0
    path = _write_yaml(tmp_path, document, f"forbidden-{dynamic_key}.yaml")

    with pytest.raises(ImpactAwareConfigError, match=f"unknown keys.*{dynamic_key}"):
        load_impact_aware_config(path, allow_synthetic=True)


def test_legacy_dynamic_geometry_schema_fails_closed(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["schema_version"] = 1
    path = _write_yaml(tmp_path, document, "legacy-schema.yaml")

    with pytest.raises(ImpactAwareConfigError, match="schema_version must be integer 3"):
        load_impact_aware_config(path, allow_synthetic=True)


def test_horizontal_enu_gravity_and_hover_authority_are_cross_checked(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["dynamics"]["gravity_world_m_per_s2"] = [0.1, 0.0, -9.80665]
    gravity_path = _write_yaml(tmp_path, document, "bad-gravity.yaml")
    with pytest.raises(ImpactAwareConfigError, match="horizontal ENU impact model"):
        load_impact_aware_config(gravity_path, allow_synthetic=True)

    document = _demo_document(project_root)
    document["rotor"]["geometry"]["lever_arms_from_com_body_m"] = [[0.0, 0.0, 0.1]] * 4
    authority_path = _write_yaml(tmp_path, document, "no-authority.yaml")
    with pytest.raises(ImpactAwareConfigError, match="rank-4 thrust allocation"):
        load_impact_aware_config(authority_path, allow_synthetic=True)


def test_nonfinite_derived_hover_wrench_fails_closed(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["dynamics"]["mass_kg"] = 1.0e308
    path = _write_yaml(tmp_path, document, "overflow-hover-wrench.yaml")

    with pytest.raises(ImpactAwareConfigError, match="nonfinite derived level-hover"):
        load_impact_aware_config(path, allow_synthetic=True)


def test_actuator_rate_bounds_must_admit_a_steady_command(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _demo_document(project_root)
    document["rotor"]["actuator"]["thrust_rate_min_n_per_s"] = [1.0] * 4
    document["rotor"]["actuator"]["thrust_rate_max_n_per_s"] = [2.0] * 4
    path = _write_yaml(tmp_path, document, "no-steady-rate.yaml")

    with pytest.raises(ImpactAwareConfigError, match="contain zero"):
        load_impact_aware_config(path, allow_synthetic=True)


def test_duplicate_yaml_keys_are_rejected_before_assembly(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "schema_version: 1\nschema_version: 1\nprofile: production\n",
        encoding="utf-8",
    )
    with pytest.raises(ImpactAwareConfigError, match="duplicate key.*schema_version"):
        load_impact_aware_config(path)
