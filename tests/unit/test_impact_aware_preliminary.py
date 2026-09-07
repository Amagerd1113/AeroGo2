"""Safety and equation tests for the offline preliminary landing model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional

import numpy as np
import pytest
import yaml

from aerogo2.common.models import Go2FootForceFeedback
from aerogo2.landing.impact_aware.go2_foot_force import (
    CalibratedGo2NormalForceSample,
    Go2FootForceSource,
    compute_go2_foot_force_mapping_hash,
)
from aerogo2.landing.impact_aware.preliminary import (
    GRAM_FORCE_TO_NEWTON,
    ContactModel,
    FootForceSemantics,
    NormalOnlyVerticalState,
    PreliminaryModelError,
    calibrated_normal_force_scalars_n,
    compute_sdk_count_contact_detector_hash,
    ideal_arresting_normal_impulse_ns,
    ideal_level_hover_thrust_per_rotor_n,
    load_preliminary_landing_model,
    manufacturer_static_thrust_n_at_throttle,
    normal_only_vertical_acceleration_m_per_s2,
    normal_only_vertical_impact_reset,
    normal_only_vertical_step,
)


def _path(project_root: Path) -> Path:
    return project_root / "configs" / "impact_aware_preliminary.yaml"


def _document(project_root: Path) -> Dict[str, Any]:
    value = yaml.safe_load(_path(project_root).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, document: Dict[str, Any]) -> Path:
    config_root = Path(__file__).resolve().parents[2] / "configs"
    go2_urdf = document.get("go2_urdf")
    if isinstance(go2_urdf, dict):
        for key in ("bundled_path", "license_path"):
            value = go2_urdf.get(key)
            if isinstance(value, str):
                copyfile(config_root / value, tmp_path / value)
    path = tmp_path / "preliminary.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _configured_rotor_document(
    project_root: Path,
    *,
    cells: int = 12,
    propeller: str = "MFP 30x11S",
) -> Dict[str, Any]:
    document = _document(project_root)
    document["rotor_prior"]["battery_series_cells"] = cells
    document["rotor_prior"]["installed_propeller_model"] = propeller
    return document


def _count_detector_document(project_root: Path) -> Dict[str, Any]:
    document = _document(project_root)
    contact = document["contact"]
    leg_order = ["FR", "FL", "RR", "RL"]
    indices = [1, 0, 2, 3]
    signs = [1, 1, 1, 1]
    mapping_version = "provisional-go2-count-map-v1"
    mapping_hash = compute_go2_foot_force_mapping_hash(
        mapping_version,
        leg_order,
        indices,
    )
    detector_values = {
        "detector_config_version": "provisional-go2-count-detector-v1",
        "lowstate_source": "foot_force",
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
        "algorithm_leg_order": leg_order,
        "sdk_indices_by_leg": indices,
        "signs_by_algorithm_leg": signs,
        "contact_on_threshold_sdk_counts": [100.0, 100.0, 100.0, 100.0],
        "contact_off_threshold_sdk_counts": [50.0, 50.0, 50.0, 50.0],
        "filter_time_constant_s": 0.001,
        "contact_confirm_s": 0.1,
        "release_confirm_s": 0.1,
        "maximum_sample_gap_s": 0.2,
        "maximum_feedback_age_s": 0.02,
        "minimum_consecutive_samples": 2,
        "maximum_source_tick_jump": 10,
    }
    detector_hash = compute_sdk_count_contact_detector_hash(**detector_values)
    contact.update(
        {
            **detector_values,
            "detector_config_hash": detector_hash,
        }
    )
    return document


def _feedback(
    *,
    timestamp_s: float,
    source_tick: int,
    raw: tuple[int, int, int, int],
    estimated: tuple[int, int, int, int] = (0, 0, 0, 0),
    raw_valid: bool = True,
    estimated_valid: bool = True,
    receipt_sequence: Optional[int] = None,
    subscription_generation: int = 1,
) -> Go2FootForceFeedback:
    return Go2FootForceFeedback(
        receipt_timestamp_s=timestamp_s,
        receipt_sequence=source_tick if receipt_sequence is None else receipt_sequence,
        subscription_generation=subscription_generation,
        source_tick=source_tick,
        source_tick_valid=True,
        source_tick_monotonic=True,
        raw_sdk_int16=raw,
        estimated_sdk_int16=estimated,
        raw_valid=raw_valid,
        estimated_valid=estimated_valid,
    )


def test_preliminary_profile_requires_opt_in_and_never_allows_hardware(
    project_root: Path,
) -> None:
    with pytest.raises(PreliminaryModelError, match="allow_provisional=True"):
        load_preliminary_landing_model(_path(project_root))
    with pytest.raises(PreliminaryModelError, match="prohibited for hardware"):
        load_preliminary_landing_model(
            _path(project_root),
            allow_provisional=True,
            for_hardware=True,
        )


def test_supplied_mass_and_fixed_x_geometry_are_loaded_without_inventing_inertia(
    project_root: Path,
) -> None:
    config = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )

    assert config.schema_version == 4
    assert config.profile == "provisional_offline"
    assert config.mass.go2_nominal_kg == pytest.approx(16.087)
    assert config.mass.added_system_nominal_kg == pytest.approx(10.0)
    assert config.mass.total_nominal_kg == pytest.approx(26.087)
    assert config.mass.total_uncertainty_kg is None
    assert not config.mass.measured_with_uncertainty
    assert not config.inertia.identified
    with pytest.raises(PreliminaryModelError, match="CAD/BOM inertia"):
        config.inertia.require_identified()
    assert config.inertia.reference_pose == "UNITREE_GO2_STANDARD_STANDING_POSE_PROVISIONAL"
    assert config.go2_urdf is not None
    assert config.go2_urdf.model_quality == "URDF_MODEL_ESTIMATE"
    assert config.go2_urdf.mass_properties.mass_kg == pytest.approx(16.087)
    assert config.go2_urdf.mass_properties.total_link_count == 42
    assert config.go2_urdf.mass_properties.inertial_link_count == 33
    assert config.go2_urdf.mass_properties.positive_mass_link_count == 31
    assert not config.go2_urdf.body_origin_B_alignment_identified
    assert config.offline_inertia_estimate is not None
    assert config.offline_inertia_estimate.quality == "PROVISIONAL_OFFLINE_ONLY"
    assert not config.offline_inertia_estimate.usable_for_hardware
    assert (
        config.offline_inertia_estimate.remaining_added_mass_effective_com_from_body_origin_B_m
        == pytest.approx(np.array([0.004847170654, 0.0, 0.191098213990]))
    )
    assert config.offline_inertia_estimate.nominal_body_kg_m2 == pytest.approx(
        np.array(
            [
                [1.960517510297, 0.000121660000, -0.021263416909],
                [0.000121660000, 2.296378784538, -0.000031200000],
                [-0.021263416909, -0.000031200000, 3.741901685208],
            ]
        )
    )

    assert config.cad_source is not None
    assert config.cad_source.file_name == "碳管铝件板总装打开.STEP"
    assert config.cad_source.sha256 == (
        "sha256:824d1a45ee9e136b392aa6bc1c66bec69d052c8e4b7a662a34e8685dc3f43d9a"
    )
    assert config.cad_source.file_size_bytes == 341_022_183
    assert config.cad_source.step_schema == "STEP_AP203_CONFIG_CONTROL_DESIGN"
    assert not config.cad_source.directly_supports_identified_inertia

    diagonal = 0.665 / np.sqrt(2.0)
    assert config.geometry.total_com_from_frame_center_body_m == pytest.approx(
        np.array([0.0, 0.0, -0.0472])
    )
    assert config.geometry.frame_center_O_from_body_origin_B_m == pytest.approx(
        np.array([0.0, 0.0, 0.0972])
    )
    assert config.geometry.total_com_C_from_body_origin_B_m == pytest.approx(
        np.array([0.0, 0.0, 0.0500])
    )
    assert config.geometry.rotor_plane_from_frame_center_O_m == pytest.approx(
        np.array([0.0, 0.0, 0.0200])
    )
    assert config.geometry.first_rotor_positive_xy_label == "LF"
    assert config.geometry.azimuths_by_rotor_order_deg == pytest.approx((225.0, 45.0, 135.0, 315.0))
    assert config.geometry.geometric_sequence_rotor_order == ("LF", "LR", "RR", "RF")
    assert np.allclose(
        config.geometry.lever_arms_from_com_body_m,
        np.array(
            [
                [-diagonal, -diagonal, 0.0672],
                [+diagonal, +diagonal, 0.0672],
                [-diagonal, +diagonal, 0.0672],
                [+diagonal, -diagonal, 0.0672],
            ]
        ),
        rtol=0.0,
        atol=1e-9,
    )
    assert np.array_equal(
        config.geometry.thrust_directions_body,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
    )
    assert config.contact.model is ContactModel.NORMAL_ONLY_VERTICAL
    assert (
        config.contact.measurement_semantics is FootForceSemantics.UNCALIBRATED_CONTACT_EVENT_ONLY
    )
    assert not config.contact.contact_event_detection_configured
    assert not config.hardware_output_permitted
    assert not config.full_six_dof_ready


def test_com_offset_and_azimuth_metadata_are_explicit_and_cross_checked(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    geometry = config.geometry
    assert geometry.total_com_from_frame_center_body_m is not None
    # p_R/O = p_C/O + p_R/C = -0.0472 + 0.0672 = 0.020 m.
    rotor_centres_from_frame = (
        geometry.lever_arms_from_com_body_m
        + geometry.total_com_from_frame_center_body_m[np.newaxis, :]
    )
    assert rotor_centres_from_frame[:, 2] == pytest.approx(np.full(4, 0.020))

    sorted_azimuths = sorted(geometry.azimuths_by_rotor_order_deg)
    wrapped = sorted_azimuths + [sorted_azimuths[0] + 360.0]
    assert np.diff(wrapped) == pytest.approx(np.full(4, 90.0))

    bad_shape = _document(project_root)
    bad_shape["geometry"]["total_com_from_frame_center_body_m"] = [0.0, 0.050]
    with pytest.raises(PreliminaryModelError, match=r"shape \(3,\)"):
        load_preliminary_landing_model(
            _write(tmp_path, bad_shape),
            allow_provisional=True,
        )

    horizontal_conflict = _document(project_root)
    horizontal_conflict["geometry"]["total_com_from_frame_center_body_m"] = [0.001, 0.0, 0.050]
    with pytest.raises(PreliminaryModelError, match="x/y must remain zero"):
        load_preliminary_landing_model(
            _write(tmp_path, horizontal_conflict),
            allow_provisional=True,
        )

    angle_conflict = _document(project_root)
    angle_conflict["geometry"]["geometric_sequence_start_azimuth_deg"] = 0.0
    with pytest.raises(PreliminaryModelError, match="do not match"):
        load_preliminary_landing_model(
            _write(tmp_path, angle_conflict),
            allow_provisional=True,
        )


def test_offline_inertia_and_first_mass_moment_are_recomputed_on_load(
    project_root: Path,
    tmp_path: Path,
) -> None:
    wrong_balance_point = _document(project_root)
    wrong_balance_point["offline_inertia_estimate"][
        "remaining_added_mass_effective_com_from_body_origin_B_m"
    ] = [0.0, 0.0, 0.0972]
    with pytest.raises(PreliminaryModelError, match="do not reproduce total COM C"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_balance_point),
            allow_provisional=True,
        )

    wrong_lower = _document(project_root)
    wrong_lower["offline_inertia_estimate"]["diagonal_lower_body_kg_m2"][0] += 1.0e-4
    with pytest.raises(PreliminaryModelError, match="lower diagonal"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_lower),
            allow_provisional=True,
        )

    wrong_nominal = _document(project_root)
    wrong_nominal["offline_inertia_estimate"]["nominal_body_kg_m2"][0][0] += 1.0e-4
    with pytest.raises(PreliminaryModelError, match="nominal inertia"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_nominal),
            allow_provisional=True,
        )


def test_legacy_v1_preliminary_config_remains_readable_without_inventing_com(
    project_root: Path,
    tmp_path: Path,
) -> None:
    legacy = _document(project_root)
    legacy["schema_version"] = 1
    legacy.pop("cad_source")
    legacy.pop("go2_urdf")
    legacy.pop("offline_inertia_estimate")
    legacy["model"].pop("dynamics_reference_point")
    legacy["model"].pop("leg_kinematics_reference_point")
    for key in (
        "total_com_from_frame_center_body_m",
        "azimuth_reference",
        "geometric_sequence_start_azimuth_deg",
        "adjacent_arm_spacing_deg",
        "geometric_sequence_rotor_order",
        "body_origin_B_definition",
        "frame_center_O_from_body_origin_B_m",
        "total_com_C_from_body_origin_B_m",
        "rotor_frame_com_from_body_origin_B_m",
        "rotor_plane_from_frame_center_O_m",
        "frame_offsets_identified",
        "first_rotor_positive_xy_label",
    ):
        legacy["geometry"].pop(key)
    for key in (
        "firmware_version",
        "airframe_type",
        "mount_orientation",
        "output_coordinate_frame",
        "quaternion_order",
        "imu_from_total_com_body_m",
    ):
        legacy["flight_controller"].pop(key)
    legacy["geometry"]["azimuth_assumption"] = "SYMMETRIC_X_45_DEG_PROVISIONAL"

    config = load_preliminary_landing_model(
        _write(tmp_path, legacy),
        allow_provisional=True,
    )

    assert config.schema_version == 1
    assert config.cad_source is None
    assert config.geometry.total_com_from_frame_center_body_m is None
    assert config.geometry.azimuths_by_rotor_order_deg == pytest.approx((225.0, 45.0, 135.0, 315.0))


def test_previous_v2_config_remains_readable_without_inventing_new_frame_data(
    project_root: Path,
    tmp_path: Path,
) -> None:
    previous = _document(project_root)
    previous["schema_version"] = 2
    previous.pop("go2_urdf")
    previous.pop("offline_inertia_estimate")
    previous["model"].pop("dynamics_reference_point")
    previous["model"].pop("leg_kinematics_reference_point")
    for key in (
        "body_origin_B_definition",
        "frame_center_O_from_body_origin_B_m",
        "total_com_C_from_body_origin_B_m",
        "rotor_frame_com_from_body_origin_B_m",
        "rotor_plane_from_frame_center_O_m",
        "frame_offsets_identified",
        "first_rotor_positive_xy_label",
    ):
        previous["geometry"].pop(key)
    for key in (
        "firmware_version",
        "airframe_type",
        "mount_orientation",
        "output_coordinate_frame",
        "quaternion_order",
        "imu_from_total_com_body_m",
    ):
        previous["flight_controller"].pop(key)

    config = load_preliminary_landing_model(
        _write(tmp_path, previous),
        allow_provisional=True,
    )

    assert config.schema_version == 2
    assert config.cad_source is not None
    assert config.go2_urdf is None
    assert config.offline_inertia_estimate is None
    assert config.geometry.frame_center_O_from_body_origin_B_m is None


def test_inconsistent_schema_v3_inertia_prior_is_retired(
    project_root: Path,
    tmp_path: Path,
) -> None:
    retired = _document(project_root)
    retired["schema_version"] = 3
    with pytest.raises(PreliminaryModelError, match="schema_version must be one of 1, 2, or 4"):
        load_preliminary_landing_model(
            _write(tmp_path, retired),
            allow_provisional=True,
        )


def test_geometric_sequence_must_name_every_rotor_once(
    project_root: Path,
    tmp_path: Path,
) -> None:
    invalid = _document(project_root)
    invalid["geometry"]["geometric_sequence_rotor_order"] = ["LF", "LR", "RR", "UNKNOWN"]
    with pytest.raises(PreliminaryModelError, match="contain RR, LF, LR, RF exactly once"):
        load_preliminary_landing_model(
            _write(tmp_path, invalid),
            allow_provisional=True,
        )


def test_mass_component_sum_and_geometry_are_cross_checked(
    project_root: Path,
    tmp_path: Path,
) -> None:
    wrong_mass = _document(project_root)
    wrong_mass["mass"]["total_nominal_kg"] = 26.0
    with pytest.raises(PreliminaryModelError, match="must equal"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_mass),
            allow_provisional=True,
        )

    wrong_geometry = _document(project_root)
    wrong_geometry["geometry"]["lever_arms_from_com_body_m"][0][0] = -0.665
    with pytest.raises(PreliminaryModelError, match="do not match"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_geometry),
            allow_provisional=True,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("physical_use_prohibited", False),
        ("allow_hardware_output", True),
        ("parameters_identified", True),
    ],
)
def test_editing_attestation_flags_cannot_upgrade_preliminary_profile(
    project_root: Path,
    tmp_path: Path,
    key: str,
    value: bool,
) -> None:
    document = _document(project_root)
    document[key] = value
    with pytest.raises(PreliminaryModelError):
        load_preliminary_landing_model(
            _write(tmp_path, document),
            allow_provisional=True,
        )


def test_inertia_nominal_and_uncertainty_must_be_commissioned_together(
    project_root: Path,
    tmp_path: Path,
) -> None:
    partial = _document(project_root)
    partial["inertia"]["nominal_body_kg_m2"] = np.eye(3).tolist()
    with pytest.raises(PreliminaryModelError, match="supplied together"):
        load_preliminary_landing_model(
            _write(tmp_path, partial),
            allow_provisional=True,
        )

    complete = deepcopy(partial)
    complete["inertia"]["uncertainty_body_kg_m2"] = (0.1 * np.eye(3)).tolist()
    complete["inertia"]["cad_bom_revision"] = "cad-bom-test-v1"
    complete["inertia"]["reference_pose"] = "nominal-landing-pose"
    config = load_preliminary_landing_model(
        _write(tmp_path, complete),
        allow_provisional=True,
    )
    nominal, uncertainty = config.inertia.require_identified()
    assert nominal == pytest.approx(np.eye(3))
    assert uncertainty == pytest.approx(0.1 * np.eye(3))

    false_exact = deepcopy(complete)
    false_exact["inertia"]["uncertainty_body_kg_m2"] = np.zeros((3, 3)).tolist()
    with pytest.raises(PreliminaryModelError, match="cannot be all zero"):
        load_preliminary_landing_model(
            _write(tmp_path, false_exact),
            allow_provisional=True,
        )

    false_exact_geometry = _document(project_root)
    false_exact_geometry["geometry"]["lever_arm_uncertainty_m"] = np.zeros((4, 3)).tolist()
    with pytest.raises(PreliminaryModelError, match="cannot be all zero"):
        load_preliminary_landing_model(
            _write(tmp_path, false_exact_geometry),
            allow_provisional=True,
        )


def test_uncalibrated_sdk_contact_configuration_cannot_claim_newtons(
    project_root: Path,
    tmp_path: Path,
) -> None:
    document = _document(project_root)
    document["contact"]["sdk_values_may_be_used_as_newtons"] = True
    with pytest.raises(PreliminaryModelError, match="cannot advertise physical force"):
        load_preliminary_landing_model(
            _write(tmp_path, document),
            allow_provisional=True,
        )

    document = _document(project_root)
    document["contact"]["lowstate_source"] = "foot_force_est"
    with pytest.raises(PreliminaryModelError, match="supplied together"):
        load_preliminary_landing_model(
            _write(tmp_path, document),
            allow_provisional=True,
        )


def test_calibrated_api_returns_scalars_without_inventing_world_vectors() -> None:
    with pytest.raises(TypeError, match="CalibratedGo2NormalForceSample"):
        calibrated_normal_force_scalars_n((1, 2, 3, 4))  # type: ignore[arg-type]

    sample = CalibratedGo2NormalForceSample(
        algorithm_leg_order=("FR", "FL", "RR", "RL"),
        normal_forces_n=(10.0, 20.0, 30.0, 40.0),
        source=Go2FootForceSource.ESTIMATED_INT16,
        source_tick=1,
        receipt_timestamp_s=1.0,
        receipt_sequence=1,
        subscription_generation=1,
        mapping_hash="sha256:" + "1" * 64,
        calibration_hash="sha256:" + "2" * 64,
    )
    forces = calibrated_normal_force_scalars_n(sample)
    assert forces == (10.0, 20.0, 30.0, 40.0)


def test_vertical_only_equations_use_mass_but_not_inertia(project_root: Path) -> None:
    config = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    hover_thrust = config.mass.total_nominal_kg * config.gravity_m_per_s2
    acceleration = normal_only_vertical_acceleration_m_per_s2(
        total_rotor_vertical_force_world_n=hover_thrust,
        total_contact_normal_force_n=0.0,
        config=config,
    )
    assert acceleration == pytest.approx(0.0, abs=1e-12)

    before = NormalOnlyVerticalState(
        height_world_m=0.1,
        vertical_velocity_world_m_per_s=-0.5,
    )
    required = ideal_arresting_normal_impulse_ns(-0.5, config)
    assert required == pytest.approx(26.087 * 0.5)
    after = normal_only_vertical_impact_reset(
        before,
        total_normal_impulse_ns=required,
        config=config,
    )
    assert after.height_world_m == before.height_world_m
    assert after.vertical_velocity_world_m_per_s == pytest.approx(0.0)

    stepped = normal_only_vertical_step(before, acceleration_m_per_s2=2.0, dt_s=0.1)
    assert stepped.height_world_m == pytest.approx(0.06)
    assert stepped.vertical_velocity_world_m_per_s == pytest.approx(-0.3)


def test_rotor_curve_selection_requires_confirmed_battery_and_propeller(
    project_root: Path,
    tmp_path: Path,
) -> None:
    unknown = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    assert unknown.rotor_prior.battery_series_cells is None
    assert unknown.rotor_prior.installed_propeller_model is None
    with pytest.raises(PreliminaryModelError, match="must both be confirmed"):
        unknown.rotor_prior.select_installed_curve()

    incomplete = _document(project_root)
    incomplete["rotor_prior"]["battery_series_cells"] = 12
    with pytest.raises(PreliminaryModelError, match="null or confirmed together"):
        load_preliminary_landing_model(
            _write(tmp_path, incomplete),
            allow_provisional=True,
        )

    unsupported_cells = _configured_rotor_document(project_root, cells=13)
    with pytest.raises(PreliminaryModelError, match="null, 12, or 14"):
        load_preliminary_landing_model(
            _write(tmp_path, unsupported_cells),
            allow_provisional=True,
        )

    wrong_propeller = load_preliminary_landing_model(
        _write(
            tmp_path,
            _configured_rotor_document(project_root, propeller="UNKNOWN PROP"),
        ),
        allow_provisional=True,
    )
    with pytest.raises(PreliminaryModelError, match="different propeller"):
        wrong_propeller.rotor_prior.select_installed_curve()


def test_reviewed_x8_curves_interpolate_without_extrapolation(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config_12s = load_preliminary_landing_model(
        _write(tmp_path, _configured_rotor_document(project_root, cells=12)),
        allow_provisional=True,
    )
    curve_12s = config_12s.rotor_prior.select_installed_curve()
    assert curve_12s.report_test_voltage_v == pytest.approx(46.0)
    assert manufacturer_static_thrust_n_at_throttle(37.0, config_12s) == pytest.approx(
        2472.0 * GRAM_FORCE_TO_NEWTON
    )
    # The official table deliberately contains a 37%-39% plateau.
    assert manufacturer_static_thrust_n_at_throttle(38.0, config_12s) == pytest.approx(
        2472.0 * GRAM_FORCE_TO_NEWTON
    )
    assert manufacturer_static_thrust_n_at_throttle(40.5, config_12s) == pytest.approx(
        2720.0 * GRAM_FORCE_TO_NEWTON
    )
    with pytest.raises(PreliminaryModelError, match="extrapolation is prohibited"):
        manufacturer_static_thrust_n_at_throttle(34.9, config_12s)
    with pytest.raises(PreliminaryModelError, match="extrapolation is prohibited"):
        manufacturer_static_thrust_n_at_throttle(100.1, config_12s)

    config_14s = load_preliminary_landing_model(
        _write(tmp_path, _configured_rotor_document(project_root, cells=14)),
        allow_provisional=True,
    )
    curve_14s = config_14s.rotor_prior.select_installed_curve()
    assert curve_14s.report_test_voltage_v == pytest.approx(54.0)
    assert manufacturer_static_thrust_n_at_throttle(38.0, config_14s) == pytest.approx(
        4965.0 * GRAM_FORCE_TO_NEWTON
    )


def test_manufacturer_semantics_and_fc_contract_cannot_be_edited_into_authority(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    assert config.flight_controller.hardware == "PIXHAWK_6X"
    assert config.flight_controller.firmware_stack is None
    assert not config.flight_controller.residual_contract_ready

    rotor_edit = _document(project_root)
    rotor_edit["rotor_prior"]["throttle_percent_is_pixhawk_normalized"] = True
    with pytest.raises(PreliminaryModelError, match="must remain false"):
        load_preliminary_landing_model(
            _write(tmp_path, rotor_edit),
            allow_provisional=True,
        )

    fc_edit = _document(project_root)
    fc_edit["flight_controller"]["per_sample_execution_ack"] = True
    with pytest.raises(PreliminaryModelError, match="must remain false"):
        load_preliminary_landing_model(
            _write(tmp_path, fc_edit),
            allow_provisional=True,
        )


def test_ideal_hover_value_is_diagnostic_not_a_curve_limit(project_root: Path) -> None:
    config = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    assert ideal_level_hover_thrust_per_rotor_n(config) == pytest.approx(63.9565196375)
    assert not hasattr(config.rotor_prior, "thrust_limit_n")


def test_sdk_count_detector_requires_complete_count_configuration(
    project_root: Path,
    tmp_path: Path,
) -> None:
    incomplete = load_preliminary_landing_model(
        _path(project_root),
        allow_provisional=True,
    )
    with pytest.raises(PreliminaryModelError, match="cannot be created"):
        incomplete.contact.new_sdk_count_contact_detector()

    document = _count_detector_document(project_root)
    document["contact"]["mapping_hash"] = "sha256:" + "0" * 64
    with pytest.raises(PreliminaryModelError, match="mapping_hash"):
        load_preliminary_landing_model(
            _write(tmp_path, document),
            allow_provisional=True,
        )


def test_sdk_count_detector_maps_selects_filters_and_confirms_events(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _write(tmp_path, _count_detector_document(project_root)),
        allow_provisional=True,
    )
    detector = config.contact.new_sdk_count_contact_detector()

    first = detector.update(
        _feedback(
            timestamp_s=1.0,
            source_tick=1,
            raw=(0, 0, 0, 0),
            estimated=(300, 300, 300, 300),
        ),
        now_s=1.0,
    )
    assert first.lowstate_source == "foot_force"
    assert first.receipt_sequence == 1
    assert first.subscription_generation == 1
    assert first.mapping_hash == config.contact.mapping_hash
    assert first.ordered_raw_sdk_counts == (0, 0, 0, 0)
    assert first.signed_sdk_counts == (0, 0, 0, 0)
    assert first.filtered_signed_sdk_counts == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert first.contacts == (False, False, False, False)

    rising = detector.update(
        _feedback(timestamp_s=1.01, source_tick=2, raw=(20, 120, 30, 40)),
        now_s=1.01,
    )
    # Configured map [1,0,2,3] places SDK index 1 at algorithm FR.
    assert rising.ordered_raw_sdk_counts == (120, 20, 30, 40)
    assert rising.contacts == (False, False, False, False)
    landed = detector.update(
        _feedback(timestamp_s=1.12, source_tick=3, raw=(20, 120, 30, 40)),
        now_s=1.12,
    )
    assert landed.contacts == (True, False, False, False)
    assert landed.contact_confirmed_events == (True, False, False, False)
    assert landed.contact_on_threshold_first_crossing_s[0] == pytest.approx(1.01)
    assert landed.contact_confirmed_at_s[0] == pytest.approx(1.12)

    detector.update(
        _feedback(timestamp_s=1.13, source_tick=4, raw=(20, 20, 30, 40)),
        now_s=1.13,
    )
    released = detector.update(
        _feedback(timestamp_s=1.24, source_tick=5, raw=(20, 20, 30, 40)),
        now_s=1.24,
    )
    assert released.contacts == (False, False, False, False)
    assert released.contact_released_events == (True, False, False, False)
    assert released.contact_off_threshold_first_crossing_s[0] == pytest.approx(1.13)
    assert released.contact_release_confirmed_at_s[0] == pytest.approx(1.24)


def test_sdk_count_detector_rejects_nonmonotonic_or_wrong_source_data(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _write(tmp_path, _count_detector_document(project_root)),
        allow_provisional=True,
    )
    detector = config.contact.new_sdk_count_contact_detector()
    detector.update(
        _feedback(timestamp_s=1.0, source_tick=1, raw=(0, 0, 0, 0)),
        now_s=1.0,
    )
    with pytest.raises(PreliminaryModelError, match="increase strictly"):
        detector.update(
            _feedback(timestamp_s=1.0, source_tick=2, raw=(0, 0, 0, 0)),
            now_s=1.0,
        )

    detector.reset()
    with pytest.raises(PreliminaryModelError, match="foot_force field is invalid"):
        detector.update(
            _feedback(
                timestamp_s=2.0,
                source_tick=3,
                raw=(0, 0, 0, 0),
                raw_valid=False,
                estimated_valid=True,
            ),
            now_s=2.0,
        )


def test_sdk_count_detector_age_gap_and_identity_faults_latch_until_reset(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _write(tmp_path, _count_detector_document(project_root)),
        allow_provisional=True,
    )
    detector = config.contact.new_sdk_count_contact_detector()
    with pytest.raises(PreliminaryModelError, match="older than"):
        detector.update(
            _feedback(timestamp_s=1.0, source_tick=1, raw=(0, 0, 0, 0)),
            now_s=1.021,
        )
    assert detector.reset_required
    with pytest.raises(PreliminaryModelError, match="reset required"):
        detector.update(
            _feedback(timestamp_s=1.01, source_tick=2, raw=(0, 0, 0, 0)),
            now_s=1.01,
        )

    detector.reset()
    with pytest.raises(PreliminaryModelError, match="in the future"):
        detector.update(
            _feedback(timestamp_s=1.1, source_tick=3, raw=(0, 0, 0, 0)),
            now_s=1.099,
        )
    assert detector.reset_required

    detector.reset()
    detector.update(
        _feedback(
            timestamp_s=2.0,
            source_tick=10,
            receipt_sequence=10,
            raw=(0, 0, 0, 0),
        ),
        now_s=2.0,
    )
    with pytest.raises(PreliminaryModelError, match="sample gap"):
        detector.update(
            _feedback(
                timestamp_s=2.201,
                source_tick=11,
                receipt_sequence=11,
                raw=(0, 0, 0, 0),
            ),
            now_s=2.201,
        )
    assert detector.reset_required
    with pytest.raises(PreliminaryModelError, match="reset required"):
        detector.update(
            _feedback(
                timestamp_s=2.21,
                source_tick=12,
                receipt_sequence=12,
                raw=(0, 0, 0, 0),
            ),
            now_s=2.21,
        )

    detector.reset()
    detector.update(
        _feedback(
            timestamp_s=3.0,
            source_tick=20,
            receipt_sequence=20,
            raw=(0, 0, 0, 0),
        ),
        now_s=3.0,
    )
    with pytest.raises(PreliminaryModelError, match="receipt_sequence"):
        detector.update(
            _feedback(
                timestamp_s=3.01,
                source_tick=21,
                receipt_sequence=20,
                raw=(0, 0, 0, 0),
            ),
            now_s=3.01,
        )


def test_sdk_count_detector_uses_rfc1982_ticks_and_maximum_jump(
    project_root: Path,
    tmp_path: Path,
) -> None:
    config = load_preliminary_landing_model(
        _write(tmp_path, _count_detector_document(project_root)),
        allow_provisional=True,
    )
    detector = config.contact.new_sdk_count_contact_detector()
    detector.update(
        _feedback(
            timestamp_s=1.0,
            source_tick=0xFFFFFFFE,
            receipt_sequence=1,
            raw=(0, 0, 0, 0),
        ),
        now_s=1.0,
    )
    wrapped = detector.update(
        _feedback(
            timestamp_s=1.01,
            source_tick=1,
            receipt_sequence=2,
            raw=(0, 0, 0, 0),
        ),
        now_s=1.01,
    )
    assert wrapped.source_tick == 1

    detector.reset()
    detector.update(
        _feedback(
            timestamp_s=2.0,
            source_tick=1,
            receipt_sequence=3,
            raw=(0, 0, 0, 0),
        ),
        now_s=2.0,
    )
    with pytest.raises(PreliminaryModelError, match="maximum_source_tick_jump"):
        detector.update(
            _feedback(
                timestamp_s=2.01,
                source_tick=12,
                receipt_sequence=4,
                raw=(0, 0, 0, 0),
            ),
            now_s=2.01,
        )

    detector.reset()
    detector.update(
        _feedback(
            timestamp_s=3.0,
            source_tick=100,
            receipt_sequence=5,
            raw=(0, 0, 0, 0),
        ),
        now_s=3.0,
    )
    with pytest.raises(PreliminaryModelError, match="RFC1982"):
        detector.update(
            _feedback(
                timestamp_s=3.01,
                source_tick=99,
                receipt_sequence=6,
                raw=(0, 0, 0, 0),
            ),
            now_s=3.01,
        )


def test_count_detector_hash_covers_sign_and_sign_is_applied(
    project_root: Path,
    tmp_path: Path,
) -> None:
    stale_hash = _count_detector_document(project_root)
    stale_hash["contact"]["signs_by_algorithm_leg"][0] = -1
    with pytest.raises(PreliminaryModelError, match="detector_config_hash"):
        load_preliminary_landing_model(
            _write(tmp_path, stale_hash),
            allow_provisional=True,
        )

    signed = _count_detector_document(project_root)
    contact = signed["contact"]
    contact["signs_by_algorithm_leg"] = [-1, 1, 1, 1]
    hash_fields = {
        key: contact[key]
        for key in (
            "detector_config_version",
            "lowstate_source",
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
        )
    }
    contact["detector_config_hash"] = compute_sdk_count_contact_detector_hash(**hash_fields)
    config = load_preliminary_landing_model(
        _write(tmp_path, signed),
        allow_provisional=True,
    )
    result = config.contact.new_sdk_count_contact_detector().update(
        _feedback(timestamp_s=1.0, source_tick=1, raw=(20, 120, 30, 40)),
        now_s=1.0,
    )
    assert result.ordered_raw_sdk_counts == (120, 20, 30, 40)
    assert result.signed_sdk_counts == (-120, 20, 30, 40)


def test_uncertainty_inertia_physics_and_azimuth_assumptions_fail_closed(
    project_root: Path,
    tmp_path: Path,
) -> None:
    zero_uncertainty = _document(project_root)
    zero_uncertainty["mass"]["total_uncertainty_kg"] = 0.0
    with pytest.raises(PreliminaryModelError, match="strictly positive"):
        load_preliminary_landing_model(
            _write(tmp_path, zero_uncertainty),
            allow_provisional=True,
        )

    nonphysical_inertia = _document(project_root)
    nonphysical_inertia["inertia"].update(
        {
            "nominal_body_kg_m2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
            "uncertainty_body_kg_m2": [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            "cad_bom_revision": "test",
            "reference_pose": "test-pose",
        }
    )
    with pytest.raises(PreliminaryModelError, match="triangle inequality"):
        load_preliminary_landing_model(
            _write(tmp_path, nonphysical_inertia),
            allow_provisional=True,
        )

    wrong_azimuth = _document(project_root)
    wrong_azimuth["geometry"]["azimuth_assumption"] = "UNKNOWN"
    with pytest.raises(PreliminaryModelError, match="provisional symmetric X frame"):
        load_preliminary_landing_model(
            _write(tmp_path, wrong_azimuth),
            allow_provisional=True,
        )


def test_duplicate_yaml_key_is_rejected(project_root: Path, tmp_path: Path) -> None:
    original = _path(project_root).read_text(encoding="utf-8")
    duplicate = original.replace(
        "schema_version: 4",
        "schema_version: 4\nschema_version: 4",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(PreliminaryModelError, match="duplicate key"):
        load_preliminary_landing_model(path, allow_provisional=True)
