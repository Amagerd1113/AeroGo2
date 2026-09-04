from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from aerogo2.common.config import (
    AppConfig,
    compute_go2_joint_mapping_hash,
    load_config,
)
from aerogo2.common.exceptions import ConfigurationError
from aerogo2.common.immutable import deep_thaw
from aerogo2.common.models import (
    Go2LowLevelStatus,
    Go2MotorFeedback,
    Go2Status,
    LowCmdOwnershipState,
    snapshot_to_dict,
)

JOINT_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
MOTOR_IDS = tuple(range(12))
DIRECTIONS = (1,) * 12
ZERO_OFFSETS = (0.0,) * 12


def _complete_low_level_config() -> Dict[str, Any]:
    mapping_version = "bench-fixture-v1"
    mapping_hash = compute_go2_joint_mapping_hash(
        mapping_version,
        JOINT_NAMES,
        MOTOR_IDS,
        DIRECTIONS,
        ZERO_OFFSETS,
    )
    return {
        "enabled": True,
        "low_state_topic": "rt/lowstate",
        "low_command_topic": "rt/lowcmd",
        "send_period_s": 0.002,
        "maximum_jitter_s": 0.0005,
        "low_state_max_age_s": 0.02,
        "target_ttl_s": 0.05,
        "acquire_timeout_s": 5.0,
        "release_timeout_s": 5.0,
        "safe_hold_policy": "configured_pose",
        "safe_hold_pose_rad": [0.0] * 12,
        "safe_hold_position_tolerance_rad": [0.02] * 12,
        "safe_hold_velocity_tolerance_rad_s": [0.05] * 12,
        "tracking_position_error_limit_rad": [0.2] * 12,
        "safe_hold_ack_timeout_s": 0.5,
        "restore_mode_form": "0",
        "restore_mode_name": "normal",
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
        "joint_names": list(JOINT_NAMES),
        "motor_ids": list(MOTOR_IDS),
        "directions": list(DIRECTIONS),
        "zero_offsets_rad": list(ZERO_OFFSETS),
        "q_min_rad": [-2.0] * 12,
        "q_max_rad": [2.0] * 12,
        "dq_max_rad_s": [1.0] * 12,
        "maximum_delta_q_rad": [0.01] * 12,
        "kp": [1.0] * 12,
        "kd": [0.1] * 12,
        "tau_ff_nm": [0.0] * 12,
        "tau_limit_nm": [1.0] * 12,
        "feedback_loss_degraded_kp": [0.1] * 12,
        "feedback_loss_degraded_kd": [0.01] * 12,
        "feedback_loss_degraded_tau_ff_nm": [0.0] * 12,
        "firmware_torque_limit_nm": [0.8] * 12,
        "firmware_torque_clamp_verified": True,
        "temperature_limit_c": [60.0] * 12,
    }


def _write_with_low_level(
    tmp_path: Path,
    app_config: AppConfig,
    low_level: Dict[str, Any],
) -> Path:
    raw = deep_thaw(app_config.raw)
    raw["go2"]["low_level"] = low_level
    path = tmp_path / "go2-low-level.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_packaged_configuration_keeps_low_level_fail_closed(app_config: AppConfig) -> None:
    config = app_config.go2.low_level

    assert config.enabled is False
    assert config.observe_only_enabled is False
    assert config.observation_enabled is False
    assert config.low_state_topic is None
    assert config.mapping_hash is None
    assert config.motor_ids is None
    assert config.safe_hold_pose_rad is None


def test_enabling_low_level_requires_every_explicit_value(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    low_level = dict(deep_thaw(app_config.raw)["go2"]["low_level"])
    low_level["enabled"] = True
    path = _write_with_low_level(tmp_path, app_config, low_level)

    with pytest.raises(ConfigurationError, match="Missing explicit configuration value"):
        load_config(path)


def test_complete_reviewed_low_level_configuration_loads(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    raw_low_level = _complete_low_level_config()
    loaded = load_config(_write_with_low_level(tmp_path, app_config, raw_low_level))

    config = loaded.go2.low_level
    assert config.enabled is True
    assert config.low_state_topic == "rt/lowstate"
    assert config.low_command_topic == "rt/lowcmd"
    assert config.joint_names == JOINT_NAMES
    assert config.motor_ids == MOTOR_IDS
    assert config.directions == DIRECTIONS
    assert config.mapping_hash == raw_low_level["mapping_hash"]


def test_observe_only_configuration_needs_mapping_but_not_actuation_values(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    complete = _complete_low_level_config()
    observe_keys = {
        "low_state_topic",
        "low_state_max_age_s",
        "mapping_version",
        "mapping_hash",
        "joint_names",
        "motor_ids",
        "directions",
        "zero_offsets_rad",
    }
    observe_only = {key: value for key, value in complete.items() if key in observe_keys}
    observe_only.update(enabled=False, observe_only_enabled=True)

    loaded = load_config(_write_with_low_level(tmp_path, app_config, observe_only))
    config = loaded.go2.low_level

    assert config.observation_enabled
    assert not config.enabled
    assert config.low_state_topic == "rt/lowstate"
    assert config.q_min_rad is None
    assert config.kp is None
    assert config.kd is None
    assert config.low_command_topic is None


def test_observe_only_configuration_still_requires_verified_mapping(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    observe_only = {
        "enabled": False,
        "observe_only_enabled": True,
        "low_state_topic": "rt/lowstate",
        "low_state_max_age_s": 0.02,
    }

    with pytest.raises(ConfigurationError, match="while LowState observation is enabled"):
        load_config(_write_with_low_level(tmp_path, app_config, observe_only))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: item.update(mapping_hash="sha256:" + "0" * 64), "does not match"),
        (lambda item: item.update(motor_ids=[0] * 12), "permutation.*0..11"),
        (lambda item: item.update(directions=[0] * 12), "only -1 or 1"),
        (lambda item: item.update(q_max_rad=[-3.0] * 12), "must be less than"),
        (lambda item: item.update(target_ttl_s=0.001), "at least send_period_s"),
        (
            lambda item: item.update(safe_hold_ack_timeout_s=6.0),
            "must not exceed release_timeout_s",
        ),
        (lambda item: item.update(maximum_jitter_s=0.002), "less than send_period_s"),
        (
            lambda item: item.update(firmware_torque_clamp_verified=False),
            "firmware_torque_clamp_verified must be true",
        ),
        (
            lambda item: item.update(feedback_loss_degraded_kp=[2.0] * 12),
            r"feedback_loss_degraded_kp\[0\] must not exceed kp\[0\]",
        ),
        (
            lambda item: item.update(feedback_loss_degraded_kd=[0.2] * 12),
            r"feedback_loss_degraded_kd\[0\] must not exceed kd\[0\]",
        ),
        (
            lambda item: item.update(firmware_torque_limit_nm=[1.1] * 12),
            r"firmware_torque_limit_nm\[0\] must not exceed tau_limit_nm\[0\]",
        ),
        (
            lambda item: item.update(feedback_loss_degraded_tau_ff_nm=[0.9] * 12),
            r"feedback_loss_degraded_tau_ff_nm\[0\].*firmware_torque_limit_nm\[0\]",
        ),
        (lambda item: item.update(unsafe_unreviewed_field=True), "unknown keys"),
    ],
)
def test_low_level_cross_checks_fail_closed(
    tmp_path: Path,
    app_config: AppConfig,
    mutator: Any,
    message: str,
) -> None:
    low_level = _complete_low_level_config()
    mutator(low_level)
    path = _write_with_low_level(tmp_path, app_config, low_level)

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_low_level_status_defaults_are_safe_and_models_are_immutable() -> None:
    go2 = Go2Status()

    assert go2.low_level_status.ownership_state is LowCmdOwnershipState.DISABLED
    assert go2.low_level_status.healthy is False
    assert go2.low_level_status.writer_alive is False
    assert go2.low_level_status.watchdog_healthy is False
    assert go2.low_level_status.owns_lowcmd is False

    feedback = Go2MotorFeedback(motor_id=3, joint_name="FL_hip", lost=False)
    status = Go2LowLevelStatus(
        connected=True,
        ownership_state=LowCmdOwnershipState.HOLDING,
        owner_epoch=7,
        healthy=True,
        writer_alive=True,
        watchdog_healthy=True,
        high_level_released=True,
        mapping_hash_verified=True,
        motors=[feedback],  # type: ignore[arg-type]
    )
    assert status.motors == (feedback,)
    assert status.owns_lowcmd is True
    with pytest.raises(FrozenInstanceError):
        status.owner_epoch = 8  # type: ignore[misc]


def test_low_level_status_serializes_without_nonfinite_age() -> None:
    converted = snapshot_to_dict(Go2LowLevelStatus())  # type: ignore[arg-type]

    assert converted["low_state_age_s"] is None
    assert converted["ownership_state"] == "DISABLED"
