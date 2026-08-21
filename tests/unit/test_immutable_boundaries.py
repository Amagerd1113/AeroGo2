"""Safety facts are defensively copied and recursively immutable."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from aerogo2.common.config import AppConfig, EscConfig
from aerogo2.common.enums import CommandStatus, F446EventType, SystemState
from aerogo2.common.immutable import deep_thaw
from aerogo2.common.models import (
    F446Event,
    HealthReport,
    PixhawkStatus,
    RCStatus,
    SystemSnapshot,
    snapshot_to_dict,
)
from aerogo2.common.results import CommandResult, OperationResult
from aerogo2.logging.schemas import to_jsonable


def test_status_mappings_copy_inputs_and_reject_mutation() -> None:
    channels = {1: 1500}
    rpm = {1: 0.0}
    online = {1: True}
    status = PixhawkStatus(rc_channels=channels, esc_rpm=rpm, esc_online=online)
    rc = RCStatus(channels=channels)

    channels[1] = 1900
    rpm[1] = 1000.0
    online[1] = False

    assert status.rc_channels[1] == 1500
    assert status.esc_rpm[1] == 0.0
    assert status.esc_online[1] is True
    assert rc.channels[1] == 1500
    with pytest.raises(TypeError):
        status.esc_rpm[1] = 10.0  # type: ignore[index]
    with pytest.raises(TypeError):
        rc.channels[1] = 1900  # type: ignore[index]


def test_event_and_health_mappings_are_recursively_frozen() -> None:
    values: dict[str, Any] = {"nested": {"samples": [1, 2]}}
    checks = {"pixhawk": True}
    event = F446Event(F446EventType.STATUS, "status", 10.0, values=values)
    report = HealthReport(True, checks)

    values["nested"]["samples"].append(3)
    checks["pixhawk"] = False

    nested = event.values["nested"]
    assert nested["samples"] == (1, 2)
    assert report.checks["pixhawk"] is True
    with pytest.raises(TypeError):
        nested["new"] = "value"
    with pytest.raises(TypeError):
        report.checks["pixhawk"] = False  # type: ignore[index]


def test_config_copies_and_deeply_freezes_raw_and_esc_slots(app_config: AppConfig) -> None:
    raw: dict[str, Any] = {
        "nested": {"items": [1, 2]},
    }
    slots = {1: "LR"}
    frozen = replace(app_config, raw=raw, esc=EscConfig(slots))

    raw["nested"]["items"].append(3)
    slots[1] = "RF"

    assert frozen.raw["nested"]["items"] == (1, 2)
    assert frozen.esc.slots[1] == "LR"
    with pytest.raises(TypeError):
        frozen.raw["nested"]["other"] = True
    with pytest.raises(TypeError):
        frozen.esc.slots[1] = "RF"  # type: ignore[index]
    assert deep_thaw(frozen.raw) == {"nested": {"items": [1, 2]}}


def test_result_objects_copy_and_deeply_freeze_caller_data() -> None:
    source: dict[str, Any] = {"checks": [{"passed": True}]}
    operation = OperationResult.success("ok", source)
    command = CommandResult(CommandStatus.SUCCESS, "ok", source)

    source["checks"][0]["passed"] = False

    assert operation.data["checks"][0]["passed"] is True
    assert command.data["checks"][0]["passed"] is True
    with pytest.raises(TypeError):
        operation.data["checks"][0]["passed"] = False


def test_frozen_snapshot_serializers_emit_plain_json_primitives() -> None:
    snapshot = SystemSnapshot(
        timestamp=10.0,
        state=SystemState.BOOT_SAFE,
        pixhawk=PixhawkStatus(
            timestamp=10.0,
            rc_channels={1: 1500},
            esc_rpm={1: 0.0},
            esc_online={1: True},
        ),
        rc=RCStatus(timestamp=10.0, channels={1: 1500}),
    )

    snapshot_data = snapshot_to_dict(snapshot)
    generic_data = to_jsonable(snapshot)

    assert snapshot_data["pixhawk"]["rc_channels"] == {"1": 1500}
    assert generic_data["pixhawk"]["rc_channels"] == {"1": 1500}
    json.dumps(snapshot_data, allow_nan=False)
    json.dumps(generic_data, allow_nan=False)
