from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aerogo2.common.config import AppConfig, load_config
from aerogo2.simulation.world import SimulationWorld


@pytest.mark.parametrize("timestamp", [0.0, -1.0, float("nan"), 11.0])
def test_rc_check_query_rejects_invalid_or_future_timestamps(
    app_config: AppConfig,
    timestamp: float,
) -> None:
    world = SimulationWorld(app_config)
    snapshot = replace(
        world.manager.snapshot,
        timestamp=10.0,
        rc=replace(world.manager.snapshot.rc, connected=True, timestamp=timestamp),
    )
    world.manager._snapshot = snapshot

    result = world.manager.query("rc check")

    assert result["valid"] is False
    assert result["fresh"] is False
    assert math.isinf(result["message_age_s"])


def test_rc_check_query_uses_inclusive_configured_freshness_boundary(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    now = 10.0
    timeout = app_config.safety.rc_timeout_s
    at_boundary = replace(
        world.manager.snapshot,
        timestamp=now,
        rc=replace(
            world.manager.snapshot.rc,
            connected=True,
            timestamp=now - timeout,
        ),
    )
    world.manager._snapshot = at_boundary

    fresh = world.manager.query("rc check")

    world.manager._snapshot = replace(
        at_boundary,
        rc=replace(at_boundary.rc, timestamp=now - timeout - 0.001),
    )
    stale = world.manager.query("rc check")

    assert fresh["valid"] is True
    assert fresh["fresh"] is True
    assert fresh["message_age_s"] == pytest.approx(timeout)
    assert stale["valid"] is True
    assert stale["fresh"] is False
    assert stale["message_age_s"] > timeout


def test_rc_check_query_rejects_the_first_float_beyond_timeout(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    now = 10.0
    timeout = app_config.safety.rc_timeout_s
    just_stale = math.nextafter(now - timeout, -math.inf)
    world.manager._snapshot = replace(
        world.manager.snapshot,
        timestamp=now,
        rc=replace(world.manager.snapshot.rc, connected=True, timestamp=just_stale),
    )

    result = world.manager.query("rc check")

    assert result["valid"] is True
    assert result["fresh"] is False


@pytest.mark.asyncio
async def test_flight_queries_and_preflight_use_authoritative_interlocks(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await world.start()
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        settled = await world._settle_for_transform()
        assert settled.ok
        transformed = await world.manager.request_transform_flight(operator_confirmed=True)
        assert transformed.ok

        ready = world.manager.query("flight ready")
        enable_low = world.manager.query("flight enable-check")
        preflight = await world.manager.preflight("flight")

        assert ready["permitted"] is True
        assert ready["flight_enable_must_be_low"] is True
        assert enable_low["permitted"] is True
        assert enable_low["flight_enable_must_be_low"] is False
        assert preflight.ok

        world._set_switches(flight_enable=1900)
        await world.manager.tick()

        ready_high = world.manager.query("flight ready")
        enable_high = world.manager.query("flight enable-check")
        preflight_high = await world.manager.preflight("flight")

        assert ready_high["permitted"] is False
        assert "FLIGHT_ENABLE_NOT_LOW" in {item["code"] for item in ready_high["checks"]}
        assert enable_high["permitted"] is True
        assert enable_high["flight_enable_requested"] is True
        assert not preflight_high.ok
        assert "FLIGHT_ENABLE_NOT_LOW" in {item["code"] for item in preflight_high.data["checks"]}
    finally:
        await world.shutdown()


@pytest.mark.parametrize("case", ["missing", "ambiguous", "mismatch"])
@pytest.mark.asyncio
async def test_flight_preflight_rejects_invalid_or_inconsistent_raw_ch5(
    app_config: AppConfig,
    case: str,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await world.start()
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        settled = await world._settle_for_transform()
        assert settled.ok
        transformed = await world.manager.request_transform_flight(operator_confirmed=True)
        assert transformed.ok

        rc = world.manager.snapshot.rc
        channels = dict(rc.channels)
        parsed = False
        if case == "missing":
            channels.pop(app_config.rc.flight_enable_channel, None)
        elif case == "ambiguous":
            channels[app_config.rc.flight_enable_channel] = app_config.rc.low_max + 1
        else:
            channels[app_config.rc.flight_enable_channel] = app_config.rc.high_min
        world.manager.accept_rc_status(
            replace(
                rc,
                channels=channels,
                flight_enable=parsed,
                timestamp=world.clock.monotonic(),
            )
        )

        preflight = await world.manager.preflight("flight")
        report = world.manager.query("flight enable-check")

        assert not preflight.ok
        assert "FLIGHT_ENABLE_INVALID" in {item["code"] for item in preflight.data["checks"]}
        assert report["permitted"] is False
        assert "FLIGHT_ENABLE_INVALID" in {item["code"] for item in report["checks"]}
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_esc_health_and_flight_enable_fail_closed_on_incomplete_views(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await world.start()
        world._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        settled = await world._settle_for_transform()
        assert settled.ok
        transformed = await world.manager.request_transform_flight(operator_confirmed=True)
        assert transformed.ok

        snapshot = world.manager.snapshot
        missing_slot = max(app_config.esc.slots)
        incomplete_rpm = {
            slot: rpm for slot, rpm in snapshot.pixhawk.esc_rpm.items() if slot != missing_slot
        }
        world.manager._snapshot = replace(
            snapshot,
            pixhawk=replace(snapshot.pixhawk, esc_rpm=incomplete_rpm),
        )

        health = world.manager.query("esc health")
        enable = world.manager.query("flight enable-check")

        assert health["healthy"] is False
        assert health["complete"] is False
        assert "ESC_TELEMETRY_UNSAFE" in {item["code"] for item in enable["checks"]}
    finally:
        await world.shutdown()


@pytest.mark.asyncio
async def test_esc_health_rejects_malformed_numeric_and_boolean_types_without_throwing(
    app_config: AppConfig,
) -> None:
    world = SimulationWorld(app_config)
    try:
        await world.start()
        snapshot = world.manager.snapshot
        slot = min(app_config.esc.slots)
        tuple_items = tuple(
            replace(item, rpm="0", healthy=1) if item.slot == slot else item
            for item in snapshot.pixhawk.esc
        )
        rpm_by_slot = dict(snapshot.pixhawk.esc_rpm)
        online_by_slot = dict(snapshot.pixhawk.esc_online)
        rpm_by_slot[slot] = "0"
        online_by_slot[slot] = 1
        malformed_pixhawk = replace(
            snapshot.pixhawk,
            esc=tuple_items,
            esc_rpm=rpm_by_slot,
            esc_online=online_by_slot,
        )
        world.manager._snapshot = replace(snapshot, pixhawk=malformed_pixhawk)

        health = world.manager.query("esc health")
        rendered_maximum = malformed_pixhawk.maximum_esc_rpm

        assert health["healthy"] is False
        assert health["consistent"] is False
        assert math.isinf(rendered_maximum)
    finally:
        await world.shutdown()


def test_configuration_queries_return_detached_mutable_data(app_config: AppConfig) -> None:
    world = SimulationWorld(app_config)
    original = app_config.raw["rc"]["low_max"]

    full = world.manager.query("config show")
    rc_mapping = world.manager.query("rc mapping")
    pixhawk = world.manager.query("pixhawk params")
    f446 = world.manager.query("motor parameters")

    full["rc"]["low_max"] = -1
    rc_mapping["mapping"]["low_max"] = -2
    pixhawk["configured"]["baud"] = -3
    f446["configured"]["baud"] = -4

    assert app_config.raw["rc"]["low_max"] == original
    assert app_config.raw["pixhawk"]["baud"] > 0
    assert app_config.raw["f446"]["baud"] > 0


@pytest.mark.asyncio
async def test_changed_config_reload_requires_restart_without_split_brain(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(app_config.source_path.parent, config_dir)
    source = config_dir / "system.yaml"
    config = load_config(source)
    world = SimulationWorld(config)
    await world.manager.start()
    manager = world.manager
    component_ids = (
        id(world.pixhawk),
        id(world.f446),
        id(world.go2),
        id(world.rc_monitor),
    )

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["system"]["loop_hz"] += 1
    source.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    try:
        result = await manager.reload_config()

        assert not result.ok
        assert result.code == "CONFIG_RESTART_REQUIRED"
        assert manager.config is config
        assert world.config is config
        assert component_ids == (
            id(world.pixhawk),
            id(world.f446),
            id(world.go2),
            id(world.rc_monitor),
        )
    finally:
        await manager.shutdown()
