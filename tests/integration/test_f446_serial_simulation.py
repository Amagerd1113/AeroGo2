from __future__ import annotations

from pathlib import Path

import pytest

from aerogo2.bridges.f446_parser import F446TextParser
from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446EventType, F446State


def test_fragmented_serial_fixture_is_parsed_without_transport_io(
    project_root: Path,
    clock: ManualClock,
) -> None:
    payload = (project_root / "tests" / "fixtures" / "f446_status_samples.txt").read_bytes()
    parser = F446TextParser(clock)
    events = []
    for offset in range(0, len(payload), 7):
        events.extend(parser.feed(payload[offset : offset + 7]))
    events.extend(parser.flush_eof())

    limit_types = {
        F446EventType.FORWARD_LIMIT_REACHED,
        F446EventType.REVERSE_LIMIT_REACHED,
    }
    assert sum(event.event_type in limit_types for event in events) == 2
    assert events[-1].event_type is F446EventType.FAULT
    assert parser.latest_status.state is F446State.FAULT
    assert parser.latest_status.duty == 0
    assert parser.buffered_bytes == 0


@pytest.mark.asyncio
async def test_fake_serial_transform_uses_only_guarded_limit_command(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    f446 = FakeF446(app_config.f446, clock)
    await f446.connect()
    result = await f446.move_to_configuration(Configuration.FLIGHT)

    commands = tuple(record.command for record in f446.command_history)
    assert result.ok
    assert commands == ("status", "limf 120", "status")
    assert f446.get_status().state is app_config.f446.expected_flight_state
    assert f446.get_status().duty == 0
    assert all(command.split()[0] not in {"mf", "mr", "raw"} for command in commands)


@pytest.mark.asyncio
async def test_fake_serial_timeout_stops_and_never_clears_faults_implicitly(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    f446 = FakeF446(app_config.f446, clock)
    await f446.connect()
    f446.inject_next_transform_timeout()
    result = await f446.move_to_configuration(Configuration.FLIGHT)

    commands = tuple(record.command for record in f446.command_history)
    assert not result.ok
    assert result.code == "F446_TRANSFORM_TIMEOUT"
    assert commands[-1] == "stop"
    assert "clear" not in commands
    assert f446.get_status().duty == 0
