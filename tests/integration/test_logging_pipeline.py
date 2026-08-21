from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerogo2.common.clock import ManualClock
from aerogo2.common.enums import SystemState
from aerogo2.common.models import SystemSnapshot
from aerogo2.logging.event_logger import EventLogger
from aerogo2.logging.ordered_event_sink import LOG_FIELDS, OrderedEventSink
from aerogo2.logging.telemetry_logger import TelemetryLogger


def test_ordered_event_sink_writes_complete_schema_and_flushes_immediately(
    tmp_path: Path,
) -> None:
    clock = ManualClock(10.0)
    sink = OrderedEventSink(tmp_path, clock=clock, filename="events.jsonl")

    record = sink.emit(
        event_type="SYSTEM_STARTED",
        system_state="BOOT_SAFE",
        command_id="cmd-001",
        command_name="connect all",
        command_result="OK",
    )

    # Reading through a separate file handle before stop() proves emit() flushed.
    persisted = json.loads(sink.path.read_text(encoding="utf-8"))
    assert set(record) == set(LOG_FIELDS)
    assert persisted == record
    assert persisted["wall_timestamp"] == "2023-11-14T22:13:20+00:00"
    assert persisted["monotonic_timestamp"] == 10.0
    assert persisted["event_type"] == "SYSTEM_STARTED"
    assert persisted["system_state"] == "BOOT_SAFE"
    assert persisted["command_id"] == "cmd-001"
    assert persisted["pixhawk_status"] is None
    assert sink.running
    assert sink.records_written == 1

    sink.stop()
    assert not sink.running


def test_ordered_event_sink_preserves_order_and_supports_tail_and_export(
    tmp_path: Path,
) -> None:
    clock = ManualClock(20.0)
    sink = OrderedEventSink(tmp_path, clock=clock, filename="ordered.jsonl")

    expected = []
    for index, event_type in enumerate(("FIRST", "SECOND", "THIRD"), start=1):
        expected.append(
            sink.emit(
                event_type=event_type,
                system_state="WALK",
                sequence=index,
            )
        )
        clock.advance(0.25)

    persisted = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    assert persisted == expected
    assert [item["event_type"] for item in persisted] == [
        "FIRST",
        "SECOND",
        "THIRD",
    ]
    assert [item["monotonic_timestamp"] for item in persisted] == [
        20.0,
        20.25,
        20.5,
    ]
    assert [item["details"]["sequence"] for item in persisted] == [1, 2, 3]
    assert sink.tail(2) == tuple(expected[-2:])
    assert sink.tail(0) == ()

    destination = tmp_path / "exports" / "events-copy.jsonl"
    exported = sink.export(destination)
    assert exported == destination
    assert destination.read_bytes() == sink.path.read_bytes()

    with pytest.raises(FileExistsError):
        sink.export(destination)

    assert sink.export(destination, overwrite=True) == destination
    sink.stop()


def test_ordered_event_sink_mark_normalizes_multiline_operator_text(
    tmp_path: Path,
) -> None:
    sink = OrderedEventSink(
        tmp_path,
        clock=ManualClock(30.0),
        filename="marks.jsonl",
    )

    record = sink.mark("  before\r\nafter  ", system_state="FLIGHT_MANUAL")

    assert record["event_type"] == "LOG_MARK"
    assert record["operator_request"] is None
    assert record["details"]["marker_text"] == "before  after"
    assert sink.tail(1) == (record,)
    sink.stop()


def test_default_event_sink_names_do_not_merge_same_clock_sessions(tmp_path: Path) -> None:
    clock = ManualClock(40.0)
    first = OrderedEventSink(tmp_path, clock=clock)
    second = OrderedEventSink(tmp_path, clock=clock)

    first.emit(event_type="FIRST_SESSION", system_state="BOOT_SAFE")
    second.emit(event_type="SECOND_SESSION", system_state="BOOT_SAFE")

    assert first.path != second.path
    assert len(first.path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(second.path.read_text(encoding="utf-8").splitlines()) == 1
    first.stop()
    second.stop()


def test_ordered_event_sink_restart_uses_linear_session_suffixes(tmp_path: Path) -> None:
    sink = OrderedEventSink(tmp_path, clock=ManualClock(40.0))

    sink.emit(event_type="SESSION_ONE", system_state="BOOT_SAFE")
    first = sink.path
    sink.stop()
    sink.start()
    sink.emit(event_type="SESSION_TWO", system_state="BOOT_SAFE")
    second = sink.path
    sink.stop()
    sink.start()
    sink.emit(event_type="SESSION_THREE", system_state="BOOT_SAFE")
    third = sink.path

    assert second == first.with_name(f"{first.stem}-1{first.suffix}")
    assert third == first.with_name(f"{first.stem}-2{first.suffix}")
    sink.stop()


def test_ordered_event_sink_rejects_marker_while_stopped(tmp_path: Path) -> None:
    sink = OrderedEventSink(tmp_path, clock=ManualClock(41.0), filename="stopped.jsonl")
    sink.emit(event_type="BEFORE_STOP", system_state="BOOT_SAFE")
    sink.stop()
    before = sink.path.read_bytes()

    with pytest.raises(RuntimeError, match="stopped"):
        sink.mark("must not be written", system_state="BOOT_SAFE")

    assert sink.path.read_bytes() == before


@pytest.mark.asyncio
async def test_event_logger_stop_requires_explicit_restart_and_sessions_are_linear(
    tmp_path: Path,
) -> None:
    logger = EventLogger(tmp_path, clock=ManualClock(42.0))
    await logger.log(
        "SESSION_ONE",
        snapshot=SystemSnapshot(timestamp=42.0, state=SystemState.BOOT_SAFE),
    )
    first = logger.path
    await logger.stop()
    before = first.read_bytes()

    with pytest.raises(RuntimeError, match="stopped"):
        await logger.log("FORBIDDEN")
    with pytest.raises(RuntimeError, match="stopped"):
        await logger.mark("forbidden")
    assert first.read_bytes() == before

    await logger.start()
    await logger.log("SESSION_TWO")
    second = logger.path
    await logger.stop()
    await logger.start()
    await logger.log("SESSION_THREE")
    third = logger.path

    assert second == first.with_name(f"{first.stem}-1{first.suffix}")
    assert third == first.with_name(f"{first.stem}-2{first.suffix}")
    await logger.stop()


@pytest.mark.parametrize("sample_hz", [float("nan"), float("inf"), float("-inf")])
def test_telemetry_logger_rejects_nonfinite_sample_rates(
    tmp_path: Path,
    sample_hz: float,
) -> None:
    event_logger = EventLogger(tmp_path / "telemetry.jsonl", clock=ManualClock(43.0))

    with pytest.raises(ValueError, match="finite and positive"):
        TelemetryLogger(event_logger, SystemSnapshot, sample_hz=sample_hz)

    telemetry = TelemetryLogger(event_logger, SystemSnapshot, sample_hz=10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        telemetry.set_sample_hz(sample_hz)
    assert telemetry.sample_hz == 10.0
    # Rejected updates leave the previously valid rate intact.
