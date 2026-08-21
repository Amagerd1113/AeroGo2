from __future__ import annotations

from aerogo2.bridges.f446_parser import F446TextParser
from aerogo2.common.clock import ManualClock
from aerogo2.common.enums import F446EventType, F446State

STATUS_LINE = "state=LIMIT_REACHED_FWD duty=0 manual_limit=350 sense=max"
CURRENT_LINE = "R_IS=123/99mV L_IS=456/367mV"
USED_LINE = "used=456/367mV threshold=1800/1450mV"
TIMING_LINE = "blank=500ms overms=180ms timeout=5000ms over_active=0"
FIRMWARE_STATUS_LINE = (
    "state=LIMIT_REACHED_FWD duty=0 manual_limit=350 sense=max "
    "R_IS=123/99mV L_IS=456/367mV used=456/367mV "
    "threshold=1800/1450mV blank=500ms overms=180ms "
    "timeout=5000ms over_active=0"
)
FIRMWARE_IS_LINE = (
    "R_IS A0: raw=123, 99mV | L_IS A1: raw=456, 367mV | "
    "MAX raw=456, 367mV | used=max raw=456, 367mV | "
    "threshold raw=1800, 1450mV"
)
FIRMWARE_LIMIT_LINE = "FORWARD LIMIT REACHED: IS_max=1900 (1531mV), threshold=1800 (1450mV)"


def test_parses_complete_status_report(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(
        ("\n".join((STATUS_LINE, CURRENT_LINE, USED_LINE, TIMING_LINE)) + "\n").encode()
    )

    assert [event.event_type for event in events] == [
        F446EventType.STATUS,
        F446EventType.CURRENT_STATUS,
        F446EventType.CURRENT_STATUS,
        F446EventType.STATUS,
    ]
    status = parser.latest_status
    assert status.state is F446State.LIMIT_REACHED_FWD
    assert status.duty == 0
    assert status.r_is_adc == 123
    assert status.l_is_adc == 456
    assert status.used_current_adc == 456
    assert status.threshold_adc == 1800
    assert status.manual_limit == 350
    assert status.sense_mode == "max"
    assert status.r_is_mv == 99
    assert status.l_is_mv == 367
    assert status.used_mv == 367
    assert status.threshold_mv == 1450
    assert status.blanking_ms == 500
    assert status.overcurrent_ms == 180
    assert status.timeout_ms == 5000
    assert status.over_active is False


def test_parses_motor_main_single_line_status_byte_exact(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    event = parser.feed((FIRMWARE_STATUS_LINE + "\r\n").encode("ascii"))[0]

    assert event.event_type is F446EventType.STATUS
    assert event.state is F446State.LIMIT_REACHED_FWD
    assert event.values["r_is_raw"] == 123
    assert event.values["threshold_mv"] == 1450
    status = parser.latest_status
    assert status.r_is_adc == 123
    assert status.l_is_adc == 456
    assert status.used_current_adc == 456
    assert status.threshold_adc == 1800
    assert status.blanking_ms == 500
    assert status.overcurrent_ms == 180
    assert status.timeout_ms == 5000
    assert not status.over_active


def test_parses_motor_main_is_report_byte_exact(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    event = parser.feed((FIRMWARE_IS_LINE + "\r\n").encode("ascii"))[0]

    assert event.event_type is F446EventType.CURRENT_STATUS
    assert event.values["max_raw"] == 456
    assert event.values["sense"] == "max"
    assert parser.latest_status.used_current_adc == 456
    assert parser.latest_status.threshold_mv == 1450


def test_parses_motor_main_detailed_limit_byte_exact(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    event = parser.feed((FIRMWARE_LIMIT_LINE + "\r\n").encode("ascii"))[0]

    assert event.event_type is F446EventType.FORWARD_LIMIT_REACHED
    assert event.values == {
        "direction": "forward",
        "sense": "max",
        "used_raw": 1900,
        "used_mv": 1531,
        "threshold_raw": 1800,
        "threshold_mv": 1450,
    }
    assert parser.latest_status.state is F446State.LIMIT_REACHED_FWD
    assert parser.latest_status.duty == 0
    assert parser.latest_status.used_current_adc == 1900


def test_parses_motor_main_fault_and_clear_lines(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(b"fault=move timeout\r\nFault cleared. State=IDLE\r\n")

    assert events[0].event_type is F446EventType.FAULT
    assert events[0].values["message"] == "move timeout"
    assert events[1].event_type is F446EventType.FAULT_CLEARED
    assert parser.latest_status.state is F446State.IDLE
    assert parser.latest_status.raw_state == "IDLE"
    assert parser.latest_status.fault_message is None


def test_parses_r_is_raw_and_millivolts(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((CURRENT_LINE + "\n").encode())[0]
    assert event.values["r_is_raw"] == 123
    assert event.values["r_is_mv"] == 99


def test_parses_l_is_raw_and_millivolts(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((CURRENT_LINE + "\n").encode())[0]
    assert event.values["l_is_raw"] == 456
    assert event.values["l_is_mv"] == 367


def test_parses_used_current(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((USED_LINE + "\n").encode())[0]
    assert event.values["used_raw"] == 456
    assert event.values["used_mv"] == 367


def test_parses_threshold(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((USED_LINE + "\n").encode())[0]
    assert event.values["threshold_raw"] == 1800
    assert event.values["threshold_mv"] == 1450


def test_parses_forward_limit_event(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed(b"FORWARD LIMIT REACHED\n")[0]
    assert event.event_type is F446EventType.FORWARD_LIMIT_REACHED
    assert event.state is F446State.LIMIT_REACHED_FWD
    assert event.values["direction"] == "forward"


def test_parses_reverse_limit_event(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed(b"REVERSE LIMIT REACHED\n")[0]
    assert event.event_type is F446EventType.REVERSE_LIMIT_REACHED
    assert event.state is F446State.LIMIT_REACHED_REV
    assert event.values["direction"] == "reverse"


def test_parses_fault_without_stopping_reader(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(b"FAULT: move timeout\nstatus\n")
    assert events[0].event_type is F446EventType.FAULT
    assert events[0].values["message"] == "move timeout"
    assert events[1].event_type is F446EventType.COMMAND_ECHO
    assert parser.latest_status.faulted


def test_accepts_crlf(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((STATUS_LINE + "\r\n").encode())[0]
    assert event.event_type is F446EventType.STATUS
    assert event.line == STATUS_LINE


def test_accepts_lf(clock: ManualClock) -> None:
    event = F446TextParser(clock).feed((STATUS_LINE + "\n").encode())[0]
    assert event.event_type is F446EventType.STATUS
    assert event.line == STATUS_LINE


def test_supports_half_line_fragmentation(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    assert parser.feed(b"state=LIMIT_REACHED_") == []
    events = parser.feed(b"FWD duty=0 manual_limit=350 sense=max\n")
    assert len(events) == 1
    assert events[0].state is F446State.LIMIT_REACHED_FWD


def test_supports_multiple_lines_in_one_chunk(clock: ManualClock) -> None:
    events = F446TextParser(clock).feed(
        (STATUS_LINE + "\n" + CURRENT_LINE + "\n" + USED_LINE + "\n").encode()
    )
    assert len(events) == 3
    assert [event.event_type for event in events] == [
        F446EventType.STATUS,
        F446EventType.CURRENT_STATUS,
        F446EventType.CURRENT_STATUS,
    ]


def test_recognizes_command_echo(clock: ManualClock) -> None:
    events = F446TextParser(clock).feed(b"status\nlimf 120\n")
    assert [event.event_type for event in events] == [
        F446EventType.COMMAND_ECHO,
        F446EventType.COMMAND_ECHO,
    ]
    assert events[1].values["command"] == "limf 120"


def test_auto_status_can_interleave_with_echo_and_events(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(
        ("status\n" + STATUS_LINE + "\nFORWARD LIMIT REACHED\n" + CURRENT_LINE + "\n").encode()
    )
    assert [event.event_type for event in events] == [
        F446EventType.COMMAND_ECHO,
        F446EventType.STATUS,
        F446EventType.FORWARD_LIMIT_REACHED,
        F446EventType.CURRENT_STATUS,
    ]
    assert parser.latest_status.r_is_adc == 123


def test_unknown_line_does_not_break_following_line(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(("not firmware output\n" + STATUS_LINE + "\n").encode())
    assert events[0].event_type is F446EventType.UNKNOWN_LINE
    assert events[1].event_type is F446EventType.STATUS


def test_non_ascii_line_is_isolated(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    events = parser.feed(b"\xff\xfe\nstatus\n")
    assert events[0].event_type is F446EventType.UNKNOWN_LINE
    assert events[0].values["reason"] == "non-ASCII serial line"
    assert events[1].event_type is F446EventType.COMMAND_ECHO


def test_unterminated_eof_fragment_is_unknown(clock: ManualClock) -> None:
    parser = F446TextParser(clock)
    assert parser.feed(b"partial") == []
    event = parser.flush_eof()[0]
    assert event.event_type is F446EventType.UNKNOWN_LINE
    assert "unterminated" in str(event.values["reason"])
