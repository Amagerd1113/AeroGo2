"""Incremental parser for the existing F446 ASCII serial protocol.

The parser is pure: it never opens a serial port and never emits a command.
It tolerates arbitrary byte fragmentation, CRLF/LF line endings, command
echoes, mixed automatic status output, malformed lines, and non-ASCII input.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.enums import F446EventType, F446State
from aerogo2.common.models import F446Event, F446Status

_STATUS_RE = re.compile(
    r"^state=(?P<state>[A-Z_]+)\s+"
    r"duty=(?P<duty>[+-]?\d+)\s+"
    r"manual_limit=(?P<manual_limit>\d+)\s+"
    r"sense=(?P<sense>[A-Za-z0-9_-]+)"
    r"(?:\s+R_IS=(?P<status_r_raw>\d+)/(?P<status_r_mv>\d+)mV\s+"
    r"L_IS=(?P<status_l_raw>\d+)/(?P<status_l_mv>\d+)mV\s+"
    r"used=(?P<status_used_raw>\d+)/(?P<status_used_mv>\d+)mV\s+"
    r"threshold=(?P<status_threshold_raw>\d+)/(?P<status_threshold_mv>\d+)mV\s+"
    r"blank=(?P<status_blank_ms>\d+)ms\s+"
    r"overms=(?P<status_overcurrent_ms>\d+)ms\s+"
    r"timeout=(?P<status_timeout_ms>\d+)ms\s+"
    r"over_active=(?P<status_over_active>[01]))?$"
)
_CURRENT_RE = re.compile(
    r"^R_IS=(?P<r_raw>\d+)/(?P<r_mv>\d+)mV\s+"
    r"L_IS=(?P<l_raw>\d+)/(?P<l_mv>\d+)mV$"
)
_IS_REPORT_RE = re.compile(
    r"^R_IS A0: raw=(?P<r_raw>\d+), (?P<r_mv>\d+)mV \| "
    r"L_IS A1: raw=(?P<l_raw>\d+), (?P<l_mv>\d+)mV \| "
    r"MAX raw=(?P<max_raw>\d+), (?P<max_mv>\d+)mV \| "
    r"used=(?P<sense>[A-Za-z0-9_-]+) raw=(?P<used_raw>\d+), "
    r"(?P<used_mv>\d+)mV \| "
    r"threshold raw=(?P<threshold_raw>\d+), (?P<threshold_mv>\d+)mV$"
)
_USED_RE = re.compile(
    r"^used=(?P<used_raw>\d+)/(?P<used_mv>\d+)mV\s+"
    r"threshold=(?P<threshold_raw>\d+)/(?P<threshold_mv>\d+)mV$"
)
_TIMING_RE = re.compile(
    r"^blank=(?P<blank_ms>\d+)ms\s+"
    r"overms=(?P<overcurrent_ms>\d+)ms\s+"
    r"timeout=(?P<timeout_ms>\d+)ms\s+"
    r"over_active=(?P<over_active>[01])$"
)
_AUTO_RE = re.compile(r"^(?:auto|auto_status)=(?P<enabled>on|off|0|1)$", re.IGNORECASE)
_LIMIT_RE = re.compile(
    r"^(?P<direction>FORWARD|REVERSE) LIMIT REACHED"
    r"(?:: IS_(?P<sense>[A-Za-z0-9_-]+)=(?P<used_raw>\d+) "
    r"\((?P<used_mv>\d+)mV\), threshold=(?P<threshold_raw>\d+) "
    r"\((?P<threshold_mv>\d+)mV\))?$"
)
_COMMAND_ECHO_RE = re.compile(
    r"^(?:"
    r"help|status|is|stop|disable|clear|"
    r"auto\s+(?:on|off)|"
    r"sense\s+(?:max|r|l)|"
    r"(?:mf|mr|raw|mlimit|limf|limr|thr|thrmv|blank|overms|timeout)"
    r"\s+[+-]?\d+"
    r")$",
    re.IGNORECASE,
)


class F446TextParser:
    """Stateful byte-stream parser with failure isolation per complete line."""

    def __init__(
        self,
        clock: Optional[Clock] = None,
        max_line_bytes: int = 4096,
        raw_history_lines: int = 16,
    ) -> None:
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be positive")
        if raw_history_lines <= 0:
            raise ValueError("raw_history_lines must be positive")
        self._clock = clock or RealClock()
        self._max_line_bytes = max_line_bytes
        self._raw_history_lines = raw_history_lines
        self._buffer = bytearray()
        self._latest_status = F446Status()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    @property
    def latest_status(self) -> F446Status:
        return self._latest_status

    def reset(self) -> None:
        self._buffer.clear()
        self._latest_status = F446Status()

    def feed(self, data: bytes) -> List[F446Event]:
        """Consume bytes and return events for every complete line."""

        if not isinstance(data, bytes):
            raise TypeError("F446TextParser.feed expects bytes")
        events: List[F446Event] = []
        self._buffer.extend(data)

        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            events.extend(self._parse_raw_line(raw))

        if len(self._buffer) > self._max_line_bytes:
            preview = bytes(self._buffer[: self._max_line_bytes])
            self._buffer.clear()
            events.append(
                self._unknown_event(
                    self._decode_for_log(preview),
                    reason="unterminated line exceeded byte limit",
                )
            )
        return events

    def flush_eof(self) -> List[F446Event]:
        """Turn a final unterminated fragment into an explicit unknown event."""

        if not self._buffer:
            return []
        raw = bytes(self._buffer)
        self._buffer.clear()
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        if not raw:
            return []
        return [
            self._unknown_event(
                self._decode_for_log(raw),
                reason="unterminated line at end of stream",
            )
        ]

    def _parse_raw_line(self, raw: bytes) -> List[F446Event]:
        if not raw:
            return []
        if len(raw) > self._max_line_bytes:
            return [
                self._unknown_event(
                    self._decode_for_log(raw[: self._max_line_bytes]),
                    reason="line exceeded byte limit",
                )
            ]
        try:
            line = raw.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            return [
                self._unknown_event(
                    self._decode_for_log(raw),
                    reason="non-ASCII serial line",
                )
            ]
        try:
            return [self._parse_line(line)]
        except (KeyError, OverflowError, ValueError) as exc:
            return [self._unknown_event(line, reason=f"parse failure: {exc}")]

    def _parse_line(self, line: str) -> F446Event:
        now = self._clock.monotonic()
        status_match = _STATUS_RE.fullmatch(line)
        if status_match is not None:
            values = status_match.groupdict()
            raw_state = values["state"]
            state = F446State.__members__.get(raw_state, F446State.UNKNOWN)
            parsed: Dict[str, object] = {
                "raw_state": raw_state,
                "duty": int(values["duty"]),
                "manual_limit": int(values["manual_limit"]),
                "sense": values["sense"],
            }
            self._latest_status = replace(
                self._latest_status,
                state=state,
                raw_state=raw_state,
                duty=int(values["duty"]),
                manual_limit=int(values["manual_limit"]),
                sense_mode=values["sense"],
                fault_message=(
                    self._latest_status.fault_message if state is F446State.FAULT else None
                ),
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            if values["status_r_raw"] is not None:
                parsed.update(
                    {
                        "r_is_raw": int(values["status_r_raw"]),
                        "r_is_mv": int(values["status_r_mv"]),
                        "l_is_raw": int(values["status_l_raw"]),
                        "l_is_mv": int(values["status_l_mv"]),
                        "used_raw": int(values["status_used_raw"]),
                        "used_mv": int(values["status_used_mv"]),
                        "threshold_raw": int(values["status_threshold_raw"]),
                        "threshold_mv": int(values["status_threshold_mv"]),
                        "blank_ms": int(values["status_blank_ms"]),
                        "overcurrent_ms": int(values["status_overcurrent_ms"]),
                        "timeout_ms": int(values["status_timeout_ms"]),
                        "over_active": values["status_over_active"] == "1",
                    }
                )
                self._latest_status = replace(
                    self._latest_status,
                    r_is_adc=int(values["status_r_raw"]),
                    r_is_raw=int(values["status_r_raw"]),
                    r_is_mv=int(values["status_r_mv"]),
                    l_is_adc=int(values["status_l_raw"]),
                    l_is_raw=int(values["status_l_raw"]),
                    l_is_mv=int(values["status_l_mv"]),
                    used_current_adc=int(values["status_used_raw"]),
                    used_raw=int(values["status_used_raw"]),
                    used_mv=int(values["status_used_mv"]),
                    threshold_adc=int(values["status_threshold_raw"]),
                    threshold_raw=int(values["status_threshold_raw"]),
                    threshold_mv=int(values["status_threshold_mv"]),
                    blanking_ms=int(values["status_blank_ms"]),
                    overcurrent_ms=int(values["status_overcurrent_ms"]),
                    timeout_ms=int(values["status_timeout_ms"]),
                    over_active=values["status_over_active"] == "1",
                )
            return F446Event(F446EventType.STATUS, line, now, state=state, values=parsed)

        current_match = _CURRENT_RE.fullmatch(line)
        if current_match is not None:
            values = current_match.groupdict()
            parsed = {
                "r_is_raw": int(values["r_raw"]),
                "r_is_mv": int(values["r_mv"]),
                "l_is_raw": int(values["l_raw"]),
                "l_is_mv": int(values["l_mv"]),
            }
            self._latest_status = replace(
                self._latest_status,
                r_is_adc=int(values["r_raw"]),
                r_is_raw=int(values["r_raw"]),
                r_is_mv=int(values["r_mv"]),
                l_is_adc=int(values["l_raw"]),
                l_is_raw=int(values["l_raw"]),
                l_is_mv=int(values["l_mv"]),
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.CURRENT_STATUS,
                line,
                now,
                state=self._latest_status.state,
                values=parsed,
            )

        is_report_match = _IS_REPORT_RE.fullmatch(line)
        if is_report_match is not None:
            values = is_report_match.groupdict()
            parsed = {
                "r_is_raw": int(values["r_raw"]),
                "r_is_mv": int(values["r_mv"]),
                "l_is_raw": int(values["l_raw"]),
                "l_is_mv": int(values["l_mv"]),
                "max_raw": int(values["max_raw"]),
                "max_mv": int(values["max_mv"]),
                "sense": values["sense"],
                "used_raw": int(values["used_raw"]),
                "used_mv": int(values["used_mv"]),
                "threshold_raw": int(values["threshold_raw"]),
                "threshold_mv": int(values["threshold_mv"]),
            }
            self._latest_status = replace(
                self._latest_status,
                sense_mode=values["sense"],
                r_is_adc=int(values["r_raw"]),
                r_is_raw=int(values["r_raw"]),
                r_is_mv=int(values["r_mv"]),
                l_is_adc=int(values["l_raw"]),
                l_is_raw=int(values["l_raw"]),
                l_is_mv=int(values["l_mv"]),
                used_current_adc=int(values["used_raw"]),
                used_raw=int(values["used_raw"]),
                used_mv=int(values["used_mv"]),
                threshold_adc=int(values["threshold_raw"]),
                threshold_raw=int(values["threshold_raw"]),
                threshold_mv=int(values["threshold_mv"]),
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.CURRENT_STATUS,
                line,
                now,
                state=self._latest_status.state,
                values=parsed,
            )

        used_match = _USED_RE.fullmatch(line)
        if used_match is not None:
            values = used_match.groupdict()
            parsed = {
                "used_raw": int(values["used_raw"]),
                "used_mv": int(values["used_mv"]),
                "threshold_raw": int(values["threshold_raw"]),
                "threshold_mv": int(values["threshold_mv"]),
            }
            self._latest_status = replace(
                self._latest_status,
                used_current_adc=int(values["used_raw"]),
                used_raw=int(values["used_raw"]),
                used_mv=int(values["used_mv"]),
                threshold_adc=int(values["threshold_raw"]),
                threshold_raw=int(values["threshold_raw"]),
                threshold_mv=int(values["threshold_mv"]),
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.CURRENT_STATUS,
                line,
                now,
                state=self._latest_status.state,
                values=parsed,
            )

        timing_match = _TIMING_RE.fullmatch(line)
        if timing_match is not None:
            values = timing_match.groupdict()
            parsed = {
                "blank_ms": int(values["blank_ms"]),
                "overcurrent_ms": int(values["overcurrent_ms"]),
                "timeout_ms": int(values["timeout_ms"]),
                "over_active": values["over_active"] == "1",
            }
            self._latest_status = replace(
                self._latest_status,
                blanking_ms=int(values["blank_ms"]),
                overcurrent_ms=int(values["overcurrent_ms"]),
                timeout_ms=int(values["timeout_ms"]),
                over_active=values["over_active"] == "1",
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.STATUS,
                line,
                now,
                state=self._latest_status.state,
                values=parsed,
            )

        auto_match = _AUTO_RE.fullmatch(line)
        if auto_match is not None:
            enabled = auto_match.group("enabled").lower() in {"on", "1"}
            self._latest_status = replace(
                self._latest_status,
                auto_status=enabled,
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.STATUS,
                line,
                now,
                state=self._latest_status.state,
                values={"auto_status": enabled},
            )

        limit_match = _LIMIT_RE.fullmatch(line)
        if limit_match is not None:
            values = limit_match.groupdict()
            direction = values["direction"].lower()
            state = (
                F446State.LIMIT_REACHED_FWD
                if direction == "forward"
                else F446State.LIMIT_REACHED_REV
            )
            return self._limit_event(
                line,
                state,
                direction,
                sense=values["sense"],
                used_raw=None if values["used_raw"] is None else int(values["used_raw"]),
                used_mv=None if values["used_mv"] is None else int(values["used_mv"]),
                threshold_raw=(
                    None if values["threshold_raw"] is None else int(values["threshold_raw"])
                ),
                threshold_mv=(
                    None if values["threshold_mv"] is None else int(values["threshold_mv"])
                ),
            )
        if line.startswith("FAULT:") or line.startswith("fault="):
            separator = ":" if line.startswith("FAULT:") else "="
            message = line.partition(separator)[2].strip() or "unspecified F446 fault"
            self._latest_status = replace(
                self._latest_status,
                state=F446State.FAULT,
                raw_state=F446State.FAULT.value,
                configuration="UNKNOWN",
                duty=0,
                fault_message=message,
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            return F446Event(
                F446EventType.FAULT,
                line,
                now,
                state=F446State.FAULT,
                values={"message": message},
            )

        notice = self._parse_notice(line)
        if notice is not None:
            state, values = notice
            self._latest_status = replace(
                self._latest_status,
                state=state,
                duty=0 if state is F446State.IDLE else self._latest_status.duty,
                raw_state=state.value,
                configuration=(
                    "UNKNOWN"
                    if line.startswith("Fault cleared.")
                    else self._latest_status.configuration
                ),
                fault_message=(
                    None if line.startswith("Fault cleared.") else self._latest_status.fault_message
                ),
                timestamp=now,
                raw_lines=self._append_raw(line),
            )
            if line.startswith("Stopped:"):
                event_type = F446EventType.STOPPED
            elif line.startswith("Disabled:"):
                event_type = F446EventType.DISABLED
            elif line.startswith("Fault cleared."):
                event_type = F446EventType.FAULT_CLEARED
            else:
                event_type = F446EventType.INFO
            return F446Event(
                event_type,
                line,
                now,
                state=state,
                values=values,
            )

        if _COMMAND_ECHO_RE.fullmatch(line) is not None:
            return F446Event(
                F446EventType.COMMAND_ECHO,
                line,
                now,
                state=self._latest_status.state,
                values={"command": line},
            )
        return self._unknown_event(line, reason="unrecognized serial line")

    def _limit_event(
        self,
        line: str,
        state: F446State,
        direction: str,
        *,
        sense: Optional[str] = None,
        used_raw: Optional[int] = None,
        used_mv: Optional[int] = None,
        threshold_raw: Optional[int] = None,
        threshold_mv: Optional[int] = None,
    ) -> F446Event:
        now = self._clock.monotonic()
        details_present = all(
            item is not None for item in (sense, used_raw, used_mv, threshold_raw, threshold_mv)
        )
        values: Dict[str, object] = {"direction": direction}
        self._latest_status = replace(
            self._latest_status,
            state=state,
            raw_state=state.value,
            duty=0,
            timestamp=now,
            raw_lines=self._append_raw(line),
        )
        if details_present:
            assert sense is not None
            assert used_raw is not None
            assert used_mv is not None
            assert threshold_raw is not None
            assert threshold_mv is not None
            values.update(
                {
                    "sense": sense,
                    "used_raw": used_raw,
                    "used_mv": used_mv,
                    "threshold_raw": threshold_raw,
                    "threshold_mv": threshold_mv,
                }
            )
            self._latest_status = replace(
                self._latest_status,
                sense_mode=sense,
                used_current_adc=used_raw,
                used_raw=used_raw,
                used_mv=used_mv,
                threshold_adc=threshold_raw,
                threshold_raw=threshold_raw,
                threshold_mv=threshold_mv,
            )
        event_type = (
            F446EventType.FORWARD_LIMIT_REACHED
            if direction == "forward"
            else F446EventType.REVERSE_LIMIT_REACHED
        )
        return F446Event(
            event_type,
            line,
            now,
            state=state,
            values=values,
        )

    @staticmethod
    def _parse_notice(line: str) -> Optional[Tuple[F446State, Dict[str, object]]]:
        if line.startswith("Stopped:"):
            return F446State.IDLE, {"notice": "stopped", "message": line.partition(":")[2].strip()}
        if line.startswith("Disabled:"):
            return F446State.IDLE, {
                "notice": "disabled",
                "message": line.partition(":")[2].strip(),
            }
        if line.startswith("Fault cleared."):
            return F446State.IDLE, {"notice": "fault_cleared"}
        if line.startswith("MANUAL forward"):
            return F446State.MANUAL_FWD, {"notice": "manual", "direction": "forward"}
        if line.startswith("MANUAL reverse"):
            return F446State.MANUAL_REV, {"notice": "manual", "direction": "reverse"}
        if line.startswith("LIMIT forward"):
            return F446State.LIMIT_FWD, {"notice": "limit_move", "direction": "forward"}
        if line.startswith("LIMIT reverse"):
            return F446State.LIMIT_REV, {"notice": "limit_move", "direction": "reverse"}
        return None

    def _unknown_event(self, line: str, reason: str) -> F446Event:
        return F446Event(
            F446EventType.UNKNOWN_LINE,
            line,
            self._clock.monotonic(),
            state=self._latest_status.state,
            values={"reason": reason},
        )

    def _append_raw(self, line: str) -> Tuple[str, ...]:
        return (self._latest_status.raw_lines + (line,))[-self._raw_history_lines :]

    @staticmethod
    def _decode_for_log(raw: bytes) -> str:
        return raw.decode("ascii", errors="replace")


__all__ = ["F446TextParser"]
