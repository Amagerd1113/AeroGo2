"""Debounced, fail-safe interpretation of RadioMaster RC channels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import RCConfig
from aerogo2.common.enums import (
    AutoLandingRequest,
    MorphologyRequest,
    RCPosition,
)
from aerogo2.common.models import RCStatus

_MIN_VALID_PWM_US = 800
_MAX_VALID_PWM_US = 2200


def classify_rc_position(config: RCConfig, value: object) -> RCPosition:
    """Classify one raw PWM value without debounce state.

    The strict integer/type check prevents booleans, strings, and malformed
    injected values from being interpreted as a safe LOW request.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        return RCPosition.UNKNOWN
    if value < _MIN_VALID_PWM_US or value > _MAX_VALID_PWM_US:
        return RCPosition.UNKNOWN
    if value <= config.low_max:
        return RCPosition.LOW
    if config.middle_min <= value <= config.middle_max:
        return RCPosition.MIDDLE
    if value >= config.high_min:
        return RCPosition.HIGH
    return RCPosition.UNKNOWN


@dataclass
class _DebounceState:
    stable: RCPosition = RCPosition.UNKNOWN
    candidate: RCPosition = RCPosition.UNKNOWN
    candidate_since: float = 0.0


class RCMonitor:
    """Interpret switch channels without retaining unsafe stale requests."""

    def __init__(
        self,
        config: RCConfig,
        clock: Optional[Clock] = None,
        stick_neutral_us: Optional[Mapping[int, int]] = None,
    ) -> None:
        self._config = config
        self._clock = clock or RealClock()
        # Phase 1 simulation uses the common 1500-us midpoint for roll, pitch,
        # throttle, and yaw. Real airframes must pass their calibrated centers.
        centers = {1: 1500, 2: 1500, 3: 1500, 4: 1500}
        if stick_neutral_us is not None:
            unknown_channels = set(stick_neutral_us) - set(centers)
            if unknown_channels:
                raise ValueError("stick calibration is limited to RC channels 1 through 4")
            centers.update(
                {int(channel): int(value) for channel, value in stick_neutral_us.items()}
            )
        if any(
            value < _MIN_VALID_PWM_US or value > _MAX_VALID_PWM_US for value in centers.values()
        ):
            raise ValueError("stick neutral values must be within valid RC PWM bounds")
        self._stick_neutral_us = centers
        self._states: Dict[int, _DebounceState] = {}
        self._last_update: Optional[float] = None
        self._status = self._safe_status(timestamp=0.0, connected=False, failsafe=True)

    def classify(self, value: Optional[int]) -> RCPosition:
        """Map PWM to a switch position; threshold gaps remain UNKNOWN."""

        return classify_rc_position(self._config, value)

    def update(
        self,
        channels: Mapping[int, int],
        failsafe: bool = False,
        connected: bool = True,
        timestamp: Optional[float] = None,
    ) -> RCStatus:
        now = self._clock.monotonic() if timestamp is None else timestamp
        if not math.isfinite(now):
            raise ValueError("RC timestamp must be finite")
        channel_copy = {int(channel): int(value) for channel, value in channels.items()}
        self._last_update = now

        if failsafe or not connected:
            self._reset_debounce()
            self._status = self._safe_status(
                timestamp=now,
                connected=connected,
                failsafe=True,
                channels=channel_copy,
            )
            return self._status

        for channel in self._configured_channels():
            position = self.classify(channel_copy.get(channel))
            self._update_channel(channel, position, now)

        flight_position = self.position(self._config.flight_enable_channel)
        morphology_position = self.position(self._config.morphology_channel)
        autoland_position = self.position(self._config.auto_landing_channel)
        auto_request = self._auto_request(autoland_position)
        manual_override = autoland_position is RCPosition.LOW or self._stick_override_requested(
            channel_copy
        )
        self._status = RCStatus(
            connected=True,
            failsafe=False,
            channels=channel_copy,
            flight_enable=flight_position is RCPosition.HIGH,
            morphology_request=self._morphology_request(morphology_position),
            auto_landing_request=auto_request,
            manual_override=manual_override,
            timestamp=now,
        )
        return self._status

    def update_from_channels(
        self,
        channels: Mapping[int, int],
        failsafe: bool = False,
        connected: bool = True,
        timestamp: Optional[float] = None,
    ) -> RCStatus:
        return self.update(channels, failsafe=failsafe, connected=connected, timestamp=timestamp)

    def get_status(self) -> RCStatus:
        if self._last_update is None:
            return self._status
        now = self._clock.monotonic()
        if now - self._last_update >= self._config.timeout_s:
            if not self._status.failsafe:
                channels = self._status.channels
                self._reset_debounce()
                self._status = self._safe_status(
                    timestamp=self._last_update,
                    connected=False,
                    failsafe=True,
                    channels=channels,
                )
        return self._status

    def latest_status(self) -> RCStatus:
        return self.get_status()

    def position(self, channel: int) -> RCPosition:
        state = self._states.get(channel)
        return RCPosition.UNKNOWN if state is None else state.stable

    @property
    def raw_channels(self) -> Mapping[int, int]:
        return dict(self.get_status().channels)

    @property
    def rtl_requested(self) -> bool:
        return self.position(self._config.rtl_channel) is RCPosition.HIGH

    @property
    def land_requested(self) -> bool:
        return self.position(self._config.land_channel) is RCPosition.HIGH

    @property
    def flight_mode_position(self) -> RCPosition:
        return self.position(self._config.flight_mode_channel)

    def reset(self) -> None:
        self._last_update = None
        self._reset_debounce()
        self._status = self._safe_status(timestamp=0.0, connected=False, failsafe=True)

    def _update_channel(self, channel: int, position: RCPosition, now: float) -> None:
        state = self._states.setdefault(channel, _DebounceState())
        if position is RCPosition.UNKNOWN:
            # Ambiguous PWM immediately removes any latched high-level request.
            state.stable = RCPosition.UNKNOWN
            state.candidate = RCPosition.UNKNOWN
            state.candidate_since = now
            return
        if position is not state.candidate:
            state.candidate = position
            state.candidate_since = now
            return
        elapsed = now - state.candidate_since
        if position is not state.stable and elapsed + 1e-9 >= self._config.debounce_s:
            state.stable = position

    def _configured_channels(self) -> Tuple[int, ...]:
        return (
            self._config.flight_enable_channel,
            self._config.flight_mode_channel,
            self._config.rtl_channel,
            self._config.land_channel,
            self._config.morphology_channel,
            self._config.auto_landing_channel,
            self._config.brake_channel,
            self._config.buzzer_channel,
        )

    def _stick_override_requested(self, channels: Mapping[int, int]) -> bool:
        for channel in (1, 2, 3, 4):
            neutral = self._stick_neutral_us[channel]
            value = channels.get(channel)
            if value is None or value < _MIN_VALID_PWM_US or value > _MAX_VALID_PWM_US:
                # In AUTO, incomplete stick telemetry cannot prove that the
                # independent RadioMaster operator has not taken over.
                return True
            if abs(value - neutral) > self._config.manual_override_deadband_us:
                return True
        return False

    @staticmethod
    def _morphology_request(position: RCPosition) -> MorphologyRequest:
        if position is RCPosition.LOW:
            return MorphologyRequest.WALK
        if position is RCPosition.MIDDLE:
            return MorphologyRequest.HOLD
        if position is RCPosition.HIGH:
            return MorphologyRequest.FLIGHT_REQUEST
        return MorphologyRequest.HOLD

    @staticmethod
    def _auto_request(position: RCPosition) -> AutoLandingRequest:
        if position is RCPosition.MIDDLE:
            return AutoLandingRequest.AUTO_READY
        if position is RCPosition.HIGH:
            return AutoLandingRequest.AUTO_EXECUTE
        return AutoLandingRequest.MANUAL

    def _reset_debounce(self) -> None:
        self._states.clear()

    @staticmethod
    def _safe_status(
        timestamp: float,
        connected: bool,
        failsafe: bool,
        channels: Optional[Mapping[int, int]] = None,
    ) -> RCStatus:
        return RCStatus(
            connected=connected,
            failsafe=failsafe,
            channels={} if channels is None else dict(channels),
            flight_enable=False,
            morphology_request=MorphologyRequest.HOLD,
            auto_landing_request=AutoLandingRequest.MANUAL,
            manual_override=False,
            timestamp=timestamp,
        )


__all__ = ["RCMonitor"]
