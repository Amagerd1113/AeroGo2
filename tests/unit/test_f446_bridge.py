from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from aerogo2.bridges.f446_text_bridge import TextF446Bridge
from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import Configuration, F446State
from aerogo2.common.exceptions import BridgeError
from aerogo2.common.models import F446Status


class RecordingWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.drain_calls = 0

    def write(self, data: bytes) -> None:
        self.chunks.append(bytes(data))

    async def drain(self) -> None:
        self.drain_calls += 1


def commands(fake: FakeF446) -> list[str]:
    return [record.command for record in fake.command_history]


@pytest.mark.asyncio
async def test_flight_forward_mapping_sends_limf(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert result.ok
    assert "limf 120" in commands(fake)


@pytest.mark.asyncio
async def test_fake_limit_event_matches_motor_main_detailed_transcript(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    bridge = FakeF446(app_config.f446, clock)
    await bridge.connect()

    result = await bridge.move_to_configuration(Configuration.FLIGHT)

    assert result.ok
    event = bridge.event_history[-1]
    assert event.line == ("FORWARD LIMIT REACHED: IS_max=1900 (1531mV), threshold=1800 (1450mV)")
    assert event.values == {
        "direction": "forward",
        "sense": "max",
        "used_raw": 1900,
        "used_mv": 1531,
        "threshold_raw": 1800,
        "threshold_mv": 1450,
    }
    assert bridge.get_status().used_current_adc == 0


@pytest.mark.asyncio
async def test_flight_reverse_mapping_sends_limr(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    reversed_config = replace(
        app_config.f446,
        flight_direction="reverse",
        walk_direction="forward",
        expected_flight_state=F446State.LIMIT_REACHED_REV,
        expected_walk_state=F446State.LIMIT_REACHED_FWD,
    )
    fake = FakeF446(reversed_config, clock)
    await fake.connect()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert result.ok
    assert "limr 120" in commands(fake)


@pytest.mark.asyncio
async def test_walk_direction_uses_configured_mapping(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(
        app_config.f446,
        clock,
        initial_configuration=Configuration.FLIGHT,
    )
    await fake.connect()
    result = await fake.move_to_configuration(Configuration.WALK)
    assert result.ok
    assert "limr 120" in commands(fake)


@pytest.mark.asyncio
async def test_missing_limit_event_returns_timeout(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_transform_timeout()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert not result.ok
    assert result.code == "F446_TRANSFORM_TIMEOUT"


@pytest.mark.asyncio
async def test_timeout_sends_stop(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_transform_timeout()
    await fake.move_to_configuration(Configuration.FLIGHT)
    assert commands(fake)[-1] == "stop"
    assert fake.get_status().duty == 0


@pytest.mark.asyncio
async def test_fault_during_transform_fails_immediately(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_transform_fault("simulated overcurrent")
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert not result.ok
    assert result.code == "F446_FAULT"
    assert fake.get_status().state is F446State.FAULT


@pytest.mark.asyncio
async def test_limit_event_is_followed_by_final_status_request(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert result.ok
    history = commands(fake)
    assert history == ["status", "limf 120", "status"]


@pytest.mark.asyncio
async def test_wrong_final_state_fails_verification(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_wrong_final_state()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert not result.ok
    assert result.code == "F446_FINAL_STATE_MISMATCH"


@pytest.mark.asyncio
async def test_nonzero_final_duty_fails_verification_and_stops(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_nonzero_final_duty()
    result = await fake.move_to_configuration(Configuration.FLIGHT)
    assert not result.ok
    assert result.code == "F446_FINAL_DUTY_NONZERO"
    assert commands(fake)[-1] == "stop"
    assert fake.get_status().duty == 0


@pytest.mark.asyncio
async def test_formal_transform_never_uses_manual_pwm_commands(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    await fake.move_to_configuration(Configuration.FLIGHT)
    await fake.move_to_configuration(Configuration.WALK)
    command_names = {command.split()[0] for command in commands(fake)}
    assert command_names.isdisjoint({"mf", "mr", "raw"})


@pytest.mark.asyncio
async def test_clear_is_never_sent_automatically(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    fake = FakeF446(app_config.f446, clock)
    await fake.connect()
    fake.inject_next_transform_fault()
    await fake.move_to_configuration(Configuration.FLIGHT)
    assert "clear" not in commands(fake)


@pytest.mark.asyncio
async def test_real_text_bridge_reports_serial_open_failure(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_open(**_: object) -> object:
        raise OSError("serial device missing")

    monkeypatch.setitem(
        sys.modules,
        "serial_asyncio",
        SimpleNamespace(open_serial_connection=fail_open),
    )
    bridge = TextF446Bridge(app_config.f446)
    with pytest.raises(BridgeError, match="Cannot open F446 serial port"):
        await bridge.connect()


@pytest.mark.asyncio
async def test_real_text_bridge_motion_is_fail_closed(
    app_config: AppConfig,
) -> None:
    bridge = TextF446Bridge(app_config.f446)
    result = await bridge.move_to_configuration(Configuration.FLIGHT)
    assert not result.ok
    assert result.code == "F446_HARDWARE_WRITE_DISABLED"


@pytest.mark.asyncio
async def test_real_text_bridge_paces_each_byte_and_uses_single_cr(
    app_config: AppConfig,
) -> None:
    bridge = TextF446Bridge(app_config.f446)
    writer = RecordingWriter()
    bridge._connected = True
    bridge._writer = writer

    await bridge._write_line("status")

    assert writer.chunks == [bytes((value,)) for value in b"status\r"]
    assert writer.drain_calls == len(b"status\r")


@pytest.mark.asyncio
async def test_real_text_bridge_accepts_negative_reverse_duty(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = TextF446Bridge(app_config.f446, allow_motion=True)
    writer = RecordingWriter()
    bridge._connected = True
    bridge._writer = writer
    initial = F446Status(
        connected=True,
        state=F446State.IDLE,
        raw_state=F446State.IDLE.value,
        duty=0,
        manual_limit=500,
    )
    responses = iter(
        (
            initial,
            replace(
                initial,
                state=F446State.MANUAL_REV,
                raw_state=F446State.MANUAL_REV.value,
                duty=-500,
            ),
        )
    )

    async def request_status() -> F446Status:
        return next(responses)

    monkeypatch.setattr(bridge, "request_status", request_status)

    result = await bridge.start_maintenance_motion("mr", 500)

    assert result.ok
    assert result.data["signed_duty"] == -500
    assert b"".join(writer.chunks) == b"mr 500\r"


@pytest.mark.asyncio
async def test_real_text_bridge_retries_only_initial_status(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = TextF446Bridge(app_config.f446)
    attempts = 0

    async def request_status() -> F446Status:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise BridgeError("incomplete startup response")
        return FakeF446(app_config.f446).get_status()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(bridge, "request_status", request_status)
    monkeypatch.setattr(
        "aerogo2.bridges.f446_text_bridge.asyncio.sleep",
        no_sleep,
    )

    status = await bridge._request_initial_status()

    assert attempts == 3
    assert status.state is not F446State.UNKNOWN


def test_fake_f446_rejects_nonfinite_transform_delay(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        FakeF446(app_config.f446, clock, transform_delay_s=float("nan"))


def test_fake_f446_defaults_match_firmware_status_parameters(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    status = FakeF446(app_config.f446, clock).get_status()
    assert status.manual_limit == 350
    assert status.threshold_raw == 1800
    assert status.threshold_adc == 1800
    assert status.threshold_mv == 1450
    assert status.blanking_ms == 500
    assert status.overcurrent_ms == 180
    assert status.timeout_ms == 5000


@pytest.mark.asyncio
async def test_transform_delay_uses_configured_deadline(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = replace(app_config.f446, transform_timeout_s=0.5)
    fake = FakeF446(config, clock, transform_delay_s=0.6)
    await fake.connect()

    result = await fake.move_to_configuration(Configuration.FLIGHT)

    assert not result.ok
    assert result.code == "F446_TRANSFORM_TIMEOUT"
    assert clock.monotonic() == 10.5
    assert commands(fake)[-1] == "stop"


@pytest.mark.asyncio
async def test_transform_delay_before_deadline_completes(
    app_config: AppConfig,
    clock: ManualClock,
) -> None:
    config = replace(app_config.f446, transform_timeout_s=0.5)
    fake = FakeF446(config, clock, transform_delay_s=0.25)
    await fake.connect()

    result = await fake.move_to_configuration(Configuration.FLIGHT)

    assert result.ok
    assert clock.monotonic() == 10.25
