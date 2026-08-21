from __future__ import annotations

import pytest

from aerogo2.bridges.go2_bridge import Go2BridgeStub
from aerogo2.bridges.pixhawk_bridge import ReadOnlyPixhawkBridge
from aerogo2.common.exceptions import UnsupportedPhaseOperation


@pytest.mark.asyncio
async def test_pixhawk_real_bridge_connect_and_write_paths_fail_closed() -> None:
    bridge = ReadOnlyPixhawkBridge()

    with pytest.raises(UnsupportedPhaseOperation, match="disabled.*Phase 1"):
        await bridge.connect()
    with pytest.raises(UnsupportedPhaseOperation, match="mode changes.*disabled"):
        await bridge.request_mode("GUIDED")
    with pytest.raises(UnsupportedPhaseOperation, match="setpoints.*disabled"):
        await bridge.send_velocity_setpoint(0.1, 0.0, -0.2, 0.0)

    # Stopping a stream that cannot exist is the sole safe hardware-side no-op.
    stop_result = await bridge.stop_external_setpoints()
    assert stop_result.ok
    assert stop_result.code == "NO_EXTERNAL_SETPOINTS"
    assert not bridge.get_status().connected
    assert not bridge.get_status().armed


@pytest.mark.asyncio
async def test_go2_real_bridge_connect_and_all_motion_paths_fail_closed() -> None:
    bridge = Go2BridgeStub()

    with pytest.raises(UnsupportedPhaseOperation, match="disabled.*Phase 1"):
        await bridge.connect()

    for operation in (
        bridge.request_stop,
        bridge.request_stand,
        bridge.request_flight_pose,
        bridge.request_landing_pose,
    ):
        with pytest.raises(UnsupportedPhaseOperation, match="disabled.*Phase 1"):
            await operation()

    assert not bridge.get_status().connected
    assert not bridge.get_status().controller_active


@pytest.mark.asyncio
async def test_real_bridge_run_loops_cannot_start_in_phase_one() -> None:
    pixhawk = ReadOnlyPixhawkBridge()
    go2 = Go2BridgeStub()

    with pytest.raises(UnsupportedPhaseOperation, match="disabled.*Phase 1"):
        await pixhawk.run()
    with pytest.raises(UnsupportedPhaseOperation, match="disabled.*Phase 1"):
        await go2.run()
