"""Production hardware runtime with no autonomous actuator decisions."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional

from aerogo2.bridges.f446_text_bridge import TextF446Bridge
from aerogo2.bridges.go2_sdk_bridge import UnitreeGo2Bridge
from aerogo2.bridges.pixhawk_mavlink_bridge import MavlinkPixhawkBridge
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import RealClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import RuntimeMode
from aerogo2.common.results import OperationResult
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.manager.system_manager import SystemManager


class HardwareWorld:
    """Own real bridges while preserving the simulation manager boundary."""

    def __init__(
        self,
        config: AppConfig,
        *,
        runtime_mode: RuntimeMode,
        event_logger: Optional[Any] = None,
    ) -> None:
        if runtime_mode not in (RuntimeMode.HARDWARE, RuntimeMode.HARDWARE_READONLY):
            raise ValueError("HardwareWorld requires a hardware runtime mode")
        self.config = config
        self.clock = RealClock()
        writes = runtime_mode is RuntimeMode.HARDWARE and config.system.hardware_write_enabled
        self.pixhawk = MavlinkPixhawkBridge(
            config.pixhawk,
            config.esc.slots,
            esc_mavlink_display_shift=config.esc.mavlink_display_shift,
            clock=self.clock,
            rc_timeout_s=config.safety.rc_timeout_s,
            allow_setpoints=False,
        )
        self.f446 = TextF446Bridge(config.f446, self.clock, allow_motion=writes)
        self.go2 = UnitreeGo2Bridge(config.go2, clock=self.clock, allow_control=writes)
        self.rc_monitor = RCMonitor(config.rc, self.clock)
        self.manager = SystemManager(
            config=config,
            pixhawk=self.pixhawk,
            f446=self.f446,
            go2=self.go2,
            landing_controller=SafeDescentController(config),
            clock=self.clock,
            runtime_mode=runtime_mode,
            event_logger=event_logger,
            rc_monitor=self.rc_monitor,
        )

    def status(self) -> Mapping[str, Any]:
        return {
            "started": self.manager.started,
            "state": self.manager.state.name,
            "runtime_mode": self.manager.runtime_mode.value,
            "hardware_write_enabled": self.config.system.hardware_write_enabled,
        }

    async def start(self) -> OperationResult:
        await self.manager.start()
        return await self.manager.connect_all()

    async def shutdown(self) -> OperationResult:
        return await self.manager.shutdown()

    async def monitor_until_stopped(self, stop_event: asyncio.Event) -> OperationResult:
        started = await self.start()
        if not started.ok:
            return started
        interval = 1.0 / self.config.system.loop_hz
        try:
            while not stop_event.is_set():
                await self.manager.tick()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            stopped = await self.shutdown()
        return stopped


__all__ = ["HardwareWorld"]
