"""Application composition root for simulation and supervised hardware."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from aerogo2.cli.commands import build_registry
from aerogo2.cli.dispatcher import CommandDispatcher
from aerogo2.cli.history import CommandHistory
from aerogo2.cli.shell import InteractiveShell
from aerogo2.common.clock import Clock, ManualClock, RealClock
from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.enums import RuntimeMode
from aerogo2.hardware.runtime import HardwareWorld
from aerogo2.logging.ordered_event_sink import OrderedEventSink
from aerogo2.simulation.world import SimulationWorld


class AeroGo2Application:
    def __init__(
        self,
        config: AppConfig,
        runtime_mode: RuntimeMode = RuntimeMode.DRY_RUN,
    ) -> None:
        self.config = config
        self.clock: Clock
        self.world: Union[SimulationWorld, HardwareWorld]
        if runtime_mode is RuntimeMode.DRY_RUN:
            simulation_clock = ManualClock(
                10.0,
                wall_initial=RealClock().wall_time(),
            )
            self.clock = simulation_clock
            self.event_sink = OrderedEventSink(
                config.system.log_directory,
                clock=simulation_clock,
            )
            self.world = SimulationWorld(
                config,
                clock=simulation_clock,
                event_logger=self.event_sink,
            )
        else:
            self.clock = RealClock()
            self.event_sink = OrderedEventSink(
                config.system.log_directory,
                clock=self.clock,
            )
            self.world = HardwareWorld(
                config, runtime_mode=runtime_mode, event_logger=self.event_sink
            )
        history_path = config.system.log_directory / "command-history.jsonl"
        self.history = CommandHistory(history_path)
        self.registry = build_registry()
        self.dispatcher = CommandDispatcher(
            registry=self.registry,
            world=self.world,
            history=self.history,
            event_sink=self.event_sink,
        )
        self.shell = InteractiveShell(self.dispatcher)

    @classmethod
    def from_path(
        cls, path: Path, runtime_mode: RuntimeMode = RuntimeMode.DRY_RUN
    ) -> AeroGo2Application:
        return cls(load_config(path), runtime_mode=runtime_mode)


__all__ = ["AeroGo2Application"]
