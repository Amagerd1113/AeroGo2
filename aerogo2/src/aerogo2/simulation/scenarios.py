"""Scenario names and descriptions exposed to the console."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str


SCENARIOS: Mapping[str, ScenarioDefinition] = MappingProxyType(
    {
        "nominal": ScenarioDefinition(
            "nominal",
            "Complete WALK -> FLIGHT -> simulated autoland -> WALK mission",
        ),
        "transform-failure": ScenarioDefinition(
            "transform-failure",
            "F446 transform timeout followed by supervised stop and FAULT",
        ),
        "rc-loss": ScenarioDefinition(
            "rc-loss",
            "RC failsafe during automatic landing and immediate manual-control return",
        ),
        "pixhawk-timeout": ScenarioDefinition(
            "pixhawk-timeout",
            "Stale Pixhawk telemetry blocks a new morphology transition",
        ),
        "f446-overcurrent": ScenarioDefinition(
            "f446-overcurrent",
            "F446 simulated overcurrent/fault during morphology movement",
        ),
        "landing": ScenarioDefinition(
            "landing",
            "Automatic landing output followed by RadioMaster manual override",
        ),
    }
)


def scenario_names() -> Tuple[str, ...]:
    return tuple(sorted(SCENARIOS))


__all__ = ["SCENARIOS", "ScenarioDefinition", "scenario_names"]
