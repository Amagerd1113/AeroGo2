"""Run a deterministic Phase 1 scenario without opening an interactive prompt."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional, Sequence

from aerogo2.app import AeroGo2Application


async def run(config: Path, scenario: str) -> int:
    app = AeroGo2Application.from_path(config)
    result = await app.world.run_scenario(scenario)
    print(f"Scenario : {result.name}")
    print("Result   : {}".format("PASS" if result.ok else "FAIL"))
    print(f"Final    : {result.final_state.name}")
    print("States   : {}".format(" -> ".join(state.name for state in result.states)))
    for message in result.messages:
        print(f"Message  : {message}")
    await app.world.shutdown()
    return 0 if result.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--scenario", default="nominal")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.config, args.scenario))


if __name__ == "__main__":
    raise SystemExit(main())
