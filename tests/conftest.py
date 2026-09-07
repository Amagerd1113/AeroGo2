from __future__ import annotations

from pathlib import Path

import pytest

from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig, load_config
from aerogo2.common.enums import (
    Configuration,
    F446State,
    MorphologyRequest,
    SystemState,
)
from aerogo2.common.models import (
    F446Status,
    Go2Status,
    PixhawkStatus,
    RCStatus,
    SystemSnapshot,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def app_config(project_root: Path) -> AppConfig:
    return load_config(project_root / "configs" / "system.yaml")


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(10.0)


@pytest.fixture
def safe_walk_snapshot(app_config: AppConfig) -> SystemSnapshot:
    now = 10.0
    return SystemSnapshot(
        timestamp=now,
        state=SystemState.WALK,
        pixhawk=PixhawkStatus(
            connected=True,
            armed=False,
            landed=True,
            heartbeat_timestamp=now,
            attitude_timestamp=now,
            kinematics_timestamp=now,
            landed_state_timestamp=now,
            esc_rpm={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            esc_online={1: True, 2: True, 3: True, 4: True},
        ),
        f446=F446Status(
            connected=True,
            state=F446State.LIMIT_REACHED_REV,
            duty=0,
            timestamp=now,
            used_current_adc=0,
            threshold_adc=1800,
        ),
        go2=Go2Status(
            connected=True,
            velocity_mps=0.0,
            stable=True,
            timestamp=now,
        ),
        rc=RCStatus(
            connected=True,
            failsafe=False,
            channels={app_config.rc.flight_enable_channel: 1000},
            flight_enable=False,
            morphology_request=MorphologyRequest.FLIGHT_REQUEST,
            timestamp=now,
        ),
        configuration=Configuration.WALK,
    )
