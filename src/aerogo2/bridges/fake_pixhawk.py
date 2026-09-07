"""Deterministic Pixhawk simulator used exclusively by Phase 1 dry-run mode."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple, cast

from aerogo2.bridges.pixhawk_interface import VelocitySetpoint
from aerogo2.common.clock import Clock, RealClock
from aerogo2.common.config import X8_ESC_SLOT_MAPPING
from aerogo2.common.models import EscTelemetry, PixhawkStatus
from aerogo2.common.results import OperationResult


class FakePixhawk:
    """In-memory Pixhawk with an explicitly separate simulation control plane.

    ``SystemManager`` should use only methods from ``PixhawkInterface``.
    Methods beginning with ``inject_`` are for ``SimulationWorld`` and tests;
    notably, the production interface exposes no arm or disarm operation.
    """

    def __init__(
        self,
        clock: Optional[Clock] = None,
        external_setpoints_enabled: bool = True,
        esc_mapping: Optional[Mapping[int, str]] = None,
    ) -> None:
        self._clock = clock or RealClock()
        mapping = dict(esc_mapping or X8_ESC_SLOT_MAPPING)
        esc = tuple(
            EscTelemetry(slot, mapping[slot], timestamp=self._clock.monotonic())
            for slot in sorted(mapping)
        )
        self._status = PixhawkStatus(esc=esc)
        self._external_setpoints_enabled = external_setpoints_enabled
        self._external_setpoints_active = False
        self._setpoint_history: list[VelocitySetpoint] = []
        self._mode_history: list[str] = []
        self._ground_arm_authorization_until = 0.0
        self._shutdown = asyncio.Event()

    async def connect(self) -> None:
        now = self._clock.monotonic()
        self._shutdown.clear()
        self.inject_status(
            connected=True,
            heartbeat_timestamp=now,
            esc=tuple(replace(item, timestamp=now) for item in self._status.esc),
        )

    async def disconnect(self) -> None:
        self._external_setpoints_active = False
        self._ground_arm_authorization_until = 0.0
        self.inject_status(connected=False)
        self._shutdown.set()

    async def run(self) -> None:
        await self._shutdown.wait()

    def get_status(self) -> PixhawkStatus:
        return self._status

    def latest_status(self) -> PixhawkStatus:
        return self.get_status()

    @property
    def external_setpoints_active(self) -> bool:
        return self._external_setpoints_active

    @property
    def setpoint_history(self) -> Tuple[VelocitySetpoint, ...]:
        return tuple(self._setpoint_history)

    @property
    def mode_history(self) -> Tuple[str, ...]:
        return tuple(self._mode_history)

    async def request_mode(self, mode: str) -> bool:
        clean_mode = mode.strip().upper()
        if not self._status.connected or not clean_mode:
            return False
        self._mode_history.append(clean_mode)
        self.inject_status(flight_mode=clean_mode)
        return True

    async def set_ground_arm_authorization(
        self,
        enabled: bool,
        ttl_s: float,
    ) -> OperationResult:
        if enabled:
            if not self._status.connected:
                return OperationResult.failure(
                    "PIXHAWK_DISCONNECTED",
                    "Cannot authorize flight while FakePixhawk is disconnected",
                )
            if not math.isfinite(ttl_s) or ttl_s <= 0.0:
                return OperationResult.failure(
                    "INVALID_AUTHORIZATION_TTL",
                    "Ground-arm authorization TTL must be finite and positive",
                )
            self._ground_arm_authorization_until = self._clock.monotonic() + ttl_s
            return OperationResult.success(
                "FakePixhawk ground-arm authorization enabled",
                data={"ttl_s": ttl_s},
            )
        self._ground_arm_authorization_until = 0.0
        return OperationResult.success("FakePixhawk ground-arm authorization revoked")

    def ground_arm_authorization_active(self) -> bool:
        return self._status.connected and (
            self._ground_arm_authorization_until > self._clock.monotonic()
        )

    async def send_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ) -> OperationResult:
        if not self._external_setpoints_enabled:
            return OperationResult.failure(
                "SETPOINT_FEATURE_DISABLED",
                "External setpoints are disabled for this Pixhawk instance",
            )
        if not self._status.connected:
            return OperationResult.failure(
                "PIXHAWK_DISCONNECTED",
                "Cannot record a setpoint while FakePixhawk is disconnected",
            )
        values = (vx, vy, vz, yaw_rate)
        if not all(math.isfinite(value) for value in values):
            return OperationResult.failure(
                "INVALID_SETPOINT",
                "Velocity setpoint values must all be finite",
            )
        command = VelocitySetpoint(
            timestamp=self._clock.monotonic(),
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            yaw_rate=float(yaw_rate),
        )
        self._setpoint_history.append(command)
        self._external_setpoints_active = True
        return OperationResult.success(
            "Setpoint recorded by FakePixhawk",
            data={
                "vx": command.vx,
                "vy": command.vy,
                "vz": command.vz,
                "yaw_rate": command.yaw_rate,
            },
        )

    async def stop_external_setpoints(self) -> OperationResult:
        was_active = self._external_setpoints_active
        self._external_setpoints_active = False
        return OperationResult.success(
            "FakePixhawk external setpoints stopped",
            data={"was_active": was_active},
        )

    def inject_status(self, **changes: object) -> PixhawkStatus:
        """Replace telemetry fields and keep public/compatibility views coherent."""

        supplied_fields = frozenset(changes)
        if "rc_failsafe" in changes and "failsafe" not in changes:
            changes["failsafe"] = changes["rc_failsafe"]
        if "attitude_rpy" in changes:
            roll, pitch, yaw = cast(Tuple[float, float, float], changes["attitude_rpy"])
            changes.setdefault("roll_rad", roll)
            changes.setdefault("pitch_rad", pitch)
            changes.setdefault("yaw_rad", yaw)
        if "local_velocity" in changes and "vertical_velocity_mps" not in changes:
            local_velocity = cast(Tuple[float, float, float], changes["local_velocity"])
            changes["vertical_velocity_mps"] = local_velocity[2]
        if "local_position" in changes and "relative_altitude_m" not in changes:
            local_position = cast(Tuple[float, float, float], changes["local_position"])
            changes["relative_altitude_m"] = -local_position[2]

        now = self._clock.monotonic()
        if supplied_fields.intersection(
            {"attitude_rpy", "roll_rad", "pitch_rad", "yaw_rad", "angular_velocity"}
        ):
            changes.setdefault("attitude_timestamp", now)
        if supplied_fields.intersection(
            {
                "local_position",
                "local_velocity",
                "relative_altitude_m",
                "vertical_velocity_mps",
            }
        ):
            changes.setdefault("kinematics_timestamp", now)
        if "landed" in supplied_fields:
            changes.setdefault("landed_state_timestamp", now)
        if "esc_rpm" in changes or "esc_online" in changes:
            rpm_by_slot = cast(
                Mapping[int, float],
                changes.get("esc_rpm", self._status.esc_rpm),
            )
            online_by_slot = cast(
                Mapping[int, bool],
                changes.get("esc_online", self._status.esc_online),
            )
            source_esc = cast(
                Tuple[EscTelemetry, ...],
                changes.get("esc", self._status.esc),
            )
            changes["esc"] = tuple(
                replace(
                    item,
                    rpm=float(rpm_by_slot.get(item.slot, item.rpm)),
                    healthy=bool(online_by_slot.get(item.slot, item.healthy)),
                    timestamp=now,
                )
                for item in source_esc
            )
        heartbeat_fields = {
            "heartbeat_timestamp",
            "connected",
            "armed",
            "flight_mode",
            "failsafe",
            "rc_failsafe",
        }
        if not supplied_fields or supplied_fields.intersection(heartbeat_fields):
            changes.setdefault("heartbeat_timestamp", now)
        status = replace(self._status, **changes)  # type: ignore[arg-type]
        self._status = replace(
            status,
            timestamp=status.heartbeat_timestamp,
            message_age_s=0.0,
            rc_failsafe=status.failsafe,
            attitude_rpy=(status.roll_rad, status.pitch_rad, status.yaw_rad),
            local_position=(
                status.local_position[0],
                status.local_position[1],
                -status.relative_altitude_m,
            ),
            local_velocity=(
                status.local_velocity[0],
                status.local_velocity[1],
                status.vertical_velocity_mps,
            ),
            esc_rpm={item.slot: item.rpm for item in status.esc},
            esc_online={item.slot: status.connected and item.healthy for item in status.esc},
        )
        if not self._status.connected:
            self._external_setpoints_active = False
        return self._status

    def inject_connection(self, connected: bool) -> PixhawkStatus:
        return self.inject_status(connected=connected)

    def inject_armed_state(self, armed: bool) -> PixhawkStatus:
        """Simulate telemetry caused by the independent RadioMaster link."""

        return self.inject_status(armed=armed)

    def inject_landed_state(
        self,
        landed: bool,
        vertical_velocity_mps: Optional[float] = None,
        relative_altitude_m: Optional[float] = None,
    ) -> PixhawkStatus:
        changes: Dict[str, object] = {"landed": landed}
        if vertical_velocity_mps is not None:
            if not math.isfinite(vertical_velocity_mps):
                raise ValueError("Vertical velocity must be finite")
            changes["vertical_velocity_mps"] = vertical_velocity_mps
        if relative_altitude_m is not None:
            if not math.isfinite(relative_altitude_m):
                raise ValueError("Relative altitude must be finite")
            changes["relative_altitude_m"] = relative_altitude_m
        return self.inject_status(**changes)

    def inject_failsafe(self, active: bool) -> PixhawkStatus:
        return self.inject_status(failsafe=active)

    def inject_attitude(self, roll_rad: float, pitch_rad: float, yaw_rad: float) -> PixhawkStatus:
        if not all(math.isfinite(value) for value in (roll_rad, pitch_rad, yaw_rad)):
            raise ValueError("Attitude values must be finite")
        return self.inject_status(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)

    def inject_esc_rpm(self, slot: int, rpm: float, healthy: bool = True) -> PixhawkStatus:
        if not math.isfinite(rpm):
            raise ValueError("ESC RPM must be finite")
        found = False
        updated = []
        now = self._clock.monotonic()
        for item in self._status.esc:
            if item.slot == slot:
                updated.append(replace(item, rpm=float(rpm), healthy=healthy, timestamp=now))
                found = True
            else:
                updated.append(item)
        if not found:
            raise ValueError(f"Unknown ESC slot {slot}")
        return self.inject_status(esc=tuple(updated))

    def inject_heartbeat(self) -> PixhawkStatus:
        return self.inject_status()

    def inject_telemetry_cycle(self) -> PixhawkStatus:
        """Refresh all independently timed touchdown sources in simulation."""

        now = self._clock.monotonic()
        return self.inject_status(
            heartbeat_timestamp=now,
            attitude_timestamp=now,
            kinematics_timestamp=now,
            landed_state_timestamp=now,
        )

    def clear_history(self) -> None:
        self._setpoint_history.clear()
        self._mode_history.clear()


__all__ = ["FakePixhawk"]
