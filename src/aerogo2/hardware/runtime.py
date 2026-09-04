"""Production hardware runtime with no autonomous actuator decisions."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Optional

from aerogo2.bridges.f446_text_bridge import TextF446Bridge
from aerogo2.bridges.go2_control_arbiter import Go2ControlArbiter
from aerogo2.bridges.go2_lowlevel_interface import Go2OwnershipPermit
from aerogo2.bridges.go2_lowlevel_sdk_bridge import UnitreeGo2LowLevelSdkBridge
from aerogo2.bridges.go2_sdk_bridge import UnitreeGo2Bridge
from aerogo2.bridges.pixhawk_mavlink_bridge import MavlinkPixhawkBridge
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.async_utils import await_nonabandonable
from aerogo2.common.clock import RealClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import RuntimeMode, SystemState
from aerogo2.common.models import ImpactLandingRecoveryEvidence, SystemSnapshot
from aerogo2.common.results import OperationResult
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.manager.system_manager import SystemManager
from aerogo2.safety.esc_telemetry import assess_esc_telemetry
from aerogo2.safety.pixhawk_freshness import pixhawk_ground_state_is_current
from aerogo2.safety.watchdog import timestamp_is_fresh


class HardwareWorld:
    """Own real bridges while preserving the simulation manager boundary."""

    def __init__(
        self,
        config: AppConfig,
        *,
        runtime_mode: RuntimeMode,
        event_logger: Optional[Any] = None,
        impact_recovery_source: Optional[Callable[[], ImpactLandingRecoveryEvidence]] = None,
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
        self.go2_control_arbiter = Go2ControlArbiter(
            domain_id=config.go2.domain_id,
            network_interface=config.go2.network_interface,
        )
        self.go2 = UnitreeGo2Bridge(
            config.go2,
            clock=self.clock,
            allow_control=writes,
            control_arbiter=self.go2_control_arbiter,
        )
        self.go2_low_level = UnitreeGo2LowLevelSdkBridge(
            config.go2,
            arbiter=self.go2_control_arbiter,
            clock=self.clock,
            allow_hardware_write=writes,
            ground_transfer_verifier=self._verify_go2_ground_transfer,
        )
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
            go2_low_level=self.go2_low_level,
            impact_recovery_source=impact_recovery_source,
        )
        self._impact_recovery_source_bound = impact_recovery_source is not None

    def _verify_go2_ground_transfer(
        self, transfer: str, permit: Go2OwnershipPermit
    ) -> OperationResult:
        """Synchronously re-read independent ground evidence at mode boundaries."""

        if transfer not in {"acquire", "release"}:
            return OperationResult.failure(
                "GO2_LIVE_GROUND_TRANSFER_INVALID",
                f"Unknown LowCmd transfer phase {transfer!r}",
            )
        if not isinstance(permit, Go2OwnershipPermit):
            return OperationResult.failure(
                "GO2_LIVE_GROUND_PERMIT_INVALID",
                "Live ground verification requires a typed ownership permit",
            )
        now = self.clock.monotonic()
        try:
            pixhawk = self.pixhawk.get_status()
            f446 = self.f446.get_status()
            go2 = self.go2.get_status()
        except Exception as exc:
            return OperationResult.failure(
                "GO2_LIVE_GROUND_STATUS_FAILED",
                f"A hardware status source raised {type(exc).__name__}: {exc}",
            )

        if not pixhawk_ground_state_is_current(
            pixhawk,
            now,
            self.config.safety.pixhawk_timeout_s,
            self.config.safety.touchdown_max_source_age_s,
        ):
            return OperationResult.failure(
                "GO2_LIVE_PIXHAWK_STALE",
                "Live Pixhawk heartbeat or landed state is disconnected or stale",
            )
        if pixhawk.armed or not pixhawk.landed:
            return OperationResult.failure(
                "GO2_LIVE_GROUND_STATE_UNSAFE",
                "Live Pixhawk evidence must remain disarmed and landed",
            )
        esc_snapshot = SystemSnapshot(
            timestamp=now,
            state=SystemState.BOOT_SAFE,
            pixhawk=pixhawk,
        )
        esc_assessment = assess_esc_telemetry(
            esc_snapshot,
            self.config.esc.slots,
            exact_zero=True,
        )
        if not esc_assessment.safe or any(
            not timestamp_is_fresh(
                now,
                item.timestamp,
                self.config.safety.pixhawk_timeout_s,
            )
            for item in pixhawk.esc
        ):
            return OperationResult.failure(
                "GO2_LIVE_ROTORS_NOT_STOPPED",
                "Every mapped ESC must remain fresh, healthy, and exactly zero RPM",
            )
        if not f446.connected or not timestamp_is_fresh(
            now,
            f446.timestamp,
            self.config.safety.f446_timeout_s,
        ):
            return OperationResult.failure(
                "GO2_LIVE_F446_STALE",
                "Live F446 deployment status is disconnected or stale",
            )
        if (
            f446.faulted
            or f446.duty != 0
            or f446.state is not self.config.f446.expected_flight_state
        ):
            return OperationResult.failure(
                "GO2_LIVE_F446_UNSAFE",
                "Rotor arms must remain fully deployed, stopped, and fault-free",
            )
        if transfer == "acquire":
            if not go2.connected or not timestamp_is_fresh(
                now,
                go2.timestamp,
                self.config.safety.go2_timeout_s,
            ):
                return OperationResult.failure(
                    "GO2_LIVE_SPORT_STATE_STALE",
                    "Fresh high-level Go2 state is required before ReleaseMode",
                )
            if not go2.joints_locked or not go2.stable or go2.moving:
                return OperationResult.failure(
                    "GO2_LIVE_JOINT_LOCK_REQUIRED",
                    "Go2 must remain stationary in authoritative mode=6 JOINT_LOCK before ReleaseMode",
                )
        return OperationResult.success(
            "Independent live ground evidence is fresh and safe",
            {
                "transfer": transfer,
                "checked_at_s": now,
                "mechanical_support_is_operator_attested": True,
            },
            code="GO2_LIVE_GROUND_VERIFIED",
        )

    def status(self) -> Mapping[str, Any]:
        return {
            "started": self.manager.started,
            "state": self.manager.state.name,
            "runtime_mode": self.manager.runtime_mode.value,
            "hardware_write_enabled": self.config.system.hardware_write_enabled,
            "go2_low_state_observation_enabled": (self.config.go2.low_level.observation_enabled),
            "go2_lowcmd_actuation_enabled": self.config.go2.low_level.enabled,
            "go2_lowcmd_actuation_ready": self.go2_low_level.actuation_readiness().ok,
            "go2_low_level": self.go2_low_level.status().ownership_state.value,
            "impact_recovery_source_bound": self._impact_recovery_source_bound,
        }

    async def start(self) -> OperationResult:
        await self.manager.start()
        return await self.manager.connect_all()

    async def shutdown(self) -> OperationResult:
        if self.manager.started:
            result = await self.manager.shutdown()
        else:
            # InteractiveShell closes the manager before control returns to
            # async_main. Do not issue a second supervised stop against
            # already-disconnected actuator bridges; only finish the separate
            # observe-only LowState reader teardown.
            try:
                ownership_pending = self.go2_low_level.status().ownership_pending
            except Exception as exc:
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_STATUS_UNAVAILABLE",
                    "Cannot prove LowCmd ownership clear after manager shutdown: "
                    f"{type(exc).__name__}: {exc}",
                )
            if ownership_pending:
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_OWNER_ACTIVE",
                    "Manager is stopped but LowCmd ownership is still pending; "
                    "the process must remain resident",
                )
            result = OperationResult.success(
                "Manager was already shut down; finishing LowState transport cleanup"
            )
        # A failed manager shutdown can mean that the airborne/unsupported
        # LowCmd writer still owns the robot.  Disconnecting it here would
        # defeat the ground-only release gate and remove the conservative
        # hold stream.  Leave every transport alive until the manager has
        # positively completed the handover.
        if not result.ok:
            return result
        low_level_result = await self.go2_low_level.disconnect()
        if not low_level_result.ok:
            return low_level_result
        return result

    async def shutdown_until_safe(self) -> OperationResult:
        """Never abandon a pending owner, even after repeated task cancellation."""

        task = asyncio.ensure_future(self._shutdown_until_safe_loop())
        stopped, cancelled = await await_nonabandonable(task)
        if cancelled:
            raise asyncio.CancelledError
        return stopped

    async def _shutdown_until_safe_loop(self) -> OperationResult:
        """Perform the resident shutdown loop used by the cancellation shield."""

        try:
            stopped = await self.shutdown()
        except Exception as exc:
            stopped = OperationResult.failure(
                "HARDWARE_SHUTDOWN_EXCEPTION",
                f"{type(exc).__name__}: {exc}",
            )
        interval = 1.0 / self.config.system.loop_hz
        while not stopped.ok:
            try:
                ownership_pending = self.go2_low_level.status().ownership_pending
            except Exception:
                # Status loss is ambiguous, never proof that the sole writer
                # or its epoch disappeared.
                ownership_pending = True
            if not ownership_pending:
                try:
                    stopped = await self.shutdown()
                except Exception as exc:
                    stopped = OperationResult.failure(
                        "HARDWARE_SHUTDOWN_EXCEPTION",
                        f"{type(exc).__name__}: {exc}",
                    )
                break
            try:
                await self.manager.tick()
            except Exception:
                # The fixed-rate owner has its own watchdog; keep the process
                # resident even if this observation cycle failed.
                pass
            await asyncio.sleep(interval)
            try:
                ownership_pending = self.go2_low_level.status().ownership_pending
            except Exception:
                ownership_pending = True
            if not ownership_pending:
                try:
                    stopped = await self.shutdown()
                except Exception as exc:
                    stopped = OperationResult.failure(
                        "HARDWARE_SHUTDOWN_EXCEPTION",
                        f"{type(exc).__name__}: {exc}",
                    )
        return stopped

    async def monitor_until_stopped(self, stop_event: asyncio.Event) -> OperationResult:
        try:
            # start() may leave a subset of transports connected when
            # connect_all() fails.  Put the whole startup transaction inside
            # the cleanup boundary so both a failure result and an exception
            # still receive the same ownership-aware shutdown handling.
            started = await self.start()
            if not started.ok:
                return started
            interval = 1.0 / self.config.system.loop_hz
            while not stop_event.is_set():
                await self.manager.tick()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            stopped = await self.shutdown_until_safe()
        return stopped


__all__ = ["HardwareWorld"]
