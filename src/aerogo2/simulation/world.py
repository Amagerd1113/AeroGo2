"""A deterministic world that exclusively owns Fake-device injection methods."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.bridges.rc_monitor import RCMonitor
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import AppConfig
from aerogo2.common.enums import SystemState
from aerogo2.common.immutable import frozen_mapping
from aerogo2.common.models import LandingEstimate
from aerogo2.common.results import OperationResult
from aerogo2.landing.safe_descent_controller import SafeDescentController
from aerogo2.manager.system_manager import SystemManager
from aerogo2.simulation.fault_injection import SimulatedFault
from aerogo2.simulation.scenarios import SCENARIOS


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    ok: bool
    final_state: SystemState
    states: Tuple[SystemState, ...]
    messages: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "details", frozen_mapping(self.details))


class SimulationWorld:
    """Own the simulation control plane; SystemManager sees only interfaces."""

    def __init__(
        self,
        config: AppConfig,
        *,
        clock: Optional[ManualClock] = None,
        event_logger: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.clock = ManualClock(10.0) if clock is None else clock
        self._event_logger = event_logger
        self.selected_scenario = "nominal"
        self.paused = False
        self._started = False
        self._channels: Dict[int, int] = {}
        self._landing_estimate: Optional[LandingEstimate] = None
        self._mutation_lock = asyncio.Lock()
        self._scenario_lock = asyncio.Lock()
        self._scenario_task: Optional[asyncio.Task[Any]] = None
        self._build_components()

    def _build_components(self) -> None:
        self._landing_estimate = None
        self.pixhawk = FakePixhawk(
            clock=self.clock,
            esc_mapping=self.config.esc.slots,
        )
        self.f446 = FakeF446(config=self.config.f446, clock=self.clock)
        self.go2 = FakeGo2(clock=self.clock)
        self.rc_monitor = RCMonitor(self.config.rc, self.clock)
        self.manager = SystemManager(
            config=self.config,
            pixhawk=self.pixhawk,
            f446=self.f446,
            go2=self.go2,
            landing_controller=SafeDescentController(self.config),
            clock=self.clock,
            event_logger=self._event_logger,
        )
        self._channels = self._safe_channels()

    async def start(self) -> OperationResult:
        if self._scenario_busy_for_caller():
            return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
            return await self._start_unlocked()

    async def _start_unlocked(self) -> OperationResult:
        if self._started:
            return OperationResult.success("Simulation already started")
        await self.manager.start()
        self._feed_rc(debounce=True)
        result = await self.manager.connect_all()
        self._started = result.ok
        return result

    async def shutdown(self) -> OperationResult:
        if self._scenario_busy_for_caller():
            return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
            return await self._shutdown_unlocked()

    async def _shutdown_unlocked(self) -> OperationResult:
        if not self.manager.started:
            return OperationResult.success("Simulation is not running")
        result = await self.manager.shutdown()
        self._started = False
        return result

    async def reset(self, start: bool = False) -> OperationResult:
        if self._scenario_busy_for_caller():
            return OperationResult.failure(
                "SCENARIO_RUNNING",
                "Simulation reset is inhibited while a scenario is running",
            )
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure(
                    "SCENARIO_RUNNING",
                    "Simulation reset is inhibited while a scenario is running",
                )
            return await self._reset_unlocked(start=start)

    async def _reset_unlocked(self, start: bool = False) -> OperationResult:
        if self.manager.started:
            shutdown = await self.manager.shutdown()
            if not shutdown.ok:
                return OperationResult.failure(
                    "SIM_RESET_INHIBITED",
                    f"Current simulation did not shut down safely: {shutdown.message}",
                )
        self._build_components()
        self.paused = False
        self._started = False
        if start:
            return await self._start_unlocked()
        return OperationResult.success("Simulation reset to disconnected BOOT_SAFE")

    def _scenario_busy_for_caller(self) -> bool:
        return self.scenario_running and asyncio.current_task() is not self._scenario_task

    def status(self) -> Mapping[str, Any]:
        return {
            "started": self.manager.started,
            "paused": self.paused,
            "selected_scenario": self.selected_scenario,
            "scenario_running": self.scenario_running,
            "state": self.manager.state.name,
            "clock": self.clock.monotonic(),
        }

    @property
    def scenario_running(self) -> bool:
        return self._scenario_lock.locked()

    async def select_scenario(self, name: str) -> OperationResult:
        if self._scenario_busy_for_caller():
            return OperationResult.failure(
                "SCENARIO_RUNNING", "Cannot change scenario while one is running"
            )
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure(
                    "SCENARIO_RUNNING", "Cannot change scenario while one is running"
                )
            return self._select_scenario_unlocked(name)

    def _select_scenario_unlocked(self, name: str) -> OperationResult:
        normalized = name.strip().lower()
        if normalized not in SCENARIOS:
            return OperationResult.failure(
                "UNKNOWN_SCENARIO",
                "Unknown scenario '{}'; choose {}".format(name, ", ".join(sorted(SCENARIOS))),
            )
        self.selected_scenario = normalized
        return OperationResult.success(f"Selected scenario {normalized}")

    async def pause(self) -> OperationResult:
        return await self._set_paused(True)

    async def resume(self) -> OperationResult:
        return await self._set_paused(False)

    async def _set_paused(self, paused: bool) -> OperationResult:
        if self._scenario_busy_for_caller():
            return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure("SCENARIO_RUNNING", "A scenario is running")
            self.paused = paused
            return OperationResult.success("Simulation paused" if paused else "Simulation resumed")

    async def run_selected(self) -> ScenarioResult:
        return await self.run_scenario(self.selected_scenario)

    async def run_scenario(self, name: str) -> ScenarioResult:
        normalized = name.strip().lower()
        if normalized not in SCENARIOS:
            return ScenarioResult(
                normalized,
                False,
                self.manager.state,
                (self.manager.state,),
                ("unknown scenario",),
            )
        if self.scenario_running:
            return ScenarioResult(
                normalized,
                False,
                self.manager.state,
                (self.manager.state,),
                ("another scenario is already running",),
                {"code": "SCENARIO_RUNNING"},
            )
        async with self._scenario_lock:
            self._scenario_task = asyncio.current_task()
            try:
                async with self._mutation_lock:
                    reset = await self._reset_unlocked(start=True)
                    if not reset.ok:
                        return ScenarioResult(
                            normalized,
                            False,
                            self.manager.state,
                            (self.manager.state,),
                            (reset.message,),
                        )
                    handlers = {
                        "nominal": self._scenario_nominal,
                        "transform-failure": self._scenario_transform_failure,
                        "rc-loss": self._scenario_rc_loss,
                        "pixhawk-timeout": self._scenario_pixhawk_timeout,
                        "f446-overcurrent": self._scenario_f446_overcurrent,
                        "landing": self._scenario_landing_override,
                    }
                    return await handlers[normalized]()
            finally:
                self._scenario_task = None

    async def step(self, seconds: float = 0.05) -> OperationResult:
        if not math.isfinite(seconds) or seconds <= 0:
            return OperationResult.failure(
                "INVALID_STEP",
                "Step duration must be finite and positive",
            )
        if asyncio.current_task() is self._scenario_task:
            return await self._step_unlocked(seconds)
        if self._scenario_busy_for_caller():
            return OperationResult.failure(
                "SCENARIO_RUNNING",
                "Background stepping is suspended while a scenario is running",
            )
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure(
                    "SCENARIO_RUNNING",
                    "Background stepping is suspended while a scenario is running",
                )
            return await self._step_unlocked(seconds)

    async def _step_unlocked(self, seconds: float) -> OperationResult:
        if self.paused:
            return OperationResult.failure("SIM_PAUSED", "Simulation is paused")
        if not self.manager.started:
            return OperationResult.failure(
                "SIM_NOT_RUNNING",
                "Simulation manager is not started; start or clear the simulation first",
            )
        next_monotonic = self.clock.monotonic() + seconds
        next_wall = self.clock.wall_time() + seconds
        if not math.isfinite(next_monotonic) or not math.isfinite(next_wall):
            return OperationResult.failure(
                "INVALID_STEP",
                "Step duration would overflow the simulation clock",
            )
        self.clock.advance(seconds)
        self._heartbeat_all()
        self._feed_rc(debounce=False)
        await self.manager.tick()
        return OperationResult.success(f"Simulation advanced {seconds:.3f}s")

    async def inject(self, fault: SimulatedFault) -> OperationResult:
        """Inject a fault without racing resets, steps, or scenario transactions."""
        if asyncio.current_task() is self._scenario_task:
            return self._inject_unlocked(fault)
        if self._scenario_busy_for_caller():
            return OperationResult.failure(
                "SCENARIO_RUNNING",
                "Fault injection is unavailable while a scenario is running",
            )
        async with self._mutation_lock:
            if self._scenario_busy_for_caller():
                return OperationResult.failure(
                    "SCENARIO_RUNNING",
                    "Fault injection is unavailable while a scenario is running",
                )
            if not self.manager.started:
                return OperationResult.failure(
                    "SIM_NOT_RUNNING",
                    "Simulation manager is not started; start or clear the simulation first",
                )
            return self._inject_unlocked(fault)

    def _inject_unlocked(self, fault: SimulatedFault) -> OperationResult:
        if fault is SimulatedFault.RC_LOSS:
            status = self.rc_monitor.update(
                self._channels,
                failsafe=True,
                connected=True,
                timestamp=self.clock.monotonic(),
            )
            self.manager.accept_rc_status(status)
        elif fault is SimulatedFault.PIXHAWK_TIMEOUT:
            self.clock.advance(self.config.safety.pixhawk_timeout_s + 0.01)
            self._heartbeat_except("pixhawk")
        elif fault is SimulatedFault.F446_TIMEOUT:
            self.f446.inject_next_transform_timeout()
        elif fault is SimulatedFault.F446_OVERCURRENT:
            self.f446.inject_next_transform_fault("simulated overcurrent", code="F446_OVERCURRENT")
        elif fault is SimulatedFault.F446_WRONG_FINAL_STATE:
            self.f446.inject_next_wrong_final_state()
        elif fault is SimulatedFault.F446_NONZERO_FINAL_DUTY:
            self.f446.inject_next_nonzero_final_duty()
        elif fault is SimulatedFault.GO2_MOVING:
            self.go2.inject_motion(0.2, stable=False)
        elif fault is SimulatedFault.ESC_RPM_NONZERO:
            self.pixhawk.inject_esc_rpm(1, 100.0)
        elif fault is SimulatedFault.MANUAL_OVERRIDE:
            self._channels[self.config.rc.auto_landing_channel] = 1000
            self._feed_rc(debounce=True)
        return OperationResult.success(f"Injected {fault.value}")

    async def _scenario_nominal(self) -> ScenarioResult:
        states: List[SystemState] = [self.manager.state]
        messages: List[str] = []
        result = await self._reach_flight_manual(states)
        messages.append(result.message)
        if not result.ok:
            return self._result("nominal", False, states, messages)

        result = await self._reach_autoland(states)
        messages.append(result.message)
        if not result.ok:
            return self._result("nominal", False, states, messages)

        self.pixhawk.inject_landed_state(True, vertical_velocity_mps=0.0, relative_altitude_m=0.0)
        self.pixhawk.inject_attitude(0.0, 0.0, 0.0)
        for slot in self.config.esc.slots:
            self.pixhawk.inject_esc_rpm(slot, 0.0)
        self._set_landing_estimate(
            LandingEstimate(
                valid=True,
                ground_detected=True,
                height_m=0.0,
                vertical_velocity_mps=0.0,
                horizontal_velocity_mps=0.0,
                timestamp=self.clock.monotonic(),
                reason="simulated ground contact",
            )
        )
        iterations = int(self.config.safety.touchdown_confirm_s / 0.05) + 2
        for _ in range(iterations):
            await self.step(0.05)
            if self.manager.state is SystemState.TOUCHDOWN_VERIFY:
                break
        states.append(self.manager.state)
        if self.manager.state is not SystemState.TOUCHDOWN_VERIFY:
            return self._result(
                "nominal", False, states, messages + ["touchdown was not confirmed"]
            )

        self.pixhawk.inject_armed_state(False)
        self._set_switches(morphology=1000, autoland=1000, flight_enable=1000)
        await self.manager.tick()
        await self._settle_for_transform()
        result = await self.manager.request_transform_walk(operator_confirmed=True)
        messages.append(result.message)
        states.append(self.manager.state)
        return self._result(
            "nominal",
            result.ok and self.manager.state.name == SystemState.WALK.name,
            states,
            messages,
            setpoints=len(self.pixhawk.setpoint_history),
        )

    async def _scenario_transform_failure(self) -> ScenarioResult:
        states = [self.manager.state]
        self._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        await self._settle_for_transform()
        self.f446.inject_next_transform_timeout()
        result = await self.manager.request_transform_flight(operator_confirmed=True)
        # The deadline advances ManualClock while the other fake telemetry
        # producers conceptually continue running concurrently.
        self._heartbeat_all()
        self._feed_rc(debounce=False)
        await self.manager.refresh_snapshot()
        states.append(self.manager.state)
        commands = tuple(item.command for item in self.f446.command_history)
        return self._result(
            "transform-failure",
            not result.ok and self.manager.state is SystemState.FAULT and "stop" in commands,
            states,
            (result.message,),
            commands=commands,
        )

    async def _scenario_rc_loss(self) -> ScenarioResult:
        states: List[SystemState] = [self.manager.state]
        first = await self._reach_flight_manual(states)
        if not first.ok:
            return self._result("rc-loss", False, states, (first.message,))
        second = await self._reach_autoland(states)
        if not second.ok:
            return self._result("rc-loss", False, states, (second.message,))
        await self.inject(SimulatedFault.RC_LOSS)
        await self.manager.tick()
        states.append(self.manager.state)
        return self._result(
            "rc-loss",
            self.manager.state is SystemState.FLIGHT_MANUAL
            and not self.pixhawk.external_setpoints_active,
            states,
            ("RC failsafe injected",),
        )

    async def _scenario_pixhawk_timeout(self) -> ScenarioResult:
        states = [self.manager.state]
        self._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        await self._settle_for_transform()
        await self.inject(SimulatedFault.PIXHAWK_TIMEOUT)
        result = await self.manager.request_transform_flight(operator_confirmed=True)
        states.append(self.manager.state)
        return self._result(
            "pixhawk-timeout",
            not result.ok and self.manager.state is SystemState.FAULT,
            states,
            (result.message,),
        )

    async def _scenario_f446_overcurrent(self) -> ScenarioResult:
        states = [self.manager.state]
        self._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        await self._settle_for_transform()
        await self.inject(SimulatedFault.F446_OVERCURRENT)
        result = await self.manager.request_transform_flight(operator_confirmed=True)
        states.append(self.manager.state)
        return self._result(
            "f446-overcurrent",
            not result.ok and self.manager.state is SystemState.FAULT,
            states,
            (result.message,),
        )

    async def _scenario_landing_override(self) -> ScenarioResult:
        states: List[SystemState] = [self.manager.state]
        first = await self._reach_flight_manual(states)
        if not first.ok:
            return self._result("landing", False, states, (first.message,))
        second = await self._reach_autoland(states)
        if not second.ok:
            return self._result("landing", False, states, (second.message,))
        self._set_switches(autoland=1000)
        await self.manager.tick()
        states.append(self.manager.state)
        return self._result(
            "landing",
            self.manager.state is SystemState.FLIGHT_MANUAL
            and not self.pixhawk.external_setpoints_active,
            states,
            ("RadioMaster manual override injected",),
        )

    async def _reach_flight_manual(self, states: List[SystemState]) -> OperationResult:
        self._set_switches(morphology=1900, autoland=1000, flight_enable=1000)
        settled = await self._settle_for_transform()
        if not settled.ok:
            return settled
        transformed = await self.manager.request_transform_flight(operator_confirmed=True)
        states.append(self.manager.state)
        if not transformed.ok:
            return transformed
        authorized = await self.manager.authorize_ground_arm()
        if not authorized.ok:
            return authorized
        self._set_switches(morphology=1900, autoland=1000, flight_enable=1900)
        self.pixhawk.inject_armed_state(True)
        self.pixhawk.inject_landed_state(False, vertical_velocity_mps=0.0, relative_altitude_m=1.0)
        for slot in self.config.esc.slots:
            self.pixhawk.inject_esc_rpm(slot, 1000.0)
        await self.manager.tick()
        states.append(self.manager.state)
        return OperationResult.success("Reached FLIGHT_MANUAL")

    async def _reach_autoland(self, states: List[SystemState]) -> OperationResult:
        self._set_landing_estimate(
            LandingEstimate(
                valid=True,
                ground_detected=True,
                height_m=1.0,
                vertical_velocity_mps=0.0,
                horizontal_velocity_mps=0.0,
                timestamp=self.clock.monotonic(),
                reason="simulated estimator valid",
            )
        )
        self._set_switches(autoland=1500)
        # Run the normal manager cycle so the prior CH10=MANUAL warning is
        # re-evaluated and cleared before the autoland preflight snapshot.
        await self.manager.tick()
        prepared = await self.manager.prepare_autoland()
        states.append(self.manager.state)
        if not prepared.ok:
            return prepared
        self._set_switches(autoland=1900)
        started = await self.manager.start_autoland()
        states.append(self.manager.state)
        return started

    async def _settle_for_transform(self) -> OperationResult:
        primed = await self.step(0.001)
        if not primed.ok:
            return primed
        return await self.step(self.config.safety.stationary_confirm_s + 0.001)

    def _set_switches(
        self,
        *,
        morphology: Optional[int] = None,
        autoland: Optional[int] = None,
        flight_enable: Optional[int] = None,
    ) -> None:
        if morphology is not None:
            self._channels[self.config.rc.morphology_channel] = morphology
        if autoland is not None:
            self._channels[self.config.rc.auto_landing_channel] = autoland
        if flight_enable is not None:
            self._channels[self.config.rc.flight_enable_channel] = flight_enable
        self._feed_rc(debounce=True)

    def _feed_rc(self, debounce: bool) -> None:
        status = self.rc_monitor.update(
            self._channels,
            connected=True,
            failsafe=False,
            timestamp=self.clock.monotonic(),
        )
        self.manager.accept_rc_status(status)
        if debounce:
            self.clock.advance(self.config.rc.debounce_s + 0.001)
            self._heartbeat_all()
            status = self.rc_monitor.update(
                self._channels,
                connected=True,
                failsafe=False,
                timestamp=self.clock.monotonic(),
            )
            self.manager.accept_rc_status(status)

    def _set_landing_estimate(self, estimate: LandingEstimate) -> None:
        self._landing_estimate = estimate
        self.manager.accept_landing_estimate(estimate)

    def _refresh_landing_estimate(self) -> None:
        if self._landing_estimate is None:
            return
        self._landing_estimate = replace(
            self._landing_estimate,
            timestamp=self.clock.monotonic(),
        )
        self.manager.accept_landing_estimate(self._landing_estimate)

    def _heartbeat_all(self) -> None:
        if self.pixhawk.get_status().connected:
            self.pixhawk.inject_heartbeat()
        if self.f446.get_status().connected:
            self.f446.inject_status()
        if self.go2.get_status().connected:
            self.go2.inject_status()
        self._refresh_landing_estimate()

    def _heartbeat_except(self, excluded: str) -> None:
        if excluded != "pixhawk" and self.pixhawk.get_status().connected:
            self.pixhawk.inject_heartbeat()
        if excluded != "f446" and self.f446.get_status().connected:
            self.f446.inject_status()
        if excluded != "go2" and self.go2.get_status().connected:
            self.go2.inject_status()
        self._refresh_landing_estimate()
        status = self.rc_monitor.update(
            self._channels,
            connected=True,
            failsafe=False,
            timestamp=self.clock.monotonic(),
        )
        self.manager.accept_rc_status(status)

    def _safe_channels(self) -> Dict[int, int]:
        channels = {
            channel: 1000
            for channel in (
                self.config.rc.flight_enable_channel,
                self.config.rc.flight_mode_channel,
                self.config.rc.rtl_channel,
                self.config.rc.land_channel,
                self.config.rc.morphology_channel,
                self.config.rc.auto_landing_channel,
                self.config.rc.brake_channel,
                self.config.rc.buzzer_channel,
            )
        }
        channels.update({1: 1500, 2: 1500, 3: 1500, 4: 1500})
        channels[self.config.rc.morphology_channel] = 1500
        return channels

    def _result(
        self,
        name: str,
        ok: bool,
        states: List[SystemState],
        messages: Any,
        **details: Any,
    ) -> ScenarioResult:
        del states
        full_trace = (SystemState.BOOT_SAFE,) + tuple(
            record.new_state for record in self.manager.transitions if record.permitted
        )
        return ScenarioResult(
            name=name,
            ok=ok,
            final_state=self.manager.state,
            states=full_trace,
            messages=tuple(messages),
            details=details,
        )
