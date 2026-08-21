"""The only component allowed to mutate the AeroGo2 system state."""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from aerogo2.common.clock import Clock
from aerogo2.common.enums import SystemState
from aerogo2.common.exceptions import TransitionRejected
from aerogo2.common.models import SystemSnapshot, TransitionRecord
from aerogo2.manager.transition_guards import TransitionGuards

EntryAction = Callable[[SystemSnapshot], Awaitable[None]]
StateSubscriber = Callable[[TransitionRecord], object]
RecordLogger = Callable[[TransitionRecord], None]


class StateMachine:
    """Guarded, auditable state holder.

    No prior state is loaded from disk: every instance starts in BOOT_SAFE.
    """

    def __init__(
        self,
        guards: TransitionGuards,
        clock: Clock,
        record_logger: Optional[RecordLogger] = None,
    ) -> None:
        self._state = SystemState.BOOT_SAFE
        self._guards = guards
        self._clock = clock
        self._record_logger = record_logger
        self._history: List[TransitionRecord] = []
        self._entry_actions: Dict[SystemState, EntryAction] = {}
        self._subscribers: List[StateSubscriber] = []
        self._transitioning = False

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def history(self) -> Tuple[TransitionRecord, ...]:
        return tuple(self._history)

    def set_entry_action(self, state: SystemState, action: EntryAction) -> None:
        self._entry_actions[state] = action

    def replace_guards(self, guards: TransitionGuards) -> None:
        if self._transitioning or self._state is not SystemState.BOOT_SAFE:
            raise TransitionRejected("Guards may only be reloaded while idle in BOOT_SAFE")
        self._guards = guards

    def subscribe(self, callback: StateSubscriber) -> None:
        self._subscribers.append(callback)

    async def transition_to(
        self,
        new_state: SystemState,
        reason: str,
        snapshot: SystemSnapshot,
    ) -> TransitionRecord:
        if self._transitioning:
            raise TransitionRejected("Nested state transitions are not allowed")

        previous = self._state
        evaluated_snapshot = snapshot.with_state(previous, self._clock.monotonic())
        guard = self._guards.evaluate(previous, new_state, evaluated_snapshot)
        if not guard.permitted:
            record = TransitionRecord(
                timestamp=self._clock.monotonic(),
                previous_state=previous,
                new_state=new_state,
                reason=reason,
                permitted=False,
                guard_codes=guard.codes,
            )
            self._record(record)
            raise TransitionRejected("; ".join(guard.messages))

        record = TransitionRecord(
            timestamp=self._clock.monotonic(),
            previous_state=previous,
            new_state=new_state,
            reason=reason,
            permitted=True,
            guard_codes=guard.codes,
        )
        self._transitioning = True
        post_transition_error: Optional[str] = None
        try:
            # Safety-critical ordering is intentional: persist the decision
            # before the sole state mutation, then notify observers, and only
            # then execute optional state-entry side effects.
            self._record(record)
            self._state = new_state
            try:
                await self._publish(record)
                action = self._entry_actions.get(new_state)
                if action is not None:
                    await action(evaluated_snapshot.with_state(new_state))
            except Exception as exc:
                post_transition_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._transitioning = False

        if post_transition_error is not None:
            failure_reason = f"state entry/publish failed: {post_transition_error}"
            if new_state is not SystemState.FAULT:
                await self.transition_to(
                    SystemState.FAULT,
                    reason=failure_reason,
                    snapshot=evaluated_snapshot.with_state(new_state),
                )
            else:
                # FAULT is already the fail-closed state. Record the failed
                # FAULT entry action without recursively attempting FAULT->FAULT.
                diagnostic = TransitionRecord(
                    timestamp=self._clock.monotonic(),
                    previous_state=SystemState.FAULT,
                    new_state=SystemState.FAULT,
                    reason=failure_reason,
                    permitted=False,
                    guard_codes=("FAULT_ENTRY_ACTION_FAILED",),
                    entry_action_error=post_transition_error,
                )
                self._record(diagnostic)
        return record

    def _record(self, record: TransitionRecord) -> None:
        self._history.append(record)
        if self._record_logger is not None:
            self._record_logger(record)

    async def _publish(self, record: TransitionRecord) -> None:
        for callback in tuple(self._subscribers):
            result = callback(record)
            if inspect.isawaitable(result):
                await result
