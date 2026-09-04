"""The only component allowed to mutate the AeroGo2 system state."""

from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from aerogo2.common.async_utils import await_nonabandonable
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
        transaction = asyncio.create_task(
            self._commit_permitted_transition(
                record=record,
                new_state=new_state,
                evaluated_snapshot=evaluated_snapshot,
            )
        )
        committed_record, cancellation_seen = await await_nonabandonable(transaction)
        if cancellation_seen:
            # The caller still observes cancellation, but only after the
            # accepted transition has crossed its non-abandonable boundary.
            raise asyncio.CancelledError
        return committed_record

    async def _commit_permitted_transition(
        self,
        *,
        record: TransitionRecord,
        new_state: SystemState,
        evaluated_snapshot: SystemSnapshot,
    ) -> TransitionRecord:
        """Linearize an accepted transition without a caller-cancellation gap."""

        post_transition_errors: List[str] = []
        subscriber_errors: Tuple[str, ...] = ()
        entry_action_error: Optional[str] = None
        try:
            # Safety-critical ordering is intentional: persist the decision
            # before the sole state mutation, then notify observers, and only
            # then execute optional state-entry side effects.
            record_logger_error = self._record(record)
            if record_logger_error is not None:
                post_transition_errors.append(record_logger_error)
            self._state = new_state
            subscriber_errors = await self._publish(record)
            post_transition_errors.extend(subscriber_errors)
            action = self._entry_actions.get(new_state)
            if action is not None:
                try:
                    await action(evaluated_snapshot.with_state(new_state))
                except (Exception, asyncio.CancelledError) as exc:
                    entry_action_error = f"{type(exc).__name__}: {exc}"
                    post_transition_errors.append(
                        f"state entry action failed: {entry_action_error}"
                    )
        finally:
            self._transitioning = False

        if post_transition_errors:
            failure_reason = "; ".join(post_transition_errors)
            if new_state is not SystemState.FAULT:
                await self.transition_to(
                    SystemState.FAULT,
                    reason=failure_reason,
                    snapshot=evaluated_snapshot.with_state(new_state),
                )
            else:
                # FAULT is already committed. Audit publish/entry failures
                # without recursively attempting FAULT->FAULT or repeating its
                # safety entry action. Logger failures were recorded by
                # ``_record`` at their point of occurrence.
                if subscriber_errors:
                    self._append_diagnostic(
                        code="FAULT_SUBSCRIBER_FAILED",
                        reason="; ".join(subscriber_errors),
                        error="; ".join(subscriber_errors),
                    )
                if entry_action_error is not None:
                    self._append_diagnostic(
                        code="FAULT_ENTRY_ACTION_FAILED",
                        reason=f"state entry action failed: {entry_action_error}",
                        error=entry_action_error,
                    )
        return record

    def _record(self, record: TransitionRecord) -> Optional[str]:
        self._history.append(record)
        if self._record_logger is not None:
            try:
                self._record_logger(record)
            except (Exception, asyncio.CancelledError) as exc:
                error = f"record logger failed: {type(exc).__name__}: {exc}"
                self._append_diagnostic(
                    code="TRANSITION_RECORD_LOGGER_FAILED",
                    reason=error,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return error
        return None

    def _append_diagnostic(self, *, code: str, reason: str, error: str) -> None:
        """Keep an in-memory audit record without re-entering a failed logger."""

        self._history.append(
            TransitionRecord(
                timestamp=self._clock.monotonic(),
                previous_state=self._state,
                new_state=self._state,
                reason=reason,
                permitted=False,
                guard_codes=(code,),
                entry_action_error=error,
            )
        )

    async def _publish(self, record: TransitionRecord) -> Tuple[str, ...]:
        errors: List[str] = []
        for callback in tuple(self._subscribers):
            try:
                result = callback(record)
                if inspect.isawaitable(result):
                    await result
            except (Exception, asyncio.CancelledError) as exc:
                errors.append(f"state subscriber failed: {type(exc).__name__}: {exc}")
        return tuple(errors)
