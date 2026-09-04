"""Safety confirmation prompts isolated from persistent command history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Iterable, Optional, cast

from prompt_toolkit import PromptSession

from aerogo2.cli.command_models import ConfirmationPolicy
from aerogo2.cli.history import CommandHistory
from aerogo2.common.enums import ConfirmationLevel

ConfirmationReader = Callable[[str], Awaitable[str]]
WarningWriter = Callable[[str], None]


@dataclass(frozen=True)
class ConfirmationDecision:
    confirmed: bool
    level: ConfirmationLevel
    completed_stage: int
    reason: str


class ConfirmationService:
    """Perform confirmations without returning or recording entered phrases."""

    def __init__(
        self,
        reader: Optional[ConfirmationReader] = None,
        warning_writer: Optional[WarningWriter] = None,
    ) -> None:
        self._session: Optional[PromptSession[str]] = None
        self._reader: ConfirmationReader = self._prompt if reader is None else reader
        self._warning_writer = warning_writer or self._default_warning_writer

    async def confirm(self, policy: ConfirmationPolicy) -> bool:
        return (await self.request(policy)).confirmed

    async def request(self, policy: ConfirmationPolicy) -> ConfirmationDecision:
        if policy.warning:
            self._warning_writer(policy.warning)

        if policy.level is ConfirmationLevel.NONE:
            return ConfirmationDecision(True, policy.level, 0, "confirmation not required")

        if policy.level is ConfirmationLevel.SIMPLE:
            accepted = await self._simple_stage(policy.prompt)
            return ConfirmationDecision(
                accepted,
                policy.level,
                1 if accepted else 0,
                "confirmed" if accepted else "operator declined",
            )

        if policy.level is ConfirmationLevel.EXACT_PHRASE:
            accepted = await self._exact_stage(policy.prompt, policy.exact_phrase)
            return ConfirmationDecision(
                accepted,
                policy.level,
                1 if accepted else 0,
                "confirmed" if accepted else "exact phrase did not match",
            )

        if policy.level is ConfirmationLevel.TWO_STAGE:
            first_stage = await self._simple_stage(policy.prompt)
            if not first_stage:
                return ConfirmationDecision(False, policy.level, 0, "operator declined first stage")
            second_stage = await self._exact_stage(
                "Type the exact phrase to execute", policy.exact_phrase
            )
            return ConfirmationDecision(
                second_stage,
                policy.level,
                2 if second_stage else 1,
                "confirmed" if second_stage else "exact phrase did not match",
            )

        return ConfirmationDecision(False, policy.level, 0, "unsupported confirmation level")

    async def _simple_stage(self, prompt: str) -> bool:
        try:
            response = await self._reader(f"{prompt} [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            return False
        return response.strip().lower() in {"y", "yes"}

    async def _exact_stage(self, prompt: str, phrase: Optional[str]) -> bool:
        if phrase is None:
            return False
        try:
            response = await self._reader(f"{prompt} [{phrase}]: ")
        except (EOFError, KeyboardInterrupt):
            return False
        # Do not strip or case-fold: the requested phrase is intentionally exact.
        return response == phrase

    async def _prompt(self, prompt: str) -> str:
        if self._session is None:
            self._session = PromptSession(history=CommandHistory.confirmation_history())
        return cast(str, await self._session.prompt_async(prompt))

    @staticmethod
    def _default_warning_writer(message: str) -> None:
        print(message)


class ScriptedConfirmationReader:
    """Deterministic reader useful for unit and integration tests."""

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses: Deque[str] = deque(responses)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise EOFError
        return self._responses.popleft()


__all__ = [
    "ConfirmationDecision",
    "ConfirmationReader",
    "ConfirmationService",
    "ScriptedConfirmationReader",
]
