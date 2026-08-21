"""Immutable command metadata shared by the console pipeline.

The CLI metadata is intentionally descriptive.  It can reject commands that are
obviously unavailable in a runtime mode, but it is not a replacement for the
fresh-snapshot guards in :class:`SystemManager`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

from aerogo2.common.enums import ConfirmationLevel, RuntimeMode, SystemState

CommandPath = Tuple[str, ...]
PathLike = Union[str, Sequence[str]]


def normalize_command_path(path: PathLike) -> CommandPath:
    """Return a lower-case, validated command path.

    Registry paths are deliberately limited to whitespace-separated words.
    Quoting belongs to argument parsing and is never meaningful in a command
    name.
    """

    if isinstance(path, str):
        parts = tuple(path.split())
    else:
        parts = tuple(str(part) for part in path)
    normalized = tuple(part.strip().lower() for part in parts if part.strip())
    if not normalized:
        raise ValueError("A command path cannot be empty")
    if any(any(character.isspace() for character in part) for part in normalized):
        raise ValueError("Each command path component must be one word")
    return normalized


class CommandPermission(Enum):
    """Coarse capability used for help text and first-pass authorization."""

    READ_ONLY = "READ_ONLY"
    SAFE_CONTROL = "SAFE_CONTROL"
    SAFETY_STOP = "SAFETY_STOP"
    SIMULATION = "SIMULATION"
    HARDWARE_WRITE = "HARDWARE_WRITE"
    F446_MAINTENANCE = "F446_MAINTENANCE"


@dataclass(frozen=True)
class CommandContext:
    """The small, immutable context used for CLI-level permission checks."""

    runtime_mode: RuntimeMode = RuntimeMode.DRY_RUN
    state: SystemState = SystemState.BOOT_SAFE
    phase: int = 1
    maintenance_mode: bool = False
    hardware_write_enabled: bool = False
    pixhawk_armed: bool = False


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    code: str = "OK"
    reason: str = ""

    @classmethod
    def allow(cls) -> PermissionDecision:
        return cls(True)

    @classmethod
    def deny(cls, code: str, reason: str) -> PermissionDecision:
        return cls(False, code, reason)


@dataclass(frozen=True)
class PermissionPolicy:
    """Static availability metadata.

    Dynamic safety facts such as ESC RPM, status freshness, and Go2 stability
    must be rechecked by ``SystemManager`` immediately before execution.
    """

    capability: CommandPermission = CommandPermission.READ_ONLY
    allowed_modes: FrozenSet[RuntimeMode] = field(default_factory=lambda: frozenset(RuntimeMode))
    allowed_states: Optional[FrozenSet[SystemState]] = None
    minimum_phase: int = 1
    requires_maintenance: bool = False
    requires_hardware_write: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_modes", frozenset(self.allowed_modes))
        if self.allowed_states is not None:
            object.__setattr__(
                self,
                "allowed_states",
                frozenset(self.allowed_states),
            )

    def evaluate(self, context: CommandContext) -> PermissionDecision:
        if context.phase < self.minimum_phase:
            return PermissionDecision.deny(
                "PHASE_NOT_AVAILABLE",
                f"This command is available from Phase {self.minimum_phase}.",
            )
        if context.runtime_mode not in self.allowed_modes:
            return PermissionDecision.deny(
                "RUNTIME_MODE_DENIED",
                f"Command is not available in {context.runtime_mode.value} mode.",
            )
        if self.allowed_states is not None and context.state not in self.allowed_states:
            return PermissionDecision.deny(
                "STATE_DENIED",
                f"Command is not available in state {context.state.name}.",
            )
        if self.requires_maintenance and not context.maintenance_mode:
            return PermissionDecision.deny(
                "MAINTENANCE_REQUIRED",
                "Enter F446 maintenance mode before using this command.",
            )
        if self.requires_hardware_write and context.runtime_mode is RuntimeMode.HARDWARE_READONLY:
            return PermissionDecision.deny(
                "HARDWARE_WRITE_DISABLED",
                "Physical hardware writes are disabled in HW-RO mode.",
            )
        if (
            self.requires_hardware_write
            and context.runtime_mode is RuntimeMode.HARDWARE
            and not context.hardware_write_enabled
        ):
            return PermissionDecision.deny(
                "HARDWARE_WRITE_DISABLED",
                "Physical hardware writes are disabled by configuration.",
            )
        return PermissionDecision.allow()


@dataclass(frozen=True)
class ConfirmationPolicy:
    """Prompt metadata without any storage for the operator's response."""

    level: ConfirmationLevel = ConfirmationLevel.NONE
    prompt: str = "Continue?"
    exact_phrase: Optional[str] = None
    warning: Optional[str] = None

    def __post_init__(self) -> None:
        if self.level in (ConfirmationLevel.EXACT_PHRASE, ConfirmationLevel.TWO_STAGE):
            if not self.exact_phrase:
                raise ValueError(f"{self.level.name} confirmation requires an exact phrase")

    @classmethod
    def none(cls) -> ConfirmationPolicy:
        return cls()

    @classmethod
    def simple(cls, prompt: str = "Continue?", warning: Optional[str] = None) -> ConfirmationPolicy:
        return cls(level=ConfirmationLevel.SIMPLE, prompt=prompt, warning=warning)

    @classmethod
    def exact(
        cls,
        phrase: str,
        prompt: str = "Type the exact phrase to continue",
        warning: Optional[str] = None,
    ) -> ConfirmationPolicy:
        return cls(
            level=ConfirmationLevel.EXACT_PHRASE,
            prompt=prompt,
            exact_phrase=phrase,
            warning=warning,
        )

    @classmethod
    def two_stage(
        cls, phrase: str, prompt: str = "Continue?", warning: Optional[str] = None
    ) -> ConfirmationPolicy:
        return cls(
            level=ConfirmationLevel.TWO_STAGE,
            prompt=prompt,
            exact_phrase=phrase,
            warning=warning,
        )


@dataclass(frozen=True)
class CommandSpec:
    """A canonical command, its aliases, and its safety metadata."""

    path: CommandPath
    description: str
    usage: str = ""
    aliases: Tuple[CommandPath, ...] = ()
    category: str = "general"
    permission: PermissionPolicy = field(default_factory=PermissionPolicy)
    confirmation: ConfirmationPolicy = field(default_factory=ConfirmationPolicy.none)
    options: Tuple[str, ...] = ()
    action: str = ""
    hidden: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_command_path(self.path))
        object.__setattr__(
            self,
            "aliases",
            tuple(normalize_command_path(alias) for alias in self.aliases),
        )
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not self.description.strip():
            raise ValueError("A command description cannot be empty")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("Command aliases must be unique")
        if self.path in self.aliases:
            raise ValueError("A canonical command path cannot also be an alias")

    @property
    def name(self) -> str:
        return " ".join(self.path)

    @property
    def qualified_name(self) -> str:
        return self.name

    @property
    def effective_usage(self) -> str:
        return self.usage or self.name

    @property
    def all_paths(self) -> Tuple[CommandPath, ...]:
        return (self.path,) + self.aliases


@dataclass(frozen=True)
class ParsedCommand:
    """The lossless result of tokenizing one console input line."""

    raw: str
    tokens: Tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.tokens

    @property
    def name(self) -> str:
        return self.tokens[0] if self.tokens else ""

    @property
    def arguments(self) -> Tuple[str, ...]:
        return self.tokens[1:]


@dataclass(frozen=True)
class CommandMatch:
    """A parsed command resolved to the longest registered command path."""

    spec: CommandSpec
    matched_path: CommandPath
    arguments: Tuple[str, ...]
    used_alias: bool = False


@dataclass(frozen=True)
class CommandInvocation:
    """A command ready to cross the dispatcher/SystemManager boundary."""

    command_id: str
    spec: CommandSpec
    raw: str
    arguments: Tuple[str, ...] = ()

    @classmethod
    def from_match(
        cls, match: CommandMatch, raw: str, command_id: Optional[str] = None
    ) -> CommandInvocation:
        return cls(
            command_id=command_id or str(uuid4()),
            spec=match.spec,
            raw=raw,
            arguments=match.arguments,
        )

    @property
    def command_name(self) -> str:
        return self.spec.name


# Compatibility names are intentionally explicit; downstream command modules can
# use either terminology without duplicating safety metadata.
CommandDefinition = CommandSpec
PermissionMetadata = PermissionPolicy
ConfirmationMetadata = ConfirmationPolicy
PermissionLevel = CommandPermission


__all__ = [
    "CommandContext",
    "CommandDefinition",
    "CommandInvocation",
    "CommandMatch",
    "CommandPath",
    "CommandPermission",
    "CommandSpec",
    "ConfirmationMetadata",
    "ConfirmationPolicy",
    "ParsedCommand",
    "PathLike",
    "PermissionDecision",
    "PermissionLevel",
    "PermissionMetadata",
    "PermissionPolicy",
    "normalize_command_path",
]
