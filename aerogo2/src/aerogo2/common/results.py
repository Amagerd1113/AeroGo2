"""Explicit operation results; expected rejections are not exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from aerogo2.common.enums import CommandStatus
from aerogo2.common.immutable import frozen_mapping


@dataclass(frozen=True)
class GuardResult:
    permitted: bool
    codes: Tuple[str, ...] = ()
    messages: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "codes", tuple(self.codes))
        object.__setattr__(self, "messages", tuple(self.messages))

    @classmethod
    def allow(cls) -> GuardResult:
        return cls(True)

    @classmethod
    def reject(cls, code: str, message: str) -> GuardResult:
        return cls(False, (code,), (message,))


@dataclass(frozen=True)
class OperationResult:
    ok: bool
    code: str
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", frozen_mapping(self.data))

    @classmethod
    def success(
        cls, message: str, data: Optional[Mapping[str, Any]] = None, code: str = "OK"
    ) -> OperationResult:
        return cls(True, code, message, {} if data is None else data)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        data: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult:
        return cls(False, code, message, {} if data is None else data)


@dataclass(frozen=True)
class CommandResult:
    status: CommandStatus
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)
    command_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", frozen_mapping(self.data))

    @property
    def ok(self) -> bool:
        return self.status is CommandStatus.SUCCESS
