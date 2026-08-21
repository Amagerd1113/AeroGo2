"""Small helpers used only while declaring command metadata."""

from __future__ import annotations

from typing import FrozenSet, Optional, Sequence

from aerogo2.cli.command_models import (
    CommandPermission,
    CommandSpec,
    ConfirmationPolicy,
    PermissionPolicy,
)
from aerogo2.common.enums import RuntimeMode, SystemState


def command(
    path: str,
    description: str,
    category: str,
    action: str,
    *,
    usage: str = "",
    aliases: Sequence[str] = (),
    options: Sequence[str] = (),
    capability: CommandPermission = CommandPermission.READ_ONLY,
    minimum_phase: int = 1,
    allowed_states: Optional[FrozenSet[SystemState]] = None,
    confirmation: Optional[ConfirmationPolicy] = None,
    dry_run_only: bool = False,
    requires_maintenance: bool = False,
    requires_hardware_write: bool = False,
) -> CommandSpec:
    modes = frozenset({RuntimeMode.DRY_RUN}) if dry_run_only else frozenset(RuntimeMode)
    return CommandSpec(
        path=tuple(path.split()),
        description=description,
        usage=usage,
        aliases=tuple(tuple(alias.split()) for alias in aliases),
        category=category,
        action=action,
        options=tuple(options),
        confirmation=ConfirmationPolicy.none() if confirmation is None else confirmation,
        permission=PermissionPolicy(
            capability=capability,
            allowed_modes=modes,
            allowed_states=allowed_states,
            minimum_phase=minimum_phase,
            requires_maintenance=requires_maintenance,
            requires_hardware_write=requires_hardware_write,
        ),
    )


def readonly(
    path: str,
    description: str,
    category: str,
    action: str,
    *,
    usage: str = "",
    options: Sequence[str] = (),
) -> CommandSpec:
    return command(
        path,
        description,
        category,
        action,
        usage=usage,
        options=options,
    )


__all__ = ["command", "readonly"]
