"""Defensive immutable-container helpers for subsystem boundaries."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

_K = TypeVar("_K")
_V = TypeVar("_V")


def deep_freeze(value: Any) -> Any:
    """Copy nested containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def frozen_mapping(value: Mapping[_K, _V]) -> Mapping[_K, _V]:
    """Return a recursively frozen defensive copy of ``value``."""

    return cast(Mapping[_K, _V], deep_freeze(value))


def deep_thaw(value: Any) -> Any:
    """Convert immutable project containers to JSON/YAML-friendly values."""

    if isinstance(value, Mapping):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [deep_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [deep_thaw(item) for item in value]
    return value


__all__ = ["deep_freeze", "deep_thaw", "frozen_mapping"]
