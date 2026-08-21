"""Hierarchical command registry with aliases and spelling suggestions."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from aerogo2.cli.command_models import (
    CommandMatch,
    CommandPath,
    CommandSpec,
    ParsedCommand,
    normalize_command_path,
)
from aerogo2.cli.parser import CommandParser


class CommandRegistryError(ValueError):
    """Base error for invalid registry operations."""


class DuplicateCommandError(CommandRegistryError):
    """A canonical path or alias is already owned by another command."""


class CommandNotFoundError(LookupError):
    """No registered command matches the supplied token prefix."""

    def __init__(self, query: str, suggestions: Sequence[str] = ()) -> None:
        self.query = query
        self.suggestions = tuple(suggestions)
        message = "Unknown command: {}".format(query or "<empty>")
        if self.suggestions:
            message += ". Did you mean {}?".format(", ".join(self.suggestions))
        super().__init__(message)


Resolvable = Union[str, ParsedCommand, Sequence[str]]


class CommandRegistry:
    """Own canonical command metadata and resolve the longest matching path."""

    def __init__(self, parser: Optional[CommandParser] = None) -> None:
        self._parser = parser or CommandParser()
        self._canonical: Dict[CommandPath, CommandSpec] = {}
        self._paths: Dict[CommandPath, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> CommandSpec:
        collisions = [path for path in spec.all_paths if path in self._paths]
        if collisions:
            raise DuplicateCommandError(
                "Command path already registered: {}".format(
                    ", ".join(" ".join(path) for path in collisions)
                )
            )
        self._canonical[spec.path] = spec
        for path in spec.all_paths:
            self._paths[path] = spec
        return spec

    def register_many(self, specs: Iterable[CommandSpec]) -> None:
        """Register a set atomically with respect to path collisions."""

        materialized = tuple(specs)
        pending: Dict[CommandPath, CommandSpec] = {}
        for spec in materialized:
            for path in spec.all_paths:
                if path in self._paths or path in pending:
                    raise DuplicateCommandError(
                        "Command path already registered: {}".format(" ".join(path))
                    )
                pending[path] = spec
        for spec in materialized:
            self.register(spec)

    def unregister(self, path: Union[str, Sequence[str]]) -> CommandSpec:
        normalized = normalize_command_path(path)
        spec = self._paths.get(normalized)
        if spec is None:
            raise CommandNotFoundError(" ".join(normalized))
        self._canonical.pop(spec.path, None)
        for owned_path in spec.all_paths:
            self._paths.pop(owned_path, None)
        return spec

    def resolve(self, command: Resolvable) -> CommandMatch:
        parsed = self._coerce(command)
        if not parsed.tokens:
            raise CommandNotFoundError("")

        normalized_tokens = tuple(token.lower() for token in parsed.tokens)
        for consumed in range(len(normalized_tokens), 0, -1):
            candidate = normalized_tokens[:consumed]
            spec = self._paths.get(candidate)
            if spec is not None:
                return CommandMatch(
                    spec=spec,
                    matched_path=candidate,
                    arguments=parsed.tokens[consumed:],
                    used_alias=candidate != spec.path,
                )

        query = " ".join(parsed.tokens)
        raise CommandNotFoundError(query, self.suggest(query))

    def get(self, path: Union[str, Sequence[str]]) -> CommandSpec:
        normalized = normalize_command_path(path)
        spec = self._paths.get(normalized)
        if spec is None:
            raise CommandNotFoundError(" ".join(normalized), self.suggest(" ".join(normalized)))
        return spec

    def find(self, path: Union[str, Sequence[str]]) -> Optional[CommandSpec]:
        try:
            normalized = normalize_command_path(path)
        except ValueError:
            return None
        return self._paths.get(normalized)

    def suggest(self, query: str, limit: int = 3) -> Tuple[str, ...]:
        if limit <= 0:
            return ()
        normalized = " ".join(query.lower().split())
        if not normalized:
            return ()
        path_names = {" ".join(path): spec.name for path, spec in self._paths.items()}
        matches = get_close_matches(
            normalized,
            tuple(path_names),
            n=max(limit * 2, limit),
            cutoff=0.42,
        )
        suggestions: List[str] = []
        for match in matches:
            canonical_name = path_names[match]
            if canonical_name not in suggestions:
                suggestions.append(canonical_name)
            if len(suggestions) == limit:
                break
        return tuple(suggestions)

    def command_names(self, include_aliases: bool = False) -> Tuple[str, ...]:
        source = self._paths if include_aliases else self._canonical
        return tuple(sorted(" ".join(path) for path in source))

    def specs(
        self, category: Optional[str] = None, include_hidden: bool = False
    ) -> Tuple[CommandSpec, ...]:
        values = sorted(self._canonical.values(), key=lambda item: (item.category, item.path))
        return tuple(
            spec
            for spec in values
            if (category is None or spec.category == category)
            and (include_hidden or not spec.hidden)
        )

    def categories(self) -> Tuple[str, ...]:
        return tuple(sorted({spec.category for spec in self._canonical.values()}))

    def paths_below(self, prefix: Union[str, Sequence[str]]) -> Tuple[CommandPath, ...]:
        try:
            normalized = normalize_command_path(prefix)
        except ValueError:
            normalized = ()
        return tuple(
            sorted(
                path
                for path in self._paths
                if len(path) >= len(normalized) and path[: len(normalized)] == normalized
            )
        )

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, (str, tuple, list)):
            return False
        return self.find(path) is not None

    def __iter__(self) -> Iterator[CommandSpec]:
        return iter(self.specs())

    def __len__(self) -> int:
        return len(self._canonical)

    def _coerce(self, command: Resolvable) -> ParsedCommand:
        if isinstance(command, ParsedCommand):
            return command
        if isinstance(command, str):
            return self._parser.parse(command)
        tokens = tuple(str(token) for token in command)
        return ParsedCommand(raw=" ".join(tokens), tokens=tokens)


__all__ = [
    "CommandNotFoundError",
    "CommandRegistry",
    "CommandRegistryError",
    "DuplicateCommandError",
]
