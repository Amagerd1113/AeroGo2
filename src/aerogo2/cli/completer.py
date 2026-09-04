"""Prompt-toolkit completion backed by :mod:`aerogo2.cli.registry`."""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from aerogo2.cli.registry import CommandRegistry


class AeroGo2Completer(Completer):  # type: ignore[misc]
    """Complete hierarchical command words, aliases, and declared options."""

    def __init__(self, registry: CommandRegistry, include_aliases: bool = True) -> None:
        self._registry = registry
        self._include_aliases = include_aliases

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        del complete_event
        before_cursor = document.text_before_cursor
        trailing_space = bool(before_cursor) and before_cursor[-1].isspace()
        words = before_cursor.split()
        if trailing_space:
            fixed = tuple(word.lower() for word in words)
            fragment = ""
        elif words:
            fixed = tuple(word.lower() for word in words[:-1])
            fragment = words[-1]
        else:
            fixed = ()
            fragment = ""
        normalized_fragment = fragment.lower()

        candidates: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for name in self._registry.command_names(include_aliases=self._include_aliases):
            path = tuple(name.split())
            if len(fixed) >= len(path) or path[: len(fixed)] != fixed:
                continue
            next_word = path[len(fixed)]
            if not next_word.startswith(normalized_fragment) or next_word in seen:
                continue
            seen.add(next_word)
            spec = self._registry.get(path)
            candidates.append((next_word, spec.description))

        # ``help <command>`` completes the command tree as its argument.
        if fixed and fixed[0] == "help":
            help_prefix = fixed[1:]
            for name in self._registry.command_names(include_aliases=self._include_aliases):
                path = tuple(name.split())
                if len(help_prefix) >= len(path) or path[: len(help_prefix)] != help_prefix:
                    continue
                next_word = path[len(help_prefix)]
                if not next_word.startswith(normalized_fragment) or next_word in seen:
                    continue
                seen.add(next_word)
                spec = self._registry.get(path)
                candidates.append((next_word, spec.description))

        exact_spec = self._registry.find(fixed)
        if exact_spec is not None:
            for option in exact_spec.options:
                if option.startswith(fragment) and option not in seen:
                    seen.add(option)
                    candidates.append((option, f"option for {exact_spec.name}"))

        for text, description in sorted(candidates):
            yield Completion(
                text=text,
                start_position=-len(fragment),
                display_meta=description,
            )


RegistryCompleter = AeroGo2Completer

__all__ = ["AeroGo2Completer", "RegistryCompleter"]
