"""Shell-like, side-effect-free command parsing."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from aerogo2.cli.command_models import ParsedCommand
from aerogo2.common.exceptions import CommandParseError


@dataclass(frozen=True)
class ParsedArguments:
    """Optional convenience view over arguments after registry resolution."""

    positionals: Tuple[str, ...]
    options: Mapping[str, Tuple[str, ...]]

    def has_flag(self, name: str) -> bool:
        values = self.options.get(name.lstrip("-"))
        return values == ()

    def values_for(self, name: str) -> Tuple[str, ...]:
        return self.options.get(name.lstrip("-"), ())


class CommandParser:
    """Tokenize input using POSIX shell quoting without executing anything."""

    def __init__(self, maximum_length: int = 16_384) -> None:
        if maximum_length <= 0:
            raise ValueError("maximum_length must be positive")
        self._maximum_length = maximum_length

    def parse(self, line: str) -> ParsedCommand:
        if "\x00" in line:
            raise CommandParseError("Command input contains a NUL byte")
        if len(line) > self._maximum_length:
            raise CommandParseError(f"Command exceeds the {self._maximum_length} character limit")
        if not line.strip():
            return ParsedCommand(raw=line, tokens=())

        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = tuple(lexer)
        except ValueError as exc:
            raise CommandParseError(f"Invalid quoting: {exc}") from exc
        return ParsedCommand(raw=line, tokens=tokens)

    def parse_line(self, line: str) -> ParsedCommand:
        """Compatibility alias for callers that prefer an explicit method name."""

        return self.parse(line)

    def try_parse(self, line: str) -> Optional[ParsedCommand]:
        try:
            return self.parse(line)
        except CommandParseError:
            return None

    @staticmethod
    def split_options(arguments: Sequence[str]) -> ParsedArguments:
        """Split GNU-style ``--name [value]`` options from positionals.

        Parsing remains deliberately conservative: short options and negative
        numeric arguments are left untouched, repeated options retain every
        value, and ``--`` ends option processing.
        """

        positionals = []
        mutable_options: Dict[str, list[str]] = {}
        index = 0
        option_mode = True
        while index < len(arguments):
            token = arguments[index]
            if option_mode and token == "--":
                option_mode = False
                index += 1
                continue
            if option_mode and token.startswith("--") and len(token) > 2:
                option = token[2:]
                if "=" in option:
                    name, value = option.split("=", 1)
                    if not name:
                        raise CommandParseError("Option name cannot be empty")
                    mutable_options.setdefault(name, []).append(value)
                    index += 1
                    continue
                if not option:
                    raise CommandParseError("Option name cannot be empty")
                if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                    mutable_options.setdefault(option, []).append(arguments[index + 1])
                    index += 2
                    continue
                mutable_options.setdefault(option, [])
                index += 1
                continue
            positionals.append(token)
            index += 1

        options = {name: tuple(values) for name, values in mutable_options.items()}
        return ParsedArguments(tuple(positionals), options)


__all__ = ["CommandParser", "ParsedArguments"]
