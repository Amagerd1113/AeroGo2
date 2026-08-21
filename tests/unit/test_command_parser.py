"""Unit tests for shell-safe command tokenization."""

from __future__ import annotations

import pytest

from aerogo2.cli.parser import CommandParser
from aerogo2.common.exceptions import CommandParseError


def test_empty_input_is_a_noop() -> None:
    parsed = CommandParser().parse("   \t")
    assert parsed.empty
    assert parsed.tokens == ()
    assert parsed.name == ""
    assert parsed.arguments == ()


def test_hierarchical_command_preserves_all_tokens() -> None:
    parsed = CommandParser().parse("motor maintenance enter")
    assert parsed.tokens == ("motor", "maintenance", "enter")
    assert parsed.name == "motor"
    assert parsed.arguments == ("maintenance", "enter")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('log mark "flight configuration verified"', "flight configuration verified"),
        ("log mark 'operator requested hold'", "operator requested hold"),
        (r'log mark "quoted \"value\""', 'quoted "value"'),
    ],
)
def test_quoted_arguments_are_preserved_as_one_argument(line: str, expected: str) -> None:
    parsed = CommandParser().parse(line)
    assert parsed.tokens == ("log", "mark", expected)


def test_unterminated_quote_is_reported_without_executing_input() -> None:
    with pytest.raises(CommandParseError, match="Invalid quoting"):
        CommandParser().parse('log mark "unterminated')


def test_nul_byte_is_rejected() -> None:
    with pytest.raises(CommandParseError, match="NUL"):
        CommandParser().parse("status\x00")


def test_maximum_length_is_enforced() -> None:
    parser = CommandParser(maximum_length=8)
    with pytest.raises(CommandParseError, match="character limit"):
        parser.parse("status --full")


def test_split_options_supports_flags_values_repetition_and_delimiter() -> None:
    parsed = CommandParser.split_options(
        ("--json", "--watch", "0.5", "--tag=one", "--tag", "two", "--", "--literal")
    )
    assert parsed.positionals == ("--literal",)
    assert parsed.has_flag("json")
    assert parsed.values_for("watch") == ("0.5",)
    assert parsed.values_for("tag") == ("one", "two")


def test_negative_motor_duty_remains_a_positional_argument() -> None:
    parsed = CommandParser.split_options(("-120",))
    assert parsed.positionals == ("-120",)
    assert parsed.options == {}
