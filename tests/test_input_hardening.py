"""Input hardening (plan 5.3; edge cases I-11, I-12, I-13, I-14, I-20)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.models import MAX_CUISINE_CHARS, MAX_FREE_TEXT_CHARS, MAX_PARTY_SIZE, Preferences


@pytest.mark.parametrize(
    "fields",
    [
        {"min_rating": -0.1},  # I-11
        {"min_rating": 5.1},
        {"budget": "cheap"},  # I-12
        {"free_text": "x" * (MAX_FREE_TEXT_CHARS + 1)},  # I-14
        {"free_text": "x" * 10_000},
        {"cuisines": ["x" * (MAX_CUISINE_CHARS + 1)]},
        {"cuisines": [f"cuisine {i}" for i in range(21)]},
        {"party_size": 0},  # I-20
        {"party_size": MAX_PARTY_SIZE + 1},
        {"location": "x" * 101},
        {"price": "low"},  # unknown fields are rejected, not ignored
    ],
    ids=lambda f: next(iter(f)),
)
def test_out_of_bounds_preferences_are_rejected(fields):
    with pytest.raises(ValidationError):
        Preferences(**fields)


def test_limits_themselves_are_accepted():
    p = Preferences(free_text="x" * MAX_FREE_TEXT_CHARS, cuisines=["x" * MAX_CUISINE_CHARS], party_size=MAX_PARTY_SIZE,
                    min_rating=5.0)
    assert len(p.free_text) == MAX_FREE_TEXT_CHARS


def test_control_characters_are_stripped():
    p = Preferences(location="Kora\x00mangala\x1b", free_text="quiet\x07 place\r\nwith\tparking", cuisines=["Chi\x08nese"])
    assert p.location == "Koramangala"
    assert p.free_text == "quiet place\nwith\tparking"
    assert p.cuisines == ["Chinese"]


def test_input_that_is_only_control_characters_counts_as_blank():  # I-13
    p = Preferences(location="\x00\x01", free_text=" \x7f ", cuisines=["\x1b"])
    assert (p.location, p.free_text, p.cuisines) == (None, None, [])
