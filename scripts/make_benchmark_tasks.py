"""Generate the ten benchmark task folders.

Each task is a tiny broken repository plus a `check.sh` that exits 0 once the
bug is fixed. Generating them from one script keeps the shape uniform and
makes the whole set reproducible from a clean clone.

The bugs are real bugs -- an off-by-one, a mutable default, a naive-datetime
comparison -- rather than syntax errors, because "find the typo" measures
nothing about an agent's ability to read code.

Run: python scripts/make_benchmark_tasks.py
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "tasks"

CHECK_SH = """#!/usr/bin/env bash
# Exits 0 when the task is solved.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q test_solution.py
"""


class Task:
    def __init__(self, task_id: str, title: str, files: dict[str, str], goal: str) -> None:
        self.task_id = task_id
        self.title = title
        self.files = files
        self.goal = goal

    def write(self) -> None:
        folder = TASKS_DIR / self.task_id
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        check = folder / "check.sh"
        check.write_text(CHECK_SH, encoding="utf-8")
        check.chmod(check.stat().st_mode | stat.S_IEXEC)

        (folder / "task.json").write_text(
            json.dumps({"id": self.task_id, "title": self.title, "goal": self.goal}, indent=2),
            encoding="utf-8",
        )


TASKS = [
    # 1 -- off-by-one in a slice
    Task(
        "01_off_by_one_slice",
        "Off-by-one in a slice bound",
        {
            "window.py": '''"""Sliding windows over a sequence."""


def windows(items, size):
    """Return every contiguous window of length `size`, in order."""
    result = []
    for start in range(len(items) - size):
        result.append(items[start : start + size])
    return result
''',
            "test_solution.py": '''from window import windows


def test_windows_covers_the_whole_sequence():
    assert windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]


def test_single_window_when_size_equals_length():
    assert windows([1, 2, 3], 3) == [[1, 2, 3]]


def test_empty_when_size_exceeds_length():
    assert windows([1, 2], 3) == []
''',
        },
        "The tests in test_solution.py fail. Fix window.py so they pass.",
    ),
    # 2 -- wrong default argument
    Task(
        "02_wrong_default_argument",
        "A wrong default argument breaks a test",
        {
            "rounding.py": '''"""Money rounding helpers."""


def round_money(amount, places=0):
    """Round a monetary amount to `places` decimal places.

    Money defaults to 2 decimal places, not 0.
    """
    return round(amount, places)
''',
            "test_solution.py": '''from rounding import round_money


def test_defaults_to_two_places():
    assert round_money(3.14159) == 3.14


def test_explicit_places_still_work():
    assert round_money(3.14159, 3) == 3.142
    assert round_money(3.7, 0) == 4
''',
        },
        "The tests in test_solution.py fail. Fix rounding.py so they pass.",
    ),
    # 3 -- missing edge-case branch
    Task(
        "03_missing_edge_case",
        "A missing edge-case branch",
        {
            "stats.py": '''"""Simple statistics."""


def mean(values):
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values)
''',
            "test_solution.py": '''from stats import mean


def test_mean_of_values():
    assert mean([1, 2, 3]) == 2.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
''',
        },
        "The tests in test_solution.py fail. Fix stats.py so they pass.",
    ),
    # 4 -- import cycle
    Task(
        "04_import_cycle",
        "An import cycle between two modules",
        {
            "models.py": '''"""Data models."""

from serializers import serialize_user


class User:
    def __init__(self, name):
        self.name = name

    def to_json(self):
        return serialize_user(self)
''',
            "serializers.py": '''"""Serialization helpers."""

from models import User


def serialize_user(user):
    """Render a User as a dict."""
    return {"name": user.name}


def is_user(value):
    return isinstance(value, User)
''',
            "test_solution.py": '''def test_user_serializes():
    from models import User

    assert User("majd").to_json() == {"name": "majd"}


def test_is_user_still_works():
    from models import User
    from serializers import is_user

    assert is_user(User("x")) is True
    assert is_user("not a user") is False
''',
        },
        "Importing these modules raises ImportError because of a circular import. "
        "Fix it so the tests in test_solution.py pass, keeping both functions working.",
    ),
    # 5 -- exception handling swallows an error
    Task(
        "05_swallowed_exception",
        "Exception handling that swallows an error",
        {
            "config_loader.py": '''"""Load JSON configuration."""

import json


def load_config(path):
    """Read a JSON config file.

    A missing file returns an empty dict. Malformed JSON must raise
    ValueError, because silently returning {} hides a broken deploy.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}
''',
            "test_solution.py": '''import json

import pytest

from config_loader import load_config


def test_missing_file_returns_empty(tmp_path):
    assert load_config(tmp_path / "nope.json") == {}


def test_valid_file_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_config(path) == {"a": 1}


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
''',
        },
        "The tests in test_solution.py fail. Fix config_loader.py so they pass.",
    ),
    # 6 -- implement from docstring
    Task(
        "06_implement_from_docstring",
        "Implement a function from its docstring and tests",
        {
            "chunker.py": '''"""Text chunking."""


def chunk_by_sentences(text, max_chars):
    """Split `text` into chunks of at most `max_chars` characters.

    Sentences end at '.', '?' or '!' followed by a space or end of string.
    A chunk never splits a sentence: sentences are packed greedily until
    adding the next one would exceed `max_chars`. A single sentence longer
    than `max_chars` becomes a chunk of its own. Leading and trailing
    whitespace is stripped from every chunk. Empty input returns [].
    """
    raise NotImplementedError
''',
            "test_solution.py": '''from chunker import chunk_by_sentences


def test_empty():
    assert chunk_by_sentences("", 10) == []


def test_packs_sentences_greedily():
    text = "One. Two. Three."
    assert chunk_by_sentences(text, 10) == ["One. Two.", "Three."]


def test_never_splits_a_sentence():
    text = "A very long single sentence that exceeds the limit."
    assert chunk_by_sentences(text, 10) == [
        "A very long single sentence that exceeds the limit."
    ]


def test_handles_question_and_exclamation():
    text = "Why? Because! Done."
    assert chunk_by_sentences(text, 14) == ["Why? Because!", "Done."]
''',
        },
        "chunker.py raises NotImplementedError. Implement it so test_solution.py passes.",
    ),
    # 7 -- regex fails on Arabic
    Task(
        "07_regex_fails_on_arabic",
        "A regex that fails on Arabic input",
        {
            "tokenize.py": r'''"""Word tokenization."""

import re

# Intended to match words in any script.
WORD = re.compile(r"[A-Za-z]+")


def words(text):
    """Return the words in `text`, in order.

    Must work for Latin script, Arabic script, and a mix of the two.
    Digits and punctuation are not words.
    """
    return WORD.findall(text)
''',
            "test_solution.py": '''from tokenize_module import words


def test_latin():
    assert words("hello there") == ["hello", "there"]


def test_arabic():
    assert words("مدة الإشعار") == ["مدة", "الإشعار"]


def test_mixed_script_and_digits():
    assert words("المادة 74 states the rule") == [
        "المادة",
        "states",
        "the",
        "rule",
    ]
''',
        },
        "The Arabic tests in test_solution.py fail. Fix tokenize.py so all of them pass. "
        "Note the test imports `tokenize_module`, so the module must be importable "
        "under that name.",
    ),
    # 8 -- timezone bug
    Task(
        "08_timezone_bug",
        "A timezone bug in a date calculation",
        {
            "deadline.py": '''"""Deadline checks."""

from datetime import datetime


def hours_remaining(deadline_utc):
    """Hours between now (UTC) and `deadline_utc`, which is timezone-aware UTC.

    Returns a float, negative when the deadline has passed.
    """
    return (deadline_utc - datetime.utcnow()).total_seconds() / 3600
''',
            "test_solution.py": '''from datetime import UTC, datetime, timedelta

from deadline import hours_remaining


def test_future_deadline_is_positive():
    later = datetime.now(UTC) + timedelta(hours=5)
    assert 4.9 < hours_remaining(later) < 5.1


def test_past_deadline_is_negative():
    earlier = datetime.now(UTC) - timedelta(hours=2)
    assert -2.1 < hours_remaining(earlier) < -1.9
''',
        },
        "The tests in test_solution.py fail with a TypeError about naive and aware "
        "datetimes. Fix deadline.py so they pass.",
    ),
    # 9 -- schema mismatch between modules
    Task(
        "09_schema_mismatch",
        "A JSON schema mismatch between two modules",
        {
            "producer.py": '''"""Produces search results."""


def search(query):
    """Return a list of result dicts."""
    return [
        {"doc": "labor_law", "article": "74", "score": 0.91, "snippet": "..."},
        {"doc": "labor_law", "article": "75", "score": 0.88, "snippet": "..."},
    ]
''',
            "consumer.py": '''"""Formats search results for display."""


def format_results(results):
    """Render results as 'article <no> (<score>)' lines, highest score first."""
    lines = []
    for result in sorted(results, key=lambda r: r["relevance"], reverse=True):
        lines.append(f"article {result['article_no']} ({result['relevance']:.2f})")
    return lines
''',
            "test_solution.py": '''from consumer import format_results
from producer import search


def test_pipeline_end_to_end():
    assert format_results(search("notice period")) == [
        "article 74 (0.91)",
        "article 75 (0.88)",
    ]
''',
        },
        "producer.py and consumer.py disagree about the result dict's field names, "
        "so the test raises KeyError. Fix the mismatch so test_solution.py passes. "
        "Do not change the test.",
    ),
    # 10 -- mutable default argument
    Task(
        "10_mutable_default",
        "A mutable default argument bug",
        {
            "accumulator.py": '''"""Collecting items."""


def add_item(item, basket=[]):
    """Append `item` to `basket` and return it.

    Called without a basket, each call starts from an empty one.
    """
    basket.append(item)
    return basket
''',
            "test_solution.py": '''from accumulator import add_item


def test_each_call_starts_fresh():
    assert add_item("a") == ["a"]
    assert add_item("b") == ["b"]


def test_explicit_basket_accumulates():
    basket = []
    add_item("a", basket)
    add_item("b", basket)
    assert basket == ["a", "b"]
''',
        },
        "The tests in test_solution.py fail. Fix accumulator.py so they pass.",
    ),
]


def main() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        task.write()
        print(f"wrote {task.task_id}")

    # Task 07's module is imported as `tokenize_module` because `tokenize` is a
    # standard-library module name, and shadowing it breaks pytest's own imports.
    seven = TASKS_DIR / "07_regex_fails_on_arabic"
    (seven / "tokenize.py").rename(seven / "tokenize_module.py")
    (seven / "task.json").write_text(
        json.dumps(
            {
                "id": "07_regex_fails_on_arabic",
                "title": "A regex that fails on Arabic input",
                "goal": "The Arabic tests in test_solution.py fail. Fix tokenize_module.py "
                "so all of them pass.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(TASKS)} tasks written to {TASKS_DIR}")


if __name__ == "__main__":
    main()
