"""Prove every benchmark task is both broken and solvable.

A task that already passes measures nothing, and a task with no fix measures
nothing either. This script checks both ends against a copy of each task, so
the originals stay broken:

  1. run check.sh on the pristine task -- it must FAIL
  2. apply the reference fix -- check.sh must now PASS

The reference fixes live here rather than in the task folders, because a fix
sitting next to the bug is a fix the agent can read.

Run: python scripts/verify_benchmark_tasks.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "tasks"

# task_id -> {filename: fixed contents}
REFERENCE_FIXES: dict[str, dict[str, str]] = {
    "01_off_by_one_slice": {
        "window.py": '''"""Sliding windows over a sequence."""


def windows(items, size):
    """Return every contiguous window of length `size`, in order."""
    result = []
    for start in range(len(items) - size + 1):
        result.append(items[start : start + size])
    return result
'''
    },
    "02_wrong_default_argument": {
        "rounding.py": '''"""Money rounding helpers."""


def round_money(amount, places=2):
    """Round a monetary amount to `places` decimal places."""
    return round(amount, places)
'''
    },
    "03_missing_edge_case": {
        "stats.py": '''"""Simple statistics."""


def mean(values):
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)
'''
    },
    "04_import_cycle": {
        "models.py": '''"""Data models."""


class User:
    def __init__(self, name):
        self.name = name

    def to_json(self):
        # Imported inside the method to break the cycle: serializers imports
        # models at module level, so models must not do the reverse.
        from serializers import serialize_user

        return serialize_user(self)
'''
    },
    "05_swallowed_exception": {
        "config_loader.py": '''"""Load JSON configuration."""

import json


def load_config(path):
    """Read a JSON config file. Missing file -> {}. Malformed JSON -> ValueError."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
'''
    },
    "06_implement_from_docstring": {
        "chunker.py": '''"""Text chunking."""

import re

SENTENCE_END = re.compile(r"(?<=[.?!])(?:\\s+|$)")


def chunk_by_sentences(text, max_chars):
    """Split `text` into chunks of at most `max_chars` characters."""
    if not text.strip():
        return []

    sentences = [s for s in SENTENCE_END.split(text.strip()) if s]
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks
'''
    },
    "07_regex_fails_on_arabic": {
        "tokenize_module.py": r'''"""Word tokenization."""

import re

# \w matches any Unicode word character, which covers Arabic script; the
# explicit digit exclusion keeps numbers from counting as words.
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text):
    """Return the words in `text`, in order."""
    return WORD.findall(text)
'''
    },
    "08_timezone_bug": {
        "deadline.py": '''"""Deadline checks."""

from datetime import UTC, datetime


def hours_remaining(deadline_utc):
    """Hours between now (UTC) and `deadline_utc`, a timezone-aware UTC datetime."""
    # datetime.utcnow() returns a naive datetime, which cannot be subtracted
    # from an aware one. datetime.now(UTC) is aware.
    return (deadline_utc - datetime.now(UTC)).total_seconds() / 3600
'''
    },
    "09_schema_mismatch": {
        "consumer.py": '''"""Formats search results for display."""


def format_results(results):
    """Render results as 'article <no> (<score>)' lines, highest score first."""
    lines = []
    for result in sorted(results, key=lambda r: r["score"], reverse=True):
        lines.append(f"article {result['article']} ({result['score']:.2f})")
    return lines
'''
    },
    "10_mutable_default": {
        "accumulator.py": '''"""Collecting items."""


def add_item(item, basket=None):
    """Append `item` to `basket` and return it.

    The default is None rather than [], because a mutable default is created
    once at definition time and shared by every call that omits it.
    """
    if basket is None:
        basket = []
    basket.append(item)
    return basket
'''
    },
}


def run_check(folder: Path) -> tuple[bool, str]:
    """Run the task's tests. Returns (passed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def main() -> int:
    task_ids = sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir())
    failures: list[str] = []

    for task_id in task_ids:
        with tempfile.TemporaryDirectory() as staging:
            work = Path(staging) / task_id
            shutil.copytree(TASKS_DIR / task_id, work)

            broken_passes, _ = run_check(work)
            if broken_passes:
                failures.append(f"{task_id}: task passes unfixed, so it measures nothing")
                print(f"FAIL  {task_id}: passes without a fix")
                continue

            fixes = REFERENCE_FIXES.get(task_id)
            if not fixes:
                failures.append(f"{task_id}: no reference fix")
                print(f"FAIL  {task_id}: no reference fix recorded")
                continue

            for name, content in fixes.items():
                (work / name).write_text(content, encoding="utf-8")

            fixed_passes, output = run_check(work)
            if not fixed_passes:
                failures.append(f"{task_id}: reference fix does not solve it")
                print(f"FAIL  {task_id}: reference fix leaves it failing\n{output[-800:]}")
                continue

            print(f"ok    {task_id}: fails unfixed, passes with the reference fix")

    print()
    if failures:
        print(f"{len(failures)} of {len(task_ids)} tasks are not usable:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"all {len(task_ids)} tasks are broken-then-solvable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
