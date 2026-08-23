"""Kill the process mid-run, resume, and check nothing ran twice.

This spawns a real subprocess and kills it uncatchably (SIGKILL on POSIX,
TerminateProcess on Windows, `Popen.kill()` is both). A try/finally cleanup
path would not be exercised by anything gentler, and the whole point is that
the journal survives a process that got no chance to tidy up.

The kill happens at ten different delays, so the interruption lands at ten
different points in the loop rather than always at the same one. Roughly half
land mid-`record`, where the side effect has already happened but its result
was never journalled: the case the idempotency rule exists for.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import completion

from agent_core.agent import Agent
from agent_core.config import Price, Settings
from agent_core.journal import Journal
from agent_core.providers.replay import ScriptedProvider
from agent_core.registry import ToolRegistry

WORKER = Path(__file__).parent / "crash_worker.py"


def run_and_kill(journal_db: Path, counter: Path, run_id: str, after_s: float) -> None:
    """Start the worker, let it get going, then kill it without warning."""
    process = subprocess.Popen(
        [sys.executable, str(WORKER), str(journal_db), str(counter), run_id],
        cwd=str(Path(__file__).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(after_s)
    process.kill()
    process.wait(timeout=10)


def resume_run(journal_db: Path, counter: Path, run_id: str) -> None:
    """Resume the killed run in this process."""
    registry = ToolRegistry()

    @registry.register(timeout_s=30, idempotent=False)
    def record(tag: str) -> str:
        """Append a tag to the counter file. NOT idempotent."""
        with counter.open("a", encoding="utf-8") as handle:
            handle.write(f"{tag}\n")
        return f"recorded {tag}"

    @registry.register(timeout_s=30, idempotent=True)
    def peek(tag: str) -> str:
        """Read-only in spirit; logs so the test can observe re-execution."""
        with counter.with_suffix(".peeks").open("a", encoding="utf-8") as handle:
            handle.write(tag + "\n")
        return f"peeked {tag}"

    agent = Agent(
        provider=ScriptedProvider([completion(text="resumed and finished")]),
        registry=registry,
        system="crash test",
        journal=Journal(journal_db),
        settings=Settings(models={}, pricing={"scripted": Price(input=0.0, output=0.0)}),
    )
    agent.resume(run_id)


# Ten kill points spanning the run, so the interruption lands at ten
# different places in the loop rather than repeatedly at one.
@pytest.mark.parametrize("kill_after_s", [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3])
def test_kill_and_resume_never_repeats_a_side_effect(tmp_path: Path, kill_after_s: float) -> None:
    journal_db = tmp_path / "runs.db"
    counter = tmp_path / "counter.txt"
    counter.write_text("", encoding="utf-8")
    run_id = "crash_run"

    run_and_kill(journal_db, counter, run_id, kill_after_s)

    tags_before = counter.read_text(encoding="utf-8").split()
    peeks_file = counter.with_suffix(".peeks")
    peeks_before = peeks_file.read_text(encoding="utf-8").split() if peeks_file.exists() else []

    journal = Journal(journal_db)
    orphans = journal.orphaned_tool_calls(run_id)
    orphan_names = [s.payload["name"] for s in orphans]
    completed_before = journal.completed_tool_call_ids(run_id)
    journal.close()

    if not tags_before and not completed_before:
        pytest.skip(f"kill at {kill_after_s}s landed before any tool ran")

    resume_run(journal_db, counter, run_id)

    tags_after = counter.read_text(encoding="utf-8").split()
    peeks_after = peeks_file.read_text(encoding="utf-8").split() if peeks_file.exists() else []

    # Property 1: no side effect is ever repeated. A journalled-complete call
    # that re-ran, or a non-idempotent orphan that re-ran, duplicates a tag.
    assert len(tags_after) == len(set(tags_after)), (
        f"a side effect was repeated. before={tags_before} after={tags_after}"
    )

    # Property 2: a non-idempotent orphan is reported, not re-executed. Its
    # side effect already happened before the kill, so the file must not grow.
    if "record" in orphan_names:
        assert tags_after == tags_before, (
            f"a non-idempotent orphan re-executed. orphans={orphan_names} "
            f"before={tags_before} after={tags_after}"
        )

    # Property 3: an idempotent orphan IS re-executed, exactly once. Without
    # this the first two properties would also pass a resume that simply
    # refused to do anything.
    if "peek" in orphan_names:
        assert len(peeks_after) == len(peeks_before) + orphan_names.count("peek"), (
            f"an idempotent orphan was not re-run. orphans={orphan_names} "
            f"before={peeks_before} after={peeks_after}"
        )


def test_the_worker_completes_normally_when_not_killed(tmp_path: Path) -> None:
    """Control: without a kill, the run finishes and every tag is recorded once."""
    journal_db = tmp_path / "runs.db"
    counter = tmp_path / "counter.txt"
    counter.write_text("", encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(WORKER), str(journal_db), str(counter), "clean_run"],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        timeout=60,
    )

    assert process.returncode == 0, process.stderr.decode(errors="replace")
    tags = counter.read_text(encoding="utf-8").split()
    # `record` runs on even indices 0,2,4,6.
    assert tags == ["0", "2", "4", "6"]
    assert Journal(journal_db).run_row("clean_run")["status"] == "complete"
