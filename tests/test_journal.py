"""Journal and resume.

The property that matters: after an interruption, a tool call whose outcome
was never recorded is re-run only when the tool declared itself idempotent.
Everything else is reported to the model as an unknown outcome. Silently
re-running a non-idempotent side effect -- charging a card twice, appending a
row twice -- is the bug this design exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

from conftest import call, completion

from agent_core.agent import Agent
from agent_core.journal import Journal
from agent_core.providers.replay import ScriptedProvider
from agent_core.registry import ToolRegistry


def test_steps_are_appended_and_read_back(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs.db")
    journal.start_run("r1", "do a thing")
    journal.append("r1", "model_call", {"content": []}, input_tokens=10, output_tokens=5, cost_usd=0.1)
    journal.append("r1", "final", {"text": "done"})
    journal.finish_run("r1", "complete")

    steps = journal.steps("r1")
    assert [s.kind for s in steps] == ["model_call", "final"]
    assert journal.totals("r1")["cost_usd"] == 0.1
    assert journal.run_row("r1")["status"] == "complete"


def test_a_started_call_with_no_result_is_orphaned(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs.db")
    journal.start_run("r1", "goal")
    journal.append("r1", "tool_call_started", {"tool_use_id": "a", "name": "t", "input": {}})
    journal.append("r1", "tool_result", {"tool_use_id": "a", "content": "ok", "is_error": False})
    journal.append("r1", "tool_call_started", {"tool_use_id": "b", "name": "t", "input": {}})
    # "b" never got a result: the process died mid-call.

    orphans = journal.orphaned_tool_calls("r1")
    assert [s.payload["tool_use_id"] for s in orphans] == ["b"]


def test_every_step_is_committed_immediately(tmp_path: Path) -> None:
    """A second connection must see the step, i.e. it was really committed."""
    path = tmp_path / "runs.db"
    journal = Journal(path)
    journal.start_run("r1", "goal")
    journal.append("r1", "model_call", {"content": []})

    observer = Journal(path)
    assert len(observer.steps("r1")) == 1


def build_agent(registry: ToolRegistry, provider: ScriptedProvider, journal: Journal, settings) -> Agent:
    return Agent(
        provider=provider,
        registry=registry,
        system="test",
        journal=journal,
        settings=settings,
    )


def test_resume_does_not_rerun_a_completed_tool_call(tmp_path: Path, settings) -> None:
    """A journalled-complete call must never execute twice."""
    counter = tmp_path / "counter.txt"
    counter.write_text("", encoding="utf-8")
    registry = ToolRegistry()

    @registry.register(idempotent=False)
    def append_row() -> str:
        """Append one row. Not idempotent."""
        with counter.open("a", encoding="utf-8") as handle:
            handle.write("x")
        return "appended"

    journal = Journal(tmp_path / "runs.db")

    # First run: one tool call completes, then the transcript is left open.
    provider = ScriptedProvider([completion(tool_calls=[call("append_row", "c1")])])
    agent = build_agent(registry, provider, journal, settings)
    journal.start_run("r1", "append a row")
    try:
        agent._loop("r1", "append a row", [])
    except AssertionError:
        pass  # provider ran out, simulating the process dying mid-run

    assert counter.read_text(encoding="utf-8") == "x"

    # Resume: the call has a journalled result, so it must not run again.
    resume_provider = ScriptedProvider([completion(text="finished")])
    resume_agent = build_agent(registry, resume_provider, journal, settings)
    resume_agent.resume("r1")

    assert counter.read_text(encoding="utf-8") == "x", "a completed tool call ran twice"


def test_resume_reruns_an_orphaned_idempotent_call(tmp_path: Path, settings) -> None:
    calls: list[str] = []
    registry = ToolRegistry()

    @registry.register(idempotent=True)
    def read_state() -> str:
        """Read something. Safe to repeat."""
        calls.append("read")
        return "state"

    journal = Journal(tmp_path / "runs.db")
    journal.start_run("r1", "read state")
    # Started, never finished.
    journal.append(
        "r1",
        "tool_call_started",
        {"tool_use_id": "c1", "name": "read_state", "input": {}},
        idempotent=True,
    )

    provider = ScriptedProvider([completion(text="done")])
    agent = build_agent(registry, provider, journal, settings)
    agent.resume("r1")

    assert calls == ["read"], "an idempotent orphan should be re-executed exactly once"


def test_resume_refuses_to_rerun_an_orphaned_non_idempotent_call(
    tmp_path: Path, settings
) -> None:
    """The headline behaviour: unknown outcome is reported, not repeated."""
    calls: list[str] = []
    registry = ToolRegistry()

    @registry.register(idempotent=False)
    def charge_card() -> str:
        """Charge a card. Absolutely not safe to repeat."""
        calls.append("charged")
        return "charged"

    journal = Journal(tmp_path / "runs.db")
    journal.start_run("r1", "charge the card")
    journal.append(
        "r1",
        "tool_call_started",
        {"tool_use_id": "c1", "name": "charge_card", "input": {}},
        idempotent=False,
    )

    provider = ScriptedProvider([completion(text="I will verify before retrying.")])
    agent = build_agent(registry, provider, journal, settings)
    agent.resume("r1")

    assert calls == [], "a non-idempotent orphan must not be re-executed"

    # The model must be told, as a failed tool_result, that the outcome is unknown.
    blocks = [
        block
        for message in provider.calls[0]["messages"]
        for block in message.content
        if block.get("type") == "tool_result"
    ]
    assert len(blocks) == 1
    assert blocks[0]["is_error"] is True
    assert "outcome is unknown" in blocks[0]["content"]


def test_resume_rebuilds_the_transcript_in_order(tmp_path: Path, settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def noop() -> str:
        """Does nothing."""
        return "ok"

    journal = Journal(tmp_path / "runs.db")
    journal.start_run("r1", "the original goal")
    journal.append(
        "r1",
        "model_call",
        {
            "content": [{"type": "tool_use", "id": "c1", "name": "noop", "input": {}}],
            "stop_reason": "tool_use",
        },
    )
    journal.append("r1", "tool_result", {"tool_use_id": "c1", "content": "ok", "is_error": False})

    provider = ScriptedProvider([completion(text="resumed")])
    agent = build_agent(registry, provider, journal, settings)
    result = agent.resume("r1")

    assert result.text == "resumed"
    replayed = provider.calls[0]["messages"]
    assert replayed[0].text() == "the original goal"
    assert replayed[1].tool_uses()[0].id == "c1"
    assert replayed[2].content[0]["tool_use_id"] == "c1"


def test_unfinished_runs_are_listed(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs.db")
    journal.start_run("done", "a")
    journal.finish_run("done", "complete")
    journal.start_run("open", "b")

    assert journal.unfinished_runs() == ["open"]
