"""Compaction.

The bug this file exists to prevent: compacting a transcript at a point that
separates a `tool_use` block from its `tool_result`. The API rejects the next
request, and the failure surfaces as an opaque 400 several steps later.
"""

from __future__ import annotations

from conftest import completion

from agent_core.context import (
    CHARS_PER_TOKEN_ARABIC,
    CHARS_PER_TOKEN_LATIN,
    Compactor,
    PinnedFacts,
    estimate_tokens_heuristic,
    settled_boundaries,
)
from agent_core.providers.base import Message, ToolResult
from agent_core.providers.replay import ScriptedProvider


def transcript() -> list[Message]:
    """A realistic transcript: goal, a parallel tool turn, a single tool turn."""
    return [
        Message.user_text("fix the failing test"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Looking around."},
                {"type": "tool_use", "id": "t1", "name": "list_dir", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "a.py"}},
            ],
        ),
        Message.tool_results([ToolResult("t1", "f a.py"), ToolResult("t2", "def f(): pass")]),
        Message(
            role="assistant",
            content=[{"type": "tool_use", "id": "t3", "name": "run_python", "input": {}}],
        ),
        Message.tool_results([ToolResult("t3", "ok")]),
        Message.assistant_text("Done."),
    ]


def test_settled_boundaries_exclude_points_inside_a_tool_exchange() -> None:
    boundaries = settled_boundaries(transcript())

    # 0 (nothing yet), 1 (after the goal), 3 (after t1+t2 answered),
    # 5 (after t3 answered), 6 (end). Never 2 or 4, which sit between a
    # tool_use turn and its results.
    assert boundaries == [0, 1, 3, 5, 6]
    assert 2 not in boundaries
    assert 4 not in boundaries


def test_compaction_never_splits_a_tool_use_from_its_result() -> None:
    """The property, checked directly on the compacted transcript."""
    messages = transcript()
    provider = ScriptedProvider([completion(text="SUMMARY: read a.py, ran the test.")])
    compactor = Compactor(provider, max_tokens=10, trigger_ratio=0.0, keep_recent=2, max_passes=1)

    compacted, changed = compactor.maybe_compact(messages)

    assert changed
    tool_use_ids = {b["id"] for m in compacted for b in m.content if b.get("type") == "tool_use"}
    tool_result_ids = {
        b["tool_use_id"] for m in compacted for b in m.content if b.get("type") == "tool_result"
    }
    assert tool_use_ids == tool_result_ids, "a tool_use lost its tool_result, or vice versa"


def test_nothing_happens_below_the_trigger() -> None:
    messages = transcript()
    provider = ScriptedProvider([])  # would raise if a summary were requested
    compactor = Compactor(provider, max_tokens=1_000_000, trigger_ratio=0.8)

    compacted, changed = compactor.maybe_compact(messages)

    assert changed is False
    assert compacted is messages


def test_recent_messages_are_kept() -> None:
    messages = transcript()
    provider = ScriptedProvider([completion(text="SUMMARY")])
    compactor = Compactor(provider, max_tokens=10, trigger_ratio=0.0, keep_recent=3, max_passes=1)

    compacted, _ = compactor.maybe_compact(messages)

    # The final assistant message survives verbatim.
    assert compacted[-1].text() == "Done."


def test_pinned_facts_survive_compaction_verbatim() -> None:
    messages = transcript()
    pinned = PinnedFacts()
    pinned.pin("The database password is in .env, never in config.yaml.")
    provider = ScriptedProvider([completion(text="SUMMARY")])
    compactor = Compactor(
        provider, max_tokens=10, trigger_ratio=0.0, keep_recent=2, pinned=pinned, max_passes=1
    )

    compacted, _ = compactor.maybe_compact(messages)

    blob = "\n".join(m.text() for m in compacted)
    assert "The database password is in .env, never in config.yaml." in blob


def test_compaction_refuses_when_the_only_cut_in_range_would_split_a_tool_call() -> None:
    """No safe boundary in range means no compaction, not an unsafe one.

    Here the keep-recent window reaches back to index 1, and index 1 sits
    between an assistant tool_use turn and the user turn answering it. The
    only settled boundaries are 0, 2 and 4, none of which is a legal cut. The
    correct answer is to leave the transcript alone: an oversized transcript
    is recoverable, a corrupted one is not.
    """
    messages = [
        Message(
            role="assistant",
            content=[{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        ),
        Message.tool_results([ToolResult("t1", "ok")]),
        Message(
            role="assistant",
            content=[{"type": "tool_use", "id": "t2", "name": "x", "input": {}}],
        ),
        Message.tool_results([ToolResult("t2", "ok")]),
    ]
    assert settled_boundaries(messages) == [0, 2, 4]

    provider = ScriptedProvider([])  # raises if a summary is even attempted
    compactor = Compactor(provider, max_tokens=10, trigger_ratio=0.0, keep_recent=3, max_passes=1)

    compacted, changed = compactor.maybe_compact(messages)

    assert changed is False
    assert compacted == messages


def test_arabic_is_estimated_at_more_tokens_per_character_than_latin() -> None:
    """One ratio for both scripts is wrong in whichever direction matters.

    Compared at identical character counts, so the only variable is the
    script. Arabic must come out as more tokens, because it tokenizes far less
    efficiently than Latin text.
    """
    assert CHARS_PER_TOKEN_ARABIC < CHARS_PER_TOKEN_LATIN

    arabic = "مدة الإشعار المطلوبة قبل إنهاء عقد العمل غير محدد المدة"
    latin = "the notice period required before terminating a contract".ljust(len(arabic))[
        : len(arabic)
    ]

    assert len(arabic) == len(latin)
    assert estimate_tokens_heuristic(arabic) > estimate_tokens_heuristic(latin)


def test_heuristic_handles_mixed_scripts() -> None:
    mixed = "المادة 74 states the notice period"
    assert estimate_tokens_heuristic(mixed) > 0


def test_counting_falls_back_to_the_heuristic_when_the_provider_fails() -> None:
    class BrokenCounter(ScriptedProvider):
        def count_tokens(self, messages, system):
            raise OSError("no network")

    compactor = Compactor(BrokenCounter([]), max_tokens=1000)
    # Must return a number rather than propagating: a counting failure is not
    # a reason to end a run.
    assert compactor.count_tokens([Message.user_text("hello")]) > 0


def test_a_second_pass_runs_when_the_first_does_not_reach_the_target() -> None:
    """Spec allows up to two passes. The second summarises the first summary."""
    messages = transcript()
    provider = ScriptedProvider(
        [completion(text="FIRST SUMMARY"), completion(text="SECOND SUMMARY")]
    )
    compactor = Compactor(
        provider, max_tokens=10, trigger_ratio=0.0, target_ratio=0.0, keep_recent=2, max_passes=2
    )

    compacted, changed = compactor.maybe_compact(messages)

    assert changed
    assert len(provider.calls) == 2, "expected exactly two summarisation calls"
    assert "SECOND SUMMARY" in " ".join(m.text() for m in compacted)


def test_compaction_is_counted_once_per_maybe_compact() -> None:
    """Two passes are one compaction event, not two, for the README stat."""
    messages = transcript()
    provider = ScriptedProvider([completion(text="A"), completion(text="B")])
    compactor = Compactor(
        provider, max_tokens=10, trigger_ratio=0.0, target_ratio=0.0, keep_recent=2, max_passes=2
    )

    compactor.maybe_compact(messages)

    assert compactor.stats()["compactions"] == 1
