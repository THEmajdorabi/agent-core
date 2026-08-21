"""Spans and the trace tree.

Every step of a run opens a span. Rendering them as a tree turns "the agent did
something for 47 seconds" into a readable account of what it actually did,
which is the fastest way to debug a loop and the most legible proof that this
is a real runtime rather than a wrapper around one API call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """One timed event inside a run."""

    name: str
    kind: str  # model_call | tool | compaction | final
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list[Span] = field(default_factory=list)

    def duration_s(self) -> float:
        return (self.ended_at or time.monotonic()) - self.started_at

    def end(self, error: str | None = None, **attributes: Any) -> None:
        self.ended_at = time.monotonic()
        self.error = error
        self.attributes.update(attributes)


class Tracer:
    """Collects spans for one run."""

    def __init__(self, run_id: str, goal: str) -> None:
        self.run_id = run_id
        self.goal = goal
        self.spans: list[Span] = []
        self.started_at = time.monotonic()

    def span(self, name: str, kind: str, **attributes: Any) -> Span:
        span = Span(name=name, kind=kind, started_at=time.monotonic(), attributes=attributes)
        self.spans.append(span)
        return span

    def total_duration_s(self) -> float:
        return time.monotonic() - self.started_at

    def render(self, cost_usd: float = 0.0, wall_s: float | None = None) -> str:
        """Render the run as a tree.

        Pure string building with box-drawing characters: no dependency, and
        the output pastes cleanly into a README or a terminal screenshot.
        """
        wall = wall_s if wall_s is not None else self.total_duration_s()
        header = (
            f"run {self.run_id[:8]}  goal={self.goal!r}  "
            f"{len(self.spans)} steps  ${cost_usd:.4f}  {wall:.1f}s"
        )
        lines = [header]

        for index, span in enumerate(self.spans):
            is_last = index == len(self.spans) - 1
            elbow = "└─ " if is_last else "├─ "
            lines.append(elbow + self._describe(span))

        return "\n".join(lines)

    @staticmethod
    def _describe(span: Span) -> str:
        attributes = span.attributes

        if span.kind == "model_call":
            label = f"{'model_call':<18}"
            tokens = f"{attributes.get('input_tokens', 0):,} in / {attributes.get('output_tokens', 0):,} out"
            return f"{label} {tokens}"

        if span.kind == "tool":
            label = f"{'tool: ' + span.name:<18}"
            status = f"ERROR {span.error}" if span.error else "ok"
            return f"{label} {span.duration_s():.2f}s  {status}"

        if span.kind == "compaction":
            label = f"{'compaction':<18}"
            before = attributes.get("tokens_before", 0)
            after = attributes.get("tokens_after", 0)
            return f"{label} {before:,} -> {after:,} tokens"

        if span.kind == "final":
            preview = str(attributes.get("text", "")).replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            return f"{'final':<18} {preview!r}"

        return f"{span.name:<18} {span.duration_s():.2f}s"
