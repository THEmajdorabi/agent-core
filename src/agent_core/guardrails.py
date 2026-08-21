"""Budgets, checked before every model call.

An agent loop without budgets is a loop that can spend your month's API credit
on a task it was never going to finish. Each limit here answers a different
failure: `max_steps` catches a loop that oscillates, `max_cost_usd` catches an
expensive one, `max_wall_s` catches one that hangs on a slow tool, and
`max_consecutive_tool_failures` catches one that keeps calling a broken tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Guardrails:
    """Limits for a single run."""

    max_steps: int = 25
    max_cost_usd: float = 1.00
    max_wall_s: float = 300.0
    tool_allowlist: set[str] | None = None
    max_consecutive_tool_failures: int = 3


@dataclass
class Budget:
    """Live tally for one run, checked against a `Guardrails`."""

    guardrails: Guardrails
    started_at: float = field(default_factory=time.monotonic)
    steps: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    consecutive_tool_failures: int = 0

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def record_model_call(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.steps += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd

    def record_tool_outcome(self, *, failed: bool) -> None:
        """Track consecutive failures.

        Counted across tools rather than per tool: an agent failing three
        different tools in a row is as stuck as one failing the same tool
        three times, and the simpler rule is the one you can reason about at
        3am.
        """
        self.consecutive_tool_failures = self.consecutive_tool_failures + 1 if failed else 0

    def check(self) -> str | None:
        """Return a human-readable breach reason, or None if within budget."""
        g = self.guardrails
        if self.steps >= g.max_steps:
            return f"step limit reached ({self.steps}/{g.max_steps})"
        if self.cost_usd >= g.max_cost_usd:
            return f"cost limit reached (${self.cost_usd:.4f}/${g.max_cost_usd:.2f})"
        elapsed = self.elapsed_s()
        if elapsed >= g.max_wall_s:
            return f"time limit reached ({elapsed:.1f}s/{g.max_wall_s:.0f}s)"
        if self.consecutive_tool_failures >= g.max_consecutive_tool_failures:
            return (
                f"{self.consecutive_tool_failures} consecutive tool failures "
                f"(limit {g.max_consecutive_tool_failures})"
            )
        return None
