"""Exception types for the agent runtime.

Every failure mode the loop can hit has its own type, so a caller can tell a
guardrail breach (expected, carries a partial result) apart from a provider
outage (retryable) apart from a bug (not caught at all).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.agent import RunResult


class AgentError(Exception):
    """Base class for everything this package raises."""


class GuardrailBreach(AgentError):
    """A budget was exhausted.

    Carries the partial run so the caller can inspect what happened before the
    stop. A bare abort would throw away the trace, which is the one thing you
    want when a run costs a dollar and returns nothing.
    """

    def __init__(self, message: str, partial: RunResult | None = None) -> None:
        super().__init__(message)
        self.partial = partial


class ToolNotFound(AgentError):
    """The model asked for a tool that is not registered."""


class ToolExecutionError(AgentError):
    """A tool raised. This is normally converted into an error tool_result
    rather than propagated, so the model gets a chance to recover."""


class ToolTimeout(ToolExecutionError):
    """A tool exceeded its per-tool timeout."""


class ProviderError(AgentError):
    """The model provider failed in a way we could not retry past."""


class StructuredOutputError(AgentError):
    """The model could not be coaxed into schema-valid JSON within the repair budget."""


class JournalError(AgentError):
    """The run journal is missing or inconsistent."""


class SandboxError(AgentError):
    """The Docker sandbox could not run the code."""


def truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` bytes, marking the cut so nobody mistakes a
    truncated payload for the whole thing."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    kept = encoded[:limit].decode("utf-8", errors="ignore")
    dropped = len(encoded) - limit
    return f"{kept}\n... [truncated {dropped} bytes of {len(encoded)}]"


def describe(exc: BaseException) -> dict[str, Any]:
    """Render an exception as loggable structured data."""
    return {"error_type": type(exc).__name__, "error": str(exc)}
