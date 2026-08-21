"""Shared fixtures.

Everything here is offline. No test in the default run may touch a paid API;
the `integration` marker exists for the few that do and is excluded by default
in pyproject.toml.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.config import Price, Settings
from agent_core.providers.base import Completion, ToolCall


@pytest.fixture
def settings() -> Settings:
    """Minimal settings with a known price, so cost assertions are exact."""
    return Settings(
        models={},
        pricing={"scripted": Price(input=1.0, output=2.0)},
    )


def completion(
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
    stop_reason: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "scripted",
) -> Completion:
    """Build a Completion for a scripted provider."""
    calls = tool_calls or []
    return Completion(
        text=text,
        tool_calls=calls,
        stop_reason=stop_reason or ("tool_use" if calls else "end_turn"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def call(name: str, call_id: str = "call_1", **kwargs: Any) -> ToolCall:
    """Build a ToolCall."""
    return ToolCall(id=call_id, name=name, input=kwargs)
