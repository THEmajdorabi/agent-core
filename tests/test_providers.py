"""Provider normalisation.

Both SDKs are mocked here. The point is not to test Anthropic or OpenAI, it is
to test that our two adapters turn two very different wire formats into the
same `Completion` and back into the right request shape.
"""

from __future__ import annotations

import json
from typing import Any

from agent_core.providers.base import Completion, Message, ToolCall, ToolResult
from agent_core.providers.openai import OpenAIProvider
from agent_core.providers.replay import (
    ReplayProvider,
    completion_from_dict,
    completion_to_dict,
)


def test_tool_results_go_back_as_one_user_message() -> None:
    """Splitting results across messages teaches the model to stop parallelising."""
    message = Message.tool_results(
        [ToolResult("a", "first"), ToolResult("b", "second", is_error=True)]
    )

    assert message.role == "user"
    assert len(message.content) == 2
    assert message.content[1]["is_error"] is True


def test_completion_round_trips_through_a_message() -> None:
    completion = Completion(
        text="Let me look.",
        tool_calls=[ToolCall(id="t1", name="read_file", input={"path": "a.py"})],
        stop_reason="tool_use",
    )
    message = completion.to_message()

    assert message.role == "assistant"
    assert message.text() == "Let me look."
    assert message.tool_uses()[0].id == "t1"


def test_raw_blocks_are_replayed_verbatim() -> None:
    """Thinking blocks must survive the round trip unchanged.

    Rebuilding an assistant turn from text plus tool_calls would silently drop
    them, and Claude requires them echoed back when the conversation continues
    on the same model.
    """
    blocks: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "", "signature": "abc123"},
        {"type": "text", "text": "Here."},
    ]
    completion = Completion(text="Here.", raw_blocks=blocks)

    assert completion.to_message().content == blocks


def test_openai_tool_definitions_are_reshaped() -> None:
    anthropic_shaped = [
        {
            "name": "search",
            "description": "Search things.",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    converted = OpenAIProvider.to_openai_tools(anthropic_shaped)

    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "search"
    assert converted[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_openai_wire_format_fans_tool_results_into_separate_messages() -> None:
    """Anthropic puts every result in one user message; OpenAI needs one each."""
    messages = [
        Message.user_text("goal"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "working"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}},
                {"type": "tool_use", "id": "t2", "name": "g", "input": {}},
            ],
        ),
        Message.tool_results([ToolResult("t1", "one"), ToolResult("t2", "two")]),
    ]

    wire = OpenAIProvider._to_wire(messages, "system prompt")

    assert wire[0] == {"role": "system", "content": "system prompt"}
    assert wire[1] == {"role": "user", "content": "goal"}

    assistant = wire[2]
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 2
    # OpenAI wants arguments as a JSON string, not an object.
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"x": 1}

    # One `role="tool"` message per result, each carrying its call id.
    assert [m["role"] for m in wire[3:]] == ["tool", "tool"]
    assert wire[3]["tool_call_id"] == "t1"
    assert wire[4]["tool_call_id"] == "t2"


def test_replay_fixture_round_trips(tmp_path) -> None:
    original = Completion(
        text="hi",
        tool_calls=[ToolCall(id="t1", name="f", input={"a": 1})],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=3,
        model="m",
    )
    restored = completion_from_dict(completion_to_dict(original))

    assert restored.text == original.text
    assert restored.tool_calls[0].input == {"a": 1}
    assert restored.input_tokens == 10

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps({"model": "m", "completions": [completion_to_dict(original)]}),
        encoding="utf-8",
    )
    provider = ReplayProvider(fixture)
    assert provider.complete([], [], "", 100).text == "hi"


def test_arabic_survives_the_openai_wire_format() -> None:
    """ensure_ascii=False keeps Arabic readable in the serialised arguments."""
    messages = [
        Message(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search",
                    "input": {"query": "مدة الإشعار"},
                }
            ],
        )
    ]
    wire = OpenAIProvider._to_wire(messages, "")
    arguments = json.loads(wire[1]["tool_calls"][0]["function"]["arguments"])

    assert arguments["query"] == "مدة الإشعار"
