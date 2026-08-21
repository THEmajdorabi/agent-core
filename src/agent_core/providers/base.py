"""The Provider protocol.

One thin interface over Anthropic and OpenAI so the loop, and the eval harness
in repo 03, can swap models without knowing whose SDK is underneath. The
normalised shapes below are the contract; each adapter's job is to translate
its provider's wire format into exactly these.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]


@dataclass
class ToolCall:
    """A single tool the model asked to run."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of running a ToolCall, on its way back to the model."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn of the transcript.

    `content` holds provider-neutral blocks. A block is one of:
      {"type": "text", "text": ...}
      {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
      {"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": bool}

    Keeping tool_use and tool_result as blocks inside messages, rather than as
    a separate list, is what lets the compactor reason about turn boundaries.
    """

    role: Role
    content: list[dict[str, Any]]

    @classmethod
    def user_text(cls, text: str) -> Message:
        return cls(role="user", content=[{"type": "text", "text": text}])

    @classmethod
    def assistant_text(cls, text: str) -> Message:
        return cls(role="assistant", content=[{"type": "text", "text": text}])

    @classmethod
    def tool_results(cls, results: list[ToolResult]) -> Message:
        """Tool results always go back as a single user message.

        Splitting them across several messages is accepted by the API but
        teaches the model to stop making parallel calls.
        """
        return cls(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_use_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        )

    def text(self) -> str:
        """Concatenate the text blocks, ignoring tool traffic."""
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text")

    def tool_uses(self) -> list[ToolCall]:
        return [
            ToolCall(id=b["id"], name=b["name"], input=b.get("input", {}))
            for b in self.content
            if b.get("type") == "tool_use"
        ]


@dataclass
class Completion:
    """A normalised model response."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any = None
    # The provider's own content blocks, kept verbatim so the assistant turn
    # can be replayed losslessly. Claude returns `thinking` blocks that must be
    # echoed back unchanged when the conversation continues on the same model;
    # rebuilding the turn from text + tool_calls alone would silently drop them.
    raw_blocks: list[dict[str, Any]] = field(default_factory=list)

    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)

    def to_message(self) -> Message:
        """Rebuild the assistant turn to append to the transcript.

        Both the text and every tool_use block must be preserved: the next
        message has to answer each tool_use by id, and dropping one is an API
        error on the following call.
        """
        if self.raw_blocks:
            return Message(role="assistant", content=list(self.raw_blocks))

        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for call in self.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return Message(role="assistant", content=blocks)


@dataclass
class Delta:
    """One streamed fragment."""

    type: Literal["text", "tool_use_start", "done"]
    text: str = ""
    tool_name: str = ""


@runtime_checkable
class Provider(Protocol):
    """What the agent loop needs from a model."""

    model_id: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Iterator[Delta]: ...

    def count_tokens(self, messages: list[Message], system: str) -> int: ...
