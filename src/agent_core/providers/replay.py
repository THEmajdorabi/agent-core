"""Record-and-replay provider.

The test suite must exercise the real loop without spending money. This
provider satisfies the `Provider` protocol two ways:

  * `ReplayProvider` serves canned `Completion` objects from a JSON fixture,
    keyed by the position in the conversation. It is what unit tests use.
  * `RecordingProvider` wraps a live provider and writes every completion to a
    fixture file, so a fixture is captured once from a real run rather than
    hand-written and slowly drifting away from what the API actually returns.

`ScriptedProvider` is the smallest of the three: a list of completions handed
in directly, for tests that do not need a file at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_core.providers.base import Completion, Delta, Message, Provider, ToolCall


def completion_to_dict(completion: Completion) -> dict[str, Any]:
    """Serialise a Completion for a fixture file, dropping the raw SDK object."""
    return {
        "text": completion.text,
        "tool_calls": [
            {"id": c.id, "name": c.name, "input": c.input} for c in completion.tool_calls
        ],
        "stop_reason": completion.stop_reason,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "model": completion.model,
        "raw_blocks": completion.raw_blocks,
    }


def completion_from_dict(data: dict[str, Any]) -> Completion:
    return Completion(
        text=data.get("text", ""),
        tool_calls=[
            ToolCall(id=c["id"], name=c["name"], input=c.get("input", {}))
            for c in data.get("tool_calls", [])
        ],
        stop_reason=data.get("stop_reason", "end_turn"),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        model=data.get("model", "replay"),
        raw_blocks=data.get("raw_blocks", []),
    )


class ScriptedProvider:
    """Returns a fixed list of completions, one per call, in order.

    Raises when the script runs out rather than looping or returning a default:
    a test that makes more model calls than it scripted has changed behaviour,
    and that should fail loudly.
    """

    def __init__(self, completions: list[Completion], model_id: str = "scripted") -> None:
        self.model_id = model_id
        self._completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion:
        self.calls.append({"messages": list(messages), "tools": tools, "system": system})
        if not self._completions:
            raise AssertionError(
                f"ScriptedProvider ran out of completions after {len(self.calls)} calls. "
                f"The loop made more model calls than the test scripted."
            )
        return self._completions.pop(0)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Iterator[Delta]:
        completion = self.complete(messages, tools, system, max_tokens)
        if completion.text:
            yield Delta(type="text", text=completion.text)
        yield Delta(type="done")

    def count_tokens(self, messages: list[Message], system: str) -> int:
        """Cheap deterministic estimate, so compaction tests are reproducible."""
        blob = system + json.dumps([m.content for m in messages], ensure_ascii=False)
        return len(blob) // 4


class ReplayProvider(ScriptedProvider):
    """A ScriptedProvider loaded from a JSON fixture on disk."""

    def __init__(self, fixture_path: str | Path) -> None:
        path = Path(fixture_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        super().__init__(
            [completion_from_dict(d) for d in payload["completions"]],
            model_id=payload.get("model", "replay"),
        )
        self.fixture_path = path


class RecordingProvider:
    """Wraps a live provider and writes each completion to a fixture file.

    Used once, by hand, to capture a fixture. Never used in the default test
    run -- it makes real, billable calls.
    """

    def __init__(self, inner: Provider, fixture_path: str | Path) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.fixture_path = Path(fixture_path)
        self._recorded: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion:
        completion = self._inner.complete(messages, tools, system, max_tokens)
        self._recorded.append(completion_to_dict(completion))
        self._flush()
        return completion

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Iterator[Delta]:
        return self._inner.stream(messages, tools, system, max_tokens)

    def count_tokens(self, messages: list[Message], system: str) -> int:
        return self._inner.count_tokens(messages, system)

    def _flush(self) -> None:
        """Write after every call, so a crash mid-capture still leaves a usable
        fixture of the turns that did complete."""
        self.fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fixture_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"model": self.model_id, "completions": self._recorded},
                handle,
                ensure_ascii=False,
                indent=2,
            )
