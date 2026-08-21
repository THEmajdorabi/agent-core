"""OpenAI adapter.

Normalises OpenAI's shape into the same `Completion` the Anthropic adapter
produces. The three differences that matter:

  * Tools are `{"type": "function", "function": {name, description, parameters}}`.
  * Tool calls arrive as `message.tool_calls`, with arguments as a JSON
    *string* that must be parsed.
  * Results go back as separate messages with `role="tool"` and a
    `tool_call_id` -- not as blocks inside a user message.

That last one is why `_to_wire` fans a single tool-result message out into
several messages.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_core.config import require_env
from agent_core.errors import ProviderError
from agent_core.providers.base import Completion, Delta, Message, ToolCall

log = structlog.get_logger(__name__)


class OpenAIProvider:
    """Provider implementation backed by the official `openai` SDK."""

    def __init__(self, model_id: str, timeout_s: float = 120.0) -> None:
        import openai

        self.model_id = model_id
        self._sdk = openai
        self._client = openai.OpenAI(
            api_key=require_env("OPENAI_API_KEY"), timeout=timeout_s, max_retries=0
        )

    # ---- wire format -----------------------------------------------------

    @staticmethod
    def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Anthropic-shaped tool dicts become OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _to_wire(messages: list[Message], system: str) -> list[dict[str, Any]]:
        """Flatten our block-based transcript into OpenAI's message list."""
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for message in messages:
            tool_results = [b for b in message.content if b.get("type") == "tool_result"]
            if tool_results:
                # One role="tool" message per result, each keyed by call id.
                for block in tool_results:
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
                continue

            text = message.text()
            tool_uses = [b for b in message.content if b.get("type") == "tool_use"]

            if message.role == "assistant" and tool_uses:
                wire.append(
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": b["id"],
                                "type": "function",
                                "function": {
                                    "name": b["name"],
                                    "arguments": json.dumps(b.get("input", {})),
                                },
                            }
                            for b in tool_uses
                        ],
                    }
                )
            else:
                wire.append({"role": message.role, "content": text})

        return wire

    def _parse(self, response: Any) -> Completion:
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []

        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                # Malformed arguments are the model's mistake, not a crash. An
                # empty input lets the tool raise a clear validation message
                # that the model can read and correct on the next turn.
                log.warning("openai.bad_tool_arguments", tool=call.function.name, error=str(exc))
                arguments = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, input=arguments))

        stop_reason = "tool_use" if tool_calls else (choice.finish_reason or "end_turn")

        return Completion(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=self.model_id,
            raw=response,
        )

    # ---- Provider protocol ----------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion:
        return self._complete_with_retry(messages, tools, system, max_tokens)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    def _complete_with_retry(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion:
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                max_completion_tokens=max_tokens,
                messages=self._to_wire(messages, system),
                tools=self.to_openai_tools(tools) or None,
            )
        except (self._sdk.RateLimitError, self._sdk.APIConnectionError) as exc:
            log.warning("openai.retryable", error=str(exc))
            raise ProviderError(f"retryable OpenAI failure: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderError(f"OpenAI {exc.status_code}: {exc}") from exc
            raise ProviderError(f"OpenAI rejected the request ({exc.status_code}): {exc}") from exc

        return self._parse(response)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Iterator[Delta]:
        stream = self._client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=max_tokens,
            messages=self._to_wire(messages, system),
            tools=self.to_openai_tools(tools) or None,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield Delta(type="text", text=delta.content)
            for call in delta.tool_calls or []:
                if call.function and call.function.name:
                    yield Delta(type="tool_use_start", tool_name=call.function.name)
        yield Delta(type="done")

    def count_tokens(self, messages: list[Message], system: str) -> int:
        """Count locally with tiktoken.

        OpenAI exposes no counting endpoint, so this is an estimate. It is
        close for English and drifts for Arabic, which is why `context.py`
        calibrates against real usage counts rather than trusting it blindly.
        """
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(self.model_id)
        except KeyError:
            # Unknown model: o200k_base is the encoding current models use.
            encoding = tiktoken.get_encoding("o200k_base")

        total = len(encoding.encode(system))
        for wire_message in self._to_wire(messages, ""):
            total += len(encoding.encode(json.dumps(wire_message, ensure_ascii=False)))
        return total
