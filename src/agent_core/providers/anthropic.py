"""Anthropic adapter.

The details that matter, and that cost a debugging session each if missed:

  * Tools are `{"name", "description", "input_schema"}`.
  * A turn that wants tools has `stop_reason == "tool_use"` and one or more
    `tool_use` content blocks carrying `id`, `name`, `input`.
  * Results go back as a **user** message of `tool_result` blocks, each with a
    `tool_use_id` matching the call. Every `tool_use` in a turn must be
    answered in the next message or the API rejects the request.
  * `thinking` blocks come back on current models and must be echoed
    unchanged; we keep raw blocks and replay them.
  * Sampling parameters (`temperature`, `top_p`, `top_k`) were removed on
    Claude Sonnet 5 / Opus 5 and are rejected with a 400. We never send them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_core.config import require_env
from agent_core.errors import ProviderError
from agent_core.providers.base import Completion, Delta, Message, ToolCall

log = structlog.get_logger(__name__)


class AnthropicProvider:
    """Provider implementation backed by the official `anthropic` SDK."""

    def __init__(self, model_id: str, timeout_s: float = 120.0) -> None:
        import anthropic

        self.model_id = model_id
        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=require_env("ANTHROPIC_API_KEY"), timeout=timeout_s, max_retries=0
        )

    # ---- wire format -----------------------------------------------------

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        """Our neutral Message list becomes Anthropic's `messages` array.

        The block shapes already match, so this is a pass-through. It exists as
        a named seam: when a provider's format drifts, it drifts here.
        """
        return [{"role": m.role, "content": m.content} for m in messages]

    def _parse(self, response: Any) -> Completion:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_blocks: list[dict[str, Any]] = []

        for block in response.content:
            # `model_dump` keeps thinking/redacted_thinking blocks intact for replay.
            raw_blocks.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return Completion(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model_id,
            raw=response,
            raw_blocks=raw_blocks,
        )

    # ---- Provider protocol ----------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Completion:
        """One model call. Retries transient failures with exponential backoff."""
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
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                system=system,
                messages=self._to_wire(messages),
                tools=tools or [],
            )
        except (self._sdk.RateLimitError, self._sdk.APIConnectionError) as exc:
            # Retryable: rate limiting and transport failures.
            log.warning("anthropic.retryable", error=str(exc))
            raise ProviderError(f"retryable Anthropic failure: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning("anthropic.server_error", status=exc.status_code)
                raise ProviderError(f"Anthropic {exc.status_code}: {exc}") from exc
            # 4xx other than 429 is our bug: a malformed request, a bad schema,
            # or an unanswered tool_use block. Retrying cannot fix it.
            raise ProviderError(f"Anthropic rejected the request ({exc.status_code}): {exc}") from exc

        return self._parse(response)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> Iterator[Delta]:
        """Stream text deltas. Used by the CLI so a long turn shows progress."""
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=self._to_wire(messages),
            tools=tools or [],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield Delta(type="text", text=event.delta.text)
                elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                    yield Delta(type="tool_use_start", tool_name=event.content_block.name)
            yield Delta(type="done")

    def count_tokens(self, messages: list[Message], system: str) -> int:
        """Exact token count from Anthropic's counting endpoint.

        Worth the network call: the character heuristic is wrong for Arabic by
        a wide enough margin to trigger compaction at the wrong time.
        """
        try:
            result = self._client.messages.count_tokens(
                model=self.model_id,
                system=system,
                messages=self._to_wire(messages),
            )
            return int(result.input_tokens)
        except self._sdk.APIError as exc:
            log.warning("anthropic.count_tokens_failed", error=str(exc))
            raise ProviderError(f"token counting failed: {exc}") from exc
