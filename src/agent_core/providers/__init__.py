"""Model providers, normalised behind one protocol."""

from agent_core.providers.base import (
    Completion,
    Delta,
    Message,
    Provider,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Completion",
    "Delta",
    "Message",
    "Provider",
    "ToolCall",
    "ToolResult",
    "build_provider",
]


def build_provider(provider: str, model_id: str, reasoning_effort: str | None = None) -> Provider:
    """Construct a provider by name.

    Imports are deferred so that using one provider does not require the
    other's SDK, or its API key, to be present.
    """
    if provider == "anthropic":
        from agent_core.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model_id)
    if provider == "openai":
        from agent_core.providers.openai import OpenAIProvider

        return OpenAIProvider(model_id, reasoning_effort=reasoning_effort)
    raise ValueError(f"unknown provider {provider!r}. Expected 'anthropic' or 'openai'.")
