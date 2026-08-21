"""Schema-validated model output, with a repair loop.

Asking a model for JSON and parsing it optimistically fails often enough to
matter. This asks for JSON, validates it against a pydantic model, and on
failure sends the validation error back and asks for a correction. The repair
count is capped, then it raises.

How often repairs are needed is itself a measurement worth publishing: it says
something real about the model and about how well the schema was described.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, ValidationError

from agent_core.errors import StructuredOutputError
from agent_core.providers.base import Message, Provider

log = structlog.get_logger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class RepairStats:
    """Tally of how often the repair loop was needed. Reported in the README."""

    attempts: int = 0
    repairs: int = 0
    failures: int = 0
    by_error: dict[str, int] = field(default_factory=dict)

    def repair_rate(self) -> float:
        return self.repairs / self.attempts if self.attempts else 0.0


STATS = RepairStats()


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or a fenced code block more often than not, so
    this strips a fence if present and otherwise takes the outermost braces.
    """
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def structured_call[T: BaseModel](
    provider: Provider,
    prompt: str,
    schema: type[T],
    max_repairs: int = 2,
    system: str = "",
    max_tokens: int = 2048,
) -> T:
    """Get schema-valid output from the model, repairing on validation failure."""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    instruction = (
        f"{system}\n\nRespond with a single JSON object matching this JSON Schema. "
        f"Output only the JSON, with no commentary.\n\n{schema_json}"
    ).strip()

    messages = [Message.user_text(prompt)]
    STATS.attempts += 1

    for attempt in range(max_repairs + 1):
        completion = provider.complete(
            messages=messages, tools=[], system=instruction, max_tokens=max_tokens
        )
        candidate = extract_json(completion.text)

        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            error_text = str(exc)
        except json.JSONDecodeError as exc:
            error_text = f"the response was not valid JSON: {exc}"

        if attempt == max_repairs:
            STATS.failures += 1
            raise StructuredOutputError(
                f"model did not produce valid {schema.__name__} after "
                f"{max_repairs} repair attempts. Last error: {error_text}"
            )

        STATS.repairs += 1
        key = type(schema).__name__
        STATS.by_error[key] = STATS.by_error.get(key, 0) + 1
        log.info("structured.repair", schema=schema.__name__, attempt=attempt + 1)

        # Feed the failed attempt and the validator's own message back. The
        # error text names the offending field and why it failed, which is far
        # more useful to the model than "try again".
        messages = [
            Message.user_text(prompt),
            Message.assistant_text(completion.text),
            Message.user_text(
                f"That did not validate against the schema:\n\n{error_text}\n\n"
                f"Return corrected JSON only."
            ),
        ]

    raise StructuredOutputError("unreachable: repair loop exited without returning")
