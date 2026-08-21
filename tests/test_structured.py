"""Structured output and the repair loop."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_core.errors import StructuredOutputError
from agent_core.providers.replay import ScriptedProvider
from agent_core.structured import extract_json, structured_call
from conftest import completion


class Verdict(BaseModel):
    score: float
    reasoning: str


def test_valid_json_needs_no_repair() -> None:
    provider = ScriptedProvider([completion(text='{"score": 0.8, "reasoning": "clear"}')])

    result = structured_call(provider, "grade this", Verdict)

    assert result.score == 0.8
    assert len(provider.calls) == 1


def test_a_fenced_code_block_is_unwrapped() -> None:
    provider = ScriptedProvider(
        [completion(text='Here you go:\n```json\n{"score": 1.0, "reasoning": "ok"}\n```')]
    )

    assert structured_call(provider, "grade", Verdict).score == 1.0


def test_prose_around_the_json_is_stripped() -> None:
    provider = ScriptedProvider(
        [completion(text='Sure. {"score": 0.5, "reasoning": "mixed"} Hope that helps.')]
    )

    assert structured_call(provider, "grade", Verdict).score == 0.5


def test_an_invalid_response_is_repaired() -> None:
    provider = ScriptedProvider(
        [
            completion(text='{"score": "not a number", "reasoning": "x"}'),
            completion(text='{"score": 0.3, "reasoning": "x"}'),
        ]
    )

    result = structured_call(provider, "grade", Verdict, max_repairs=2)

    assert result.score == 0.3
    assert len(provider.calls) == 2
    # The repair turn must carry the validator's own message, which names the
    # offending field. "Try again" would not tell the model what to change.
    repair_prompt = provider.calls[1]["messages"][-1].text()
    assert "score" in repair_prompt


def test_repairs_are_capped_then_it_raises() -> None:
    provider = ScriptedProvider([completion(text="not json at all") for _ in range(5)])

    with pytest.raises(StructuredOutputError, match="after 2 repair attempts"):
        structured_call(provider, "grade", Verdict, max_repairs=2)

    # One initial attempt plus two repairs.
    assert len(provider.calls) == 3


def test_the_schema_is_given_to_the_model() -> None:
    provider = ScriptedProvider([completion(text='{"score": 1, "reasoning": "y"}')])
    structured_call(provider, "grade", Verdict)

    system = provider.calls[0]["system"]
    assert "reasoning" in system
    assert "score" in system


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('prefix {"a": 1} suffix', '{"a": 1}'),
        ("no json here", "no json here"),
    ],
)
def test_extract_json(raw: str, expected: str) -> None:
    assert extract_json(raw) == expected
