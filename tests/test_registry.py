"""Registry: schema generation from type hints."""

from __future__ import annotations

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from agent_core.errors import ToolNotFound
from agent_core.registry import ToolRegistry


def test_schema_is_generated_from_signature() -> None:
    registry = ToolRegistry()

    @registry.register(timeout_s=10)
    def search_corpus(query: str, top_k: int = 5) -> list[dict]:
        """Search the document corpus and return matching passages."""
        return []

    schema = registry.schemas()[0]
    assert schema["name"] == "search_corpus"
    assert schema["description"] == "Search the document corpus and return matching passages."
    properties = schema["input_schema"]["properties"]
    assert properties["query"]["type"] == "string"
    assert properties["top_k"]["type"] == "integer"
    assert properties["top_k"]["default"] == 5
    # A parameter with a default must not be required.
    assert schema["input_schema"]["required"] == ["query"]


def test_annotated_supplies_a_parameter_description() -> None:
    registry = ToolRegistry()

    @registry.register
    def fetch(url: Annotated[str, "Absolute http(s) URL to fetch."]) -> str:
        """Fetch a URL."""
        return ""

    properties = registry.schemas()[0]["input_schema"]["properties"]
    assert properties["url"]["description"] == "Absolute http(s) URL to fetch."


def test_bare_decorator_and_called_decorator_both_work() -> None:
    registry = ToolRegistry()

    @registry.register
    def bare() -> str:
        """A tool registered with a bare decorator."""
        return ""

    @registry.register(timeout_s=1)
    def called() -> str:
        """A tool registered with a called decorator."""
        return ""

    assert registry.names() == ["bare", "called"]
    assert registry.get("called").timeout_s == 1


def test_missing_docstring_is_refused_at_registration() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="no docstring"):

        @registry.register
        def undocumented(x: int) -> int:
            return x


def test_duplicate_registration_is_refused() -> None:
    registry = ToolRegistry()

    @registry.register
    def dup() -> str:
        """First."""
        return ""

    with pytest.raises(ValueError, match="already registered"):

        @registry.register(name="dup")
        def other() -> str:
            """Second."""
            return ""


def test_varargs_are_refused() -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="JSON Schema"):

        @registry.register
        def variadic(*args: int) -> int:
            """Cannot be described to a model."""
            return 0


class Filter(BaseModel):
    """A nested model used by the complex-type test.

    Defined at module level on purpose: see
    test_function_local_types_raise_an_actionable_error.
    """

    field_name: str
    value: str


def test_complex_types_survive() -> None:
    """Literals, unions and nested models are pydantic's problem, not ours."""
    registry = ToolRegistry()

    @registry.register
    def query(
        mode: Literal["dense", "hybrid"],
        limit: int | None = None,
        filters: list[Filter] | None = None,
    ) -> str:
        """Query with a constrained mode."""
        return ""

    schema = registry.schemas()[0]["input_schema"]
    assert schema["properties"]["mode"]["enum"] == ["dense", "hybrid"]
    # Nested models land in $defs, which is valid JSON Schema and passes through.
    assert "$defs" in schema


def test_allowlist_hides_tools() -> None:
    registry = ToolRegistry()

    @registry.register
    def visible() -> str:
        """Shown."""
        return ""

    @registry.register
    def hidden() -> str:
        """Not shown."""
        return ""

    names = [s["name"] for s in registry.schemas(allowlist={"visible"})]
    assert names == ["visible"]


def test_unknown_tool_raises_with_the_known_names() -> None:
    registry = ToolRegistry()

    @registry.register
    def known() -> str:
        """Known."""
        return ""

    with pytest.raises(ToolNotFound, match="known"):
        registry.get("nope")


def test_registered_function_stays_callable() -> None:
    """The decorator returns the function unchanged, so it stays unit-testable."""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add(2, 3) == 5


def test_function_local_types_raise_an_actionable_error() -> None:
    """A type defined inside a function cannot be resolved from annotations.

    This is a Python limitation, not a bug here: under postponed annotation
    evaluation the annotation is the string "Local", and `get_type_hints`
    resolves it against module globals where no such name exists. The registry
    turns the bare NameError into a message that says what to do about it.
    """
    registry = ToolRegistry()

    class Local(BaseModel):
        x: int

    with pytest.raises(NameError, match="must be importable at module level"):

        @registry.register
        def uses_local(item: Local) -> str:
            """Takes a function-local type."""
            return ""
