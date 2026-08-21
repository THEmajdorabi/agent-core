"""Tool registration.

A tool is a plain Python function. The decorator reads its signature and turns
it into the JSON Schema the model sees, so nobody hand-writes a schema and
nobody's schema drifts from their function.

The schema is produced by building a pydantic model from the signature and
calling `.model_json_schema()`. Hand-rolling a type-to-schema mapper means
reimplementing pydantic's handling of unions, enums, literals, nested models
and defaults, badly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import Field, create_model

from agent_core.errors import ToolNotFound


@dataclass
class Tool:
    """A registered tool and the metadata the loop needs to run it safely."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    timeout_s: float = 30.0
    dangerous: bool = False
    idempotent: bool = False

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def _split_annotated(annotation: Any) -> tuple[Any, str | None]:
    """Pull a description out of `Annotated[T, "description"]`.

    Returns the underlying type and the first string metadata entry, if any.
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base = args[0]
        for extra in args[1:]:
            if isinstance(extra, str):
                return base, extra
        return base, None
    return annotation, None


def build_input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema for `fn`'s parameters.

    Parameters with defaults become optional; parameters without become
    required. `Annotated[T, "..."]` supplies a per-parameter description.
    """
    signature = inspect.signature(fn)
    # include_extras keeps Annotated metadata alive, which is where the
    # per-argument descriptions come from.
    hints = get_type_hints(fn, include_extras=True)

    fields: dict[str, Any] = {}
    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # *args / **kwargs cannot be described to the model; a tool that
            # needs them should declare explicit parameters instead.
            raise TypeError(
                f"tool {fn.__name__!r} uses *args/**kwargs, which cannot be "
                f"expressed as JSON Schema. Declare explicit parameters."
            )

        annotation, description = _split_annotated(hints.get(param_name, str))
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, Field(default, description=description))

    model = create_model(f"{fn.__name__}_Input", **fields)
    schema = model.model_json_schema()

    # Anthropic expects a plain object schema. pydantic emits $defs for nested
    # models, which is valid JSON Schema and is passed through untouched.
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class ToolRegistry:
    """Holds the tools an agent is allowed to call."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        timeout_s: float = 30.0,
        dangerous: bool = False,
        idempotent: bool = False,
    ) -> Any:
        """Register a function as a tool.

        Usable bare (`@registry.register`) or with arguments
        (`@registry.register(timeout_s=10)`). The function is returned
        unchanged, so it stays directly callable and directly testable.
        """

        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            docstring = inspect.getdoc(func)
            if not docstring:
                # A tool with no description is a tool the model will misuse.
                raise ValueError(
                    f"tool {func.__name__!r} has no docstring. The docstring is "
                    f"the description the model uses to decide when to call it."
                )
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=docstring,
                input_schema=build_input_schema(func),
                fn=func,
                timeout_s=timeout_s,
                dangerous=dangerous,
                idempotent=idempotent,
            )
            return func

        if fn is not None:
            return decorate(fn)
        return decorate

    def get(self, name: str) -> Tool:
        """Look up a registered tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise ToolNotFound(f"no tool named {name!r}. Registered: {known}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, allowlist: set[str] | None = None) -> list[dict[str, Any]]:
        """The tool list to send to the provider.

        `allowlist` restricts which tools the model is even told about, which
        is a stronger control than refusing the call afterwards: a tool the
        model cannot see is a tool it cannot decide to use.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in (self._tools[n] for n in sorted(self._tools))
            if allowlist is None or tool.name in allowlist
        ]
