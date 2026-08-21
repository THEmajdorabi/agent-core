"""The loop: tool execution, failures as data, timeouts, guardrails."""

from __future__ import annotations

import threading
import time

from conftest import call, completion

from agent_core.agent import Agent
from agent_core.guardrails import Guardrails
from agent_core.providers.replay import ScriptedProvider
from agent_core.registry import ToolRegistry


def build(provider: ScriptedProvider, registry: ToolRegistry, settings, **kwargs) -> Agent:
    return Agent(
        provider=provider,
        registry=registry,
        system="You are a test agent.",
        settings=settings,
        **kwargs,
    )


def test_a_turn_without_tools_finishes(settings) -> None:
    provider = ScriptedProvider([completion(text="The answer is 42.")])
    agent = build(provider, ToolRegistry(), settings)

    result = agent.run("What is the answer?")

    assert result.text == "The answer is 42."
    assert result.steps == 1
    assert result.stop_reason == "end_turn"


def test_tool_result_goes_back_as_a_user_message(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def echo(value: str) -> str:
        """Echo the value back."""
        return f"echoed:{value}"

    provider = ScriptedProvider(
        [
            completion(tool_calls=[call("echo", "c1", value="hi")]),
            completion(text="done"),
        ]
    )
    agent = build(provider, registry, settings)
    result = agent.run("echo hi")

    assert result.text == "done"
    assert result.tool_calls == 1
    # The second model call must have seen a user message carrying the result.
    second_call_messages = provider.calls[1]["messages"]
    tool_result_blocks = [
        block
        for message in second_call_messages
        for block in message.content
        if block.get("type") == "tool_result"
    ]
    assert len(tool_result_blocks) == 1
    assert tool_result_blocks[0]["tool_use_id"] == "c1"
    assert "echoed:hi" in tool_result_blocks[0]["content"]


def test_every_tool_use_is_answered_even_when_one_fails(settings) -> None:
    """An unanswered tool_use block is an API error. All of them get a result."""
    registry = ToolRegistry()

    @registry.register
    def good() -> str:
        """Succeeds."""
        return "ok"

    @registry.register
    def bad() -> str:
        """Always raises."""
        raise RuntimeError("this tool is broken")

    provider = ScriptedProvider(
        [
            completion(tool_calls=[call("good", "c1"), call("bad", "c2")]),
            completion(text="handled"),
        ]
    )
    agent = build(provider, registry, settings)
    result = agent.run("run both")

    assert result.tool_calls == 2
    assert result.tool_failures == 1

    blocks = [
        block
        for message in provider.calls[1]["messages"]
        for block in message.content
        if block.get("type") == "tool_result"
    ]
    answered = {b["tool_use_id"] for b in blocks}
    assert answered == {"c1", "c2"}
    errored = [b for b in blocks if b["is_error"]]
    assert len(errored) == 1
    assert "this tool is broken" in errored[0]["content"]


def test_tools_in_one_turn_run_concurrently(settings) -> None:
    """Three 0.3s tools in one turn should finish in well under 0.9s."""
    registry = ToolRegistry()
    active = []
    peak = 0
    lock = threading.Lock()

    @registry.register(timeout_s=5)
    def slow(tag: str) -> str:
        """Sleep briefly, recording how many copies ran at once."""
        nonlocal peak
        with lock:
            active.append(tag)
            peak = max(peak, len(active))
        time.sleep(0.3)
        with lock:
            active.remove(tag)
        return tag

    provider = ScriptedProvider(
        [
            completion(
                tool_calls=[
                    call("slow", "c1", tag="a"),
                    call("slow", "c2", tag="b"),
                    call("slow", "c3", tag="c"),
                ]
            ),
            completion(text="done"),
        ]
    )
    agent = build(provider, registry, settings)

    started = time.monotonic()
    agent.run("run three")
    elapsed = time.monotonic() - started

    assert peak >= 2, "tools did not overlap; execution was serial"
    assert elapsed < 0.8, f"three concurrent 0.3s tools took {elapsed:.2f}s"


def test_results_keep_call_order(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def wait_then(tag: str, seconds: float) -> str:
        """Sleep, then return the tag."""
        time.sleep(seconds)
        return tag

    # The first call is the slowest, so an order-losing implementation returns
    # them the other way round.
    provider = ScriptedProvider(
        [
            completion(
                tool_calls=[
                    call("wait_then", "c1", tag="first", seconds=0.25),
                    call("wait_then", "c2", tag="second", seconds=0.01),
                ]
            ),
            completion(text="done"),
        ]
    )
    agent = build(provider, registry, settings)
    agent.run("ordered")

    blocks = [
        block
        for message in provider.calls[1]["messages"]
        for block in message.content
        if block.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]


def test_a_tool_that_exceeds_its_timeout_returns_an_error_result(settings) -> None:
    registry = ToolRegistry()

    @registry.register(timeout_s=0.1)
    def hangs() -> str:
        """Sleeps far longer than its timeout."""
        time.sleep(3)
        return "never"

    provider = ScriptedProvider(
        [completion(tool_calls=[call("hangs", "c1")]), completion(text="recovered")]
    )
    agent = build(provider, registry, settings)
    result = agent.run("hang")

    assert result.tool_failures == 1
    blocks = [
        b
        for m in provider.calls[1]["messages"]
        for b in m.content
        if b.get("type") == "tool_result"
    ]
    assert blocks[0]["is_error"]
    assert "timed out" in blocks[0]["content"]


def test_bad_arguments_come_back_as_an_error_the_model_can_read(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def needs_int(n: int) -> int:
        """Double an integer."""
        return n * 2

    provider = ScriptedProvider(
        [
            completion(tool_calls=[call("needs_int", "c1", wrong_name="x")]),
            completion(text="corrected"),
        ]
    )
    agent = build(provider, registry, settings)
    result = agent.run("call it wrong")

    assert result.tool_failures == 1
    assert result.text == "corrected"


def test_unknown_tool_does_not_crash_the_run(settings) -> None:
    provider = ScriptedProvider(
        [completion(tool_calls=[call("nonexistent", "c1")]), completion(text="ok")]
    )
    agent = build(provider, ToolRegistry(), settings)
    result = agent.run("call a missing tool")

    assert result.tool_failures == 1
    assert result.text == "ok"


def test_tool_allowlist_blocks_execution(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def permitted() -> str:
        """Allowed."""
        return "yes"

    @registry.register
    def forbidden() -> str:
        """Not allowed."""
        return "should not run"

    provider = ScriptedProvider(
        [completion(tool_calls=[call("forbidden", "c1")]), completion(text="blocked")]
    )
    agent = build(
        provider,
        registry,
        settings,
        guardrails=Guardrails(tool_allowlist={"permitted"}),
    )
    result = agent.run("try the forbidden tool")

    assert result.tool_failures == 1
    # The model was never even shown the forbidden tool.
    offered = {t["name"] for t in provider.calls[0]["tools"]}
    assert offered == {"permitted"}


def test_step_limit_stops_the_run_and_returns_a_partial(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def spin() -> str:
        """Always succeeds, so the model keeps going."""
        return "again"

    provider = ScriptedProvider([completion(tool_calls=[call("spin", f"c{i}")]) for i in range(20)])
    agent = build(provider, registry, settings, guardrails=Guardrails(max_steps=3))
    result = agent.run("loop forever")

    assert result.stop_reason == "guardrail"
    assert "step limit" in result.text
    assert result.steps == 3


def test_cost_is_computed_from_the_price_table(settings) -> None:
    # settings prices "scripted" at $1.00/1M input, $2.00/1M output.
    provider = ScriptedProvider([completion(text="hi", input_tokens=1_000_000, output_tokens=500_000)])
    agent = build(provider, ToolRegistry(), settings)
    result = agent.run("cost me")

    assert result.cost_usd == 1.0 + 1.0
    assert result.cost_is_estimated is False


def test_cost_limit_stops_the_run(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def spin() -> str:
        """Keeps the loop going."""
        return "again"

    provider = ScriptedProvider(
        [
            completion(tool_calls=[call("spin", f"c{i}")], input_tokens=1_000_000, output_tokens=0)
            for i in range(10)
        ]
    )
    agent = build(provider, registry, settings, guardrails=Guardrails(max_cost_usd=2.5))
    result = agent.run("spend")

    assert result.stop_reason == "guardrail"
    assert "cost limit" in result.text


def test_consecutive_tool_failures_abort(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def always_fails() -> str:
        """Always raises."""
        raise RuntimeError("nope")

    provider = ScriptedProvider(
        [completion(tool_calls=[call("always_fails", f"c{i}")]) for i in range(10)]
    )
    agent = build(
        provider, registry, settings, guardrails=Guardrails(max_consecutive_tool_failures=2)
    )
    result = agent.run("fail repeatedly")

    assert result.stop_reason == "guardrail"
    assert "consecutive tool failures" in result.text


def test_trace_renders_the_run(settings) -> None:
    registry = ToolRegistry()

    @registry.register
    def peek() -> str:
        """Look at something."""
        return "looked"

    provider = ScriptedProvider(
        [completion(tool_calls=[call("peek", "c1")]), completion(text="Fixed it")]
    )
    agent = build(provider, registry, settings)
    result = agent.run("fix the failing test")

    assert "goal='fix the failing test'" in result.trace
    assert "tool: peek" in result.trace
    assert "model_call" in result.trace
    assert "final" in result.trace
