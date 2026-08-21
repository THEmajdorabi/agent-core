"""The agent loop.

Every agent framework hides this file. It is about two hundred lines:

    1. call the model with the transcript and the tool schemas
    2. if the model did not ask for tools, it is done
    3. run the tools it asked for, concurrently
    4. append the results as one user message
    5. compact the transcript if it has grown too large
    6. check the budgets, then go back to 1

The rest of the length here is the four things that make it survivable:
tool failures become data instead of crashes, every tool has a timeout, every
step is journalled before the next call, and a resumed run refuses to re-run a
side effect whose outcome it cannot establish.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import structlog

from agent_core.config import Settings, load_settings
from agent_core.context import Compactor
from agent_core.errors import GuardrailBreach, ToolNotFound, describe, truncate
from agent_core.guardrails import Budget, Guardrails
from agent_core.journal import Journal
from agent_core.providers.base import Completion, Message, Provider, ToolCall, ToolResult
from agent_core.registry import ToolRegistry
from agent_core.tracing import Tracer

log = structlog.get_logger(__name__)

MAX_TOOL_RESULT_BYTES = 10_000


@dataclass
class RunResult:
    """Everything a caller needs to judge what a run did and what it cost."""

    run_id: str
    goal: str
    text: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_s: float
    stop_reason: str
    trace: str = ""
    compactions: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    cost_is_estimated: bool = False
    messages: list[Message] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.stop_reason} in {self.steps} steps, "
            f"{self.tool_calls} tool calls ({self.tool_failures} failed), "
            f"${self.cost_usd:.4f}, {self.wall_s:.1f}s"
        )


class Agent:
    """A model, a set of tools, and the loop that runs between them."""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        system: str,
        guardrails: Guardrails | None = None,
        journal: Journal | None = None,
        compactor: Compactor | None = None,
        settings: Settings | None = None,
        max_tokens: int = 4096,
        max_tool_workers: int = 8,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.system = system
        self.guardrails = guardrails or Guardrails()
        self.journal = journal
        self.compactor = compactor
        self.settings = settings or load_settings()
        self.max_tokens = max_tokens
        self.max_tool_workers = max_tool_workers

    # ---- cost ------------------------------------------------------------

    def _cost(self, completion: Completion) -> float:
        """USD for one completion, from the price table in config.yaml."""
        price = self.settings.price_for(completion.model or self.provider.model_id)
        return (
            completion.input_tokens * price.input + completion.output_tokens * price.output
        ) / 1_000_000

    # ---- tool execution --------------------------------------------------

    def _run_one_tool(self, call: ToolCall, tracer: Tracer) -> ToolResult:
        """Run a single tool call, converting every failure into a result.

        The model can recover from a tool that failed if it is told what went
        wrong. It cannot recover from an exception that ends the process. So
        the only thing that escapes this method is a bug in this method.
        """
        span = tracer.span(call.name, "tool", tool_use_id=call.id)

        if self.guardrails.tool_allowlist is not None and call.name not in self.guardrails.tool_allowlist:
            span.end(error="not allowlisted")
            return ToolResult(
                tool_use_id=call.id,
                content=f"Tool {call.name!r} is not permitted in this run.",
                is_error=True,
            )

        try:
            tool = self.registry.get(call.name)
        except ToolNotFound as exc:
            span.end(error="unknown tool")
            return ToolResult(tool_use_id=call.id, content=str(exc), is_error=True)

        try:
            # Each tool gets its own single-worker pool purely to enforce a
            # timeout: ThreadPoolExecutor.result(timeout=) is the only portable
            # way to bound a synchronous call. The worker itself is not
            # cancelled -- Python cannot interrupt arbitrary code -- so a tool
            # that hangs leaks a thread. Tools doing real I/O set their own
            # socket timeouts; run_python enforces its limit in Docker.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.fn, **call.input)
                value = future.result(timeout=tool.timeout_s)
        except TimeoutError:
            span.end(error=f"timeout after {tool.timeout_s}s")
            return ToolResult(
                tool_use_id=call.id,
                content=f"Tool {call.name!r} timed out after {tool.timeout_s}s.",
                is_error=True,
            )
        except (TypeError, ValueError) as exc:
            # Almost always the model supplying arguments that do not match the
            # signature. Handing the message back lets it correct itself.
            span.end(error=type(exc).__name__)
            return ToolResult(
                tool_use_id=call.id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad, and the one place in this package that is:
            # a tool is arbitrary user code, and any exception it raises is
            # information for the model rather than a reason to end the run.
            log.warning("tool.failed", tool=call.name, **describe(exc))
            span.end(error=type(exc).__name__)
            return ToolResult(
                tool_use_id=call.id, content=f"{type(exc).__name__}: {exc}", is_error=True
            )

        span.end()
        return ToolResult(
            tool_use_id=call.id, content=truncate(str(value), MAX_TOOL_RESULT_BYTES)
        )

    def _run_tools(self, calls: list[ToolCall], tracer: Tracer, run_id: str) -> list[ToolResult]:
        """Run every requested tool concurrently, preserving call order.

        Order matters because each result carries its `tool_use_id`, but
        keeping the list aligned with the calls makes the journal and the trace
        readable, and costs nothing.
        """
        if self.journal is not None:
            for call in calls:
                # Journalled *before* execution: this is the record that lets
                # resume tell "never started" from "started, outcome unknown".
                tool = self.registry.get(call.name) if self.registry.has(call.name) else None
                self.journal.append(
                    run_id,
                    "tool_call_started",
                    {"tool_use_id": call.id, "name": call.name, "input": call.input},
                    idempotent=bool(tool and tool.idempotent),
                )

        if len(calls) == 1:
            results = [self._run_one_tool(calls[0], tracer)]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_tool_workers, len(calls))) as pool:
                results = list(pool.map(lambda c: self._run_one_tool(c, tracer), calls))

        if self.journal is not None:
            for result in results:
                self.journal.append(
                    run_id,
                    "tool_result",
                    {
                        "tool_use_id": result.tool_use_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    },
                )
        return results

    # ---- the loop --------------------------------------------------------

    def run(self, goal: str, run_id: str | None = None) -> RunResult:
        """Run until the model stops asking for tools, or a budget is spent."""
        run_id = run_id or uuid.uuid4().hex
        messages = [Message.user_text(goal)]
        if self.journal is not None:
            self.journal.start_run(run_id, goal)
        return self._loop(run_id, goal, messages)

    def resume(self, run_id: str) -> RunResult:
        """Continue a run that was interrupted.

        Rebuilds the transcript from the journal, then applies the idempotency
        rule to any tool call that was started but never finished.
        """
        if self.journal is None:
            raise ValueError("resume requires a journal")

        row = self.journal.run_row(run_id)
        if row is None:
            raise ValueError(f"no journalled run with id {run_id!r}")

        goal = row["goal"]
        messages = self._replay(run_id, goal)
        log.info("agent.resume", run_id=run_id, messages=len(messages))
        return self._loop(run_id, goal, messages)

    def _replay(self, run_id: str, goal: str) -> list[Message]:
        """Rebuild the message list from journalled steps.

        Orphaned tool calls -- started, never finished -- are resolved here:
        idempotent tools are re-executed, everything else is replayed as a
        failed result saying the outcome is unknown, and the model decides how
        to proceed.
        """
        assert self.journal is not None
        messages: list[Message] = [Message.user_text(goal)]
        orphans = {s.payload["tool_use_id"]: s for s in self.journal.orphaned_tool_calls(run_id)}
        pending_results: list[ToolResult] = []

        for step in self.journal.steps(run_id):
            if step.kind == "model_call":
                if pending_results:
                    messages.append(Message.tool_results(pending_results))
                    pending_results = []
                messages.append(Message(role="assistant", content=step.payload["content"]))

            elif step.kind == "tool_result":
                pending_results.append(
                    ToolResult(
                        tool_use_id=step.payload["tool_use_id"],
                        content=step.payload["content"],
                        is_error=step.payload.get("is_error", False),
                    )
                )

            elif step.kind == "compaction":
                messages = [Message.user_text(goal), Message.user_text(step.payload["summary"])]
                pending_results = []

        for call_id, step in orphans.items():
            name = step.payload["name"]
            if step.idempotent:
                log.info("resume.rerun_idempotent", tool=name, tool_use_id=call_id)
                tracer = Tracer(run_id, goal)
                call = ToolCall(id=call_id, name=name, input=step.payload["input"])
                result = self._run_one_tool(call, tracer)
                self.journal.append(
                    run_id,
                    "tool_result",
                    {
                        "tool_use_id": result.tool_use_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    },
                )
                pending_results.append(result)
            else:
                log.warning("resume.unknown_outcome", tool=name, tool_use_id=call_id)
                result = ToolResult(
                    tool_use_id=call_id,
                    content=(
                        f"The run was interrupted while {name!r} was executing. "
                        f"It is not marked idempotent, so it was NOT re-run: its "
                        f"outcome is unknown. Verify the current state before "
                        f"relying on it or calling it again."
                    ),
                    is_error=True,
                )
                self.journal.append(
                    run_id,
                    "tool_result",
                    {
                        "tool_use_id": call_id,
                        "content": result.content,
                        "is_error": True,
                    },
                )
                pending_results.append(result)

        if pending_results:
            messages.append(Message.tool_results(pending_results))
        return messages

    def _loop(self, run_id: str, goal: str, messages: list[Message]) -> RunResult:
        tracer = Tracer(run_id, goal)
        budget = Budget(guardrails=self.guardrails)
        tool_calls = 0
        tool_failures = 0
        stop_reason = "end_turn"
        final_text = ""
        unpriced = not self.settings.is_priced(self.provider.model_id)

        while True:
            breach = budget.check()
            if breach is not None:
                stop_reason = "guardrail"
                final_text = f"Stopped: {breach}"
                log.warning("agent.guardrail", run_id=run_id, reason=breach)
                break

            span = tracer.span("model_call", "model_call")
            completion = self.provider.complete(
                messages=messages,
                tools=self.registry.schemas(self.guardrails.tool_allowlist),
                system=self.system,
                max_tokens=self.max_tokens,
            )
            cost = self._cost(completion)
            span.end(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            budget.record_model_call(completion.input_tokens, completion.output_tokens, cost)

            assistant_message = completion.to_message()
            messages.append(assistant_message)

            if self.journal is not None:
                self.journal.append(
                    run_id,
                    "model_call",
                    {"content": assistant_message.content, "stop_reason": completion.stop_reason},
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cost_usd=cost,
                )

            if not completion.wants_tools():
                stop_reason = completion.stop_reason
                final_text = completion.text
                break

            results = self._run_tools(completion.tool_calls, tracer, run_id)
            tool_calls += len(results)
            failed = sum(1 for r in results if r.is_error)
            tool_failures += failed
            # One consecutive-failure verdict per turn: a turn where every tool
            # failed counts as one failure, not N, so parallel calls do not
            # trip the limit faster than serial ones.
            budget.record_tool_outcome(failed=failed == len(results) and bool(results))

            messages.append(Message.tool_results(results))

            if self.compactor is not None:
                before = self.compactor.count_tokens(messages, self.system)
                messages, changed = self.compactor.maybe_compact(messages, self.system)
                if changed:
                    after = self.compactor.count_tokens(messages, self.system)
                    compaction_span = tracer.span("compaction", "compaction")
                    compaction_span.end(tokens_before=before, tokens_after=after)
                    if self.journal is not None:
                        self.journal.append(
                            run_id,
                            "compaction",
                            {"summary": messages[1].text() if len(messages) > 1 else ""},
                        )

        final_span = tracer.span("final", "final")
        final_span.end(text=final_text)

        if self.journal is not None:
            self.journal.append(run_id, "final", {"text": final_text, "stop_reason": stop_reason})
            self.journal.finish_run(run_id, "complete" if stop_reason != "guardrail" else "stopped")

        wall_s = tracer.total_duration_s()
        result = RunResult(
            run_id=run_id,
            goal=goal,
            text=final_text,
            steps=budget.steps,
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            cost_usd=budget.cost_usd,
            wall_s=wall_s,
            stop_reason=stop_reason,
            trace=tracer.render(budget.cost_usd, wall_s),
            compactions=self.compactor.compactions if self.compactor else 0,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
            cost_is_estimated=unpriced,
            messages=messages,
        )

        log.info("agent.done", run_id=run_id, outcome=result.summary())
        return result


def raise_on_breach(result: RunResult) -> RunResult:
    """Turn a guardrail stop into an exception, for callers that want one.

    The loop returns a partial result rather than raising, because a benchmark
    wants to record the partial run. A caller that would rather fail fast wraps
    the call in this.
    """
    if result.stop_reason == "guardrail":
        raise GuardrailBreach(result.text, partial=result)
    return result
