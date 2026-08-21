"""Command line interface.

    agent-core run "goal"        run an agent against the current directory
    agent-core resume <run_id>   continue an interrupted run
    agent-core runs              list unfinished runs
    agent-core trace <run_id>    re-render a finished run's trace tree

No web UI by design. The CLI is enough to drive the loop, and the trace tree
is the thing worth looking at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from agent_core.agent import Agent
from agent_core.config import load_settings
from agent_core.context import Compactor, PinnedFacts
from agent_core.guardrails import Guardrails
from agent_core.journal import Journal
from agent_core.providers import build_provider
from agent_core.registry import ToolRegistry
from agent_core.tools import (
    docker_available,
    register_filesystem_tools,
    register_python_sandbox,
    register_shell_tool,
)

SYSTEM_PROMPT = """\
You are a software engineering agent working inside a single directory.

Work by using the tools rather than by describing what you would do. Read
before you write. After changing code, run the tests to check the change.
When a tool fails, read the error and adapt; do not repeat the same call.

Use pin_fact to record anything you must not forget, such as a reproduction
step or a constraint you discovered. Pinned facts survive context compaction.

Stop as soon as the goal is met, and say plainly what you changed.
"""


def configure_logging(verbose: bool) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20 if verbose else 30),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


def build_agent(workdir: Path, model_key: str, allow_shell: bool, verbose: bool) -> Agent:
    """Assemble an agent: provider, tools, guardrails, journal, compactor."""
    settings = load_settings()
    spec = settings.models[model_key]
    provider = build_provider(spec.provider, spec.id)

    registry = ToolRegistry()
    register_filesystem_tools(registry, workdir)

    pinned = PinnedFacts()

    @registry.register(idempotent=True)
    def pin_fact(fact: str) -> str:
        """Record a fact that must survive context compaction. Use for constraints and findings you must not forget."""
        pinned.pin(fact)
        return f"pinned ({len(pinned.facts)} total)"

    if allow_shell:
        register_shell_tool(registry, workdir)

    if docker_available():
        register_python_sandbox(registry, settings.sandbox)
    elif verbose:
        print("note: docker is unavailable, so run_python is not registered", file=sys.stderr)

    journal = Journal(settings.journal.path)
    compactor = Compactor(
        provider,
        max_tokens=settings.context.max_tokens,
        trigger_ratio=settings.context.trigger_ratio,
        target_ratio=settings.context.target_ratio,
        keep_recent=settings.context.keep_recent,
        pinned=pinned,
    )

    return Agent(
        provider=provider,
        registry=registry,
        system=SYSTEM_PROMPT,
        guardrails=Guardrails(
            max_steps=settings.guardrails.max_steps,
            max_cost_usd=settings.guardrails.max_cost_usd,
            max_wall_s=settings.guardrails.max_wall_s,
            max_consecutive_tool_failures=settings.guardrails.max_consecutive_tool_failures,
        ),
        journal=journal,
        compactor=compactor,
        settings=settings,
        max_tokens=spec.max_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-core", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="log every step")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an agent against a goal")
    run_parser.add_argument("goal")
    run_parser.add_argument("--workdir", default=".", help="directory the agent may touch")
    run_parser.add_argument(
        "--model", default="default", help="a key under `models:` in config.yaml"
    )
    run_parser.add_argument("--allow-shell", action="store_true", help="register run_command")

    resume_parser = subparsers.add_parser("resume", help="continue an interrupted run")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--workdir", default=".")
    resume_parser.add_argument("--model", default="default")
    resume_parser.add_argument("--allow-shell", action="store_true")

    subparsers.add_parser("runs", help="list unfinished runs")

    trace_parser = subparsers.add_parser("trace", help="re-render a run's trace")
    trace_parser.add_argument("run_id")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "runs":
        journal = Journal(load_settings().journal.path)
        unfinished = journal.unfinished_runs()
        if not unfinished:
            print("no unfinished runs")
            return 0
        for run_id in unfinished:
            row = journal.run_row(run_id)
            # run_row can return None if the runs row was deleted while its
            # steps survived. Skipping is better than crashing the listing.
            if row is not None:
                print(f"{run_id}  {row['goal']}")
        return 0

    if args.command == "trace":
        from agent_core.tracing import render_journalled_run

        print(render_journalled_run(Journal(load_settings().journal.path), args.run_id))
        return 0

    agent = build_agent(Path(args.workdir).resolve(), args.model, args.allow_shell, args.verbose)

    result = agent.run(args.goal) if args.command == "run" else agent.resume(args.run_id)

    print(result.trace)
    print()
    print(result.text)
    print()
    print(f"run_id: {result.run_id}  {result.summary()}")
    return 0 if result.stop_reason != "guardrail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
