"""Run the benchmark and produce the README numbers.

Every task is copied to a scratch directory before the agent touches it, so
the task folders stay pristine and repeats are independent. `check.sh` decides
whether a task was solved -- the agent's own claim that it fixed something is
not evidence.

Writes benchmark/results.json and prints the markdown table.

Run:
    python scripts/run_benchmark.py                  # all tasks, 3 repeats
    python scripts/run_benchmark.py --repeats 1
    python scripts/run_benchmark.py --task 01_off_by_one_slice
    python scripts/run_benchmark.py --model compare  # the OpenAI model
    python scripts/run_benchmark.py --dry-run        # no API calls; checks wiring
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "benchmark" / "tasks"
RESULTS_PATH = REPO_ROOT / "benchmark" / "results.json"
DRY_RUN_PATH = REPO_ROOT / "benchmark" / "results.dry-run.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_core.agent import Agent  # noqa: E402
from agent_core.config import load_settings  # noqa: E402
from agent_core.guardrails import Guardrails  # noqa: E402
from agent_core.providers import build_provider  # noqa: E402
from agent_core.registry import ToolRegistry  # noqa: E402
from agent_core.tools import register_filesystem_tools, register_shell_tool  # noqa: E402

SYSTEM_PROMPT = """\
You are fixing a bug in a small Python repository.

Read the failing test first, then the source. Make the smallest change that
makes the test pass. Do not edit the test file. Run the test with
run_command to confirm your fix before you finish.

When you are done, say in one sentence what was wrong and what you changed.
"""


@dataclass
class Attempt:
    """One agent run against one task."""

    task_id: str
    repeat: int
    solved: bool
    steps: int
    tool_calls: int
    tool_failures: int
    cost_usd: float
    wall_s: float
    stop_reason: str
    input_tokens: int
    output_tokens: int
    error: str | None = None


def run_check(folder: Path) -> bool:
    """The task's own verdict. Exit code 0 means solved."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
            cwd=folder,
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def build_agent(workdir: Path, model_key: str) -> Agent:
    settings = load_settings()
    spec = settings.models[model_key]
    registry = ToolRegistry()
    register_filesystem_tools(registry, workdir)
    register_shell_tool(registry, workdir, timeout_s=60)

    return Agent(
        provider=build_provider(spec.provider, spec.id, spec.reasoning_effort),
        registry=registry,
        system=SYSTEM_PROMPT.format(python=sys.executable),
        guardrails=Guardrails(
            max_steps=settings.guardrails.max_steps,
            max_cost_usd=settings.guardrails.max_cost_usd,
            max_wall_s=settings.guardrails.max_wall_s,
        ),
        settings=settings,
        max_tokens=spec.max_tokens,
    )


def attempt_task(task_id: str, repeat: int, model_key: str, dry_run: bool) -> Attempt:
    """Copy the task, let the agent work, then ask check.sh for the verdict."""
    goal = json.loads((TASKS_DIR / task_id / "task.json").read_text(encoding="utf-8"))["goal"]

    with tempfile.TemporaryDirectory() as staging:
        workdir = Path(staging) / task_id
        shutil.copytree(TASKS_DIR / task_id, workdir)

        if dry_run:
            # Prove the harness works without spending anything: no agent runs,
            # so every task stays broken and every attempt is honestly unsolved.
            return Attempt(
                task_id=task_id,
                repeat=repeat,
                solved=run_check(workdir),
                steps=0,
                tool_calls=0,
                tool_failures=0,
                cost_usd=0.0,
                wall_s=0.0,
                stop_reason="dry_run",
                input_tokens=0,
                output_tokens=0,
            )

        started = time.monotonic()
        try:
            agent = build_agent(workdir, model_key)
            result = agent.run(goal)
        except Exception as exc:  # noqa: BLE001
            # One task blowing up must not lose the other nineteen attempts.
            return Attempt(
                task_id=task_id,
                repeat=repeat,
                solved=False,
                steps=0,
                tool_calls=0,
                tool_failures=0,
                cost_usd=0.0,
                wall_s=time.monotonic() - started,
                stop_reason="error",
                input_tokens=0,
                output_tokens=0,
                error=f"{type(exc).__name__}: {exc}",
            )

        return Attempt(
            task_id=task_id,
            repeat=repeat,
            solved=run_check(workdir),
            steps=result.steps,
            tool_calls=result.tool_calls,
            tool_failures=result.tool_failures,
            cost_usd=result.cost_usd,
            wall_s=result.wall_s,
            stop_reason=result.stop_reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


def median(values: list[float]) -> float | None:
    """Median, or None when there is nothing to take a median of.

    None rather than 0.0: a task solved zero times has no median cost, and
    reporting $0.00 would read as "free" instead of "no data".
    """
    return statistics.median(values) if values else None


def summarise(attempts: list[Attempt]) -> dict:
    """Aggregate per task and overall. Medians are over *solved* attempts only."""
    by_task: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_task.setdefault(attempt.task_id, []).append(attempt)

    per_task = []
    for task_id in sorted(by_task):
        runs = by_task[task_id]
        solved = [r for r in runs if r.solved]
        per_task.append(
            {
                "task_id": task_id,
                "attempts": len(runs),
                "solved": len(solved),
                "solve_rate": len(solved) / len(runs),
                "median_steps": median([r.steps for r in solved]),
                "median_cost_usd": median([r.cost_usd for r in solved]),
                "median_wall_s": median([r.wall_s for r in solved]),
            }
        )

    solved_all = [a for a in attempts if a.solved]
    # A task counts as solved if it was solved in at least one repeat.
    tasks_ever_solved = len({a.task_id for a in solved_all})

    return {
        "per_task": per_task,
        "overall": {
            "tasks": len(by_task),
            "tasks_solved_at_least_once": tasks_ever_solved,
            "total_attempts": len(attempts),
            "attempts_solved": len(solved_all),
            "attempt_solve_rate": len(solved_all) / len(attempts) if attempts else 0.0,
            "median_steps_per_solved": median([a.steps for a in solved_all]),
            "median_cost_usd_per_solved": median([a.cost_usd for a in solved_all]),
            "median_wall_s_per_solved": median([a.wall_s for a in solved_all]),
            "total_cost_usd": sum(a.cost_usd for a in attempts),
        },
    }


def fmt(value: float | None, spec: str, prefix: str = "", suffix: str = "") -> str:
    """Format a number, or a bare `n/a` when it was never measured.

    The prefix and suffix are dropped for n/a, so a task that was never solved
    reads as "n/a" rather than "$n/a" or "n/as".
    """
    if value is None:
        return "n/a"
    return f"{prefix}{format(value, spec)}{suffix}"


def markdown_table(summary: dict, model_id: str) -> str:
    lines = [
        f"Model: `{model_id}`  ",
        f"Tasks solved at least once: "
        f"**{summary['overall']['tasks_solved_at_least_once']} of {summary['overall']['tasks']}**  ",
        f"Attempt solve rate: {summary['overall']['attempt_solve_rate']:.0%} "
        f"({summary['overall']['attempts_solved']}/{summary['overall']['total_attempts']})",
        "",
        "| Task | Solved | Median steps | Median cost | Median wall |",
        "|---|---|---|---|---|",
    ]
    for row in summary["per_task"]:
        lines.append(
            f"| `{row['task_id']}` | {row['solved']}/{row['attempts']} | "
            f"{fmt(row['median_steps'], '.0f')} | "
            f"{fmt(row['median_cost_usd'], '.4f', prefix='$')} | "
            f"{fmt(row['median_wall_s'], '.1f', suffix='s')} |"
        )
    overall = summary["overall"]
    lines += [
        "",
        f"Median over all solved attempts: {fmt(overall['median_steps_per_solved'], '.0f')} steps, "
        f"{fmt(overall['median_cost_usd_per_solved'], '.4f', prefix='$')}, "
        f"{fmt(overall['median_wall_s_per_solved'], '.1f', suffix='s')}.  ",
        f"Total benchmark cost: ${overall['total_cost_usd']:.2f}.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task", action="append", help="run only these task ids")
    parser.add_argument("--model", default="default", help="a key under `models:` in config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the harness with no API calls; every task stays unsolved",
    )
    args = parser.parse_args()

    task_ids = args.task or sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir())
    model_id = "dry-run" if args.dry_run else load_settings().models[args.model].id

    attempts: list[Attempt] = []
    for task_id in task_ids:
        for repeat in range(args.repeats):
            attempt = attempt_task(task_id, repeat, args.model, args.dry_run)
            attempts.append(attempt)
            verdict = "solved" if attempt.solved else "unsolved"
            note = f"  [{attempt.error}]" if attempt.error else ""
            print(
                f"{task_id} repeat {repeat}: {verdict} "
                f"({attempt.steps} steps, ${attempt.cost_usd:.4f}, {attempt.wall_s:.1f}s){note}"
            )

    summary = summarise(attempts)
    # A dry run writes elsewhere, so its all-zero table can never be mistaken
    # for a measured result sitting in results.json.
    output_path = DRY_RUN_PATH if args.dry_run else RESULTS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": model_id,
                "repeats": args.repeats,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dry_run": args.dry_run,
                "summary": summary,
                "attempts": [asdict(a) for a in attempts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(markdown_table(summary, model_id))
    print()
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
