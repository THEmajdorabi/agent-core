"""Render a trace tree.

Two modes:

    python scripts/render_trace.py <run_id>   re-render a real journalled run
    python scripts/render_trace.py --demo     a scripted run, no API key needed

The demo mode exists so the README can show real `Tracer.render` output rather
than a hand-typed illustration. It runs the actual agent loop against real
tools; only the model's replies are scripted.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_core.agent import Agent  # noqa: E402
from agent_core.config import Price, Settings, load_settings  # noqa: E402
from agent_core.journal import Journal  # noqa: E402
from agent_core.providers.base import Completion, ToolCall  # noqa: E402
from agent_core.providers.replay import ScriptedProvider  # noqa: E402
from agent_core.registry import ToolRegistry  # noqa: E402
from agent_core.tools import register_filesystem_tools, register_shell_tool  # noqa: E402
from agent_core.tracing import render_journalled_run, use_utf8_stdout  # noqa: E402

BROKEN = '''def windows(items, size):
    """Return every contiguous window of length `size`."""
    return [items[i : i + size] for i in range(len(items) - size)]
'''

FIXED = '''def windows(items, size):
    """Return every contiguous window of length `size`."""
    return [items[i : i + size] for i in range(len(items) - size + 1)]
'''


def scripted(text: str = "", calls: list[ToolCall] | None = None, out: int = 0) -> Completion:
    tool_calls = calls or []
    return Completion(
        text=text,
        tool_calls=tool_calls,
        stop_reason="tool_use" if tool_calls else "end_turn",
        input_tokens=1200 + 900 * len(tool_calls),
        output_tokens=out or 150,
        model="scripted",
    )


def demo() -> str:
    """Run the real loop with real tools against a scripted model."""
    workdir = Path(tempfile.mkdtemp())
    (workdir / "window.py").write_text(BROKEN, encoding="utf-8")

    registry = ToolRegistry()
    register_filesystem_tools(registry, workdir)
    register_shell_tool(registry, workdir, timeout_s=30)

    provider = ScriptedProvider(
        [
            scripted(
                "Let me look at the code and the failing case.",
                [
                    ToolCall("c1", "list_dir", {"path": "."}),
                    ToolCall("c2", "read_file", {"path": "window.py"}),
                ],
                out=189,
            ),
            scripted(
                "The range stops one short. Checking the current behaviour first.",
                [
                    ToolCall(
                        "c3",
                        "run_command",
                        {
                            "command": f'"{sys.executable}" -c '
                            f"\"import sys; sys.path.insert(0, r'{workdir}'); "
                            f"from window import windows; "
                            f'assert windows([1,2,3,4],2)==[[1,2],[2,3],[3,4]]"'
                        },
                    )
                ],
                out=402,
            ),
            scripted(
                "Confirmed. Fixing the slice bound.",
                [ToolCall("c4", "write_file", {"path": "window.py", "content": FIXED})],
                out=288,
            ),
            scripted(
                "Fixed: the range excluded the final window. Changed "
                "`len(items) - size` to `len(items) - size + 1`.",
                out=64,
            ),
        ]
    )

    agent = Agent(
        provider=provider,
        registry=registry,
        system="You are a software engineering agent.",
        settings=Settings(models={}, pricing={"scripted": Price(input=3.0, output=15.0)}),
        journal=Journal(workdir / "runs.db"),
    )
    result = agent.run("fix the failing test")
    return result.trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", help="a journalled run id")
    parser.add_argument("--demo", action="store_true", help="scripted run, no API key needed")
    args = parser.parse_args()
    use_utf8_stdout()

    if args.demo:
        print(demo())
        return 0

    if not args.run_id:
        parser.error("give a run_id, or --demo")

    print(render_journalled_run(Journal(load_settings().journal.path), args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
