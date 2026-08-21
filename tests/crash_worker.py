"""Worker process for the crash-resume test. Not a test module itself.

Runs a real agent, with a real journal, against a scripted provider so the
crash test costs nothing. The `record` tool appends a line to a counter file
*before* sleeping, so a kill during the sleep produces exactly the dangerous
state we care about: the side effect happened, but its result was never
journalled.

Usage: python crash_worker.py <journal_db> <counter_file> <run_id>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from agent_core.agent import Agent
from agent_core.config import Price, Settings
from agent_core.journal import Journal
from agent_core.providers.base import Completion, ToolCall
from agent_core.providers.replay import ScriptedProvider
from agent_core.registry import ToolRegistry

SLEEP_S = 0.4


def main() -> None:
    journal_db, counter_file, run_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    registry = ToolRegistry()

    @registry.register(timeout_s=30, idempotent=False)
    def record(tag: str) -> str:
        """Append a tag to the counter file, then sleep. NOT idempotent."""
        with counter_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{tag}\n")
        time.sleep(SLEEP_S)
        return f"recorded {tag}"

    @registry.register(timeout_s=30, idempotent=True)
    def peek(tag: str) -> str:
        """Read-only in spirit; logs to a side file so the test can observe
        whether resume re-ran it. Safe to repeat."""
        with counter_file.with_suffix(".peeks").open("a", encoding="utf-8") as handle:
            handle.write(tag + "\n")
        time.sleep(SLEEP_S)
        return f"peeked {tag}"

    # Alternating non-idempotent and idempotent calls, so a kill at different
    # delays lands on different kinds of tool.
    script = []
    for index in range(8):
        name = "record" if index % 2 == 0 else "peek"
        script.append(
            Completion(
                text="",
                tool_calls=[ToolCall(id=f"c{index}", name=name, input={"tag": str(index)})],
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
                model="scripted",
            )
        )
    script.append(Completion(text="all done", stop_reason="end_turn", model="scripted"))

    journal = Journal(journal_db)
    agent = Agent(
        provider=ScriptedProvider(script),
        registry=registry,
        system="crash test",
        journal=journal,
        settings=Settings(models={}, pricing={"scripted": Price(input=0.0, output=0.0)}),
    )
    agent.run("record several tags", run_id=run_id)


if __name__ == "__main__":
    main()
