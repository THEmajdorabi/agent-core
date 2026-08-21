"""Shell tool.

Marked `dangerous=True`, so it only reaches the model when a run explicitly
allowlists it. Commands run without a shell interpreter and inside the working
directory, which removes the shell-injection surface: the model supplies an
argument list, not a string a shell will re-parse.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Annotated

from agent_core.errors import truncate
from agent_core.registry import ToolRegistry

MAX_OUTPUT_BYTES = 10_000


def register_shell_tool(
    registry: ToolRegistry, workdir: str | Path, timeout_s: float = 30.0
) -> None:
    """Register `run_command`, restricted to `workdir`."""
    root = Path(workdir).resolve()

    @registry.register(timeout_s=timeout_s, dangerous=True, idempotent=False)
    def run_command(
        command: Annotated[str, "The command line to run, e.g. 'pytest -q tests/'."],
    ) -> str:
        """Run a shell command in the working directory and return its exit code, stdout and stderr."""
        # shlex.split, then no shell: the model's string is parsed into argv
        # here rather than by a shell, so `; rm -rf /` becomes a literal
        # argument instead of a second command.
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"could not parse command {command!r}: {exc}") from exc
        if not argv:
            raise ValueError("empty command")

        completed = subprocess.run(  # noqa: S603 - argv list, shell=False
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

        parts = [f"exit_code: {completed.returncode}"]
        if completed.stdout:
            parts.append(f"stdout:\n{truncate(completed.stdout, MAX_OUTPUT_BYTES)}")
        if completed.stderr:
            parts.append(f"stderr:\n{truncate(completed.stderr, MAX_OUTPUT_BYTES)}")
        return "\n".join(parts)
