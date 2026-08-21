"""Filesystem tools, jailed to one root directory.

The jail is the point. An agent given `read_file` without a root can read
`~/.ssh/id_rsa` the moment a prompt talks it into trying. Every path is
resolved to an absolute real path and checked against the root, so `../`,
symlinks and absolute paths all fail the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from agent_core.registry import ToolRegistry

MAX_READ_BYTES = 100_000


class PathJail:
    """Resolves paths and refuses anything outside the root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve `relative` inside the jail, or raise ValueError.

        `Path.resolve()` follows symlinks, so a link pointing outside the root
        resolves to its target and fails the containment check -- which is the
        behaviour we want.
        """
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(
                f"path {relative!r} resolves outside the permitted root {self.root}"
            )
        return candidate


def register_filesystem_tools(registry: ToolRegistry, root: str | Path) -> PathJail:
    """Register read_file, write_file and list_dir against `root`."""
    jail = PathJail(root)

    @registry.register(timeout_s=10, idempotent=True)
    def read_file(
        path: Annotated[str, "Path to the file, relative to the working directory."],
    ) -> str:
        """Read a UTF-8 text file and return its contents."""
        target = jail.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        data = target.read_bytes()[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")

    @registry.register(timeout_s=10, idempotent=False)
    def write_file(
        path: Annotated[str, "Path to write, relative to the working directory."],
        content: Annotated[str, "Full new contents of the file."],
    ) -> str:
        """Write text to a file, creating parent directories and overwriting any existing file."""
        target = jail.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    @registry.register(timeout_s=10, idempotent=True)
    def list_dir(
        path: Annotated[str, "Directory to list, relative to the working directory."] = ".",
    ) -> str:
        """List the files and directories at a path, one per line."""
        target = jail.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return "(empty directory)"
        return "\n".join(
            f"{'d' if entry.is_dir() else 'f'} {entry.name}" for entry in entries
        )

    return jail
