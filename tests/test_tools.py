"""Tools: the path jail, the shell parser, and the HTTP allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.registry import ToolRegistry
from agent_core.tools.filesystem import PathJail, register_filesystem_tools
from agent_core.tools.http import host_is_allowed
from agent_core.tools.shell import register_shell_tool


# ---- path jail ----------------------------------------------------------


def test_jail_allows_paths_inside_the_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("hello", encoding="utf-8")
    jail = PathJail(tmp_path)

    assert jail.resolve("sub/a.txt").read_text(encoding="utf-8") == "hello"


def test_jail_refuses_parent_traversal(tmp_path: Path) -> None:
    jail = PathJail(tmp_path / "root")
    (tmp_path / "root").mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the permitted root"):
        jail.resolve("../secret.txt")


def test_jail_refuses_an_absolute_path(tmp_path: Path) -> None:
    jail = PathJail(tmp_path)

    with pytest.raises(ValueError, match="outside the permitted root"):
        # An absolute path passed to `/` replaces the left operand entirely,
        # which is exactly how a naive jail gets escaped.
        jail.resolve("C:/Windows/System32/drivers/etc/hosts" if Path("C:/").exists() else "/etc/passwd")


def test_jail_refuses_deep_traversal(tmp_path: Path) -> None:
    jail = PathJail(tmp_path)

    with pytest.raises(ValueError, match="outside the permitted root"):
        jail.resolve("a/b/../../../../../../etc/passwd")


def test_read_write_and_list_round_trip(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    write = registry.get("write_file")
    read = registry.get("read_file")
    listing = registry.get("list_dir")

    write.fn(path="notes/todo.txt", content="fix the bug")
    assert read.fn(path="notes/todo.txt") == "fix the bug"
    assert "d notes" in listing.fn(path=".")


def test_read_file_reports_a_missing_file(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.get("read_file").fn(path="nope.txt")


def test_write_file_is_marked_non_idempotent(tmp_path: Path) -> None:
    """Overwriting is a side effect, so resume must not repeat it blindly."""
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    assert registry.get("read_file").idempotent is True
    assert registry.get("list_dir").idempotent is True
    assert registry.get("write_file").idempotent is False


# ---- shell --------------------------------------------------------------


def test_shell_tool_is_marked_dangerous(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_shell_tool(registry, tmp_path)

    assert registry.get("run_command").dangerous is True


def test_shell_reports_exit_code_and_output(tmp_path: Path) -> None:
    import sys

    registry = ToolRegistry()
    register_shell_tool(registry, tmp_path)

    output = registry.get("run_command").fn(
        command=f'"{sys.executable}" -c "print(41 + 1)"'
    )
    assert "exit_code: 0" in output
    assert "42" in output


def test_shell_reports_a_nonzero_exit_rather_than_raising(tmp_path: Path) -> None:
    import sys

    registry = ToolRegistry()
    register_shell_tool(registry, tmp_path)

    output = registry.get("run_command").fn(
        command=f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    )
    assert "exit_code: 3" in output


def test_shell_refuses_an_unparseable_command(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_shell_tool(registry, tmp_path)

    with pytest.raises(ValueError, match="could not parse"):
        registry.get("run_command").fn(command='unmatched "quote')


def test_shell_does_not_let_a_shell_reinterpret_the_string(tmp_path: Path) -> None:
    """`;` is a literal argument, not a command separator.

    The command runs with shell=False on an argv list, so the classic
    injection has nothing to inject into: the whole thing is one argument
    list, and `echo` is asked to print the semicolon.
    """
    import sys

    registry = ToolRegistry()
    register_shell_tool(registry, tmp_path)
    marker = tmp_path / "pwned.txt"

    output = registry.get("run_command").fn(
        command=f'"{sys.executable}" -c "print(1)" ; "{sys.executable}" -c "open(r\'{marker}\',\'w\')"'
    )

    assert not marker.exists(), "a second command was executed: shell=False was bypassed"
    assert "exit_code:" in output


# ---- http allowlist -----------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/page", True),
        ("https://docs.example.com/page", True),  # subdomain
        ("http://example.com", True),
        ("https://evil.com/page", False),
        ("https://notexample.com", False),  # suffix must be on a dot boundary
        ("https://example.com.evil.com", False),
        ("file:///etc/passwd", False),  # non-http scheme
        ("ftp://example.com", False),
    ],
)
def test_host_allowlist(url: str, expected: bool) -> None:
    assert host_is_allowed(url, {"example.com"}) is expected
