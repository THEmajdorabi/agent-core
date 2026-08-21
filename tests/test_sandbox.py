"""Sandbox tests.

Marked `docker` and excluded from the default run, because they need a live
daemon. They are not integration tests in the paid-API sense -- they cost
nothing -- but a developer without Docker should still get a green `make test`.

The claims these check are the ones the README makes about isolation. An
unverified "runs in an isolated container with a 512MB limit" is the kind of
sentence an interviewer will ask you to prove.
"""

from __future__ import annotations

import json

import pytest

from agent_core.config import SandboxSettings, load_settings
from agent_core.errors import SandboxError
from agent_core.registry import ToolRegistry
from agent_core.tools.python_sandbox import (
    docker_available,
    register_python_sandbox,
    run_python_in_docker,
    verify_memory_limit,
)

pytestmark = pytest.mark.docker

requires_docker = pytest.mark.skipif(not docker_available(), reason="no reachable Docker daemon")


@pytest.fixture
def sandbox() -> SandboxSettings:
    return load_settings().sandbox


@requires_docker
def test_code_runs_and_returns_stdout(sandbox: SandboxSettings) -> None:
    result = run_python_in_docker("print(6 * 7)", sandbox)

    assert result["exit_code"] == 0
    assert "42" in result["stdout"]
    assert result["timed_out"] is False


@requires_docker
def test_a_traceback_comes_back_as_stderr_not_an_exception(sandbox: SandboxSettings) -> None:
    result = run_python_in_docker("raise ValueError('boom')", sandbox)

    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


@requires_docker
def test_the_network_is_unreachable(sandbox: SandboxSettings) -> None:
    """--network none, checked rather than asserted in a README."""
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('NETWORK_REACHABLE')\n"
        "except OSError as exc:\n"
        "    print('NETWORK_BLOCKED')\n"
    )
    result = run_python_in_docker(code, sandbox)

    assert "NETWORK_BLOCKED" in result["stdout"]
    assert "NETWORK_REACHABLE" not in result["stdout"]


@requires_docker
def test_the_root_filesystem_is_read_only(sandbox: SandboxSettings) -> None:
    code = (
        "try:\n"
        "    open('/etc/passwd', 'a').write('x')\n"
        "    print('WROTE_TO_ROOT')\n"
        "except OSError:\n"
        "    print('ROOT_READ_ONLY')\n"
    )
    result = run_python_in_docker(code, sandbox)

    assert "ROOT_READ_ONLY" in result["stdout"]


@requires_docker
def test_workspace_is_writable(sandbox: SandboxSettings) -> None:
    """The tmpfs at /workspace must still work, or nothing useful can run."""
    code = (
        "open('/workspace/scratch.txt', 'w').write('ok')\n"
        "print(open('/workspace/scratch.txt').read())\n"
    )
    result = run_python_in_docker(code, sandbox)

    assert result["exit_code"] == 0
    assert "ok" in result["stdout"]


@requires_docker
def test_the_process_is_not_root(sandbox: SandboxSettings) -> None:
    result = run_python_in_docker("import os; print(os.getuid())", sandbox)

    assert result["stdout"].strip() != "0", "sandbox ran as root"


@requires_docker
def test_a_hanging_script_is_killed_at_the_timeout(sandbox: SandboxSettings) -> None:
    fast = sandbox.model_copy(update={"timeout_s": 3.0})
    result = run_python_in_docker("import time; time.sleep(60)", fast)

    assert result["timed_out"] is True
    assert result["exit_code"] == 124


@requires_docker
def test_output_is_truncated_with_a_marker(sandbox: SandboxSettings) -> None:
    small = sandbox.model_copy(update={"max_output_bytes": 200})
    result = run_python_in_docker("print('x' * 5000)", small)

    assert "truncated" in result["stdout"]
    assert len(result["stdout"].encode()) < 1000


@requires_docker
def test_the_memory_limit_is_actually_enforced(sandbox: SandboxSettings) -> None:
    """Docker silently ignores --memory on some hosts.

    The README states a 512m cap, so this checks the cap exists rather than
    trusting the flag. If it fails, the correct response is to change the
    README, not to delete the test.
    """
    assert verify_memory_limit(sandbox), (
        "--memory was not enforced on this host: the container allocated 2GB "
        "under a 512m limit. Do not claim a memory cap in the README."
    )


@requires_docker
def test_the_registered_tool_returns_json(sandbox: SandboxSettings) -> None:
    registry = ToolRegistry()
    register_python_sandbox(registry, sandbox)
    tool = registry.get("run_python")

    assert tool.dangerous is True
    assert tool.idempotent is True

    payload = json.loads(tool.fn(code="print('hello')"))
    assert payload["exit_code"] == 0
    assert "hello" in payload["stdout"]


def test_a_missing_docker_binary_raises_rather_than_running_on_the_host(
    monkeypatch: pytest.MonkeyPatch, sandbox: SandboxSettings
) -> None:
    """The most important test in this file, and it needs no daemon.

    There must be no host-Python fallback. A sandbox that quietly stops
    sandboxing is worse than one that is plainly unavailable.
    """
    monkeypatch.setattr("agent_core.tools.python_sandbox.shutil.which", lambda _: None)

    with pytest.raises(SandboxError, match="no host-Python fallback"):
        run_python_in_docker("print('should never run')", sandbox)
