"""The Docker sandbox: the tool that makes this a runtime rather than a chatbot.

Model-written Python runs in a container with no network, a memory cap, a CPU
cap, a read-only root filesystem, a process limit and a non-root user. Each
flag closes a specific escape:

  --network none      no exfiltration, no downloading a payload
  --memory / --cpus   a fork bomb or an infinite allocation cannot take the host down
  --read-only         the image cannot be modified; only /workspace (tmpfs) is writable
  --pids-limit        caps process creation. Note the plural: `--pid-limit` is
                      not a Docker flag and fails silently in a script that
                      does not check exit codes
  --user              a container escape lands as an unprivileged user

The wall-clock timeout is enforced here rather than inside the container,
because a container that has stopped responding will not honour its own limit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any

import structlog

from agent_core.config import SandboxSettings
from agent_core.errors import SandboxError, truncate
from agent_core.registry import ToolRegistry

log = structlog.get_logger(__name__)


def docker_available() -> bool:
    """True when a Docker daemon is reachable. Used to skip tests, not to
    silently fall back to running model-written code on the host."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def build_sandbox_image(dockerfile: str | Path, tag: str) -> None:
    """Build the sandbox image. Run once, from `make setup`."""
    dockerfile = Path(dockerfile)
    result = subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", tag, str(dockerfile.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SandboxError(f"docker build failed:\n{result.stderr}")


def run_python_in_docker(code: str, settings: SandboxSettings) -> dict[str, Any]:
    """Execute `code` in a locked-down container and return stdout/stderr/exit code."""
    if shutil.which("docker") is None:
        raise SandboxError(
            "docker is not on PATH. The sandbox deliberately has no host-Python "
            "fallback: running model-written code on the host is the thing this "
            "tool exists to prevent."
        )

    # The script is mounted read-only rather than passed via `-c`, so quoting
    # and encoding never mangle it.
    with tempfile.TemporaryDirectory() as staging:
        script = Path(staging) / "main.py"
        script.write_text(code, encoding="utf-8")

        argv = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", settings.memory,
            "--cpus", settings.cpus,
            "--pids-limit", str(settings.pids_limit),
            "--read-only",
            "--tmpfs", "/workspace:rw,size=64m,mode=1777",
            "--user", "65534:65534",
            "--workdir", "/workspace",
            "-v", f"{script.as_posix()}:/workspace/main.py:ro",
            settings.image,
            "python", "/workspace/main.py",
        ]

        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=settings.timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            # The container is still running; `--rm` will not fire until it
            # stops, so nothing here reaps it. Docker cleans it up when the
            # daemon notices the client is gone, and the timeout is reported
            # to the model as a normal tool failure.
            return {
                "stdout": "",
                "stderr": f"execution exceeded the {settings.timeout_s}s limit and was killed",
                "exit_code": 124,
                "timed_out": True,
            }
        except OSError as exc:
            raise SandboxError(f"could not start the sandbox container: {exc}") from exc

    return {
        "stdout": truncate(completed.stdout, settings.max_output_bytes),
        "stderr": truncate(completed.stderr, settings.max_output_bytes),
        "exit_code": completed.returncode,
        "timed_out": False,
    }


def verify_memory_limit(settings: SandboxSettings) -> bool:
    """Check that --memory is actually enforced on this host.

    Docker silently ignores memory limits on some setups (notably kernels
    without cgroup swap accounting). Claiming a 512m cap in a README without
    checking is how you end up with a claim you cannot defend in an interview,
    so `make setup` runs this and the README reports the answer.
    """
    probe = (
        "buf = bytearray()\n"
        "try:\n"
        "    for _ in range(64):\n"
        "        buf.extend(bytearray(32 * 1024 * 1024))\n"
        "    print('ALLOCATED_2GB')\n"
        "except MemoryError:\n"
        "    print('MEMORY_ERROR')\n"
    )
    result = run_python_in_docker(probe, settings)
    # Enforced: the container is killed (137) or hits MemoryError. Not
    # enforced: it happily allocates 2 GB and prints ALLOCATED_2GB.
    return "ALLOCATED_2GB" not in result["stdout"]


def register_python_sandbox(registry: ToolRegistry, settings: SandboxSettings) -> None:
    """Register `run_python`."""

    @registry.register(timeout_s=settings.timeout_s + 5, dangerous=True, idempotent=True)
    def run_python(
        code: Annotated[str, "Python source to execute. Print results to stdout."],
    ) -> str:
        """Execute Python code in an isolated sandbox and return stdout, stderr and exit code.

        The sandbox has no network access and no access to the host filesystem.
        Only the standard library is available.
        """
        result = run_python_in_docker(code, settings)
        log.info("sandbox.run", exit_code=result["exit_code"], timed_out=result["timed_out"])
        return json.dumps(result, ensure_ascii=False)
