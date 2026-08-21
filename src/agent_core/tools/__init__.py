"""Built-in tools.

Each `register_*` function takes a registry and the settings that tool needs,
so an agent decides its own capabilities rather than inheriting a global set.
"""

from agent_core.tools.filesystem import PathJail, register_filesystem_tools
from agent_core.tools.http import register_http_tool
from agent_core.tools.python_sandbox import (
    docker_available,
    register_python_sandbox,
    run_python_in_docker,
)
from agent_core.tools.shell import register_shell_tool

__all__ = [
    "PathJail",
    "docker_available",
    "register_filesystem_tools",
    "register_http_tool",
    "register_python_sandbox",
    "register_shell_tool",
    "run_python_in_docker",
]
