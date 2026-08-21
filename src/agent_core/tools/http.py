"""HTTP fetch, restricted to an explicit domain allowlist.

An agent with unrestricted outbound HTTP is an exfiltration channel: anything
it has read can be appended to a URL. The allowlist is checked against the
resolved host after redirects are disabled, so a permitted domain cannot
bounce the request somewhere else.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlparse

from agent_core.errors import truncate
from agent_core.registry import ToolRegistry

MAX_BODY_BYTES = 50_000


def host_is_allowed(url: str, allowlist: set[str]) -> bool:
    """True when the URL's host is in the allowlist, or is a subdomain of one."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowlist)


def register_http_tool(
    registry: ToolRegistry, allowlist: set[str], timeout_s: float = 15.0
) -> None:
    """Register `fetch_url`, limited to `allowlist`."""

    @registry.register(timeout_s=timeout_s, idempotent=True)
    def fetch_url(
        url: Annotated[str, "Absolute http(s) URL to fetch."],
    ) -> str:
        """Fetch a URL and return the response body as text. Only allowlisted domains are permitted."""
        import httpx

        if not host_is_allowed(url, allowlist):
            permitted = ", ".join(sorted(allowlist)) or "<none>"
            raise ValueError(f"host not permitted. Allowed domains: {permitted}")

        # follow_redirects=False keeps the allowlist meaningful: a permitted
        # host cannot redirect the request to an unpermitted one.
        response = httpx.get(url, timeout=timeout_s, follow_redirects=False)
        response.raise_for_status()
        return truncate(response.text, MAX_BODY_BYTES)
