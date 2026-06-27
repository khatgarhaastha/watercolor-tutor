"""MCP client — consume an external web-search MCP server (the CLIENT side of MCP).

Our agent acts as an MCP client here. Over the stdio transport it launches the
configured search server as a subprocess, performs the MCP handshake, then:
  1. DISCOVER — `session.list_tools()` (the server advertises its tools); we use
     the `search` tool the DuckDuckGo server exposes.
  2. CALL     — `session.call_tool("search", {"query": ...})`.
  3. CONSUME  — parse the returned `CallToolResult` (a formatted text block).

Because we speak MCP, the server is swappable: DuckDuckGo (no key) today, a keyed
Tavily/Brave server later — without changing this seam or any node.

`web_search()` is the seam the rest of the app sees. Tests stub it, so the real
server is NEVER launched in the suite. It also NEVER raises: any failure (server
missing, timeout, tool error) returns "" so the calling node degrades gracefully.
"""

import asyncio
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)

# The tool name the DuckDuckGo MCP server advertises (discovered via list_tools).
SEARCH_TOOL = "search"


def _resolve_command(command: str) -> str:
    """Resolve a bare server command to a launchable path.

    Prefer the executable next to the current interpreter (our venv's bin), so the
    subprocess launches regardless of PATH; fall back to a PATH lookup, then the
    bare name.
    """
    sibling = Path(sys.executable).parent / command
    if sibling.exists():
        return str(sibling)
    return shutil.which(command) or command


async def _search_async(command: str, query: str, max_results: int) -> str:
    """Connect to the MCP server over stdio, call its search tool, return text."""
    params = StdioServerParameters(command=_resolve_command(command), args=[])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP handshake
            result = await session.call_tool(
                SEARCH_TOOL, {"query": query, "max_results": max_results}
            )
            if getattr(result, "isError", False):
                raise RuntimeError("search tool reported an error")
            # The DuckDuckGo server returns the results as text content block(s).
            return "\n".join(
                getattr(block, "text", "")
                for block in result.content
                if getattr(block, "type", None) == "text"
            )


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via the MCP server; return formatted results text.

    The swappable seam. A synchronous wrapper around the async MCP client (a fresh
    connection per search — fine for a CLI). Returns "" if search is disabled or on
    ANY failure, so routing/teaching never crashes on a flaky external server.
    """
    settings = get_settings()
    if not settings.web_search_enabled:
        logger.info("web search disabled via config")
        return ""
    try:
        return asyncio.run(_search_async(settings.mcp_search_command, query, max_results))
    except Exception as exc:  # never let a flaky external server crash the agent
        logger.warning("web search failed (%s); degrading gracefully", exc)
        return ""
