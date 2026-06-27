"""MCP client — consume external search MCP servers (the CLIENT side of MCP).

Our agent is a MULTI-SERVER MCP client here. Over the stdio transport it launches
a configured search server as a subprocess, performs the MCP handshake, then
DISCOVERS/CALLS/CONSUMES a tool. Two servers, two purposes:
  - web_search()   -> the text server's `search` tool (3b-1): live/buyable info.
  - image_search() -> the image server's `ddg-image-search` tool (3b-2): visual
    references, with a graceful fall back to that server's `ddg-text-search` for
    reference *page* links when the image endpoint is rate-limited (DuckDuckGo
    throttles images aggressively).

Because we speak MCP, each server is swappable via config without touching nodes.
The public functions are the seam tests stub, so the real servers are NEVER
launched in the suite. They NEVER raise: disabled or all-failed -> "" so the
calling node degrades gracefully.
"""

import asyncio
import shutil
import sys
from pathlib import Path

from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


def _resolve_command(command: str) -> str:
    """Resolve a bare server command to a launchable path.

    Prefer the executable next to the current interpreter (our venv's bin) so the
    subprocess launches regardless of PATH; fall back to a PATH lookup, then the
    bare name.
    """
    sibling = Path(sys.executable).parent / command
    if sibling.exists():
        return str(sibling)
    return shutil.which(command) or command


async def _call_tool_async(command: str, tool_name: str, arguments: dict) -> str:
    """Launch the MCP server `command`, call `tool_name`, return its text content.

    This is the generalized discover→call→consume core, reused by every search
    function. Raises on a tool error so callers can fall back / degrade.
    """
    params = StdioServerParameters(command=_resolve_command(command), args=[])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP handshake
            result = await session.call_tool(tool_name, arguments)
            if getattr(result, "isError", False):
                raise RuntimeError(f"{tool_name} reported an error")
            # Keep only text blocks (titles/links/snippets); ignore any binary
            # image blocks — we link to references, never reproduce them.
            return "\n".join(
                getattr(block, "text", "")
                for block in result.content
                if getattr(block, "type", None) == "text"
            )


@traceable  # shows up as a web-search tool run in LangSmith (no-op when tracing off)
def web_search(query: str, max_results: int = 5) -> str:
    """Live web search via the text MCP server. "" if disabled or on any failure."""
    settings = get_settings()
    if not settings.web_search_enabled:
        logger.info("web search disabled via config")
        return ""
    try:
        return asyncio.run(
            _call_tool_async(
                settings.mcp_search_command, "search", {"query": query, "max_results": max_results}
            )
        )
    except Exception as exc:  # never let a flaky external server crash the agent
        logger.warning("web search failed (%s); degrading gracefully", exc)
        return ""


@traceable  # shows up as an image-search tool run in LangSmith (no-op when tracing off)
def image_search(query: str, max_results: int = 5) -> str:
    """Reference search via the image MCP server, resilient to image rate-limits.

    Tries `ddg-image-search` first (real image references). DuckDuckGo throttles
    its image endpoint aggressively, so on failure we fall back to the same
    server's `ddg-text-search` for reference *page* links. Returns "" only if
    disabled or BOTH attempts fail — so the reference node always degrades cleanly.
    """
    settings = get_settings()
    if not settings.image_search_enabled:
        logger.info("image search disabled via config")
        return ""

    command = settings.mcp_image_command
    args = {"keywords": query, "max_results": max_results}

    try:
        return asyncio.run(_call_tool_async(command, "ddg-image-search", args))
    except Exception as exc:
        logger.warning("image search failed (%s); falling back to text reference search", exc)

    try:
        return asyncio.run(_call_tool_async(command, "ddg-text-search", args))
    except Exception as exc:
        logger.warning("reference text fallback failed (%s); degrading gracefully", exc)
        return ""
