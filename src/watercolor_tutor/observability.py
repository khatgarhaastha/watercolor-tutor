"""Optional LangSmith tracing — production-grade observability, additive & key-gated.

Tracing turns on ONLY when a LANGSMITH_API_KEY is configured. Without one this is a
no-op: the app (and the tests) run identically with no LangSmith and no network.

When it's on, three things light up the LangSmith UI together:
  - LangGraph auto-traces the node flow (each node becomes a nested "run").
  - `wrap_anthropic` (in llm.py) traces every LLM call with token usage + latency.
  - `@traceable` functions (RAG retrieval, MCP web/image search) appear as runs.

A "run" is one unit of work (an LLM/tool call, a node, the whole invoke) with its
inputs, outputs, timing, and metadata; a "trace" is the tree of runs for one
end-to-end execution. Unlike our structured logs (flat, hand-written lines), the
trace tree + token/latency metrics are captured automatically and viewed in a UI.
"""

import os

from dotenv import load_dotenv

from .logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_PROJECT = "watercolor-tutor"


def tracing_enabled() -> bool:
    """True if LangSmith tracing is currently switched on in the environment."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")


def setup_tracing() -> bool:
    """Enable LangSmith tracing IFF an API key is configured. Returns whether it's on.

    Called once at startup. We `load_dotenv()` so LANGSMITH_* from .env reach
    os.environ — the langsmith SDK reads os.environ, whereas pydantic-settings reads
    .env separately and does not populate the process environment. No key -> no-op.
    An explicit LANGSMITH_TRACING=false is respected even when a key is present (a
    kill-switch), since we only `setdefault` the flag.
    """
    load_dotenv()
    if not os.environ.get("LANGSMITH_API_KEY"):
        logger.info("LangSmith tracing off (no LANGSMITH_API_KEY set)")
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    enabled = tracing_enabled()
    logger.info(
        "LangSmith tracing %s (project=%s)",
        "ON" if enabled else "off",
        os.environ.get("LANGSMITH_PROJECT"),
    )
    return enabled
