"""The web_search node — fetch live info via the MCP web-search client.

Reached only when the classifier labels the reply 'needs_web_info' (e.g. asking
what beginner supplies to BUY, or for prices/recent tutorials) — so external tool
use is gated by the same intent machinery as everything else. It calls the MCP
search seam, folds the results into ONE Claude synthesis for a beginner, and stays
on the current step. If search returns nothing (server down/disabled), it degrades
gracefully and tells the learner live search was unavailable — no crash.
"""

from .. import llm, mcp_search
from ..logging_config import get_logger
from ..prompts import SYSTEM_PROMPT, WEB_SEARCH_PREAMBLE, WEB_SEARCH_UNAVAILABLE
from ..routing import last_user_message
from ..state import TutorState

logger = get_logger(__name__)


def web_search(state: TutorState) -> dict:
    """Answer the learner's live-info question using MCP web search; stay on-step."""
    query = last_user_message(state) or ""
    logger.info("web search for query=%r", query)

    results = mcp_search.web_search(query)  # "" if disabled or on any failure
    if results:
        # f-string (not .format) so result text containing braces is harmless.
        content = (
            f"{WEB_SEARCH_PREAMBLE}\n\nLearner's question: {query}\n\nSearch results:\n{results}"
        )
    else:
        content = f"{WEB_SEARCH_UNAVAILABLE}\n\nLearner's question: {query}"

    reply = llm.generate(SYSTEM_PROMPT, [{"role": "user", "content": content}])
    return {"messages": [{"role": "assistant", "content": reply}]}
