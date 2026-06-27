"""The reference node — fetch a step-anchored reference image via the MCP client.

Reached only when the classifier labels the reply 'needs_reference_image' (the
learner asked to SEE an example). Two appropriateness mechanisms, framed honestly
as intelligent search (not a curated guarantee):
  1. SMART QUERY — we search a per-step, BEGINNER-AIMED query (REFERENCE_QUERIES),
     not the learner's literal words, so results skew toward teaching references.
  2. METADATA FILTERING — the LLM reasons over the result titles/links/sources to
     pick beginner-appropriate references and skip masterpieces/products, and says
     so honestly if nothing fits.

Copyright: we return source links + descriptions, never reproduced artwork
(terminal can't display images anyway; inline display is a later UI concern).

Deferred (documented, not built): vision-VETTING candidate images with our own
vision capability, and a curated reference library — the more rigorous upgrades to
appropriateness. Step-anchored search + LLM metadata filtering is enough here.
"""

from .. import llm, mcp_search
from ..logging_config import get_logger
from ..prompts import (
    REFERENCE_QUERIES,
    REFERENCE_SELECTION_PREAMBLE,
    REFERENCE_UNAVAILABLE,
    STEP_TITLES,
    SYSTEM_PROMPT,
)
from ..state import TutorState

logger = get_logger(__name__)


def reference(state: TutorState) -> dict:
    """Find a beginner-appropriate reference for the current step; stay on-step."""
    step = state["step"]
    query = REFERENCE_QUERIES[step]  # step-anchored + beginner-aimed (mechanism #1)
    logger.info("reference search step=%s query=%r", step, query)

    results = mcp_search.image_search(query)  # "" if disabled or all attempts fail
    if results:
        # f-string (not .format) so result text containing braces is harmless.
        content = (
            f"{REFERENCE_SELECTION_PREAMBLE}\n\n"
            f"The learner is on Step {step}: {STEP_TITLES[step]}.\n"
            f"Search query: {query}\n\nResults:\n{results}"
        )
    else:
        content = f"{REFERENCE_UNAVAILABLE}\n\nQuery they could try: {query}"

    reply = llm.generate(SYSTEM_PROMPT, [{"role": "user", "content": content}])
    return {"messages": [{"role": "assistant", "content": reply}]}
