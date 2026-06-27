"""LLM intent classifier (v1) — the "smarter router" brain.

This is where the keyword heuristic from v0 is replaced by an LLM that actually
understands the learner's reply, using STRUCTURED OUTPUT: we hand Claude a schema
and get back a validated object whose `intent` is guaranteed to be one of our
labels — no fragile JSON parsing, no out-of-enum surprises.

The classifier never hard-crashes routing: if the API call (or validation) fails,
we fall back to the v0 keyword heuristic so the lesson keeps moving.
"""

from typing import Literal

from pydantic import BaseModel, Field

from . import llm
from .logging_config import get_logger
from .prompts import INTENT_SYSTEM_PROMPT
from .routing import is_ready_signal

logger = get_logger(__name__)


class IntentResult(BaseModel):
    """The structured result the model must return.

    `reasoning` comes FIRST on purpose: the model fills fields in order, so it
    reasons briefly before committing to a label (a small chain-of-thought nudge)
    — and the sentence is invaluable when debugging why a reply was classified a
    certain way in a learning project.
    """

    reasoning: str = Field(description="One brief sentence explaining the choice.")
    intent: Literal[
        "question",  # asks something, not ready to move on
        "ready",  # wants to continue to the next step
        "both",  # asks a question AND wants to move on
        "confused",  # doesn't understand; wants the current step re-explained
        "skip_ahead",  # wants to jump forward a step
        "go_back",  # wants to return to an earlier step
        "off_topic",  # not about the lesson
        "sharing_progress",  # describes what they painted / how it's going
        "needs_web_info",  # wants current/external info (products to buy, prices, links)
        "needs_reference_image",  # wants to SEE an example/reference of the technique
    ] = Field(description="The single best label for the learner's latest reply.")


def classify_intent(text: str) -> str:
    """Classify the learner's reply as 'question' | 'ready' | 'both'.

    Falls back to the v0 keyword heuristic if the structured call fails for any
    reason — routing must never crash on a bad/slow API response. (Note the
    fallback can only produce 'question'/'ready'; 'both' needs the LLM.)
    """
    # Future optimization — "route cheap, teach strong": classification is a great
    # fit for a smaller/cheaper model (e.g. Haiku 4.5, which also supports
    # structured outputs). For now we use the configured model to keep one setting.
    try:
        result = llm.parse(INTENT_SYSTEM_PROMPT, [{"role": "user", "content": text}], IntentResult)
        logger.info("classified intent=%s reasoning=%r", result.intent, result.reasoning)
        return result.intent
    except Exception as exc:  # never let a bad API call break routing
        logger.warning("intent classifier failed (%s); using keyword fallback", exc)
        return "ready" if is_ready_signal(text) else "question"
