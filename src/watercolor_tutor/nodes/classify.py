"""The classify node — runs the LLM intent classifier and records the result.

This node exists so the LLM call (the "do") is isolated and the routers stay
pure (the "decide"): a conditional edge can only RETURN a destination, it can't
remember the intent for a later node. By writing `intent` to state here, BOTH
routers — the one after classify and the one after answer — can read it, which
is what lets the "both" case answer the question AND then advance.
"""

from .. import classifier
from ..logging_config import get_logger
from ..routing import last_user_message
from ..state import TutorState

logger = get_logger(__name__)


def classify(state: TutorState) -> dict:
    """Classify the learner's latest reply and store the intent in state."""
    text = last_user_message(state) or ""
    intent = classifier.classify_intent(text)
    logger.info("classify node: intent=%s step=%s", intent, state["step"])
    return {"intent": intent}
