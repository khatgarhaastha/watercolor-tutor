"""Regression guards for the two bugs fixed in the prompt text."""

from watercolor_tutor.prompts import RESPONSE_INSTRUCTIONS, STEP_PROMPTS, SYSTEM_PROMPT, TOTAL_STEPS


def test_no_mid_conversation_completion_congratulation() -> None:
    """Bug 2: completion fires once at the real end (CLI), not on every step-3 turn.

    Neither the system prompt (sent on every call) nor the final step's teach prompt
    should instruct the model to congratulate on finishing.
    """
    assert "congratulat" not in SYSTEM_PROMPT.lower()
    assert "congratulat" not in STEP_PROMPTS[TOTAL_STEPS].lower()


def test_sharing_progress_does_not_invent_visuals() -> None:
    """Bug 1b: the respond path must not describe a painting it hasn't seen."""
    instruction = RESPONSE_INSTRUCTIONS["sharing_progress"].lower()
    assert "not" in instruction and "describe" in instruction
    assert "/feedback" in instruction  # points the learner to real vision feedback
