"""Proof that RAG grounding reaches the model — REAL retrieval, stubbed LLM.

We stub `llm.generate` to capture the prompt the node would send, then assert it
contains corpus-only specifics. That proves the retrieved technique notes are
actually injected into the teaching prompt (not invented by the model).
"""

import base64
from pathlib import Path

import pytest

from watercolor_tutor.nodes.reexplain import reexplain
from watercolor_tutor.nodes.teach import teach
from watercolor_tutor.nodes.vision_feedback import vision_feedback
from watercolor_tutor.state import TutorState

# Grounding runs the REAL retriever, so the index must be built first.
pytestmark = pytest.mark.usefixtures("rag_index")

# Specifics that exist only in the corpus, so finding them in the prompt proves
# they came from retrieval.
_CORPUS_SPECIFICS = ("1:3", "15 degrees", "bead", "backrun", "wet edge")

# Diagnostic terms (fault → cause → fix) the vision critique should be able to draw
# on — present in the corpus, not generic praise.
_CORPUS_DIAGNOSTICS = ("backrun", "streak", "banding", "underloaded", "wet edge", "bloom")

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _state(step: int) -> TutorState:
    return TutorState(
        messages=[{"role": "user", "content": "..."}],
        step=step,
        awaiting_question=True,
        intent="",
        image_path="",
    )


def _capture_prompt(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["prompt"] = messages[0]["content"]
        return "lesson"

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)
    return captured


def test_teach_prompt_is_grounded_in_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_prompt(monkeypatch)
    teach(_state(step=3))
    prompt = captured["prompt"]
    assert "Reference notes" in prompt  # the grounding block was attached
    assert any(s in prompt for s in _CORPUS_SPECIFICS), prompt


def test_reexplain_prompt_is_grounded_in_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_prompt(monkeypatch)
    reexplain(_state(step=3))
    prompt = captured["prompt"]
    assert "Reference notes" in prompt
    assert any(s in prompt for s in _CORPUS_SPECIFICS), prompt


def test_vision_feedback_prompt_is_grounded_in_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Capture the text the vision call would send (image goes first, text second).
    captured: dict = {}

    def fake_see(system: str, image_b64: str, media_type: str, prompt: str) -> str:
        captured["prompt"] = prompt
        return "feedback"

    monkeypatch.setattr("watercolor_tutor.llm.see", fake_see)

    image = tmp_path / "wash.png"
    image.write_bytes(_PNG_1X1)
    state = _state(step=3)
    state["image_path"] = str(image)

    vision_feedback(state)

    prompt = captured["prompt"].lower()
    assert "diagnostics" in prompt  # the grounding block is present
    assert any(term in prompt for term in _CORPUS_DIAGNOSTICS), prompt
