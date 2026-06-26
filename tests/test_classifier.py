"""Tests for the LLM intent classifier and its keyword fallback (offline).

We stub `llm.parse` so no real structured-output call is made: returning an
IntentResult exercises the happy path, raising exercises the fallback.
"""

import pytest

from watercolor_tutor import classifier, llm
from watercolor_tutor.classifier import IntentResult


def test_classify_intent_returns_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "parse", lambda *a, **k: IntentResult(reasoning="asks a question", intent="question")
    )
    assert classifier.classify_intent("what brush should I use?") == "question"


def test_classify_intent_returns_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "parse", lambda *a, **k: IntentResult(reasoning="asks and moves on", intent="both")
    )
    assert classifier.classify_intent("sounds good, but what color first?") == "both"


def test_classify_intent_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> IntentResult:
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(llm, "parse", boom)
    # Fallback uses the keyword heuristic: ready-signal -> "ready", else "question".
    assert classifier.classify_intent("I'm ready!") == "ready"
    assert classifier.classify_intent("what paper do I need?") == "question"
