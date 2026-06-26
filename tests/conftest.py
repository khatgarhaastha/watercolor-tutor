"""Shared pytest fixtures.

Tests must NOT hit the real Anthropic API. As LLM-backed nodes arrive, inject a
fake client via the `fake_llm` fixture below rather than calling out.
"""

import pytest

from watercolor_tutor.state import TutorState


@pytest.fixture
def initial_state() -> TutorState:
    """A fresh, empty tutor state as the graph would be invoked with."""
    return TutorState(messages=[], step=0, awaiting_question=False, intent="", image_path="")


@pytest.fixture
def stub_grounding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable RAG grounding so flow/node tests stay fully offline.

    Real retrieval loads the embedding model (a one-time download); flow tests
    that only exercise control flow don't need it. Tests that PROVE grounding
    (test_knowledge, test_grounding) deliberately don't use this fixture.
    """
    monkeypatch.setattr("watercolor_tutor.retrieval.grounding_for", lambda *a, **k: "")


@pytest.fixture
def fake_llm():
    """A stand-in for the Anthropic client.

    Returns a canned response object shaped like the real SDK's, so node tests
    stay fast and offline. Extend `text` per-test as needed.
    """

    class _FakeContentBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.content = [_FakeContentBlock(text)]

    class _FakeMessages:
        def create(self, *args, **kwargs) -> _FakeResponse:
            return _FakeResponse("fake lesson text")

    class _FakeClient:
        messages = _FakeMessages()

    return _FakeClient()
