"""Test for the classify node (offline; classifier stubbed)."""

import pytest

from watercolor_tutor import classifier
from watercolor_tutor.nodes.classify import classify
from watercolor_tutor.state import TutorState


def test_classify_node_writes_intent_to_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classifier, "classify_intent", lambda text: "both")
    state = TutorState(
        messages=[{"role": "user", "content": "let's move on, what should I paint?"}],
        step=1,
        awaiting_question=True,
        intent="",
        image_path="",
    )

    update = classify(state)

    # The node's only job: record the classifier's verdict in state.
    assert update == {"intent": "both"}
