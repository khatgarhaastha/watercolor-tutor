"""Tests for the vision feedback path: llm.see, the node, and the CLI parser."""

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from watercolor_tutor import llm
from watercolor_tutor.__main__ import _parse_image_message
from watercolor_tutor.nodes.vision_feedback import vision_feedback
from watercolor_tutor.state import TutorState

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

# vision_feedback now retrieves via RAG; stub it off so these tests stay offline.
# Grounding of the vision prompt is proven in test_grounding.py.
pytestmark = pytest.mark.usefixtures("stub_rag")


def _state(step: int, image_path: str) -> TutorState:
    return TutorState(
        messages=[{"role": "user", "content": "[photo]"}],
        step=step,
        awaiting_question=True,
        intent="",
        image_path=image_path,
    )


# --- llm.see: builds a valid multimodal request ------------------------------


def test_see_sends_image_before_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="looks smooth")])

    monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(model="claude-sonnet-4-6"))

    out = llm.see("system", "BASE64DATA", "image/png", "describe it")

    assert out == "looks smooth"
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"  # image FIRST (recommended ordering)
    assert content[1]["type"] == "text"  # text second
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "BASE64DATA"


# --- vision_feedback node ----------------------------------------------------


def test_vision_feedback_is_step_anchored_and_clears_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "wash.png"
    image.write_bytes(PNG_1X1)

    captured: dict = {}

    def fake_see(system: str, b64: str, media_type: str, prompt: str) -> str:
        captured["prompt"] = prompt
        captured["media_type"] = media_type
        return "Smooth overall, but a little streaky on the right edge."

    monkeypatch.setattr("watercolor_tutor.llm.see", fake_see)

    update = vision_feedback(_state(step=3, image_path=str(image)))

    assert update["messages"][0]["role"] == "assistant"
    assert "streaky" in update["messages"][0]["content"]
    assert update["image_path"] == ""  # consumed/cleared
    assert "step" not in update  # stays on the current step
    assert captured["media_type"] == "image/png"
    assert "wash" in captured["prompt"].lower()  # step-3 focus anchored the prompt


# --- CLI command parser ------------------------------------------------------


def test_parse_image_message_detects_paths_in_any_form(tmp_path: Path) -> None:
    image = tmp_path / "wash.png"
    image.write_bytes(PNG_1X1)
    p = str(image)

    # A bare pasted path, with no command word.
    assert _parse_image_message(p) == (p, "")
    # "feedback <path>" (no slash) — the original bug.
    assert _parse_image_message(f"feedback {p}") == (p, "")
    # "/feedback <path> <message>".
    assert _parse_image_message(f"/feedback {p} does this look smooth?") == (
        p,
        "does this look smooth?",
    )
    # A bare path with surrounding words still resolves to the image.
    assert _parse_image_message(f"{p} is this ok") == (p, "is this ok")


def test_parse_image_message_ignores_non_images(tmp_path: Path) -> None:
    assert _parse_image_message("what brush should I use?") is None
    # A path-shaped string that isn't a real file must NOT be treated as an image.
    assert _parse_image_message(str(tmp_path / "does-not-exist.png")) is None
