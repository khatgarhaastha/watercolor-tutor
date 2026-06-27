"""Offline tests for the FastAPI layer.

Fully offline, like the rest of the suite: the LLM (`generate`/`classify_intent`/
`see`) and RAG (`stub_rag`) are stubbed, and the checkpointer is a real SQLite file
in `tmp_path`. We drive the app through Starlette's TestClient — real request/response
round-trips, no network, no live model.
"""

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from watercolor_tutor.api.server import create_app
from watercolor_tutor.config import Settings

# A 1x1 PNG — the smallest valid image, enough for images.load_image to accept it.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

pytestmark = pytest.mark.usefixtures("stub_rag")  # keep teaching/vision retrieval offline


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every model call with deterministic, offline stand-ins."""
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *a, **k: "LESSON TEXT")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", lambda *a, **k: "ready")
    monkeypatch.setattr("watercolor_tutor.llm.see", lambda *a, **k: "Smooth wash — nice even tone.")


@pytest.fixture
def client(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient whose DB + uploads live under tmp_path (isolated per test)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # Settings requires the key field
    monkeypatch.setenv("WATERCOLOR_DB_PATH", str(tmp_path / "sessions.sqlite"))  # type: ignore[operator]
    monkeypatch.setenv("WATERCOLOR_UPLOADS_DIR", str(tmp_path / "uploads"))  # type: ignore[operator]
    # Keep tests fully offline: don't let the app's lifespan enable real LangSmith
    # tracing from a developer's .env (which would ship test traces to the cloud).
    monkeypatch.setattr("watercolor_tutor.api.server.setup_tracing", lambda: False)
    app = create_app(Settings())  # type: ignore[call-arg]  # values come from the env above
    with TestClient(app) as test_client:
        yield test_client


def _new_session(client: TestClient) -> str:
    """Create a session and return its thread_id."""
    response = client.post("/sessions")
    assert response.status_code == 200
    return response.json()["thread_id"]


# --- basics ------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_session_starts_the_lesson(client: TestClient) -> None:
    body = client.post("/sessions").json()

    assert body["thread_id"]
    assert body["step"] == 1  # welcome -> teach lands on step 1
    assert body["status"] == "awaiting"  # paused for the learner
    # Opening = the welcome message + the (stubbed) Step 1 lesson, both assistant.
    assert len(body["messages"]) == 2
    assert all(m["role"] == "assistant" for m in body["messages"])
    assert any("LESSON TEXT" in m["content"] for m in body["messages"])


# --- the core resume-per-request cycle ---------------------------------------


def test_message_advances_and_returns_only_new_messages(client: TestClient) -> None:
    thread_id = _new_session(client)

    body = client.post(f"/sessions/{thread_id}/messages", json={"text": "ready"}).json()

    assert body["step"] == 2  # 'ready' (stubbed) advanced to step 2
    assert body["status"] == "awaiting"
    # Only THIS turn's tutor message comes back — not the whole history.
    assert body["messages"] == [{"role": "assistant", "content": "LESSON TEXT"}]


def test_get_session_returns_full_history_with_roles(client: TestClient) -> None:
    thread_id = _new_session(client)
    client.post(f"/sessions/{thread_id}/messages", json={"text": "ready"})

    body = client.get(f"/sessions/{thread_id}").json()

    assert body["step"] == 2
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles and "assistant" in roles  # full convo, both sides
    assert any(m["content"] == "ready" for m in body["messages"])  # the learner's turn


# --- image upload routes to REAL vision (never confabulates) ------------------


def test_feedback_upload_routes_to_vision(client: TestClient) -> None:
    thread_id = _new_session(client)

    response = client.post(
        f"/sessions/{thread_id}/feedback",
        files={"file": ("wash.png", PNG_1X1, "image/png")},
        data={"text": "is this smooth?"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["step"] == 1  # vision feedback stays on the current step
    # The reply is the vision node's output (stubbed `see`), proving it went through
    # vision_feedback — not the text 'respond' path that would invent praise.
    assert body["messages"] == [{"role": "assistant", "content": "Smooth wash — nice even tone."}]


def test_feedback_rejects_non_image(client: TestClient) -> None:
    thread_id = _new_session(client)

    response = client.post(
        f"/sessions/{thread_id}/feedback",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


# --- error paths -------------------------------------------------------------


def test_message_to_unknown_session_is_404(client: TestClient) -> None:
    response = client.post("/sessions/does-not-exist/messages", json={"text": "hi"})
    assert response.status_code == 404


def test_messaging_a_completed_lesson_is_409(client: TestClient) -> None:
    thread_id = _new_session(client)
    # Three 'ready' turns walk steps 1 -> 2 -> 3 -> END (lesson complete).
    for _ in range(3):
        client.post(f"/sessions/{thread_id}/messages", json={"text": "ready"})

    assert client.get(f"/sessions/{thread_id}").json()["status"] == "complete"
    response = client.post(f"/sessions/{thread_id}/messages", json={"text": "ready"})
    assert response.status_code == 409
