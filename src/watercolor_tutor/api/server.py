"""FastAPI app — an HTTP entry point that wraps the same compiled graph as the CLI.

THE RESHAPE (see conversation.py for the shared mechanics): the terminal app is a
blocking interrupt()/resume loop; HTTP is request -> response. So one HTTP request =
one trip around that loop. Each request opens the SQLite checkpointer, loads the
conversation for its thread_id (the session id the client holds), resumes the paused
graph to the next interrupt(), and returns what the tutor produced. All continuity
lives in SQLite keyed by thread_id, so the server keeps NO per-session memory between
requests — it's genuinely stateless, which is exactly what HTTP wants.

Two choices verified for langgraph 1.2.x + a local SQLite checkpointer:

  - Endpoints are sync `def`, so FastAPI runs them in its threadpool. The MCP nodes
    call asyncio.run() internally; on an `async def` endpoint (which runs on the main
    event loop) that would raise "asyncio.run() cannot be called from a running event
    loop". A threadpool thread has no running loop, so the existing sync graph works.

  - The SQLite connection is opened PER REQUEST (SqliteSaver.from_conn_string), not
    shared, to avoid SQLite's "objects created in a thread can only be used in that
    same thread" error across threadpool workers. The compiled graph is stateless and
    cheap to rebuild; the heavy bits (embedding model, Anthropic client) are cached
    process-wide. This is right for a LOCAL, low-concurrency demo; production would
    move to a lifespan-managed async/Postgres saver — a config seam, not a rewrite.

The frontend (static HTML/JS/CSS in ./static) is served by this SAME app at "/", so
the UI and API share an origin — no CORS — and a single `uvicorn` command runs both.
"""

import contextlib
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from .. import conversation, images
from ..config import Settings, get_settings
from ..graph import compile_graph
from ..logging_config import configure_logging, get_logger
from ..observability import setup_tracing
from ..prompts import TOTAL_STEPS
from .schemas import (
    CreateSessionRequest,
    Message,
    MessageRequest,
    SessionResponse,
    SessionSummary,
)

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def slugify(name: str) -> str:
    """Turn a display name into a stable thread_id ('Aastha K' -> 'aastha-k').

    Lowercase, non-alphanumerics collapsed to single hyphens, trimmed. Deterministic
    so the SAME name always maps to the SAME session (that's how resume-by-name
    works). Falls back to 'learner' when nothing usable remains.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "learner"


def _config(thread_id: str) -> RunnableConfig:
    """The run config that ties a graph invocation to a session's saved state."""
    return {"configurable": {"thread_id": thread_id}}


@contextlib.contextmanager
def _open_graph(db_path: str) -> Iterator[tuple[CompiledStateGraph, SqliteSaver]]:
    """Open a per-request SQLite checkpointer and compile the graph against it.

    Yields the graph AND the saver (the saver is needed to enumerate threads for the
    session list). The `with` block owns the connection for exactly one request, then
    releases it — see the module docstring for why per-request beats a shared one.
    """
    with SqliteSaver.from_conn_string(db_path) as saver:
        saver.setup()  # create tables on first use (idempotent)
        yield compile_graph(saver), saver


def _save_upload(file: UploadFile, uploads_dir: Path) -> str:
    """Persist an uploaded photo to the uploads dir and return its path.

    Rejects unsupported types up front (the HTTP analog of the CLI's path guard).
    The saved path is later handed to vision_feedback as image_path — so an upload
    always routes to the REAL vision node, never the text 'respond' path. That's how
    the never-confabulate guarantee holds by construction for uploads, too.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in images.SUPPORTED_MEDIA_TYPES:
        raise HTTPException(
            status_code=400, detail="Unsupported image type. Use jpg, png, webp, or gif."
        )
    destination = uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(file.file.read())
    return str(destination)


def _require_active(status: str) -> None:
    """Translate a non-resumable session into the right HTTP error."""
    if status == "absent":
        raise HTTPException(
            status_code=404, detail="No such session. Create one with POST /sessions."
        )
    if status == "complete":
        raise HTTPException(status_code=409, detail="This lesson is already complete.")


def _status_of(snapshot: object) -> str:
    """'awaiting' while paused at the interrupt, 'complete' once the graph ended."""
    return "awaiting" if snapshot.next else "complete"  # type: ignore[attr-defined]


def _full_response(
    thread_id: str, graph: CompiledStateGraph, config: RunnableConfig
) -> SessionResponse:
    """Standard response carrying the FULL history — for start/resume of a session."""
    snapshot = graph.get_state(config)
    values = snapshot.values
    return SessionResponse(
        thread_id=thread_id,
        name=values.get("name", ""),
        step=values["step"],
        total_steps=TOTAL_STEPS,
        status=_status_of(snapshot),
        messages=[Message(**m) for m in conversation.history(values["messages"])],
    )


def _turn_response(
    thread_id: str, graph: CompiledStateGraph, config: RunnableConfig, new_texts: list[str]
) -> SessionResponse:
    """Standard response carrying only the tutor's NEW messages — for a single turn."""
    snapshot = graph.get_state(config)
    values = snapshot.values
    return SessionResponse(
        thread_id=thread_id,
        name=values.get("name", ""),
        step=values["step"],
        total_steps=TOTAL_STEPS,
        status=_status_of(snapshot),
        messages=[Message(role="assistant", content=text) for text in new_texts],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    A factory (not a module-level app) so tests can inject Settings pointing at a
    temp DB / uploads dir without touching real ones. `settings`, `db_path`, and
    `uploads_dir` are captured in the endpoint closures below.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    uploads_dir = Path(settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Enable LangSmith tracing once at startup (no-op without a key), exactly as
        # the CLI does — RAG, MCP, and persistence are otherwise unchanged behind us.
        tracing = setup_tracing()
        logger.info(
            "watercolor-tutor API up db=%s uploads=%s tracing=%s",
            settings.db_path,
            settings.uploads_dir,
            tracing,
        )
        yield

    app = FastAPI(title="Watercolor Tutor API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.get("/sessions", response_model=list[SessionSummary])
    def list_sessions() -> list[SessionSummary]:
        """List existing sessions for the picker (newest first)."""
        summaries: list[SessionSummary] = []
        seen: set[str] = set()
        with _open_graph(settings.db_path) as (graph, saver):
            # saver.list(None) yields every checkpoint across all threads, newest
            # first; we keep the first (latest) sighting of each distinct thread.
            for checkpoint in saver.list(None):
                thread_id = checkpoint.config["configurable"]["thread_id"]
                if thread_id in seen:
                    continue
                seen.add(thread_id)
                values = graph.get_state(_config(thread_id)).values
                if not values:
                    continue
                summaries.append(
                    SessionSummary(
                        thread_id=thread_id,
                        name=values.get("name", "") or thread_id,
                        step=values["step"],
                        status=_status_of(graph.get_state(_config(thread_id))),
                    )
                )
        return summaries

    @app.post("/sessions", response_model=SessionResponse)
    def create_or_resume_session(request: CreateSessionRequest | None = None) -> SessionResponse:
        """Enter the tutor as a name: resume that session if it exists, else start fresh.

        The name slugifies to the thread_id, so re-entering the same name continues
        the same lesson. Returns the FULL conversation either way, so the UI just
        renders whatever comes back.
        """
        request = request or CreateSessionRequest()
        name = (request.name or "").strip() or "Learner"
        thread_id = slugify(name)
        config = _config(thread_id)
        with _open_graph(settings.db_path) as (graph, _saver):
            if conversation.session_status(graph, config) == "absent":
                conversation.start_lesson(graph, config, name)
            return _full_response(thread_id, graph, config)

    @app.get("/sessions/{thread_id}", response_model=SessionResponse)
    def get_session(thread_id: str) -> SessionResponse:
        """Return the full history + current standing — lets a UI rehydrate a session."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as (graph, _saver):
            if not graph.get_state(config).values:
                raise HTTPException(status_code=404, detail="No such session.")
            return _full_response(thread_id, graph, config)

    @app.post("/sessions/{thread_id}/messages", response_model=SessionResponse)
    def send_message(thread_id: str, request: MessageRequest) -> SessionResponse:
        """Send one learner turn of text; resume the graph to the next pause."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as (graph, _saver):
            _require_active(conversation.session_status(graph, config))
            _, _, new = conversation.resume_turn(graph, config, request.text)
            return _turn_response(thread_id, graph, config, new)

    @app.post("/sessions/{thread_id}/feedback", response_model=SessionResponse)
    def send_feedback(thread_id: str, file: UploadFile, text: str = Form("")) -> SessionResponse:
        """Upload a painting photo for REAL vision feedback on the current step."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as (graph, _saver):
            _require_active(conversation.session_status(graph, config))
            path = _save_upload(file, uploads_dir)
            try:
                images.load_image(path)  # validate (size/type) before sending to the model
            except (ValueError, OSError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Couldn't use that image: {exc}"
                ) from exc
            # The dict resume routes straight to vision_feedback (never 'respond').
            resume = {"text": text, "image_path": path}
            _, _, new = conversation.resume_turn(graph, config, resume)
            return _turn_response(thread_id, graph, config, new)

    # Serve the static frontend at "/" LAST, so the API routes above take precedence.
    # app.frontend (FastAPI >=0.138) falls back to index.html for unknown PAGES while
    # still 404-ing missing assets — and never shadows a real API route.
    app.frontend("/", directory=str(_STATIC_DIR))

    return app
