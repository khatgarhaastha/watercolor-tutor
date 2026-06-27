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
"""

import contextlib
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
from .schemas import CreateSessionRequest, Message, MessageRequest, SessionResponse

logger = get_logger(__name__)


def _config(thread_id: str) -> RunnableConfig:
    """The run config that ties a graph invocation to a session's saved state."""
    return {"configurable": {"thread_id": thread_id}}


@contextlib.contextmanager
def _open_graph(db_path: str) -> Iterator[CompiledStateGraph]:
    """Open a per-request SQLite checkpointer and compile the graph against it.

    The `with` block owns the connection for exactly one request, then releases it —
    see the module docstring for why per-request beats a shared connection here.
    """
    with SqliteSaver.from_conn_string(db_path) as saver:
        saver.setup()  # create tables on first use (idempotent)
        yield compile_graph(saver)


def _require_active(status: str) -> None:
    """Translate a non-resumable session into the right HTTP error."""
    if status == "absent":
        raise HTTPException(
            status_code=404, detail="No such session. Create one with POST /sessions."
        )
    if status == "complete":
        raise HTTPException(status_code=409, detail="This lesson is already complete.")


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


def _assistant_response(
    thread_id: str, step: int, status: str, texts: list[str]
) -> SessionResponse:
    """Wrap the tutor's new messages from a turn into the standard response shape."""
    return SessionResponse(
        thread_id=thread_id,
        step=step,
        status=status,
        messages=[Message(role="assistant", content=text) for text in texts],
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

    @app.post("/sessions", response_model=SessionResponse)
    def create_session(request: CreateSessionRequest | None = None) -> SessionResponse:
        """Start a brand-new lesson and return its opening (welcome + Step 1)."""
        request = request or CreateSessionRequest()
        label = request.session_id or "session"
        # thread_id IS the session identity. A random suffix guarantees a fresh
        # session every call (no accidental restart of an existing one).
        thread_id = f"{label}-{uuid.uuid4().hex[:8]}"
        with _open_graph(settings.db_path) as graph:
            step, status, new = conversation.start_lesson(graph, _config(thread_id))
        return _assistant_response(thread_id, step, status, new)

    @app.get("/sessions/{thread_id}", response_model=SessionResponse)
    def get_session(thread_id: str) -> SessionResponse:
        """Return the full history + current standing — lets a UI rehydrate a session."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as graph:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise HTTPException(status_code=404, detail="No such session.")
            status = "awaiting" if snapshot.next else "complete"
            return SessionResponse(
                thread_id=thread_id,
                step=snapshot.values["step"],
                status=status,
                messages=[Message(**m) for m in conversation.history(snapshot.values["messages"])],
            )

    @app.post("/sessions/{thread_id}/messages", response_model=SessionResponse)
    def send_message(thread_id: str, request: MessageRequest) -> SessionResponse:
        """Send one learner turn of text; resume the graph to the next pause."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as graph:
            _require_active(conversation.session_status(graph, config))
            step, status, new = conversation.resume_turn(graph, config, request.text)
        return _assistant_response(thread_id, step, status, new)

    @app.post("/sessions/{thread_id}/feedback", response_model=SessionResponse)
    def send_feedback(thread_id: str, file: UploadFile, text: str = Form("")) -> SessionResponse:
        """Upload a painting photo for REAL vision feedback on the current step."""
        config = _config(thread_id)
        with _open_graph(settings.db_path) as graph:
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
            step, status, new = conversation.resume_turn(graph, config, resume)
        return _assistant_response(thread_id, step, status, new)

    return app
