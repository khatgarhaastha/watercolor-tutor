"""Application configuration — the ONLY place that reads environment variables.

Everything else in the codebase asks `get_settings()` for values instead of
touching `os.environ` directly. This keeps secret handling in one auditable spot.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from the environment (and `.env` for local development)."""

    # `extra="ignore"` so unrelated env vars don't blow up startup.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The Anthropic API key. Required for live runs; never hard-code it.
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")

    # Which Claude model the tutor uses. Sonnet is a good speed/cost default.
    model: str = Field("claude-sonnet-4-6", alias="WATERCOLOR_MODEL")

    # Logging verbosity (DEBUG/INFO/WARNING/ERROR).
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Where durable session state (the LangGraph SQLite checkpointer) is stored.
    # User/runtime data, not source — gitignored. Delete the file for a full reset.
    db_path: str = Field("watercolor_tutor.sqlite", alias="WATERCOLOR_DB_PATH")

    # Where the web API saves uploaded painting photos before sending them to the
    # vision model. Runtime data, not source — gitignored. Kept on disk (not a
    # tempdir) so an uploaded image is easy to inspect while debugging a demo.
    uploads_dir: str = Field("uploads", alias="WATERCOLOR_UPLOADS_DIR")

    # Live web search via an external MCP server (v2 Slice 3b-1). Set False to turn
    # the tool off entirely. The command launches the (swappable) search server.
    web_search_enabled: bool = Field(True, alias="WATERCOLOR_WEB_SEARCH")
    mcp_search_command: str = Field("duckduckgo-mcp-server", alias="WATERCOLOR_MCP_SEARCH_COMMAND")

    # Reference-image search via a SECOND (image) MCP server (v2 Slice 3b-2).
    image_search_enabled: bool = Field(True, alias="WATERCOLOR_IMAGE_SEARCH")
    mcp_image_command: str = Field("ddg-mcp", alias="WATERCOLOR_MCP_IMAGE_COMMAND")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Construction is deferred to first call (not import time) so that tests which
    inject a fake LLM never need a real API key just to import a module.
    """
    return Settings()  # type: ignore[call-arg]  # values come from the environment
