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


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Construction is deferred to first call (not import time) so that tests which
    inject a fake LLM never need a real API key just to import a module.
    """
    return Settings()  # type: ignore[call-arg]  # values come from the environment
