"""Thin wrapper around the Anthropic client.

Centralizing client construction here means nodes never import `anthropic`
directly — they call into this module, which makes them trivial to stub in tests.
The client is built lazily so importing this module never requires an API key.
"""

from functools import lru_cache

from anthropic import Anthropic

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


@lru_cache
def get_client() -> Anthropic:
    """Return a cached Anthropic client built from configured settings."""
    settings = get_settings()
    logger.debug("constructing Anthropic client model=%s", settings.model)
    return Anthropic(api_key=settings.anthropic_api_key)
