"""Thin wrapper around the Anthropic client.

Centralizing client construction here means nodes never import `anthropic`
directly — they call into this module, which makes them trivial to stub in tests.
The client is built lazily so importing this module never requires an API key.
"""

from functools import lru_cache
from typing import TypeVar, cast

from anthropic import Anthropic
from anthropic.types import MessageParam
from pydantic import BaseModel

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)

# A few sentences per reply is plenty for a beginner lesson; cap tokens so a
# runaway response can't surprise us on cost or latency.
MAX_TOKENS = 1024

# Classification is tiny (a short reasoning sentence + a label), so it needs far
# fewer tokens than a lesson.
PARSE_MAX_TOKENS = 512

# A schema type, so parse() returns the exact model you pass in (typed result).
T = TypeVar("T", bound=BaseModel)


@lru_cache
def get_client() -> Anthropic:
    """Return a cached Anthropic client built from configured settings."""
    settings = get_settings()
    logger.debug("constructing Anthropic client model=%s", settings.model)
    return Anthropic(api_key=settings.anthropic_api_key)


def generate(system: str, messages: list[dict[str, str]]) -> str:
    """Send one request to Claude and return its text reply.

    Nodes call this instead of touching the Anthropic SDK directly — that keeps
    nodes provider-agnostic (plain `{"role", "content"}` dicts) and trivial to
    stub in tests. The SDK-specific type lives only here, behind a cast.

    Note: the Anthropic API requires the FIRST message to use the "user" role.
    """
    client = get_client()
    settings = get_settings()
    logger.debug("calling model=%s messages=%d", settings.model, len(messages))

    response = client.messages.create(
        model=settings.model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=cast(list[MessageParam], messages),
    )

    # A response is a list of content blocks; concatenate the text ones into a
    # single string. (For our text-only tutor there's normally just one.)
    return "".join(getattr(block, "text", "") for block in response.content)


def parse(system: str, messages: list[dict[str, str]], schema: type[T]) -> T:
    """Call Claude with STRUCTURED OUTPUT and return a validated `schema` instance.

    `messages.parse()` constrains the model's decoding to the given Pydantic
    schema, so the response is guaranteed to match it — no fragile JSON parsing,
    no chance of an out-of-enum value. `parsed_output` is already a validated
    instance of `schema` (or None on a safety refusal, which we treat as failure
    so the caller can fall back).
    """
    client = get_client()
    settings = get_settings()
    logger.debug("structured call model=%s schema=%s", settings.model, schema.__name__)

    parsed = client.messages.parse(
        model=settings.model,
        max_tokens=PARSE_MAX_TOKENS,
        system=system,
        messages=cast(list[MessageParam], messages),
        output_format=schema,
    ).parsed_output

    if parsed is None:
        raise ValueError("structured output returned no parsed result")
    return parsed
