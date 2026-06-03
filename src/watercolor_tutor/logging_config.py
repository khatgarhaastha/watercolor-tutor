"""Structured logging setup, configured in exactly one place.

We emit logs as `key=value` pairs so they stay greppable in a terminal but can
also be parsed by log tooling. Call `configure_logging()` once at startup.
"""

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call multiple times (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    # Structured, single-line records: timestamp, level, logger name, message.
    handler.setFormatter(
        logging.Formatter(
            fmt="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Prefer this over the root logger."""
    return logging.getLogger(name)
