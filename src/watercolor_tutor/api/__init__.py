"""Web API package wrapping the tutor graph over HTTP.

Run it locally with:

    uvicorn watercolor_tutor.api:app --reload      # interactive docs at /docs

`app` is built LAZILY (PEP 562 module __getattr__) on first access, not at import
time, so importing this package in tests doesn't require real settings/secrets —
tests call `create_app(test_settings)` with their own temp DB instead. (The factory
lives in `server.py`, NOT `app.py`, so the submodule name doesn't shadow the lazy
`app` attribute that `uvicorn watercolor_tutor.api:app` resolves.)
"""

from typing import Any

from .server import create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
