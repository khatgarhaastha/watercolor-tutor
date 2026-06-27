"""Tests for the optional, key-gated LangSmith tracing setup (offline, no network).

`load_dotenv` is stubbed so we never read the real .env, and `monkeypatch` env
changes auto-restore — so these tests neither enable global tracing nor touch the
network. setup_tracing() only sets env vars; it makes no LangSmith calls.
"""

import os

import pytest

from watercolor_tutor import observability


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't read the real .env (it may hold a real LANGSMITH key)."""
    monkeypatch.setattr(observability, "load_dotenv", lambda *a, **k: None)


def test_tracing_off_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert observability.setup_tracing() is False


def test_tracing_on_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    assert observability.setup_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "watercolor-tutor"


def test_explicit_tracing_false_is_a_killswitch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A key is present, but the operator explicitly turned tracing off.
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert observability.setup_tracing() is False
