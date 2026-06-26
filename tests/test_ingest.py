"""Tests for the offline ingestion step (the index-once half of the split)."""

import json
from pathlib import Path

import faiss
import pytest

from watercolor_tutor import ingest, retrieval
from watercolor_tutor.retrieval import INDEX_FILE, META_FILE


def test_build_index_writes_artifacts(tmp_path: Path) -> None:
    """Ingestion writes a FAISS index + aligned metadata, one vector per chunk."""
    ingest.build_index(tmp_path)

    index_path = tmp_path / INDEX_FILE
    meta_path = tmp_path / META_FILE
    assert index_path.exists()
    assert meta_path.exists()

    metadata = json.loads(meta_path.read_text())
    assert len(metadata) == len(ingest._load_chunks())
    assert all("text" in entry and "step" in entry for entry in metadata)

    index = faiss.read_index(str(index_path))
    assert index.ntotal == len(metadata)  # exactly one vector per chunk


def test_load_index_errors_clearly_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Query time does NOT build lazily — a missing index points you at ingest."""
    monkeypatch.setattr(retrieval, "INDEX_DIR", tmp_path / "absent")
    retrieval._load_index.cache_clear()
    with pytest.raises(FileNotFoundError, match="ingest"):
        retrieval._load_index()
    retrieval._load_index.cache_clear()  # don't leak the empty-dir state to other tests
