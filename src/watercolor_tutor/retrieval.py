"""Local semantic retrieval (RAG) over the curated corpus — FAISS-backed.

PRODUCTION-SHAPE NOTE (read this first): our corpus is 4 short docs. In-memory
cosine was entirely sufficient, and this FAISS + transformer-embedding backend is
NOT required for correctness. We use it deliberately to rehearse the scalable,
production RAG architecture (index once, query many) behind the SAME `retrieve()`
interface the nodes already call — so swapping the backend touched zero node code.

Split of responsibilities:
  - ingest.py          : OFFLINE — read docs, chunk, embed, WRITE the index to disk.
  - retrieval.py (here): QUERY TIME — LOAD the prebuilt index, embed the query,
    SEARCH for the top-k. It never re-reads or re-embeds the corpus.

How the search works: chunk vectors are L2-normalized, so FAISS's inner-product
index (`IndexFlatIP`) ranks by cosine similarity. `IndexFlat` is exact (brute
force) — right at this scale; a large corpus would use an approximate index
(IVF/HNSW) that searches without comparing against every vector.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langsmith import traceable
from sentence_transformers import SentenceTransformer

from .logging_config import get_logger

logger = get_logger(__name__)

# Small, standard sentence-transformer (384-dim). Downloaded once, then cached.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunks from 00-*.md are shared — eligible for retrieval on every step.
SHARED_STEP = 0

# Where the prebuilt index lives: inside the package for discoverability, and
# gitignored (it's a derived artifact, rebuilt by `python -m watercolor_tutor.ingest`).
# A read-only deployment would override INDEX_DIR to an external writable path.
INDEX_DIR = Path(__file__).resolve().parent / "knowledge" / "_index"
INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"


@dataclass(frozen=True)
class Chunk:
    """One retrievable section of a doc, tagged with the step it belongs to."""

    text: str
    step: int


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    """Load (and cache) the embedding model. First call downloads it; then cached."""
    logger.info("loading embedding model %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts as L2-normalized float32 vectors (cosine-ready for IndexFlatIP)."""
    vectors = _embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=1)
def _load_index() -> tuple[Any, list[Chunk]]:
    """Load the prebuilt FAISS index + chunk metadata from disk (cached).

    By design we do NOT build the index lazily here — a missing index raises a
    clear error, keeping the index-once / query-many split explicit.
    """
    index_path = INDEX_DIR / INDEX_FILE
    meta_path = INDEX_DIR / META_FILE
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"RAG index not found in {INDEX_DIR}. Build it once with:\n"
            "    python -m watercolor_tutor.ingest"
        )
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [Chunk(text=entry["text"], step=entry["step"]) for entry in metadata]
    return index, chunks


@traceable  # shows up as a RAG-retrieval run in LangSmith (no-op when tracing off)
def retrieve(step: int, query: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant chunk texts for `query`, scoped to `step`.

    The step SCOPES the candidates: only this step's chunks plus the shared (00)
    chunks count. We search the whole (tiny) index, then filter the ranked results
    by step. NOTE: at scale you'd push the step filter INTO the search (FAISS
    `IDSelector`, or a metadata-filtering store) rather than search-all-then-filter.
    """
    index, chunks = _load_index()
    query_vec = _embed([query])
    _, ranked_idx = index.search(query_vec, len(chunks))  # rank every chunk (exact)

    results: list[str] = []
    for idx in ranked_idx[0]:
        chunk = chunks[int(idx)]
        if chunk.step in (step, SHARED_STEP):
            results.append(chunk.text)
            if len(results) == k:
                break
    return results


GROUNDING_PREAMBLE = (
    "Reference notes from the studio handbook — base your explanation on these and "
    "cite the specific numbers, ratios, and named techniques; do not contradict them:"
)


def grounding_for(step: int, query: str, k: int = 3) -> str:
    """Retrieve step-relevant notes, formatted as a grounding block for a prompt.

    Returns "" when nothing is found, so callers can append it unconditionally.
    """
    notes = retrieve(step, query, k)
    if not notes:
        return ""
    return GROUNDING_PREAMBLE + "\n\n" + "\n\n---\n\n".join(notes)
