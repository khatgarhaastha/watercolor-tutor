"""Local semantic retrieval (RAG) over the curated corpus in `knowledge/`.

The pipeline, end to end:
  1. CHUNK   — load each markdown doc and split it on `## ` headings; tag each
               chunk with the STEP from its filename (00-* = shared, eligible for
               every step).
  2. EMBED   — turn every chunk into a vector with a small LOCAL static-embedding
               model (model2vec). "Static embeddings" = each token has a
               precomputed vector and encoding just averages them, so it's fast,
               deterministic, needs no GPU/torch, and runs offline. The ~30 MB
               model is downloaded once from the HuggingFace hub, then cached.
  3. RETRIEVE — embed the query, cosine-rank the current step's candidate chunks,
               return the top-k. Cosine similarity = how aligned two vectors are
               (1.0 = same direction/meaning, 0 = unrelated).

`retrieve()` is the swappable seam: only `_model()` knows about model2vec, so a
different backend (Voyage, a vector DB) could replace it without touching callers.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from model2vec import StaticModel

from .logging_config import get_logger

logger = get_logger(__name__)

# Tiny static-embedding model (~30 MB, numpy-only inference). Downloaded once.
EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Chunks from 00-*.md are shared — eligible for retrieval on every step.
SHARED_STEP = 0


@dataclass(frozen=True)
class Chunk:
    """One retrievable section of a doc, tagged with the step it belongs to."""

    text: str
    step: int


@lru_cache(maxsize=1)
def _model() -> StaticModel:
    """Load (and cache) the embedding model. First call downloads it; then cached."""
    logger.info("loading embedding model %s", EMBEDDING_MODEL)
    return StaticModel.from_pretrained(EMBEDDING_MODEL)


def _split_sections(markdown: str) -> list[str]:
    """Split a doc into chunks on `## ` headings (each heading kept with its body).

    The text before the first `## ` (the `# Title`) carries no technique content,
    so it's dropped.
    """
    parts = re.split(r"(?m)^(?=## )", markdown)
    return [p.strip() for p in parts if p.strip().startswith("## ")]


@lru_cache(maxsize=1)
def _chunks() -> tuple[Chunk, ...]:
    """Load + chunk every doc in knowledge/, tagging each chunk with its step."""
    chunks: list[Chunk] = []
    knowledge_dir = resources.files("watercolor_tutor.knowledge")
    for entry in sorted(knowledge_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".md"):
            continue
        step = int(entry.name[:2])  # "03-flat-wash.md" -> 3 ; "00-*.md" -> SHARED_STEP
        for section in _split_sections(entry.read_text(encoding="utf-8")):
            chunks.append(Chunk(text=section, step=step))
    logger.info("loaded %d knowledge chunks", len(chunks))
    return tuple(chunks)


@lru_cache(maxsize=1)
def _chunk_matrix() -> np.ndarray:
    """Embed all chunks once; row i is the vector for _chunks()[i]."""
    return _model().encode([chunk.text for chunk in _chunks()])


def _cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and each row of `matrix`."""
    query_unit = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    row_units = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return row_units @ query_unit


def retrieve(step: int, query: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant chunk texts for `query`, scoped to `step`.

    The step SCOPES the candidates: only this step's chunks plus the shared (00)
    chunks are considered, so Step 3 can't surface Step 1's materials notes.
    """
    chunks = _chunks()
    candidate_idx = [i for i, chunk in enumerate(chunks) if chunk.step in (step, SHARED_STEP)]
    if not candidate_idx:
        return []

    query_vec = _model().encode([query])[0]
    scores = _cosine(query_vec, _chunk_matrix()[candidate_idx])
    ranked = sorted(zip(candidate_idx, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [chunks[i].text for i, _ in ranked[:k]]


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
