"""Offline ingestion: build the FAISS index from the knowledge corpus.

INDEX-ONCE / QUERY-MANY: this script reads the docs, chunks them, embeds the
chunks, and WRITES the FAISS index + metadata to disk. `retrieval.py` then just
LOADS and searches that index — it never re-reads or re-embeds the corpus at query
time. That decoupling (an offline indexing job vs. cheap online queries) is the
production-shape part we're rehearsing.

Honest note: 4 short docs do NOT need FAISS or transformer embeddings — in-memory
cosine was plenty. We build this offline step deliberately to demonstrate the
scalable RAG architecture behind the unchanged `retrieve()` seam.

Run once (rebuild whenever the corpus changes):

    python -m watercolor_tutor.ingest
"""

import json
import re
from pathlib import Path

import faiss

from .logging_config import configure_logging, get_logger
from .retrieval import INDEX_DIR, INDEX_FILE, META_FILE, Chunk, _embed

logger = get_logger(__name__)

# The corpus lives next to this module, in the `knowledge/` package directory.
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _split_sections(markdown: str) -> list[str]:
    """Split a doc into chunks on `## ` headings (each heading kept with its body).

    The text before the first `## ` (the `# Title`) carries no technique content,
    so it's dropped.
    """
    parts = re.split(r"(?m)^(?=## )", markdown)
    return [p.strip() for p in parts if p.strip().startswith("## ")]


def _load_chunks() -> list[Chunk]:
    """Read every doc in knowledge/, chunk it, and tag each chunk with its step."""
    chunks: list[Chunk] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        step = int(path.name[:2])  # "03-flat-wash.md" -> 3 ; "00-*.md" -> shared
        for section in _split_sections(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(text=section, step=step))
    return chunks


def build_index(index_dir: Path = INDEX_DIR) -> None:
    """Chunk + embed the corpus and write the FAISS index + metadata to `index_dir`."""
    chunks = _load_chunks()
    logger.info("embedding %d chunks", len(chunks))
    vectors = _embed([chunk.text for chunk in chunks])  # (n, 384) normalized float32

    # Inner-product over L2-normalized vectors == cosine similarity. IndexFlat is
    # exact brute force — fine at this scale (a large corpus would use IVF/HNSW).
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    metadata = [{"text": chunk.text, "step": chunk.step} for chunk in chunks]
    (index_dir / META_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("wrote index (%d vectors) to %s", index.ntotal, index_dir)


def main() -> None:
    """Entry point for `python -m watercolor_tutor.ingest`."""
    configure_logging("INFO")
    build_index()


if __name__ == "__main__":
    main()
