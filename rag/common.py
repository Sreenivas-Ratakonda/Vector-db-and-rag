"""
Shared helpers for the demos: loading documents, chunking, and embeddings.

Embeddings
----------
By default we use the small, free, local model that ChromaDB ships with
(all-MiniLM-L6-v2 running on ONNX, 384 dimensions). It downloads once
(~80 MB) on first use and then works offline. No API key, no GPU.

If that model can't be loaded (no internet on first run, corporate proxy,
etc.) we fall back to a tiny "hashing" embedder so the demos still run.
The fallback only matches on shared words, NOT meaning, which makes it a
nice illustration of why real embedding models matter.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "be", "to", "of", "i", "my", "do", "does",
    "how", "what", "can", "for", "in", "on", "and", "or", "it", "if", "me", "you",
    "your", "we", "our", "with", "at", "by", "from", "this", "that", "should", "want",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------
# 1. Documents and chunking
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Read every .md / .txt file in data/ -> {filename: text}."""
    docs = {}
    for p in sorted(data_dir.glob("*")):
        if p.suffix.lower() in {".md", ".txt"}:
            docs[p.name] = p.read_text(encoding="utf-8")
    if not docs:
        raise SystemExit(f"No .md/.txt files found in {data_dir}")
    return docs


def chunk_text(text: str, max_chars: int = 350) -> list[str]:
    """
    Split a document into chunks of roughly `max_chars`.

    Strategy: split on blank lines (paragraphs / sections), then pack
    paragraphs together until the chunk would get too big. Simple, and it
    keeps related sentences together, which is what retrieval needs.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(max_chars: int = 350) -> list[Chunk]:
    """Load all documents and turn them into Chunk objects with metadata."""
    chunks = []
    for name, text in load_documents().items():
        title = text.splitlines()[0].lstrip("# ").strip()
        for i, piece in enumerate(chunk_text(text, max_chars)):
            chunks.append(
                Chunk(
                    id=f"{name}#{i}",
                    text=piece,
                    metadata={"source": name, "title": title, "chunk": i},
                )
            )
    return chunks


# --------------------------------------------------------------------------
# 2. Embeddings
# --------------------------------------------------------------------------
class HashingEmbedder:
    """
    Fallback embedder: words -> hashed buckets -> normalised vector.

    It is a "bag of words" vector, so "refund" and "money back" are NOT
    similar here. Real embedding models fix exactly that.
    """

    name = "hashing-bag-of-words (fallback, keyword-level only)"
    dim = 384

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in STOPWORDS:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class MiniLMEmbedder:
    """Chroma's built-in local model: all-MiniLM-L6-v2 (ONNX, 384 dims)."""

    name = "all-MiniLM-L6-v2 (local ONNX model via ChromaDB)"
    dim = 384

    def __init__(self):
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()
        self._fn(["warm-up"])  # triggers the one-time download now

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


_EMBEDDER = None


def get_embedder():
    """Return the best available embedder (cached)."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if os.getenv("RAG_EMBEDDER", "").lower() == "hashing":
        _EMBEDDER = HashingEmbedder()
    else:
        try:
            print("Loading embedding model (first run downloads ~80 MB)...")
            _EMBEDDER = MiniLMEmbedder()
        except Exception as e:  # offline, proxy, missing package...
            print(f"[warn] Could not load MiniLM model ({type(e).__name__}).")
            print("[warn] Falling back to a keyword-level hashing embedder.\n")
            _EMBEDDER = HashingEmbedder()
    print(f"Embedder: {_EMBEDDER.name}\n")
    return _EMBEDDER


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


# --------------------------------------------------------------------------
# 3. Pretty printing
# --------------------------------------------------------------------------
def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def preview(text: str, n: int = 110) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= n else one_line[: n - 3] + "..."


def keyword_score(query: str, text: str) -> float:
    """Classic keyword overlap score (what a LIKE/full-text search does)."""
    q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOPWORDS}
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q & t) / math.sqrt(len(q) or 1)
