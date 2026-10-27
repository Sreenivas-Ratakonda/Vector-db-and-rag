"""
STEP 3 - RAG with a real vector database (ChromaDB, runs locally, no server)

    python rag/step3_rag_with_chromadb.py                       # interactive Q&A
    python rag/step3_rag_with_chromadb.py "How do I report phishing?"
    python rag/step3_rag_with_chromadb.py --source leave-policy.md "carry over days?"
    python rag/step3_rag_with_chromadb.py --reset               # rebuild the index

What the vector DB gives us over step 2:
  * Persistence  - vectors saved to ./chroma_db, no re-embedding on every run
  * ANN index    - HNSW graph: fast search even with millions of vectors
  * Metadata     - store source/title with each vector, filter on it (--source)
  * CRUD         - add / update / delete documents without rebuilding everything
"""
import argparse
import shutil
import time
from pathlib import Path

import chromadb

from common import banner, build_chunks, get_embedder, preview
from llm import generate

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION = "brewly_docs"
TOP_K = 3


def get_collection(reset: bool):
    if reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))       # a folder on disk
    embed = get_embedder()
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "embedder": embed.name},
    )
    if col.metadata.get("embedder") != embed.name:
        print("[info] Index was built with a different embedder - rebuilding.")
        client.delete_collection(COLLECTION)
        col = client.create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine", "embedder": embed.name})

    if col.count() == 0:                                         # ---- INGEST ----
        chunks = build_chunks(max_chars=350)
        t0 = time.perf_counter()
        col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embed([c.text for c in chunks]),
        )
        print(f"Indexed {len(chunks)} chunks into ChromaDB at {DB_DIR.name}/ "
              f"in {time.perf_counter() - t0:.2f}s")
    else:
        print(f"Loaded existing index: {col.count()} chunks (no re-embedding needed)")
    return col, embed


def ask(col, embed, question: str, source: str | None = None) -> None:
    where = {"source": source} if source else None               # metadata filter
    t0 = time.perf_counter()
    res = col.query(
        query_embeddings=embed([question]),
        n_results=TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ms = (time.perf_counter() - t0) * 1000

    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    banner(f"Question: {question}" + (f"   (filter: source={source})" if source else ""))
    print(f"Vector search took {ms:.1f} ms\n")
    for rank, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"#{rank}  similarity={1 - dist:.3f}  [{m['source']}]")
        print(f"     {preview(d)}")

    contexts = [{"text": d, "source": m["source"]} for d, m in zip(docs, metas)]
    provider, answer = generate(question, contexts)
    banner(f"Answer (LLM: {provider})")
    print(answer)


def main():
    p = argparse.ArgumentParser(description="RAG over data/ using ChromaDB")
    p.add_argument("question", nargs="*", help="question to ask (omit for interactive)")
    p.add_argument("--source", help="only search this file, e.g. leave-policy.md")
    p.add_argument("--reset", action="store_true", help="delete and rebuild the index")
    a = p.parse_args()

    col, embed = get_collection(a.reset)
    if a.question:
        ask(col, embed, " ".join(a.question), a.source)
        return
    print("\nInteractive mode. Type a question (or 'exit').")
    print("Try: 'Can I bring unused leave into next year?'  |  'beans in the fridge?'")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            ask(col, embed, q, a.source)


if __name__ == "__main__":
    main()
