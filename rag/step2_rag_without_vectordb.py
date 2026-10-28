"""
STEP 2 - RAG with NO vector database (vectors in a plain Python list / numpy array)

    python rag/step2_rag_without_vectordb.py
    python rag/step2_rag_without_vectordb.py "How many sick days do I get?"

The whole RAG pipeline in ~40 lines:
    load docs -> chunk -> embed -> (store in memory) -> embed question
    -> brute-force similarity over every chunk -> top-k -> prompt -> LLM

For a few hundred / few thousand chunks this is perfectly fine.
It stops being fine when you need: millions of vectors, persistence,
metadata filters, updates, concurrency -> that's what a vector DB adds (step 3).
"""
import sys
import time

import numpy as np

from common import banner, build_chunks, get_embedder, preview
from llm import generate

TOP_K = 3
question = " ".join(sys.argv[1:]) or "What should I do if my coffee machine arrives broken?"

# 1-2. Load + chunk ----------------------------------------------------------
chunks = build_chunks(max_chars=350)
print(f"Loaded {len(chunks)} chunks from data/")

# 3. Embed every chunk, keep them in RAM -------------------------------------
embed = get_embedder()
t0 = time.perf_counter()
matrix = np.array(embed([c.text for c in chunks]), dtype=np.float32)   # shape (N, 384)
matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)                 # normalise once
print(f"Embedded into a {matrix.shape[0]} x {matrix.shape[1]} matrix "
      f"in {time.perf_counter() - t0:.2f}s  (this IS our 'vector store')")

# 4. Embed the question and compare against EVERY chunk ----------------------
q = np.array(embed([question])[0], dtype=np.float32)
q /= np.linalg.norm(q)
t0 = time.perf_counter()
scores = matrix @ q                         # cosine similarity for all chunks at once
top = np.argsort(-scores)[:TOP_K]
search_ms = (time.perf_counter() - t0) * 1000

banner(f"Question: {question}")
print(f"Brute-force search over {len(chunks)} vectors took {search_ms:.3f} ms\n")
for rank, i in enumerate(top, 1):
    c = chunks[i]
    print(f"#{rank}  score={scores[i]:.3f}  [{c.metadata['source']}]")
    print(f"     {preview(c.text)}")

# 5. Augment + Generate ------------------------------------------------------
contexts = [{"text": chunks[i].text, "source": chunks[i].metadata["source"]} for i in top]
provider, answer = generate(question, contexts)
banner(f"Answer (LLM: {provider})")
print(answer)
