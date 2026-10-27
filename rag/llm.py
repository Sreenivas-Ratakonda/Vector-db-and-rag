"""
The "G" in RAG: turn (question + retrieved context) into an answer.

Pick a provider with the RAG_LLM environment variable, or let it auto-detect:

    RAG_LLM=ollama     -> local Ollama (free, offline)  e.g. `ollama pull llama3.2`
    RAG_LLM=anthropic  -> needs ANTHROPIC_API_KEY
    RAG_LLM=openai     -> needs OPENAI_API_KEY
    RAG_LLM=none       -> no LLM: shows the retrieved context + best sentences

Auto-detect order: ANTHROPIC_API_KEY -> OPENAI_API_KEY -> Ollama running -> none.
Uses only the Python standard library (no SDKs to install).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

PROMPT_TEMPLATE = """You are a helpful assistant for Brewly Coffee Co. employees.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Cite the source file names you used in square brackets, e.g. [refund-policy.md].

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, contexts: list[dict]) -> str:
    """contexts: [{'text': ..., 'source': ...}, ...]"""
    blocks = [f"[{c['source']}]\n{c['text']}" for c in contexts]
    return PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(blocks), question=question)


# --------------------------------------------------------------------------
def _post(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(_ollama_url() + "/api/tags", timeout=1)
        return True
    except Exception:
        return False


def _ollama_url() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _anthropic(prompt: str) -> str:
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    r = _post(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 500,
         "messages": [{"role": "user", "content": prompt}]},
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
         "anthropic-version": "2023-06-01"},
    )
    return "".join(b.get("text", "") for b in r["content"])


def _openai(prompt: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    r = _post(
        "https://api.openai.com/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}]},
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )
    return r["choices"][0]["message"]["content"]


def _ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    r = _post(
        _ollama_url() + "/api/generate",
        {"model": model, "prompt": prompt, "stream": False},
        {},
        timeout=300,
    )
    return r["response"]


def _no_llm(question: str, contexts: list[dict]) -> str:
    """
    No LLM available: return the sentences from the retrieved chunks that
    overlap most with the question. Retrieval still did the hard work.
    """
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    best = []
    for c in contexts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            s = s.strip(" -*#")
            if len(s) < 25:
                continue
            score = len(q & set(re.findall(r"[a-z0-9]+", s.lower())))
            best.append((score, s, c["source"]))
    best.sort(key=lambda x: -x[0])
    lines = [f"- {s} [{src}]" for _, s, src in best[:3]]
    return ("(No LLM configured: showing the most relevant sentences "
            "from the retrieved context.)\n" + "\n".join(lines))


def pick_provider() -> str:
    choice = os.getenv("RAG_LLM", "").lower()
    if choice:
        return choice
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if _ollama_up():
        return "ollama"
    return "none"


def generate(question: str, contexts: list[dict]) -> tuple[str, str]:
    """Returns (provider_used, answer)."""
    provider = pick_provider()
    if provider == "none":
        return provider, _no_llm(question, contexts)
    fns = {"anthropic": _anthropic, "openai": _openai, "ollama": _ollama}
    if provider not in fns:
        raise SystemExit(f"Unknown RAG_LLM={provider!r}. Use ollama|anthropic|openai|none.")
    prompt = build_prompt(question, contexts)
    try:
        return provider, fns[provider](prompt).strip()
    except (urllib.error.URLError, KeyError, TimeoutError, OSError) as e:
        print(f"[warn] {provider} call failed: {e}. Falling back to no-LLM mode.")
        return "none", _no_llm(question, contexts)
