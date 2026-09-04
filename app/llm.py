"""Answer generation on top of retrieved context.

With OPENAI_API_KEY set, answers come from an OpenAI-compatible chat endpoint.
Without it the app stays fully usable: it returns the retrieved passages as an
extractive answer so the retrieval half can be exercised offline.
"""
from __future__ import annotations

import os

import httpx

from .store import SearchResult

SYSTEM_PROMPT = (
    "You answer questions strictly from the provided context passages. "
    "Cite the passages you used as [1], [2], ... matching their numbering. "
    "If the context does not contain the answer, say so plainly."
)


def llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, results: list[SearchResult]) -> tuple[str, bool]:
    """Return the answer text and whether it was produced by an LLM."""
    if not results:
        return ("No indexed document matches this question yet.", False)
    if not llm_enabled():
        return (_extractive_answer(results), False)
    return (_llm_answer(question, results), True)


def _format_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{index}] ({result.chunk.document_name}) {result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )


def _extractive_answer(results: list[SearchResult]) -> str:
    passages = "\n\n".join(
        f"[{index}] {result.chunk.text.strip()}"
        for index, result in enumerate(results, start=1)
    )
    return (
        "No LLM configured (set OPENAI_API_KEY to get a synthesized answer). "
        "Most relevant passages:\n\n" + passages
    )


def _llm_answer(question: str, results: list[SearchResult]) -> str:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{_format_context(results)}\n\nQuestion: {question}",
                },
            ],
        },
        timeout=90.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
