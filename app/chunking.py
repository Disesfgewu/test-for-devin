"""Split documents into overlapping chunks suitable for retrieval."""
from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 150

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long(paragraph, chunk_size, overlap))
            continue
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = _tail(buffer, overlap) + "\n\n" + paragraph if overlap else paragraph
    if buffer:
        chunks.append(buffer)
    return [c.strip() for c in chunks if c.strip()]


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def _tail(text: str, overlap: int) -> str:
    return text[-overlap:] if len(text) > overlap else text
