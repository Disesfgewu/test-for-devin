"""Extract plain text from uploaded documents."""
from __future__ import annotations

import io

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class UnsupportedFileType(Exception):
    pass


def extract_text(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return _extract_pdf(content)
    if any(lowered.endswith(ext) for ext in (".txt", ".md", ".markdown")):
        return content.decode("utf-8", errors="replace")
    raise UnsupportedFileType(
        f"Unsupported file type: {filename}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {index}]\n{text}")
    return "\n\n".join(pages)
