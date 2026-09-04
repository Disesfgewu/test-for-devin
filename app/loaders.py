"""Extract plain text from uploaded documents."""
from __future__ import annotations

import io
import os

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 500))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", 2_000_000))


class UnsupportedFileType(Exception):
    pass


class DocumentTooLarge(Exception):
    pass


def extract_text(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return _extract_pdf(content)
    if any(lowered.endswith(ext) for ext in (".txt", ".md", ".markdown")):
        return _cap(content.decode("utf-8", errors="replace"))
    raise UnsupportedFileType(
        f"Unsupported file type: {filename}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _cap(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentTooLarge(
            f"Document text exceeds the {MAX_TEXT_CHARS} character limit"
        )
    return text


def _extract_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
    except Exception as exc:
        raise UnsupportedFileType("The PDF could not be parsed") from exc

    if page_count > MAX_PDF_PAGES:
        raise DocumentTooLarge(f"The PDF has {page_count} pages, limit is {MAX_PDF_PAGES}")

    pages: list[str] = []
    total = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        if not text.strip():
            continue
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise DocumentTooLarge(
                f"Extracted text exceeds the {MAX_TEXT_CHARS} character limit"
            )
        pages.append(f"[page {index}]\n{text}")
    return "\n\n".join(pages)
