import pytest

from app.chunking import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_paragraphs_are_packed_under_chunk_size():
    text = "\n\n".join(["a" * 100 for _ in range(10)])
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_long_paragraph_is_split_with_overlap():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=100)
