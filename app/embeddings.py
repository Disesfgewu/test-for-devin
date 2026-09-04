"""Embedding backends.

Three backends are supported, picked automatically unless EMBEDDING_BACKEND is set:

* ``openai``     - OpenAI-compatible embedding endpoint (needs OPENAI_API_KEY)
* ``local``      - sentence-transformers model running on this machine
* ``tfidf``      - pure scikit-learn fallback, no model download and no API key
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class UnembeddableContent(Exception):
    """Raised when a document holds no content the backend can represent."""


class Embedder(Protocol):
    name: str

    @property
    def identity(self) -> str:
        """Identifies the vector space, so persisted vectors can be validated."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class CorpusEmbedder(Protocol):
    """An embedder whose vector space depends on every document seen so far."""

    def reset(self) -> None: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        return f"openai:{self._base_url}:{self._model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return normalize(np.array([item["embedding"] for item in data], dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class LocalEmbedder:
    name = "local"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def identity(self) -> str:
        return f"local:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class TfidfEmbedder:
    """Fallback embedder: keeps a vectorizer fitted on everything seen so far.

    The vocabulary changes with every document, so vectors are only comparable
    when they all come from the same fit. ``embed_documents`` therefore returns
    vectors for the *whole* corpus, and callers must replace their matrix
    rather than append to it.
    """

    name = "tfidf"
    refits_whole_corpus = True

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._corpus: list[str] = []
        self._fitted = False

    @property
    def identity(self) -> str:
        return "tfidf"

    def reset(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._corpus = []
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        previous_corpus = list(self._corpus)
        self._corpus.extend(texts)
        try:
            self._vectorizer.fit(self._corpus)
        except ValueError as exc:
            self._corpus = previous_corpus
            if previous_corpus:
                self._vectorizer.fit(previous_corpus)
            raise UnembeddableContent(
                "The document contains no indexable words."
            ) from exc
        self._fitted = True
        return normalize(self._vectorizer.transform(self._corpus).toarray().astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("No documents indexed yet")
        return normalize(self._vectorizer.transform([text]).toarray().astype(np.float32))[0]


def build_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    api_key = os.getenv("OPENAI_API_KEY")

    if backend in {"auto", "openai"} and api_key:
        return OpenAIEmbedder(
            api_key=api_key,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    if backend == "openai" and not api_key:
        raise RuntimeError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY")

    if backend in {"auto", "local"}:
        try:
            return LocalEmbedder(os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        except Exception:
            if backend == "local":
                raise

    return TfidfEmbedder()
