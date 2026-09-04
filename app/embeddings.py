"""Embedding backends.

Three backends are supported, picked automatically unless EMBEDDING_BACKEND is set:

* ``openai``     - OpenAI-compatible embedding endpoint (needs OPENAI_API_KEY)
* ``local``      - sentence-transformers model running on this machine
* ``tfidf``      - pure scikit-learn fallback, no model download and no API key
"""
from __future__ import annotations

import os
from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


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

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class TfidfEmbedder:
    """Fallback embedder: keeps a vectorizer fitted on everything seen so far."""

    name = "tfidf"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._corpus: list[str] = []
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._corpus.extend(texts)
        self._vectorizer.fit(self._corpus)
        self._fitted = True
        return normalize(self._vectorizer.transform(self._corpus).toarray().astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("No documents indexed yet")
        return normalize(self._vectorizer.transform([text]).toarray().astype(np.float32))[0]

    @property
    def refits_whole_corpus(self) -> bool:
        return True


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
