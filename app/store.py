"""In-process vector store with JSON persistence."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .embeddings import CorpusEmbedder, Embedder


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    position: int
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder, persist_path: Path | None = None) -> None:
        self._embedder = embedder
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._load()

    @property
    def backend(self) -> str:
        return self._embedder.name

    @property
    def _corpus_scoped(self) -> bool:
        return isinstance(self._embedder, CorpusEmbedder)

    def documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for chunk in self._chunks:
            entry = seen.setdefault(
                chunk.document_id,
                {"id": chunk.document_id, "name": chunk.document_name, "chunks": 0},
            )
            entry["chunks"] += 1
        return list(seen.values())

    def add_document(self, name: str, chunks: list[str]) -> dict:
        if not chunks:
            raise ValueError("Document produced no text to index")

        document_id = uuid.uuid4().hex
        new_chunks = [
            Chunk(
                id=uuid.uuid4().hex,
                document_id=document_id,
                document_name=name,
                position=position,
                text=text,
            )
            for position, text in enumerate(chunks)
        ]

        with self._lock:
            vectors = self._embedder.embed_documents([c.text for c in new_chunks])
            self._chunks.extend(new_chunks)
            if self._corpus_scoped:
                # These embedders re-fit on every document, so the returned
                # matrix already covers the whole corpus.
                self._vectors = vectors
            elif self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self._save()

        return {"id": document_id, "name": name, "chunks": len(new_chunks)}

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
            if len(keep) == len(self._chunks):
                return False
            self._chunks = [self._chunks[i] for i in keep]
            if self._corpus_scoped:
                self._rebuild_corpus()
            elif self._vectors is not None:
                self._vectors = self._vectors[keep] if keep else None
            self._save()
            return True

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        with self._lock:
            if self._vectors is None or not self._chunks:
                return []
            query_vector = self._embedder.embed_query(query)
            scores = self._vectors @ query_vector
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(chunk=self._chunks[i], score=float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]

    def _rebuild_corpus(self) -> None:
        """Re-fit a corpus-scoped embedder on the surviving chunks."""
        self._embedder.reset()
        self._vectors = (
            self._embedder.embed_documents([c.text for c in self._chunks])
            if self._chunks
            else None
        )

    def _save(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "identity": self._embedder.identity,
            "chunks": [asdict(c) for c in self._chunks],
            "vectors": self._vectors.tolist() if self._vectors is not None else None,
        }
        self._persist_path.write_text(json.dumps(payload))

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        payload = json.loads(self._persist_path.read_text())
        if payload.get("identity") != self._embedder.identity:
            # Vectors from another model or backend are not comparable.
            return
        self._chunks = [Chunk(**c) for c in payload.get("chunks", [])]
        if self._corpus_scoped:
            self._rebuild_corpus()
            return
        vectors = payload.get("vectors")
        self._vectors = np.array(vectors, dtype=np.float32) if vectors else None
