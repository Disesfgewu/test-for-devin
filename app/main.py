"""FastAPI application exposing document ingestion and RAG question answering."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .embeddings import build_embedder
from .llm import generate_answer, llm_enabled
from .loaders import UnsupportedFileType, extract_text
from .store import VectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DATA_PATH = Path(os.getenv("INDEX_PATH", BASE_DIR / "data" / "index.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

app = FastAPI(title="RAG Docs QA", version="1.0.0")
store = VectorStore(embedder=build_embedder(), persist_path=DATA_PATH)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class Citation(BaseModel):
    document_name: str
    position: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    llm_generated: bool
    citations: list[Citation]


@app.get("/api/status")
def status() -> dict:
    return {
        "embedding_backend": store.backend,
        "llm_enabled": llm_enabled(),
        "documents": store.documents(),
    }


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return store.documents()


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the size limit")

    try:
        text = extract_text(file.filename or "document", content)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the document")

    return store.add_document(name=file.filename or "document", chunks=chunks)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> Response:
    if not store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    results = store.search(request.question, top_k=request.top_k)
    answer, llm_generated = generate_answer(request.question, results)
    return AskResponse(
        answer=answer,
        llm_generated=llm_generated,
        citations=[
            Citation(
                document_name=result.chunk.document_name,
                position=result.chunk.position,
                score=round(result.score, 4),
                excerpt=result.chunk.text[:400],
            )
            for result in results
        ],
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
