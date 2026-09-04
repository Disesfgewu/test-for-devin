"""FastAPI application exposing document ingestion and RAG question answering."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .embeddings import UnembeddableContent, build_embedder
from .llm import generate_answer, llm_enabled
from .loaders import DocumentTooLarge, UnsupportedFileType, extract_text
from .store import VectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DATA_PATH = Path(os.getenv("INDEX_PATH", BASE_DIR / "data" / "index.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
READ_CHUNK_BYTES = 1024 * 1024
API_TOKEN = os.getenv("API_TOKEN")


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Guard the API when API_TOKEN is configured; a no-op otherwise."""
    if not API_TOKEN:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


app = FastAPI(title="RAG Docs QA", version="1.0.0")
api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])
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


@api.get("/status")
def status() -> dict:
    return {
        "embedding_backend": store.backend,
        "llm_enabled": llm_enabled(),
        "auth_required": bool(API_TOKEN),
        "documents": store.documents(),
    }


@api.get("/documents")
def list_documents() -> list[dict]:
    return store.documents()


@api.post("/documents", status_code=201)
async def upload_document(file: UploadFile) -> dict:
    content = await _read_within_limit(file)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return await run_in_threadpool(_index_document, file.filename or "document", content)


def _index_document(filename: str, content: bytes) -> dict:
    try:
        text = extract_text(filename, content)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except DocumentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the document")

    try:
        return store.add_document(name=filename, chunks=chunks)
    except UnembeddableContent as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _read_within_limit(file: UploadFile) -> bytes:
    buffer = bytearray()
    while chunk := await file.read(READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the size limit")
    return bytes(buffer)


@api.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> Response:
    if not store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@api.post("/ask", response_model=AskResponse)
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


app.include_router(api)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
