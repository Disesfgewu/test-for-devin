# RAG Docs QA

Upload PDF/TXT/Markdown documents, index them into a vector store, and ask
questions that are answered from the indexed content with citations.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000.

## Configuration

All settings are environment variables; every one of them is optional.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Enables LLM-generated answers and OpenAI embeddings |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model used for answering |
| `EMBEDDING_BACKEND` | `auto` | `openai`, `local`, or `tfidf` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `INDEX_PATH` | `data/index.json` | Where the index is persisted |
| `MAX_UPLOAD_BYTES` | `20971520` | Upload size limit |

Backend selection with `EMBEDDING_BACKEND=auto`: OpenAI if a key is present,
otherwise a local sentence-transformers model, otherwise TF-IDF. The TF-IDF
fallback needs no key and no model download, so the app works fully offline.

Local semantic embeddings need the optional extra:

```bash
pip install -r requirements-local-embeddings.txt
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/status` | Active backends and indexed documents |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Upload and index a file (multipart `file`) |
| `DELETE` | `/api/documents/{id}` | Remove a document and its chunks |
| `POST` | `/api/ask` | `{"question": "...", "top_k": 4}` → answer + citations |

## Tests

```bash
pytest
```
