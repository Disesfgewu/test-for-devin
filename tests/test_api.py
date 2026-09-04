import importlib

import pytest
from fastapi.testclient import TestClient

DOC = b"""Vacation policy

Full-time employees accrue 20 days of paid vacation per year.

Expense policy

Meals during business travel are reimbursed up to 50 USD per day.
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main = importlib.reload(importlib.import_module("app.main"))
    with TestClient(main.app) as test_client:
        yield test_client


def test_status_reports_backend(client):
    body = client.get("/api/status").json()
    assert body["embedding_backend"] == "tfidf"
    assert body["llm_enabled"] is False
    assert body["documents"] == []


def test_upload_search_and_delete(client):
    upload = client.post("/api/documents", files={"file": ("policy.md", DOC, "text/markdown")})
    assert upload.status_code == 201
    document_id = upload.json()["id"]
    assert upload.json()["chunks"] >= 1

    answer = client.post("/api/ask", json={"question": "How many vacation days?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["llm_generated"] is False
    assert body["citations"]
    assert "vacation" in body["citations"][0]["excerpt"].lower()

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_rejects_unsupported_file(client):
    response = client.post("/api/documents", files={"file": ("a.docx", b"data", "application/msword")})
    assert response.status_code == 415


def test_ask_without_documents(client):
    body = client.post("/api/ask", json={"question": "anything"}).json()
    assert body["citations"] == []
