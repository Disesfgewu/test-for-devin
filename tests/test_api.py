import importlib

import pytest
from fastapi.testclient import TestClient

VACATION = b"""Vacation policy

Full-time employees accrue 20 days of paid vacation per year.
"""

EXPENSES = b"""Expense policy

Meals during business travel are reimbursed up to 50 USD per day.
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)
    main = importlib.reload(importlib.import_module("app.main"))
    with TestClient(main.app) as test_client:
        yield test_client


def upload(client, name, content):
    response = client.post("/api/documents", files={"file": (name, content, "text/markdown")})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_status_reports_backend(client):
    body = client.get("/api/status").json()
    assert body["embedding_backend"] == "tfidf"
    assert body["llm_enabled"] is False
    assert body["auth_required"] is False
    assert body["documents"] == []


def test_upload_search_and_delete(client):
    document_id = upload(client, "policy.md", VACATION)

    body = client.post("/api/ask", json={"question": "How many vacation days?"}).json()
    assert body["llm_generated"] is False
    assert "vacation" in body["citations"][0]["excerpt"].lower()

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_surviving_documents_stay_citable_after_delete_then_upload(client):
    stale_id = upload(client, "stale.md", b"Parking policy\n\nParking passes cost 10 USD.\n")
    upload(client, "vacation.md", VACATION)
    assert client.delete(f"/api/documents/{stale_id}").status_code == 204
    upload(client, "expenses.md", EXPENSES)

    body = client.post("/api/ask", json={"question": "How many vacation days?"}).json()
    names = [citation["document_name"] for citation in body["citations"]]
    assert "vacation.md" in names
    assert "stale.md" not in names
    top = body["citations"][0]
    assert top["document_name"] == "vacation.md"
    assert "vacation" in top["excerpt"].lower()


def test_deleting_last_document_resets_the_index(client):
    document_id = upload(client, "policy.md", VACATION)
    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    upload(client, "expenses.md", EXPENSES)

    body = client.post("/api/ask", json={"question": "Meals reimbursed during travel"}).json()
    assert [c["document_name"] for c in body["citations"]] == ["expenses.md"]


def test_document_without_indexable_words_is_rejected(client):
    response = client.post("/api/documents", files={"file": ("noise.txt", b"!!! ??? ...", "text/plain")})
    assert response.status_code == 422
    assert client.get("/api/documents").json() == []

    # The rejected document must not poison later uploads.
    upload(client, "policy.md", VACATION)
    body = client.post("/api/ask", json={"question": "vacation"}).json()
    assert body["citations"][0]["document_name"] == "policy.md"


def test_rejects_unsupported_file(client):
    response = client.post("/api/documents", files={"file": ("a.docx", b"data", "application/msword")})
    assert response.status_code == 415


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 32)
    response = client.post("/api/documents", files={"file": ("big.txt", b"x" * 1024, "text/plain")})
    assert response.status_code == 413


def test_ask_without_documents(client):
    body = client.post("/api/ask", json={"question": "anything"}).json()
    assert body["citations"] == []


def test_api_token_guards_the_api_but_not_the_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("API_TOKEN", "s3cret")
    main = importlib.reload(importlib.import_module("app.main"))
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        authorized = client.get("/api/status", headers={"Authorization": "Bearer s3cret"})
        assert authorized.status_code == 200
        assert authorized.json()["auth_required"] is True
