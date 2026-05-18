import uuid


def test_empty_document_body(client):
    resp = client.post("/documents", json={"document_text": ""})
    assert resp.status_code == 400
    assert resp.json()["error"]["category"] == "validation_error"


def test_document_too_short(client):
    resp = client.post("/documents", json={"document_text": "short"})
    assert resp.status_code == 400
    assert "too short" in resp.json()["error"]["message"]


def test_document_too_large(client):
    text = "x" * 100001
    resp = client.post("/documents", json={"document_text": text})
    assert resp.status_code == 413
    assert resp.json()["error"]["category"] == "validation_error"


def test_missing_document_text(client):
    resp = client.post("/documents", json={})
    assert resp.status_code == 422


def test_empty_question(client, sample_document_text):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]
    resp = client.post(f"/documents/{doc_id}/questions", json={"question": ""})
    assert resp.status_code == 400


def test_question_too_long(client, sample_document_text):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]
    resp = client.post(f"/documents/{doc_id}/questions", json={"question": "x" * 2001})
    assert resp.status_code == 400


def test_unknown_document_id(client):
    fake_id = str(uuid.uuid4())
    resp = client.post(f"/documents/{fake_id}/questions", json={"question": "hello?"})
    assert resp.status_code == 404
    assert resp.json()["error"]["category"] == "document_not_found"
