def test_successful_ingestion(client, sample_document_text):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    assert resp.status_code == 201
    data = resp.json()
    assert "document_id" in data
    assert data["chunks_created"] > 0
    assert data["limits"]["max_document_chars"] == 100000


def test_pii_redacted_in_response(client, sample_document_text):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    data = resp.json()
    assert data["pii"]["redactions_applied"] > 0
    assert "email" in data["pii"]["types_detected"]


def test_chunk_hashes_format(client, sample_document_text):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]

    from app.store import document_store

    chunks = document_store.get_document(doc_id)
    for chunk in chunks:
        assert chunk.chunk_hash.startswith("sha256:")
        assert len(chunk.chunk_hash) == 71  # sha256: + 64 hex chars
