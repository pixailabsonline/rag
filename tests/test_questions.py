from unittest.mock import patch, AsyncMock
import sys


def test_answered_with_sources(client, sample_document_text, mock_llm_tool_then_answer):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]

    with patch.object(sys.modules["app.main"], "llm_client") as mock_llm:
        mock_llm.chat = AsyncMock(side_effect=mock_llm_tool_then_answer)

        resp = client.post(
            f"/documents/{doc_id}/questions",
            json={"question": "What is the interest rate?"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert len(data["source_chunks"]) > 0
    assert data["usage"] is not None
    assert data["evidence"]["tool_calls_made"] >= 1


def test_insufficient_context_no_tool_call(
    client, sample_document_text, mock_llm_no_tool
):
    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]

    with patch.object(sys.modules["app.main"], "llm_client") as mock_llm:
        mock_llm.chat = AsyncMock(side_effect=mock_llm_no_tool)

        resp = client.post(
            f"/documents/{doc_id}/questions", json={"question": "Random question"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_context"


def test_llm_timeout_returns_503(client, sample_document_text):
    from app.llm import LLMError

    resp = client.post("/documents", json={"document_text": sample_document_text})
    doc_id = resp.json()["document_id"]

    async def raise_timeout(messages, tools=None):
        raise LLMError("The model provider did not respond in time.")

    with patch.object(sys.modules["app.main"], "llm_client") as mock_llm:
        mock_llm.chat = AsyncMock(side_effect=raise_timeout)

        resp = client.post(f"/documents/{doc_id}/questions", json={"question": "test?"})

    assert resp.status_code == 503
