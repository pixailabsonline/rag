import json
from unittest.mock import patch, AsyncMock
import sys

from app.llm import LLMResponse


def test_cross_document_isolation(client):
    doc_a_text = (
        "Document A content about apples and oranges. " * 20
        + " The price of apples is £2.50 per kg."
    )
    doc_b_text = (
        "Document B content about cars and motorcycles. " * 20
        + " The top speed is 150 mph."
    )

    resp_a = client.post("/documents", json={"document_text": doc_a_text})
    assert resp_a.status_code == 201
    doc_a_id = resp_a.json()["document_id"]

    resp_b = client.post("/documents", json={"document_text": doc_b_text})
    assert resp_b.status_code == 201

    call_count = {"n": 0}

    async def mock_chat(messages, tools=None):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1 and tools:
            return LLMResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_iso",
                        "function": {
                            "name": "retrieve_document_context",
                            "arguments": json.dumps({"query": "price of apples"}),
                        },
                    }
                ],
                model_name="us.anthropic.claude-opus-4-6-v1",
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            )
        return LLMResponse(
            content="The price of apples is £2.50 per kg.",
            tool_calls=[],
            model_name="us.anthropic.claude-opus-4-6-v1",
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
        )

    with patch.object(sys.modules["app.main"], "llm_client") as mock_llm:
        mock_llm.chat = AsyncMock(side_effect=mock_chat)
        resp = client.post(
            f"/documents/{doc_a_id}/questions",
            json={"question": "What is the price of apples?"},
        )

    assert resp.status_code == 200
    data = resp.json()
    for chunk in data["source_chunks"]:
        assert chunk["chunk_id"].startswith(doc_a_id)
