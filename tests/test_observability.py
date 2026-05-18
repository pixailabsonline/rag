import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from langfuse import observe

from app.config import Settings
from app.observability import log_question_event


def test_observe_decorator_on_llm_chat():
    from app.llm import LLMClient

    assert hasattr(LLMClient.chat, "__wrapped__") or "observe" in str(
        getattr(LLMClient.chat, "__qualname__", "")
    ), "LLMClient.chat should be decorated with @observe"


def test_observe_decorator_on_run_agent():
    from app.agent import run_agent

    assert hasattr(run_agent, "__wrapped__") or "observe" in str(
        getattr(run_agent, "__qualname__", "")
    ), "run_agent should be decorated with @observe"


def test_langfuse_observe_importable():
    assert callable(observe)


def test_content_logging_disabled_omits_chunk_texts(caplog):
    settings = Settings(ENABLE_CONTENT_LOGGING=False, TESTING=True)
    with caplog.at_level("INFO", logger="document_qa"):
        log_question_event(
            request_id="r1",
            document_id="d1",
            status="answered",
            question="What is the rate?",
            answer="The rate is 4.5%",
            retrieved_chunk_texts=["Annual interest rate: 4.5% AER"],
            settings=settings,
        )
    event = json.loads(caplog.records[-1].message)
    assert "retrieved_chunk_texts" not in event
    assert event["question_preview"] == "What is the rate?"
    assert event["answer_preview"] == "The rate is 4.5%"


def test_content_logging_enabled_includes_chunk_texts(caplog):
    settings = Settings(ENABLE_CONTENT_LOGGING=True, TESTING=True)
    with caplog.at_level("INFO", logger="document_qa"):
        log_question_event(
            request_id="r1",
            document_id="d1",
            status="answered",
            question="What is the rate?",
            answer="The rate is 4.5%",
            retrieved_chunk_texts=["Annual interest rate: 4.5% AER"],
            settings=settings,
        )
    event = json.loads(caplog.records[-1].message)
    assert event["retrieved_chunk_texts"] == ["Annual interest rate: 4.5% AER"]
    assert event["question_preview"] == "What is the rate?"
    assert event["answer_preview"] == "The rate is 4.5%"


@pytest.mark.asyncio
async def test_llm_chat_calls_update_current_generation():
    settings = Settings(TESTING=True)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="hello")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.model = "us.anthropic.claude-opus-4-6-v1"

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("app.llm.Langfuse") as mock_langfuse_cls:
        mock_langfuse_instance = MagicMock()
        mock_langfuse_cls.return_value = mock_langfuse_instance

        from app.llm import LLMClient

        client = LLMClient(settings, client=mock_client)
        result = await client.chat([{"role": "user", "content": "hi"}])

        assert result.content == "hello"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        mock_langfuse_instance.update_current_generation.assert_called_once_with(
            model="us.anthropic.claude-opus-4-6-v1",
            usage_details={"input": 10, "output": 5},
        )
