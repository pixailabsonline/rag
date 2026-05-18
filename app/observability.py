import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.config import Settings

logger = logging.getLogger("document_qa")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def log_question_event(
    request_id: str,
    document_id: str,
    status: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    latency_ms: Optional[int] = None,
    model_name: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    retrieved_chunk_ids: Optional[list[str]] = None,
    retrieved_chunk_hashes: Optional[list[str]] = None,
    retrieved_chunk_texts: Optional[list[str]] = None,
    tool_calls_made: Optional[int] = None,
    grounding_status: Optional[str] = None,
    error_category: Optional[str] = None,
    settings: Optional[Settings] = None,
):
    log_content = settings.ENABLE_CONTENT_LOGGING if settings else False

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "event": "question_completed",
        "endpoint": f"/documents/{document_id}/questions",
        "document_id": document_id,
        "status": status,
        "latency_ms": latency_ms,
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "retrieved_chunk_ids": retrieved_chunk_ids or [],
        "retrieved_chunk_hashes": retrieved_chunk_hashes or [],
        "tool_calls_made": tool_calls_made,
        "grounding_status": grounding_status,
        "error_category": error_category,
    }

    if log_content:
        event["question_preview"] = question
        event["answer_preview"] = answer
        event["retrieved_chunk_texts"] = [
            (text[:300] + "...") if len(text) > 300 else text
            for text in (retrieved_chunk_texts or [])
        ]
    else:
        event["question_preview"] = (
            (question[:80] + "...") if question and len(question) > 80 else question
        )
        event["answer_preview"] = (
            (answer[:80] + "...") if answer and len(answer) > 80 else answer
        )

    logger.info(json.dumps(event))
