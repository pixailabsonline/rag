import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.models import (
    DocumentRequest,
    DocumentResponse,
    QuestionRequest,
    QuestionResponse,
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
    LimitsInfo,
    PiiInfo,
    SourceChunk,
    UsageInfo,
    Evidence,
)
from app.pii import redact_pii
from app.chunking import chunk_text
from app.store import document_store, StoredChunk
from app.agent import run_agent
from app.llm import LLMClient
from app.observability import log_question_event

import hashlib


settings: Settings = None
llm_client: LLMClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, llm_client
    settings = get_settings()
    llm_client = LLMClient(settings)
    if settings.ENABLE_LANGFUSE:
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    yield
    if settings.ENABLE_LANGFUSE:
        from langfuse import Langfuse

        Langfuse().flush()


app = FastAPI(title="Document Q&A Agent", version="0.1.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    category = "internal_error"
    if exc.status_code == 404:
        category = "document_not_found"
    elif exc.status_code in (400, 422, 413):
        category = "validation_error"
    elif exc.status_code == 503:
        category = "llm_error"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                category=category,
                message=exc.detail,
                request_id=request_id,
            )
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        app_version=settings.APP_VERSION,
        git_commit=settings.GIT_COMMIT,
        model_provider="bedrock",
        model_name=settings.LLM_MODEL,
    )


@app.post("/documents", response_model=DocumentResponse, status_code=201)
async def ingest_document(req: DocumentRequest):
    if not req.document_text or not req.document_text.strip():
        raise HTTPException(status_code=400, detail="document_text must not be empty")
    if len(req.document_text) < 10:
        raise HTTPException(
            status_code=400, detail="document_text too short (minimum 10 characters)"
        )
    if len(req.document_text) > settings.MAX_DOCUMENT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"document_text exceeds maximum {settings.MAX_DOCUMENT_CHARS} characters",
        )

    try:
        pii_result = redact_pii(req.document_text, settings.PII_MODE)
    except Exception:
        raise HTTPException(
            status_code=500, detail="PII redaction failed; request rejected"
        )

    text_to_store = pii_result.redacted_text
    chunks = chunk_text(text_to_store, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    document_id = str(uuid.uuid4())
    stored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_hash = "sha256:" + hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        stored_chunks.append(
            StoredChunk(
                chunk_id=f"{document_id}:chunk:{i}",
                text=chunk,
                chunk_hash=chunk_hash,
                index=i,
            )
        )

    document_store.store_document(document_id, stored_chunks)

    return DocumentResponse(
        document_id=document_id,
        chunks_created=len(stored_chunks),
        limits=LimitsInfo(max_document_chars=settings.MAX_DOCUMENT_CHARS),
        pii=PiiInfo(
            mode=settings.PII_MODE,
            redactions_applied=pii_result.redactions_applied,
            types_detected=pii_result.types_detected,
        ),
    )


@app.post("/documents/{document_id}/questions", response_model=QuestionResponse)
async def ask_question(document_id: str, req: QuestionRequest):
    request_id = str(uuid.uuid4())

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if len(req.question) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"question exceeds maximum {settings.MAX_QUESTION_CHARS} characters",
        )

    question_pii = redact_pii(req.question, settings.PII_MODE)
    safe_question = question_pii.redacted_text

    doc = document_store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    start = time.time()
    try:
        result = await run_agent(document_id, safe_question, llm_client, settings)
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        err_str = str(e).lower()
        if "mismatch" in err_str:
            error_category = "model_mismatch"
        elif "timeout" in err_str or "provider" in err_str:
            error_category = "llm_error"
        else:
            error_category = "internal_error"
        log_question_event(
            request_id=request_id,
            document_id=document_id,
            status="error",
            question=safe_question,
            answer=None,
            latency_ms=latency_ms,
            error_category=error_category,
            settings=settings,
        )
        status_code = 503 if error_category == "llm_error" else 500
        raise HTTPException(status_code=status_code, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)

    source_chunks = [
        SourceChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            chunk_hash=c.chunk_hash,
            score=c.score,
        )
        for c in result.source_chunks
    ]

    usage = None
    if result.usage:
        usage = UsageInfo(
            prompt_tokens=result.usage["prompt_tokens"],
            completion_tokens=result.usage["completion_tokens"],
            total_tokens=result.usage["total_tokens"],
        )

    grounding = result.grounding

    evidence = Evidence(
        request_id=request_id,
        trace_id=request_id,
        document_id=document_id,
        model_provider="bedrock" if result.status == "answered" else None,
        model_name=result.model_name,
        model_verified=result.model_verified,
        tool_calls_made=result.tool_calls_made,
        app_version=settings.APP_VERSION,
        git_commit=settings.GIT_COMMIT,
    )

    response = QuestionResponse(
        answer=result.answer,
        status=result.status,
        source_chunks=source_chunks,
        usage=usage,
        latency_ms=latency_ms,
        grounding=grounding,
        evidence=evidence,
    )

    log_question_event(
        request_id=request_id,
        document_id=document_id,
        status=result.status,
        question=safe_question,
        answer=result.answer,
        latency_ms=latency_ms,
        model_name=result.model_name,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        retrieved_chunk_ids=[c.chunk_id for c in source_chunks],
        retrieved_chunk_hashes=[c.chunk_hash for c in source_chunks],
        retrieved_chunk_texts=[c.text for c in source_chunks],
        tool_calls_made=result.tool_calls_made,
        grounding_status=grounding.status,
        settings=settings,
    )

    return response
