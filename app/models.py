from pydantic import BaseModel
from typing import Optional


class DocumentRequest(BaseModel):
    document_text: str


class PiiInfo(BaseModel):
    mode: str
    redactions_applied: int
    types_detected: list[str]


class LimitsInfo(BaseModel):
    max_document_chars: int


class DocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    limits: LimitsInfo
    pii: PiiInfo


class QuestionRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    chunk_hash: str
    score: float


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GroundingInfo(BaseModel):
    status: str
    method: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class Evidence(BaseModel):
    request_id: str
    trace_id: str
    document_id: str
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    model_verified: Optional[bool] = None
    tool_calls_made: int
    prompt_version: str = "qa-system-v1"
    retrieval_tool_version: str = "retrieval-v1"
    policy_version: str = "readonly-qa-v1"
    app_version: str = "0.1.0"
    git_commit: str = "unknown-local"


class QuestionResponse(BaseModel):
    answer: str
    status: str
    source_chunks: list[SourceChunk]
    usage: Optional[UsageInfo] = None
    latency_ms: int
    grounding: GroundingInfo
    evidence: Evidence


class ErrorDetail(BaseModel):
    category: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = "ok"
    app_version: str
    git_commit: str
    model_provider: str
    model_name: str
    prompt_version: str = "qa-system-v1"
    policy_version: str = "readonly-qa-v1"
