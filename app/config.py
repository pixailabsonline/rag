from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    LLM_MODEL: str = "us.anthropic.claude-opus-4-6-v1"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 800
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 2

    AWS_REGION: str = "us-east-1"

    MAX_DOCUMENT_CHARS: int = 100000
    MAX_QUESTION_CHARS: int = 2000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 3
    MIN_RETRIEVAL_SCORE: float = 0.05
    MAX_RETRIEVED_CONTEXT_CHARS: int = 8000
    MAX_TOOL_CALLS: int = 3

    PII_MODE: str = "redact_before_llm"
    ENABLE_CONTENT_LOGGING: bool = False

    ENABLE_LANGFUSE: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    GIT_COMMIT: str = "unknown-local"
    APP_VERSION: str = "0.1.0"
    TESTING: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}

    @field_validator(
        "MAX_DOCUMENT_CHARS",
        "MAX_QUESTION_CHARS",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_TOP_K",
        "MAX_RETRIEVED_CONTEXT_CHARS",
        "MAX_TOOL_CALLS",
        "LLM_MAX_TOKENS",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
    )
    @classmethod
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError("Must be a positive integer")
        return v


def get_settings() -> Settings:
    return Settings()
