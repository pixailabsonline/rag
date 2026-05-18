# Document Q&A Agent API

FastAPI service for document Q&A using a bounded LangGraph retrieval agent. It demonstrates the assignment requirements: document ingestion, source-grounded answers, abstention, Langfuse tracing, PII-aware handling, evaluation, tests, Docker, and CI controls.

## Run

```bash
cp .env.example .env
# edit .env with Bedrock credentials
sudo docker compose up --build
```

Local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

For tests and CI tools:

```bash
pip install -r requirements-dev.txt
```

Run the demo after the API is up:

```bash
bash examples/demo.sh
```

## Configuration

Bedrock Anthropic:

```env
LLM_MODEL=us.anthropic.claude-opus-4-6-v1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

Operational controls:

```env
PII_MODE=redact_before_llm
MAX_DOCUMENT_CHARS=100000
MAX_QUESTION_CHARS=2000
RETRIEVAL_TOP_K=3
MIN_RETRIEVAL_SCORE=0.05
MAX_TOOL_CALLS=3
ENABLE_LANGFUSE=false
ENABLE_CONTENT_LOGGING=false
```

The application uses an internal `LLMClient` boundary around Bedrock Anthropic. In a bank deployment this boundary would call an approved private model gateway with identity, allowlists, DLP, audit, budgets, and data residency controls.

## API

Health:

```bash
curl http://localhost:8000/health
```

Ingest:

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"document_text":"Annual interest rate: 4.5% AER. Withdrawals beyond three per month cost £10."}'
```

Ask:

```bash
curl -X POST http://localhost:8000/documents/{document_id}/questions \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the annual interest rate?"}'
```

Question responses include:

```json
{
  "answer": "The annual interest rate is 4.5% AER.",
  "status": "answered",
  "source_chunks": [{"chunk_id": "...", "text": "...", "chunk_hash": "sha256:...", "score": 0.87}],
  "usage": {"prompt_tokens": 1000, "completion_tokens": 120, "total_tokens": 1120},
  "latency_ms": 850,
  "grounding": {"status": "passed", "method": "keyword_overlap", "score": 0.72},
  "evidence": {"request_id": "...", "trace_id": "...", "model_name": "...", "tool_calls_made": 1}
}
```

If the document lacks evidence, the API returns `status=insufficient_context` rather than fabricating an answer.

## Evaluation

Run the self-contained faithfulness eval:

```bash
pip install -r requirements-dev.txt
python eval/run_eval.py
```

Optionally add DeepEval semantic judging:

```bash
OPENAI_API_KEY=... EVAL_WITH_DEEPEVAL=true python eval/run_eval.py
```

The eval script uses FastAPI `TestClient`, ingests `examples/sample_document.txt`, mocks the application LLM path for stable API outputs, and checks faithfulness against returned source chunks. When `EVAL_WITH_DEEPEVAL=true`, it also uses DeepEval `FaithfulnessMetric`. Cases cover grounded direct answer, grounded multi-fact answer, unanswerable question, and prompt injection.

## Langfuse

Enable tracing:

```env
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Each question request emits structured JSON logs to stdout. When Langfuse is enabled, the `@observe()` decorator on `run_agent` creates a trace span, and `@observe(as_type="generation")` on `LLMClient.chat` records the LLM call with model name and token usage. Tracing failures are logged and do not fail the API request.

### Local Langfuse stack

Run the full Langfuse v3 stack (Postgres, ClickHouse, Redis, MinIO) locally:

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

If running the app on the host (e.g. `uvicorn`), set:

```env
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
```

If running both the app and Langfuse in Docker, combine the compose files so they share a network:

```bash
docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up --build
```

In that case set `LANGFUSE_HOST=http://langfuse-web:3000` in `.env`.

Langfuse UI: http://localhost:3000 (login: `admin@local.dev` / `password`).

## Tests and CI

```bash
pytest --tb=short -q
ruff check .
ruff format --check .
```

Development and CI tools live in `requirements-dev.txt`; the Docker image installs only `requirements.txt`.

GitHub Actions runs dependency install, ruff, pytest, self-contained eval, `pip-audit`, SBOM generation, and Docker build. DeepEval is an optional local/reviewer quality gate because it requires an evaluator API key.

## Architecture

```text
FastAPI -> validation -> PII redaction -> chunk/hash/store
        -> LangGraph model/tool loop -> scoped TF-IDF retrieval
        -> grounded response + evidence -> logs/Langfuse
```

Security controls implemented for the assignment:

- document-scoped retrieval
- bounded tool calls
- no LLM-visible `document_id` tool argument
- size limits and max token limits
- PII redaction before storage/LLM by default
- abstention on missing evidence
- source chunks and chunk hashes in responses
- provider timeouts/retries and controlled errors
- Docker non-root runtime and compose hardening
- `pip-audit` and SBOM in CI

## Production Gaps

This is not a production banking platform. A real deployment still needs:

- **Auth/RBAC** and tenant isolation
- **Bedrock Guardrails** for defence-in-depth content filtering, PII detection, denied topics, and grounding checks at the infrastructure layer (complementing app-level controls)
- **CloudTrail** with a persistent trail to S3 for long-term audit of all Bedrock invocations, tied to IAM role identity
- **Least-privilege IAM** scoped per-service (separate task roles for the app vs developer tooling), with VPC endpoint conditions and session duration limits
- Enterprise DLP, persistent governed storage
- **Semantic/embedding retrieval** replacing TF-IDF keyword search (e.g. Pinecone, OpenSearch, or pgvector), with hybrid keyword+vector scoring for better recall on paraphrased queries
- Encryption and retention controls, API gateway/WAF/rate limits, circuit breakers
- **Model Risk Management (MRM)** per SR 11-7 / SS1/23: model inventory registration, independent validation, risk tiering, committee sign-off, and periodic re-validation — the app provides the hooks (model versioning, evidence fields, eval framework) but not the governance process
- Drift monitoring, larger golden eval sets
- Monitoring/runbooks, secret rotation, signed images, SLSA provenance, and promotion gates
