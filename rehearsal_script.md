# Senior RAG Interview — Rehearsal Script

Read each answer out loud. Time yourself — every answer should land in 30-45 seconds. If you're over a minute, you're rambling. Practice until these feel like your own words, not recitation.

---

## HOW TO USE THIS SCRIPT

1. Cover the answer. Read only the question.
2. Answer out loud from memory.
3. Uncover. Compare. Note what you missed.
4. Repeat daily until you can nail every answer cold in one breath.

The structure for every answer is: **Choice → Why → Limitation → Production alternative.** That four-beat rhythm is what signals senior.

---

## BLOCK 1: YOUR SYSTEM DESIGN CHOICES

---

### Q: Why TF-IDF and not vector search?

> TF-IDF was deliberate. I'm working with short financial documents where relevant terms — AER, interest rate, withdrawal fee — appear verbatim in both the question and the text. TF-IDF gives near-perfect recall for exact terminology, zero infra cost, no embedding latency. The failure mode is synonyms and paraphrases, but for structured product T&Cs that rarely matters.
>
> In production I'd use hybrid retrieval: BM25 for keyword precision plus dense embeddings — Cohere Embed or Titan — for semantic recall. Store in pgvector if scale is moderate, OpenSearch Serverless for enterprise. Add a re-ranking layer — Cohere Rerank or a cross-encoder — on the top 20 chunks before sending to the LLM. That catches both "AER" the exact term and "what's my return on savings" the paraphrase.

**Beat check:** Choice (TF-IDF) → Why (exact terms, zero cost) → Limitation (synonyms) → Production (hybrid + rerank).

---

### Q: Why LangGraph?

> Without LangGraph I'd write a while loop — ask the model, if it wants to search then retrieve, feed results back, repeat until it answers or hits a limit. LangGraph structures that as an explicit state machine: two nodes — model and tool — a routing function, and typed state that carries the conversation between steps.
>
> I chose it for three things: bounded recursion so the graph can't loop forever, typed inspectable state for debugging, and native async so retrieval and LLM calls don't block each other.
>
> For single-document Q&A it's slightly heavy. Its value compounds when you add tools — search internal docs AND call a pricing API AND check a regulatory database. The pattern is right even if it's overbuilt for the demo. For Barclays production you'd want LangGraph's persistence layer for resumable multi-turn conversations plus human-in-the-loop nodes for high-stakes decisions.

**Beat check:** Choice (LangGraph) → Why (bounded loops, typed state, async) → Limitation (heavy for one tool) → Production (multi-tool, persistence, human-in-loop).

---

### Q: How is your solution agentic?

> The agentic property is that the model drives the retrieval strategy. In standard RAG, retrieval is deterministic — always fetch top-K for the user's question, always the same path. Here the model decides whether to search, what to search for, and whether to search again with a refined query.
>
> Concretely: the model might retrieve interest rate info, realize it still needs withdrawal fee info, and issue a second search with a different query. That's adaptive planning under uncertainty — it can't see the document directly, only what it explicitly asks for.
>
> The limitation is it's read-only — it can only retrieve, not write or call external APIs. For a banking assistant you'd expand the tool set: look up account data, check product eligibility, file a complaint ticket. Each new tool expands the decision space, which is exactly why bounded tool calls, model verification, and prompt injection defences matter — they constrain the blast radius of any single agent action.

**Beat check:** Choice (model-driven retrieval) → Why (adaptive planning, multi-step reasoning) → Limitation (read-only, narrow) → Production (multi-tool, bounded blast radius).

---

### Q: What is PYTHONUNBUFFERED in the Dockerfile?

> It disables Python's stdout and stderr buffering. Python normally batches output writes for efficiency. In a container, if the process crashes, buffered output that hasn't flushed is lost — your last ten seconds of logs disappear. Setting it to 1 makes every print and every logger call hit the container log stream immediately. It's standard practice for any Python service running in Docker — you never want to debug a crash with missing logs.

**Beat check:** What it does → Why it matters (crash = lost logs) → One sentence showing you understand the operational reality.

---

## BLOCK 2: OBSERVABILITY & EVALUATION

---

### Q: What would you capture in Langfuse?

> I'd segment into three dashboards. First, reliability: latency percentiles — p50, p95, p99 — error rates by category, timeout frequency. Second, quality: grounding scores, abstention rate, faithfulness trends over time, tool call count distribution. Third, economics: tokens per request, cost per answered question, model ROI by endpoint.
>
> The one I'd alert on hardest is grounding failure rate. If the percentage of answers that fail grounding checks exceeds five percent, something changed — document quality dropped, model drifted, or a prompt edit introduced regression. That's the signal a product owner and a risk team both care about.
>
> I'd also track abstention rate as a product metric. High abstention might mean documents are insufficient or users are asking out-of-scope questions. That drives product decisions — what documents to ingest next.

**Beat check:** Structure (three dashboards) → Specifics (grounding alert, abstention as product signal) → Who cares (dev, product owner, risk).

---

### Q: Tell me about your evaluation approach.

> Honest framing: my implementation is a CI sanity check. Four deterministic cases with a mocked LLM. It verifies the pipeline doesn't catastrophically break. It does not measure quality.
>
> A production eval has five layers. One, a golden dataset — 100 to 500 question-answer-source triples curated by domain experts. In banking, lawyers or product specialists. Two, LLM-as-judge — DeepEval's FaithfulnessMetric using Claude to score: is the answer faithful to context, is it relevant, does it hallucinate. Three, regression suite — run on every model upgrade or prompt change, block deployment if faithfulness drops more than five percent. Four, shadow eval in production — sample one to five percent of live traffic, score asynchronously, alert on degradation. Five, for a bank specifically — FCA-aligned test cases: does it give accurate regulatory information, does it correctly abstain on questions requiring human advice.
>
> One non-trivial integration detail: DeepEval defaults to OpenAI as the judge. In a bank not using OpenAI, you configure Bedrock Claude as both app model and judge model. Different token limits, different prompting, async client — it's real work.

**Beat check:** Honest limitation → Five-layer production answer → Bank-specific detail → Integration nuance.

---

## BLOCK 3: PRODUCTION & BANK-GRADE CONCERNS

---

### Q: What does production RAG look like at a bank?

> A production system is not search plus LLM. It's a controlled platform.
>
> Ingestion: S3-triggered processing, OCR via Textract for PDFs, PII detection via Comprehend — ML-based, not regex — chunking with sentence boundary awareness, embeddings stored in OpenSearch, document metadata in DynamoDB — owner, classification, retention policy. All behind VPC endpoints, KMS encryption at rest.
>
> Query path: API Gateway with WAF and rate limiting at the front. Hybrid retrieval — BM25 plus vector, re-ranked by Cohere Rerank. ACL check before anything reaches the LLM — does this user have access to this document. LLM via Bedrock with Guardrails for content filtering and denied topics. Response validated against retrieved chunks. Human escalation path for high-confidence-required queries.
>
> Governance: model registered in internal inventory, independent validation team reviews eval methodology, change control on prompt updates — versioned, reviewed, approved — quarterly re-validation, and a kill switch via feature flag to disable or degrade to keyword-only.

**Beat check:** Three layers (ingestion, query, governance) → Specific AWS services → Access control as first-class → Kill switch.

---

### Q: How do you think about Model Risk Management for RAG?

> MRM for an LLM system is broader than "test the model." It means defining intended use and excluded use, identifying failure modes beyond just accuracy — hallucination, wrong evidence, unauthorized retrieval, prompt injection, user over-reliance — and validating the full system, not just the model in isolation.
>
> Concretely: I'd validate retrieval quality, answer faithfulness, abstention behavior, access control boundaries, prompt injection resistance, and regression after any change. Passing offline eval is necessary but not sufficient — in production I'd monitor continuously and treat quality drift as a model risk issue.
>
> Change governance matters: prompts, model versions, and retrieval logic are governed artifacts. Any material change triggers re-validation because it can change user-facing behavior. And for high-risk use cases, the model supports human decision-making — it doesn't replace it.
>
> The frame is: this is a governed system with a defined purpose, operating within limits, validated before release, monitored after, and re-validated when materially changed.

**Beat check:** Definition (broader than accuracy) → Failure modes → Validation scope → Change governance → Human-in-loop → One-sentence frame.

---

### Q: How do you handle PII?

> PII handling in a bank is not just "redact before the LLM." It's a data flow decision. You define where PII may exist, where it may flow, how it's controlled at each boundary, and how you prove that control.
>
> In my implementation I have a PII mode that redacts before the LLM sees the text. In production you'd use Comprehend for ML-based detection — not regex — with policy that specifies: what may be sent to the model, what must be redacted, what may be logged, what may be retained, and what must stay inside approved infrastructure.
>
> The strongest position is: content logging is off by default. Traces capture structure — latency, token count, status — but not user content unless explicitly enabled with appropriate retention and access controls. That's a legal and policy decision, not just a technical one.

**Beat check:** Frame (data flow, not just redaction) → Implementation → Production (ML detection, policy) → Logging stance.

---

### Q: How do you handle security beyond prompt injection?

> Prompt injection is one risk but not the whole security story. Production RAG security includes: document-level access control — retrieval must be permission-aware before anything reaches the model. API authentication and rate limiting. Secret management — no credentials in code or environment. Least-privilege IAM scoping. Network isolation — model endpoints behind VPC. Data retention rules. Output controls — the model can't return content the user shouldn't see even if it retrieved it.
>
> The design principle is: assume any model can make mistakes. The architecture must limit the blast radius. The trust boundary sits around the full system, not just around the model. That means approved endpoints, credential scoping, prompt controls, output handling rules, and human escalation where the stakes are high.

**Beat check:** Enumerate beyond injection → Design principle (blast radius) → Trust boundary framing.

---

## BLOCK 4: SCALING & ARCHITECTURE

---

### Q: How would you scale this from prototype to production?

> Four dimensions. Storage: in-memory store to Postgres with pgvector, or OpenSearch Serverless for enterprise scale. Retrieval: TF-IDF to hybrid — BM25 plus dense vector plus reranking. Compute: single Uvicorn worker to containerized service behind a load balancer, with queue-based async processing for ingestion. Observability: basic logging to Langfuse traces plus CloudWatch infrastructure metrics plus CloudTrail for audit — every Bedrock API call logged to S3, immutable.
>
> The non-obvious scaling concern is document governance. When you go from one document to millions, you need metadata — owner, classification, retention policy, access rules — attached at ingest time. Without that, retrieval can't be permission-aware and you can't scope queries to what the user is allowed to see.

**Beat check:** Four dimensions with specifics → Non-obvious concern (document governance at scale).

---

### Q: pgvector vs Pinecone vs OpenSearch — when would you use each?

> pgvector when you're already on Postgres and scale is moderate — say under ten million vectors. One fewer vendor, one fewer data residency question, familiar operational model. Good fit for a bank that wants to keep everything in existing Postgres estate.
>
> Pinecone when you want zero operational overhead and you're allowed to use a managed SaaS — it handles indexing, scaling, and filtering. Less likely in a bank because of data residency and vendor risk appetite.
>
> OpenSearch Serverless when you need hybrid search built in — BM25 plus neural sparse encoding in one system — at enterprise scale with fine-grained access control. That's usually the right answer at a large bank: it handles millions of documents, supports metadata filtering and ACLs natively, and runs within AWS so it satisfies data residency.

**Beat check:** Three options → When each fits → Bank default (OpenSearch) with reasoning.

---

## BLOCK 5: BEHAVIOURAL / META QUESTIONS

---

### Q: What would you do differently if you built this again?

> Three things. First, I'd use hybrid retrieval from the start — even a simple BM25 plus embedding baseline — to show I understand the production path, not just the prototype shortcut. Second, I'd build a proper eval dataset — twenty to thirty curated cases with expected answers and sources — so I could demonstrate quality measurement, not just pipeline testing. Third, I'd add a proper abstention threshold with a confidence signal so I could show the system knows when it doesn't know, and I could quantify that behavior.
>
> The meta-lesson: a take-home that shows the production path — even partially — signals more seniority than a polished demo of just the happy path.

---

### Q: Walk me through how you'd debug a bad answer in production.

> Start with the trace. Pull the Langfuse trace for that request. Check: what chunks were retrieved and what were their scores? If scores are low, the retrieval failed — wrong query, document not indexed, embedding drift. If scores are high but the answer is wrong, the model ignored good evidence or hallucinated — check the prompt, check token limits, check if the context was truncated.
>
> Then check: did the model make multiple tool calls? What query did it use? Sometimes the model searches for the wrong thing. That's a prompt issue or a system prompt regression.
>
> Then zoom out: is this a one-off or a pattern? Check the grounding failure rate over the last 24 hours. If it spiked, something changed — a deployment, a prompt edit, a new document batch with different structure. That's where version tracking on prompts and retrieval config pays off.
>
> The discipline is: trace first, then pattern, then root cause. Never guess.

---

### Q: How do you work with other teams on a system like this?

> A production RAG system touches platform engineering, security, risk, legal, and product. My job as the engineer is to make the system legible to each of those teams.
>
> For security: clear documentation of data flows, trust boundaries, and access controls. For risk: model card with intended use, limitations, validation evidence, and monitoring plan. For product: dashboards showing abstention rate, question themes, and quality trends. For legal: evidence that PII controls work, that content logging respects retention policy, and that audit trails are complete.
>
> The senior signal is: I don't just build the system. I build it so that non-engineers can govern it, challenge it, and trust it. That's what separates a feature from a platform.

---

## BLOCK 6: RAPID-FIRE (one-breath answers)

Practice these as 10-15 second responses for follow-up questions.

---

**Q: What's the difference between RAG and fine-tuning?**
> RAG retrieves external knowledge at query time — current, auditable, no retraining. Fine-tuning bakes knowledge into model weights — static, opaque, expensive to update. For a bank, RAG wins because documents change quarterly and you need to trace every answer back to a source.

**Q: What's a grounding check?**
> Verifying that the model's answer is actually supported by the retrieved context. Either keyword overlap as a fast heuristic or an LLM judge that scores faithfulness. If the answer contains claims not present in the chunks, it fails grounding.

**Q: What's chunking and how do you choose a strategy?**
> Splitting documents into retrieval units. Fixed-size is simple but can split mid-sentence. Sentence-boundary is better for coherence. Semantic chunking groups related sentences. Hierarchical uses parent-child relationships. For financial T&Cs, sentence-boundary with overlap works well because clauses are self-contained.

**Q: What's the difference between BM25 and TF-IDF?**
> Both are keyword-based. TF-IDF weighs term frequency against how rare a term is across all documents. BM25 adds length normalization and a saturation function — repeated terms don't keep gaining weight. BM25 is generally better for retrieval; TF-IDF is fine for small corpora.

**Q: What does bounded agency mean?**
> The agent can make decisions — what to search, when to stop — but within fixed constraints: maximum tool calls, approved tools only, typed state, no side effects beyond retrieval. The system enforces the boundaries; the model operates within them.

**Q: Why not just use ChatGPT with a document upload?**
> No access control, no audit trail, data leaves your perimeter, no custom retrieval tuning, no integration with internal systems, no governance over model changes. For a bank, the question is never "can it answer" — it's "can we control, audit, and trust the answer."

---

## DAILY PRACTICE ROUTINE

1. **Morning (10 min):** Pick 3 questions randomly. Answer out loud. Time yourself.
2. **Commute (15 min):** Run through Block 3 (production/bank-grade) — this is where most candidates are weakest and where you differentiate.
3. **Before bed (5 min):** Rapid-fire block. All six in under 90 seconds total.

After 5 days of this, these answers will feel like your own thinking, not memorized text. That's the goal — not recitation, but rewired intuition.

---

## THE ONE SENTENCE THAT FRAMES EVERYTHING

If you remember nothing else, open every system design answer with this mental model:

> "A take-home proves I can build the loop. Production proves I understand control. In a bank, the hard parts are access-controlled retrieval, auditability, evaluation, observability, PII handling, and safe failure behavior."

That sentence is your north star. Every answer should orient around it.
