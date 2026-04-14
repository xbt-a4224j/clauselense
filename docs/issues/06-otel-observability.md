# Issue 6: OpenTelemetry tracing on every LLM call

**Labels:** `enhancement`, `observability`, `platform`

## Context

For an LLM-backed service, tokens, cost, latency, and the chain from request → retrieval → generation are critical signals. Without them you can't debug why a specific answer went wrong.

## Proposed approach

1. Add `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, and `opentelemetry-exporter-otlp`.
2. Wrap `rag.ask()` in a parent span with attributes: `question_len`, `k`, `score_threshold`.
3. Wrap each sub-step in a child span:
   - `rag.embed` — `embed.model`, `embed.latency_ms`, `embed.token_count`
   - `rag.retrieve` — `retrieve.n_candidates`, `retrieve.top_score`, `retrieve.latency_ms`
   - `rag.generate` — `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`, `llm.latency_ms`
4. Exporter defaults to console for local dev; OTLP endpoint configurable via env.
5. Docker compose (future issue) will add a Jaeger/Tempo sink for local visualization.

## Acceptance criteria

- [ ] All three phases emit spans with the attributes above
- [ ] Parent `trace_id` propagates from FastAPI request through to LLM call
- [ ] `llm.cost_usd` computed from a small price table in `clauselens/pricing.py`
- [ ] README adds a "Tracing" section with a screenshot of a single trace in Jaeger
- [ ] No PII (question text) in span attributes by default; optional `CLAUSELENS_TRACE_PAYLOADS=1` env var to include for dev

