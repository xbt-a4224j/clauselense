# Issue 5: Cross-encoder reranker

**Labels:** `enhancement`, `retrieval`

## Context

Bi-encoder retrieval (what we do now) embeds query and documents independently and compares. That's fast but loses cross-attention between query and document. A cross-encoder model scores each (query, document) pair jointly — much more accurate, much slower, which is why it's typically used to *rerank* the top 20-50 retrieved by a bi-encoder.

## Proposed approach

1. Retrieve top-N (e.g., N=20) with existing hybrid retrieval (Issue 4).
2. Run a cross-encoder (`sentence-transformers/ms-marco-MiniLM-L-6-v2` is a reasonable small baseline) over those 20 pairs.
3. Return top-k from the reranked list.

## Acceptance criteria

- [ ] `clauselens/reranker.py` with a `CrossEncoderReranker` class
- [ ] Toggle via `RetrievalConfig.use_reranker`
- [ ] `--reranker` CLI flag
- [ ] Eval comparison: hybrid with/without reranker across 2+ datasets
- [ ] Latency column added to results DB so we can see the speed tradeoff

## Expected outcome

Rerankers typically add +5-10 points of citation precision at the cost of +500ms-1s latency per query. Interesting question: is the tradeoff worth it for a contract-QA use case where latency budgets are loose but accuracy matters?

