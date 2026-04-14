# Issue 4: Hybrid retrieval (BM25 + vector)

**Labels:** `enhancement`, `retrieval`

## Context

Pure vector retrieval is weak on exact-term matches — things like defined terms in contracts (*"Licensee"*, *"Effective Date"*) that embeddings tend to smooth out. BM25 is strong exactly there. Real RAG systems combine both.

## Proposed approach

1. Add a BM25 index alongside the vector index (start with `rank_bm25` — pure Python, no new infra).
2. For each query, retrieve top-k from each, normalize scores, and compute a weighted combined score:
   `final = alpha * vector_score + (1 - alpha) * bm25_score`
3. Expose `alpha` in `RetrievalConfig` (from Issue 1).

## Acceptance criteria

- [ ] `clauselens/bm25.py` with a `BM25Index` that mirrors the `ClauseStore` interface
- [ ] Hybrid search function that merges results
- [ ] `alpha` tunable via config and CLI
- [ ] Eval comparison (leaning on Issue 3): pure vector vs pure BM25 vs hybrid at alpha=0.5 on at least 2 datasets
- [ ] README "what I learned" section updated with findings

## Expected outcome

On contract-like text where defined terms matter, hybrid should beat pure vector on citation precision by a meaningful margin.
