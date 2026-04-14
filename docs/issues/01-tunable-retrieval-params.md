# Issue 1: Expose retrieval parameters as tunable knobs

**Labels:** `enhancement`, `retrieval`, `evals`

## Context

Today, retrieval uses a hardcoded `k=4` and no score threshold. Every question retrieves 4 clauses whether or not any of them are actually relevant. This artificially inflates recall on a small corpus and tanks precision.

To measure and improve precision/recall meaningfully, the core retrieval behavior needs to be configurable — not hardcoded in `rag.py`.

## Proposed knobs

| Param | What it controls | Effect |
|---|---|---|
| `top_k` | Max clauses returned | ↑ k → ↑ recall, ↓ precision |
| `score_threshold` | Min cosine similarity to include | ↑ threshold → ↑ precision, ↓ recall |
| `max_context_tokens` | Truncate context if too long | Guardrail |

## Acceptance criteria

- [ ] `RetrievalConfig` dataclass in `clauselens/config.py` with sane defaults
- [ ] `ask()` and `ClauseStore.search()` accept the config
- [ ] CLI flags in the eval runner: `--top-k`, `--score-threshold`
- [ ] One entry in `docs/tuning.md` showing a before/after precision/recall comparison at k=4 vs k=8 on the toy dataset
