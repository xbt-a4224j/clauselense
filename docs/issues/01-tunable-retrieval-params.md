# Issue 1: Expose retrieval parameters as tunable knobs

**Labels:** `enhancement`, `retrieval`, `evals`

## Context

(Updated Jul 2026 — the original text predated the tunable-params work.)

`top_k` and `score_threshold` are now threaded through `ask()`, the `/ask` request
model, the playground controls, and the eval runner CLI (`--top-k`,
`--score-threshold`). What's still missing is a single typed config object rather
than parameters repeated at four call sites, plus a context-length guardrail and a
recorded before/after comparison.

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
