# ClauseLens — Context for Claude Code

This file exists so the next Claude Code session (or any AI coding assistant) picks up exactly where the previous Cowork session left off. Read this first before touching anything.

## What this repo is

A tiny RAG + eval harness over contract clauses. Deliberately small — SQLite + numpy, no Docker — so every piece of the pipeline is legible.

Personal project to play with RAG, evals, and observability on contract-like text.

Don't over-engineer. Keep the surface small and readable. Leave complexity for the issue tracker.

## Current state (as of Apr 14, 2026)

### Implemented

- `README.md` — project overview. Don't rewrite the voice.
- `requirements.txt`, `.env.example`, `.gitignore`
- `clauselens/__init__.py`
- `clauselens/store.py` — SQLite-backed vector store, cosine similarity in numpy
- `clauselens/rag.py` — embed → retrieve → generate with cited JSON output
- `clauselens/app.py` — FastAPI with `POST /ask` and `GET /healthz`
- `clauselens/evals.py` — citation precision/recall + LLM-as-judge faithfulness

### Not yet implemented (see "Immediate next steps")

- `clauselens/seed.py` — CLI to embed clauses from `data/sample_clauses.json` into the DB
- `data/sample_clauses.json` — 10 hand-picked contract clauses
- `data/eval_set.json` — 10 labeled Q&A pairs
- `tests/test_evals.py` — pytest wrapper that runs the full eval and asserts thresholds
- `.github/workflows/ci.yml` — runs pytest on every push
- Any actual end-to-end test run (no one has run `pytest` or `uvicorn` yet)

## Architectural decisions (and why)

1. **SQLite + numpy, not pgvector** — Keeps v0 shippable in an evening. Issue #2 is the pgvector / multi-dataset upgrade path. Explicitly called out in the README as a limitation.
2. **JSON-structured LLM output (answer + citations + confidence)** — Lets us grade citation accuracy separately from answer faithfulness. This is the core eval methodology point.
3. **Two-metric eval (citation P/R + LLM-as-judge faithfulness)** — Citation P/R catches retrieval problems; faithfulness catches generation problems where the model cites the right clause but misstates it. They fail in different ways.
4. **`temperature=0` everywhere** — Evals must be reproducible.
5. **No auth, no rate limiting, no CORS** — Out of scope for v0. Issue-worthy for a "prod-ify" story.

## Immediate next steps (pick up here)

In order:

1. **Write `clauselens/seed.py`** — loads `data/sample_clauses.json`, embeds each clause via OpenAI, upserts to the store. Needs to handle the case where `OPENAI_API_KEY` is missing by printing a helpful error.
2. **Create `data/sample_clauses.json`** — 10 clauses with fields `{id, contract, text}`. Pull real-sounding snippets from 2-3 different contract types (NDA, SaaS MSA, vendor agreement). Keep each under ~500 chars. Assign IDs like `NDA-01`, `MSA-03`, `VND-02`.
3. **Create `data/eval_set.json`** — 10 Q&A cases with `{question, expected_clause_ids, notes}`. Mix of easy ("what's the termination period?") and harder ("is assignment allowed without consent?"). Make sure at least two questions reference the same clause so citation recall has room to fail.
4. **Write `tests/test_evals.py`** — imports `run_eval`, `aggregate`, `load_eval_set`, runs over the eval set, asserts `faithfulness >= 0.8` and `citation_f1 >= 0.7`. Use `pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"))` so the test skips cleanly in CI when no key is set.
5. **Write `.github/workflows/ci.yml`** — lint (ruff), typecheck (mypy), pytest. Store `OPENAI_API_KEY` as a repo secret so the eval actually runs on push; if no key, the test should skip, not fail.
6. **Smoke test locally** — `python -m clauselens.seed && pytest -v`. Expect faithfulness ~0.8, citation F1 ~0.6-0.7 on the toy set. If numbers look "too good" (F1 > 0.9 on 10 clauses), you're probably testing retrieval against a corpus smaller than your top-k. That's a feature of the v0, called out in the README.
7. **Commit in logical chunks** — not one giant "initial commit." The git log is part of the artifact.
8. **Push to GitHub** — `gh repo create <user>/clauselens --private --source=. --push`
9. **Turn `docs/issues/*.md` into real GitHub issues**:
   ```bash
   for f in docs/issues/*.md; do
     title=$(head -n1 "$f" | sed 's/^# Issue [0-9]*: //')
     gh issue create --title "$title" --body-file "$f"
   done
   ```

## The issues roadmap

Seven issues are drafted as markdown in `docs/issues/`:

1. **Tunable retrieval params** — expose `top_k` and `score_threshold` as config
2. **Multi-dataset harness** — `data/datasets/<name>/` layout + `--dataset` flag
3. **Persist eval runs** — append-only SQLite table with git_sha, model, metrics
4. **Hybrid retrieval (BM25 + vector)** — with a tunable `alpha`
5. **Cross-encoder reranker** — two-stage retrieval
6. **OpenTelemetry tracing** — spans on embed/retrieve/generate with token + cost attributes
7. **Abstain logic + eval taxonomy expansion** — multi-hop, comparative, unanswerable cases

Issues 1, 3, and 6 are the most interesting to tackle first.

## Code conventions

- Python 3.12, type hints on all public functions
- Format with `ruff format`, lint with `ruff check`
- Docstrings on module and class level; inline comments sparingly and only where the *why* isn't obvious
- Keep the public surface of each module small (one class / a few functions)
- No new dependencies without a one-line justification in the commit message

## Things to NOT do

- Don't rewrite the README voice — it's deliberately first-person and slightly self-effacing. That's the point.
- Don't add Docker / Postgres / Redis in v0. Those are the upgrade path (Issues 2, 6), not the starting point.
- Don't add auth or user accounts. Out of scope.
- Don't silently change eval thresholds to make CI pass. If the numbers drop, that's a signal.
- Don't expand the eval set past 15-20 cases until Issue 7 lands — the taxonomy matters more than the count.

