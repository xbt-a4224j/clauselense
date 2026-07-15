# ClauseLens

Contract clause Q&A with retrieval-augmented generation. Retrieves relevant clauses from a vector store, generates cited answers, and evaluates quality against labeled ground-truth.

![ClauseLens playground — a cited answer with per-clause similarity scores and a faithfulness-graded confidence badge.](docs/screenshots/playground.png)

*Ask a question → get a **cited** answer with a confidence grade, plus every retrieved clause and its similarity score. The **CITED** clause is what the answer is grounded in — citation accuracy and faithfulness are scored separately, because a wrong citation and a misstated-but-cited clause are different failures.*

## Architecture

| Component | Implementation |
|-----------|---------------|
| Vector store | SQLite + numpy cosine similarity |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini`, structured JSON output |
| API | FastAPI (`POST /ask`, `GET /healthz`) |
| Eval metrics | Citation precision/recall, LLM-as-judge faithfulness |
| Playground | Browser UI at `/` with retrieval parameter controls |

## Project structure

```
clauselens/
  store.py          Vector store (embed, index, search)
  rag.py            Retrieve → generate pipeline
  evals.py          Eval harness and metric computation
  app.py            API and playground server
  static/
    playground.html
data/
  sample_clauses.json
  eval_set.json
tests/
  test_evals.py
```

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python -m clauselens.seed      # index clauses
uvicorn clauselens.app:app     # serve at http://localhost:8000
```

## Adding clauses

Clauses live in `data/sample_clauses.json` as an array of objects:

```json
[
  {"id": "NDA-01", "contract": "Acme NDA", "text": "The Receiving Party shall not disclose..."},
  {"id": "MSA-01", "contract": "SaaS MSA",  "text": "Either party may terminate with 30 days..."}
]
```

After editing the file, re-index and verify:

```bash
python -m clauselens.seed      # re-embeds and upserts all clauses
uvicorn clauselens.app:app     # corpus stats visible at http://localhost:8000
```

The playground UI shows the current corpus breakdown (clause count per contract) in the sidebar. If the corpus is empty, it tells you what to do.

## Iteration loop

The workflow for testing retrieval quality as you expand the corpus:

1. Add or modify clauses in `data/sample_clauses.json`
2. Add corresponding Q&A cases in `data/eval_set.json`
3. Re-index: `python -m clauselens.seed`
4. Run evals: `pytest tests/test_evals.py -v`
5. Tune retrieval parameters (`top_k`, `score_threshold`) in the playground
6. Repeat

As the corpus grows, expect citation F1 to drop — that's the signal to improve retrieval (hybrid search, reranking, better thresholds).

## Eval harness

```bash
pytest tests/test_evals.py -v
```

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | >= 0.8 | Every claim supported by retrieved clauses (LLM-as-judge) |
| Citation F1 | >= 0.7 | Cited clause IDs match labeled ground-truth |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `CLAUSELENS_DB` | `clauselens.db` | SQLite database path |

Retrieval parameters (`top_k`, `score_threshold`) are configurable per request via the API or the playground UI.

## Known limitations

- Vector-only retrieval; no BM25 or reranking stage
- In-memory similarity search; practical up to ~10k clauses
- Eval set is small — metrics on a toy corpus with top_k=4 will overfit

## Roadmap

See [open issues](../../issues).
