# ClauseLens

Contract clause Q&A with retrieval-augmented generation. Retrieves relevant clauses from a vector store, generates cited answers, and evaluates quality against labeled ground-truth.

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
