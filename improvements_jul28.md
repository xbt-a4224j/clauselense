# ClauseLens Tightening Pass — Implementation Plan (Jul 28, 2026)

> Work this plan one task at a time. Each task ends at a commit and is independently
> testable — don't start the next until the previous one's steps all pass. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every claim the repo makes true, provable, and runnable by a stranger without an API key — without growing the surface area.

**Architecture:** Three threads, in dependency order. (1) *Truth* — fix the two places the docs/prompt describe behavior the code doesn't have. (2) *Eval infrastructure* — add the eval runner CLI that Issues 1, 2, and 3 all assume exists, plus a retrieval-only metric path and an embedding cache so evals run offline and CI can gate retrieval on every commit without secrets. (3) *Demoability* — a fresh clone should show something working in one command.

**Tech Stack:** Python 3.12, numpy, SQLite, FastAPI, pytest, ruff, mypy. No new runtime dependencies except `numpy.savez` (already present).

## Global Constraints

- Python 3.12; type hints on all public functions.
- `ruff format` / `ruff check` clean; `mypy clauselens/ --ignore-missing-imports` clean. CI enforces all three.
- **No new dependencies.** If a task seems to need one, it's the wrong task.
- Do not rewrite the README voice — first-person, slightly self-effacing. Tighten claims, keep the register.
- Do not lower an eval threshold to make CI pass. If a number drops, that is the finding.
- `temperature=0.0` everywhere. Evals must be reproducible.
- Commit in logical chunks, one per task minimum. The git log is part of the artifact.
- **Do not push.** Batch commits locally; pushing is a separate, explicit decision.

---

## Findings this plan addresses

Numbered so tasks can reference them.

| # | Finding | Severity |
|---|---|---|
| F1 | `SYSTEM_PROMPT` says "numbered clauses"; `rag.py:70` builds ID-tagged context (`[NDA-01] …`). The prompt then demands `citations: ["<clause id>"]`. Citation F1 is scored on exact string match, so any nudge toward `1`/`[2]` reads as a retrieval failure that is actually a prompt bug. | High |
| F2 | README caption claims a "faithfulness-graded confidence badge." `judge_faithfulness` never runs in the request path — `confidence` is the model's own self-report. | High |
| F3 | No eval runner CLI exists. Issues 1, 2 and 3 all reference `python -m clauselens.evals --…`. Only `pytest` can run evals today. Blocks three issues. | High |
| F4 | A repo about evaluation commits no eval numbers. `AggregateReport.as_markdown()` exists and nothing calls it to a file. | High |
| F5 | Retrieval is barely under test: 10 clauses, `k=4` → every query returns 40% of the corpus. Citation recall near 1.0 is not evidence retrieval works. | High |
| F6 | `store.search()` loads every row and re-normalizes the whole corpus per query. README bills it as good to "~10k clauses"; the honest ceiling is lower. | Medium |
| F7 | `citation_f1` is F1-of-means, not mean-of-F1s (`evals.py:157-159`). Defensible, undocumented, and it hides variance. | Medium |
| F8 | No dev/holdout split. `top_k`, `score_threshold` and prompt edits are all tuned against the same 10 cases. | Medium |
| F9 | Fresh clone → empty DB. `seed.py` requires `OPENAI_API_KEY`. A reviewer without a key sees nothing at all. | Medium |
| F10 | `app.py:60` reaches into `_store._conn` from outside the class. | Low |
| F11 | `.idea/` is tracked. JetBrains config in a public repo. | Low |
| F12 | Issue 1's text is stale — it says `k` is hardcoded, but `ask()`, `AskRequest` and the playground all expose `k` and `score_threshold` as of the Jul 14 commit. | Low |

## Explicitly not in scope

Issue 4 (hybrid BM25), Issue 5 (cross-encoder reranker), Issue 6 (OTel tracing), Issue 7 (abstain logic), and the multi-dataset layout from Issue 2 beyond the dev/holdout split. This plan builds the measurement floor those issues need; it does not start them.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `clauselens/embed_cache.py` | Create | Content-addressed embedding cache (`sha256(text) → vector`), backed by a single `.npz`. Used by both seeding and query embedding so evals run offline. |
| `clauselens/metrics.py` | Create | Retrieval-only metrics (recall@k, MRR). No LLM calls, no OpenAI import. |
| `clauselens/rag.py` | Modify | Fix `SYSTEM_PROMPT` (F1); `embed()` accepts an optional cache. |
| `clauselens/store.py` | Modify | Unit-normalize at write time; cache the stacked matrix; add `contract_counts()`. |
| `clauselens/evals.py` | Modify | Thread `k`/`score_threshold` through `run_eval`; add both F1 flavors; add `main()` CLI. |
| `clauselens/seed.py` | Modify | `--offline` flag reading from the embedding cache; `--write-cache` to populate it. |
| `clauselens/app.py` | Modify | Use `store.contract_counts()` instead of `_conn`. |
| `data/eval_set.json` | Modify | Add a `split` field to every case. |
| `data/embeddings.npz` | Create | Committed cache: 10 clause vectors + 10 question vectors. ~123 KB. |
| `tests/test_prompt_contract.py` | Create | Stub-client tests for the prompt/context contract. No API key. |
| `tests/test_store.py` | Create | Normalization, caching, invalidation. No API key. |
| `tests/test_retrieval_offline.py` | Create | Retrieval metrics over the cached embeddings. No API key. |
| `tests/test_evals.py` | Modify | Split-aware; still skips without a key. |
| `results/README.md`, `results/latest.md` | Create | Committed eval output. |
| `Makefile` | Create | `make demo`, `make eval`, `make eval-offline`, `make test`. |
| `README.md` | Modify | Real numbers, corrected claims, honest limits. |
| `.github/workflows/ci.yml` | Modify | Add an offline retrieval job that needs no secret. |

---

## Task 1: Fix the prompt/context contract

Addresses F1.

**Files:**
- Modify: `clauselens/rag.py:25-39` (`SYSTEM_PROMPT`)
- Test: `tests/test_prompt_contract.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `SYSTEM_PROMPT` continues to be importable from `clauselens.rag`. Tests rely on a `StubClient` defined in this task's test file; later tasks reuse it by importing from `tests.test_prompt_contract`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_contract.py`:

```python
"""Prompt/context contract tests. No API key required — the OpenAI client is stubbed."""
from __future__ import annotations

import json
import types

import numpy as np

from clauselens.rag import SYSTEM_PROMPT, ask
from clauselens.store import ClauseStore

DIM = 8


class StubClient:
    """Minimal OpenAI stand-in. Records the chat kwargs, returns a canned JSON payload."""

    def __init__(self, payload: dict | None = None) -> None:
        self.captured: dict = {}
        self._payload = payload or {
            "answer": "Sixty days.",
            "citations": ["MSA-01"],
            "confidence": "high",
        }
        self.embeddings = types.SimpleNamespace(create=self._embed)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._chat)
        )

    def _embed(self, model: str, input: str | list[str]):
        n = 1 if isinstance(input, str) else len(input)
        data = [types.SimpleNamespace(embedding=[1.0] * DIM) for _ in range(n)]
        return types.SimpleNamespace(data=data)

    def _chat(self, **kwargs):
        self.captured = kwargs
        msg = types.SimpleNamespace(content=json.dumps(self._payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _seeded_store(tmp_path) -> ClauseStore:
    store = ClauseStore(tmp_path / "t.db")
    store.upsert(
        [
            ("MSA-01", "SaaS MSA", "Either party may terminate on 60 days notice.",
             np.ones(DIM, dtype=np.float32)),
            ("NDA-01", "Acme NDA", "The Receiving Party shall not disclose.",
             np.full(DIM, 0.5, dtype=np.float32)),
        ]
    )
    return store


def test_system_prompt_does_not_say_numbered() -> None:
    assert "numbered" not in SYSTEM_PROMPT.lower()


def test_system_prompt_names_the_id_format() -> None:
    assert "ID-tagged" in SYSTEM_PROMPT
    assert "MSA-01" in SYSTEM_PROMPT


def test_context_lines_are_id_tagged(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    client = StubClient()
    ask(store, "What is the termination period?", client=client)
    user_msg = client.captured["messages"][1]["content"]
    assert "[MSA-01]" in user_msg
    assert "[NDA-01]" in user_msg
    store.close()


def test_citations_pass_through_verbatim(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    client = StubClient(
        {"answer": "a", "citations": ["MSA-01"], "confidence": "medium"}
    )
    resp = ask(store, "What is the termination period?", client=client)
    assert resp.citations == ["MSA-01"]
    assert resp.confidence == "medium"
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_prompt_contract.py -v`
Expected: `test_system_prompt_does_not_say_numbered` and `test_system_prompt_names_the_id_format` FAIL. The other two should already pass.

- [ ] **Step 3: Fix the prompt**

Replace `SYSTEM_PROMPT` in `clauselens/rag.py`:

```python
SYSTEM_PROMPT = """You are a careful legal research assistant. Answer the user's \
question using ONLY the ID-tagged clauses provided. Do not invent facts.

Each clause is supplied as: [<clause id>] (from <contract>): <text>

Return STRICT JSON with this shape:
{
  "answer": "<concise natural-language answer>",
  "citations": ["<clause id>", ...],
  "confidence": "high" | "medium" | "low"
}

Rules:
- Every factual claim in "answer" must be supported by at least one cited clause.
- "citations" must contain clause IDs copied verbatim from the brackets, e.g. "MSA-01".
  Never use ordinal positions, list numbers, or contract names as citations.
- If the clauses do not contain the answer, say so in "answer" and set citations to [].
- Do not cite a clause you did not use.
"""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_prompt_contract.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
ruff format clauselens/ tests/ && ruff check clauselens/ tests/
mypy clauselens/ --ignore-missing-imports
git add clauselens/rag.py tests/test_prompt_contract.py
git commit -m "fix(prompt): context is ID-tagged, not numbered

The system prompt told the model to expect numbered clauses while rag.py
supplies bracketed clause IDs. Citation F1 is scored on exact string match
against expected_clause_ids, so a model emitting '1' instead of 'MSA-01'
looked like a retrieval failure. Adds stub-client contract tests that need
no API key."
```

---

## Task 2: Normalize at write time and cache the matrix

Addresses F6, F10.

**Files:**
- Modify: `clauselens/store.py:50-91`
- Modify: `clauselens/app.py:57-66`
- Test: `tests/test_store.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `ClauseStore.upsert()` now stores unit-norm vectors. `ClauseStore.contract_counts() -> dict[str, int]`. `ClauseStore.search()` signature unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
"""ClauseStore behavior. No API key required."""
from __future__ import annotations

import numpy as np

from clauselens.store import ClauseStore


def _store(tmp_path) -> ClauseStore:
    return ClauseStore(tmp_path / "s.db")


def test_upsert_stores_unit_norm_vectors(tmp_path) -> None:
    store = _store(tmp_path)
    store.upsert([("A-01", "C", "text", np.array([3.0, 4.0], dtype=np.float32))])
    raw = store._conn.execute("SELECT embedding FROM clauses").fetchone()[0]
    vec = np.frombuffer(raw, dtype=np.float32)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    store.close()


def test_search_ranks_by_cosine(tmp_path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [
            ("A-01", "C", "near", np.array([1.0, 0.0], dtype=np.float32)),
            ("A-02", "C", "far", np.array([0.0, 1.0], dtype=np.float32)),
        ]
    )
    hits = store.search(np.array([1.0, 0.1], dtype=np.float32), k=2)
    assert [h.id for h in hits] == ["A-01", "A-02"]
    assert hits[0].score > hits[1].score
    store.close()


def test_score_threshold_filters(tmp_path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [
            ("A-01", "C", "near", np.array([1.0, 0.0], dtype=np.float32)),
            ("A-02", "C", "orthogonal", np.array([0.0, 1.0], dtype=np.float32)),
        ]
    )
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2, score_threshold=0.5)
    assert [h.id for h in hits] == ["A-01"]
    store.close()


def test_upsert_invalidates_the_cache(tmp_path) -> None:
    store = _store(tmp_path)
    store.upsert([("A-01", "C", "one", np.array([1.0, 0.0], dtype=np.float32))])
    assert len(store.search(np.array([1.0, 0.0], dtype=np.float32), k=10)) == 1
    store.upsert([("A-02", "C", "two", np.array([0.9, 0.1], dtype=np.float32))])
    assert len(store.search(np.array([1.0, 0.0], dtype=np.float32), k=10)) == 2
    store.close()


def test_contract_counts(tmp_path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [
            ("A-01", "NDA", "x", np.array([1.0, 0.0], dtype=np.float32)),
            ("A-02", "NDA", "y", np.array([0.0, 1.0], dtype=np.float32)),
            ("B-01", "MSA", "z", np.array([1.0, 1.0], dtype=np.float32)),
        ]
    )
    assert store.contract_counts() == {"MSA": 1, "NDA": 2}
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: `test_upsert_stores_unit_norm_vectors` FAILS (vectors stored raw), `test_contract_counts` FAILS with `AttributeError`. The rest pass.

- [ ] **Step 3: Rewrite the store internals**

Replace the body of `ClauseStore` below `__init__` in `clauselens/store.py`:

```python
    def _invalidate(self) -> None:
        self._rows: list[tuple[str, str, str]] | None = None
        self._embs: np.ndarray | None = None

    def _load(self) -> tuple[list[tuple[str, str, str]], np.ndarray]:
        """Load and cache the corpus matrix. Vectors are already unit-norm on disk."""
        if self._rows is None or self._embs is None:
            raw = self._conn.execute(
                "SELECT id, contract, text, embedding FROM clauses"
            ).fetchall()
            self._rows = [(r[0], r[1], r[2]) for r in raw]
            self._embs = (
                np.stack([np.frombuffer(r[3], dtype=np.float32) for r in raw])
                if raw
                else np.zeros((0, 0), dtype=np.float32)
            )
        return self._rows, self._embs

    def upsert(self, clauses: Iterable[tuple[str, str, str, np.ndarray]]) -> None:
        """clauses is an iterable of (id, contract, text, embedding).

        Embeddings are unit-normalized here so search is a plain matmul.
        """
        rows = []
        for cid, contract, text, emb in clauses:
            vec = np.asarray(emb, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            rows.append((cid, contract, text, vec.tobytes()))
        self._conn.executemany(
            "INSERT OR REPLACE INTO clauses (id, contract, text, embedding) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._invalidate()

    def search(
        self, query_embedding: np.ndarray, k: int = 4, score_threshold: float = 0.0
    ) -> list[Clause]:
        """Return top-k clauses by cosine similarity.

        Corpus vectors are unit-norm at rest and the stacked matrix is cached, so a
        query costs one normalize + one matmul. Still O(n) in corpus size per query —
        swap for pgvector well before you reach six figures of clauses.
        """
        rows, embs = self._load()
        if not rows:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        scores = embs @ q
        top_idx = np.argsort(-scores)[:k]

        return [
            Clause(id=rows[i][0], contract=rows[i][1], text=rows[i][2], score=float(scores[i]))
            for i in top_idx
            if float(scores[i]) >= score_threshold
        ]

    def contract_counts(self) -> dict[str, int]:
        """Clause count per contract, ascending by contract name."""
        rows = self._conn.execute(
            "SELECT contract, COUNT(*) FROM clauses GROUP BY contract ORDER BY contract"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
```

Add `self._invalidate()` as the last line of `__init__`.

- [ ] **Step 4: Use the new accessor in the API**

In `clauselens/app.py`, replace the body of `corpus_info()`:

```python
@app.get("/corpus")
def corpus_info() -> dict:
    """Summary stats about the indexed corpus."""
    return {
        "total_clauses": _store.count(),
        "contracts": _store.contract_counts(),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_store.py tests/test_prompt_contract.py -v`
Expected: 9 passed.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
ruff format clauselens/ tests/ && ruff check clauselens/ tests/
mypy clauselens/ --ignore-missing-imports
git add clauselens/store.py clauselens/app.py tests/test_store.py
git commit -m "perf(store): normalize at write time, cache the corpus matrix

search() previously re-read every BLOB and re-normalized the whole corpus on
every query. Vectors are now unit-norm at rest and the stacked matrix is
cached, so a query is one normalize plus one matmul. Adds contract_counts()
so app.py stops reaching into _store._conn."
```

---

## Task 3: Content-addressed embedding cache

Addresses F9 and unblocks offline evals in Task 5.

**Files:**
- Create: `clauselens/embed_cache.py`
- Modify: `clauselens/rag.py` (`embed()`)
- Modify: `clauselens/seed.py`
- Test: `tests/test_embed_cache.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `EmbedCache(path)` with `.get(text) -> np.ndarray | None`, `.put(text, vec)`, `.save()`, `.__len__()`. `rag.embed(client, text, cache=None)` — when `cache` is supplied, a hit skips the API call and a miss stores the result. `seed(db_path, data_path, cache_path=None, offline=False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed_cache.py`:

```python
"""Embedding cache. No API key required."""
from __future__ import annotations

import numpy as np
import pytest

from clauselens.embed_cache import EmbedCache


def test_roundtrip(tmp_path) -> None:
    path = tmp_path / "e.npz"
    cache = EmbedCache(path)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.put("hello", vec)
    cache.save()

    reloaded = EmbedCache(path)
    got = reloaded.get("hello")
    assert got is not None
    assert np.allclose(got, vec)


def test_miss_returns_none(tmp_path) -> None:
    cache = EmbedCache(tmp_path / "e.npz")
    assert cache.get("never seen") is None


def test_missing_file_is_an_empty_cache(tmp_path) -> None:
    cache = EmbedCache(tmp_path / "does-not-exist.npz")
    assert len(cache) == 0


def test_key_is_content_addressed(tmp_path) -> None:
    cache = EmbedCache(tmp_path / "e.npz")
    cache.put("same text", np.ones(3, dtype=np.float32))
    cache.put("same text", np.zeros(3, dtype=np.float32))
    assert len(cache) == 1
    assert np.allclose(cache.get("same text"), np.zeros(3))


def test_strict_get_raises_on_miss(tmp_path) -> None:
    cache = EmbedCache(tmp_path / "e.npz")
    with pytest.raises(KeyError, match="not in the embedding cache"):
        cache.require("absent")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_embed_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clauselens.embed_cache'`.

- [ ] **Step 3: Write the cache**

Create `clauselens/embed_cache.py`:

```python
"""
Content-addressed embedding cache.

Keyed by sha256 of the embedded text, so the same string always resolves to the
same vector regardless of where it came from — clause or question. Committing a
populated cache is what lets a stranger clone the repo and run retrieval and the
retrieval-only evals with no OPENAI_API_KEY.

Backed by a single .npz. At 1536 dims and float32 that is ~6 KB per entry.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def cache_key(text: str) -> str:
    """Stable 16-hex-char key for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbedCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as npz:
                self._data = {k: npz[k].astype(np.float32) for k in npz.files}

    def get(self, text: str) -> np.ndarray | None:
        return self._data.get(cache_key(text))

    def require(self, text: str) -> np.ndarray:
        """Like get(), but raises instead of silently falling back to an API call."""
        vec = self.get(text)
        if vec is None:
            preview = text[:60].replace("\n", " ")
            raise KeyError(
                f"{cache_key(text)} ({preview!r}) is not in the embedding cache. "
                f"Re-run with an API key and --write-cache to populate it."
            )
        return vec

    def put(self, text: str, vec: np.ndarray) -> None:
        self._data[cache_key(text)] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self._data)

    def __len__(self) -> int:
        return len(self._data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_embed_cache.py -v`
Expected: 5 passed.

- [ ] **Step 5: Wire the cache into `embed()`**

In `clauselens/rag.py`, replace `embed()`:

```python
def embed(client: OpenAI, text: str, cache: "EmbedCache | None" = None) -> np.ndarray:
    """Embed a single string, consulting `cache` first when one is supplied."""
    if cache is not None:
        hit = cache.get(text)
        if hit is not None:
            return hit
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    if cache is not None:
        cache.put(text, vec)
    return vec
```

Add the import at the top of `rag.py`:

```python
from .embed_cache import EmbedCache  # noqa: TC001  (runtime use in default arg annotation)
```

Then thread it through `ask()` — change the signature and the retrieve step:

```python
def ask(
    store: ClauseStore,
    question: str,
    k: int = 4,
    score_threshold: float = 0.0,
    client: OpenAI | None = None,
    cache: EmbedCache | None = None,
) -> RagResponse:
    client = client or OpenAI()

    # 1. retrieve
    q_emb = embed(client, question, cache=cache)
    retrieved = store.search(q_emb, k=k, score_threshold=score_threshold)
```

- [ ] **Step 6: Add `--offline` and `--write-cache` to seed**

In `clauselens/seed.py`, replace `seed()` and `main()`:

```python
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings.npz"


def seed(
    db_path: str,
    data_path: str | Path,
    cache_path: str | Path | None = None,
    offline: bool = False,
    write_cache: bool = False,
) -> None:
    """Load clauses from JSON, embed them, and upsert into the store.

    offline=True embeds entirely from the cache and never calls the API.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    clauses = json.loads(data_path.read_text())
    if not clauses:
        print("Error: no clauses found in data file.", file=sys.stderr)
        sys.exit(1)

    cache = EmbedCache(cache_path or DEFAULT_CACHE)
    texts = [c["text"] for c in clauses]

    if offline:
        try:
            embeddings = [cache.require(t) for t in texts]
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Embedding {len(clauses)} clauses from cache (offline).")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY is not set.\n"
                "Set it in your environment or in a .env file:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "Or run offline from the committed cache:\n"
                "  python -m clauselens.seed --offline",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        print(f"Embedding {len(clauses)} clauses with {EMBED_MODEL}...")
        embeddings = _embed_batch(client, texts)
        if write_cache:
            for text, emb in zip(texts, embeddings):
                cache.put(text, emb)
            cache.save()
            print(f"Wrote {len(cache)} vectors to {cache.path}.")

    store = ClauseStore(db_path)
    store.upsert(
        [(c["id"], c["contract"], c["text"], emb) for c, emb in zip(clauses, embeddings)]
    )
    contracts = {c["contract"] for c in clauses}
    print(f"Indexed {len(clauses)} clauses from {len(contracts)} contracts.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the ClauseLens clause store.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to clauses JSON")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Embedding cache .npz")
    parser.add_argument(
        "--offline", action="store_true", help="Embed from the cache; never call the API"
    )
    parser.add_argument(
        "--write-cache", action="store_true", help="Persist fresh embeddings to the cache"
    )
    args = parser.parse_args()
    seed(args.db, args.data, args.cache, offline=args.offline, write_cache=args.write_cache)
```

Add `from .embed_cache import EmbedCache` to the imports.

- [ ] **Step 7: Populate and commit the cache**

Requires `OPENAI_API_KEY` once. This is the only step in the plan that does.

```bash
export OPENAI_API_KEY=sk-...
python -m clauselens.seed --write-cache
python - <<'PY'
import json, os
from openai import OpenAI
from clauselens.embed_cache import EmbedCache
from clauselens.rag import EMBED_MODEL
from clauselens.seed import DEFAULT_CACHE

cases = json.load(open("data/eval_set.json"))
questions = [c["question"] for c in cases]
cache = EmbedCache(DEFAULT_CACHE)
client = OpenAI()
resp = client.embeddings.create(model=EMBED_MODEL, input=questions)
for q, d in zip(questions, resp.data):
    cache.put(q, d.embedding)
cache.save()
print(f"cache now holds {len(cache)} vectors")
PY
```

Expected: `cache now holds 20 vectors`.

- [ ] **Step 8: Allow the cache past .gitignore**

`.gitignore` currently has `*.db` but nothing excluding `.npz`. Confirm the cache is not ignored:

Run: `git check-ignore -v data/embeddings.npz`
Expected: no output (exit 1) — the file is not ignored.

- [ ] **Step 9: Run the full offline suite**

```bash
rm -f clauselens.db
unset OPENAI_API_KEY
python -m clauselens.seed --offline
pytest tests/test_embed_cache.py tests/test_store.py tests/test_prompt_contract.py -v
```

Expected: seed prints `Embedding 10 clauses from cache (offline).` then `Indexed 10 clauses from 3 contracts.`; tests pass.

- [ ] **Step 10: Commit**

```bash
git add clauselens/embed_cache.py clauselens/rag.py clauselens/seed.py \
        tests/test_embed_cache.py data/embeddings.npz
git commit -m "feat(embed): content-addressed embedding cache + offline seeding

Commits the 20 vectors (10 clauses, 10 eval questions) the toy corpus needs so
a fresh clone can seed and retrieve with no OPENAI_API_KEY. Unblocks
retrieval-only evals in CI without a repo secret."
```

---

## Task 4: Retrieval-only metrics

Addresses F5.

**Files:**
- Create: `clauselens/metrics.py`
- Test: `tests/test_metrics.py` (create)

**Interfaces:**
- Consumes: nothing (deliberately no OpenAI import — this module is pure math).
- Produces: `recall_at_k(expected, retrieved) -> float`, `mrr(expected, retrieved) -> float`, `precision_at_k(expected, retrieved) -> float`. All take `expected: set[str] | list[str]` and `retrieved: list[str]` in rank order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Retrieval metrics. Pure math — no API key, no network."""
from __future__ import annotations

import pytest

from clauselens.metrics import mrr, precision_at_k, recall_at_k


@pytest.mark.parametrize(
    "expected,retrieved,want",
    [
        (["A"], ["A", "B", "C"], 1.0),
        (["A"], ["B", "C"], 0.0),
        (["A", "B"], ["A", "C"], 0.5),
        (["A", "B"], ["A", "B"], 1.0),
    ],
)
def test_recall_at_k(expected, retrieved, want) -> None:
    assert recall_at_k(expected, retrieved) == want


@pytest.mark.parametrize(
    "expected,retrieved,want",
    [
        (["A"], ["A", "B"], 1.0),
        (["A"], ["B", "A"], 0.5),
        (["A"], ["B", "C", "A"], pytest.approx(1 / 3)),
        (["A"], ["B", "C"], 0.0),
        (["B", "C"], ["A", "C", "B"], 0.5),
    ],
)
def test_mrr_uses_the_first_relevant_rank(expected, retrieved, want) -> None:
    assert mrr(expected, retrieved) == want


def test_precision_at_k() -> None:
    assert precision_at_k(["A", "B"], ["A", "X", "B", "Y"]) == 0.5


def test_empty_expected_is_perfect_recall() -> None:
    assert recall_at_k([], ["A"]) == 1.0


def test_empty_retrieved_is_zero() -> None:
    assert recall_at_k(["A"], []) == 0.0
    assert mrr(["A"], []) == 0.0
    assert precision_at_k(["A"], []) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clauselens.metrics'`.

- [ ] **Step 3: Write the metrics**

Create `clauselens/metrics.py`:

```python
"""
Retrieval-only metrics.

These grade what the *search* returned, independent of what the LLM chose to cite.
That separation matters: citation F1 conflates a retrieval miss with a generation
mistake, and on a 10-clause corpus at k=4 it mostly measures the latter.

No OpenAI import here on purpose — these run offline, in CI, on every commit.
"""
from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(expected: Iterable[str], retrieved: list[str]) -> float:
    """Fraction of expected ids that appear anywhere in `retrieved`."""
    exp = set(expected)
    if not exp:
        return 1.0
    return len(exp & set(retrieved)) / len(exp)


def precision_at_k(expected: Iterable[str], retrieved: list[str]) -> float:
    """Fraction of retrieved ids that were expected."""
    if not retrieved:
        return 0.0
    return len(set(expected) & set(retrieved)) / len(retrieved)


def mrr(expected: Iterable[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first relevant hit. 0.0 if none was retrieved."""
    exp = set(expected)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in exp:
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 12 passed (4 recall cases + 5 MRR cases + 3 standalone).

- [ ] **Step 5: Commit**

```bash
ruff format clauselens/ tests/ && ruff check clauselens/ tests/
mypy clauselens/ --ignore-missing-imports
git add clauselens/metrics.py tests/test_metrics.py
git commit -m "feat(evals): retrieval-only metrics (recall@k, MRR, P@k)

Grades what search returned, independent of what the model cited. Pure math,
no API calls — these can gate every commit."
```

---

## Task 5: Split the eval set, add both F1 flavors, ship the runner CLI

Addresses F3, F7, F8.

**Files:**
- Modify: `data/eval_set.json`
- Modify: `clauselens/evals.py`
- Test: `tests/test_retrieval_offline.py` (create), `tests/test_evals.py` (modify)

**Interfaces:**
- Consumes: `metrics.recall_at_k`, `metrics.mrr`, `metrics.precision_at_k` from Task 4; `EmbedCache` from Task 3.
- Produces: `EvalCase` gains `split: str = "dev"`. `load_eval_set(path, split="all")`. `run_eval(store, case, client=None, k=4, score_threshold=0.0, cache=None)`. `run_retrieval_eval(store, case, client=None, k=4, cache=None) -> RetrievalResult`. `AggregateReport` gains `citation_f1_mean` and `retrieval_recall`. `python -m clauselens.evals` CLI.

- [ ] **Step 1: Add splits to the eval set**

Edit `data/eval_set.json`, adding `"split"` to each of the 10 objects. Assign so that both hard shapes appear in dev *and* holdout — the adversarial pair 3/4 splits across, and each side keeps one multi-clause case:

| # | Question topic | split |
|---|---|---|
| 1 | SaaS termination notice | `dev` |
| 2 | NDA confidentiality survival | `dev` |
| 3 | NDA assignment without consent | `dev` |
| 4 | SaaS assignment without consent | `holdout` |
| 5 | Assignment across all contracts (multi) | `dev` |
| 6 | SaaS liability cap | `dev` |
| 7 | Vendor insurance | `holdout` |
| 8 | Vendor breach (multi) | `holdout` |
| 9 | Security incident reporting | `dev` |
| 10 | SaaS excluded damages | `holdout` |

Result: 6 dev, 4 holdout.

- [ ] **Step 2: Write the failing offline retrieval test**

Create `tests/test_retrieval_offline.py`:

```python
"""End-to-end retrieval quality over the committed embedding cache. No API key."""
from __future__ import annotations

from pathlib import Path

import pytest

from clauselens.embed_cache import EmbedCache
from clauselens.evals import load_eval_set, run_retrieval_eval
from clauselens.seed import DEFAULT_CACHE, seed
from clauselens.store import ClauseStore

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "data" / "eval_set.json"
CLAUSES = ROOT / "data" / "sample_clauses.json"


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> ClauseStore:
    db = str(tmp_path_factory.mktemp("cl") / "offline.db")
    seed(db, CLAUSES, DEFAULT_CACHE, offline=True)
    s = ClauseStore(db)
    yield s
    s.close()


@pytest.fixture(scope="module")
def cache() -> EmbedCache:
    return EmbedCache(DEFAULT_CACHE)


def test_every_eval_question_is_cached(cache) -> None:
    for case in load_eval_set(EVAL_SET):
        assert cache.get(case.question) is not None, f"uncached: {case.question}"


def test_splits_are_balanced() -> None:
    cases = load_eval_set(EVAL_SET)
    dev = [c for c in cases if c.split == "dev"]
    holdout = [c for c in cases if c.split == "holdout"]
    assert len(dev) == 6
    assert len(holdout) == 4


def test_retrieval_recall_at_4(store, cache) -> None:
    cases = load_eval_set(EVAL_SET)
    results = [run_retrieval_eval(store, c, k=4, cache=cache) for c in cases]
    recall = sum(r.recall_at_k for r in results) / len(results)
    assert recall >= 0.9, f"recall@4 {recall:.2f} below 0.90"


def test_retrieval_recall_degrades_at_k1(store, cache) -> None:
    """Sanity: k=1 must be strictly harder than k=4, or the metric is not measuring."""
    cases = load_eval_set(EVAL_SET)
    r4 = sum(run_retrieval_eval(store, c, k=4, cache=cache).recall_at_k for c in cases)
    r1 = sum(run_retrieval_eval(store, c, k=1, cache=cache).recall_at_k for c in cases)
    assert r1 < r4
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_retrieval_offline.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_retrieval_eval'`.

- [ ] **Step 4: Extend `evals.py`**

Add to the imports:

```python
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from .embed_cache import EmbedCache
from .metrics import mrr, precision_at_k, recall_at_k
from .rag import embed
```

Change `EvalCase` and `load_eval_set`:

```python
@dataclass
class EvalCase:
    question: str
    expected_clause_ids: list[str]
    notes: str = ""
    split: str = "dev"


def load_eval_set(path: str | Path, split: str = "all") -> list[EvalCase]:
    """Load eval cases. `split` is 'dev', 'holdout', or 'all'."""
    data = json.loads(Path(path).read_text())
    cases = [EvalCase(**row) for row in data]
    if split == "all":
        return cases
    return [c for c in cases if c.split == split]
```

Add the retrieval-only path:

```python
@dataclass
class RetrievalResult:
    question: str
    expected: list[str]
    retrieved: list[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float


def run_retrieval_eval(
    store: ClauseStore,
    case: EvalCase,
    client: OpenAI | None = None,
    k: int = 4,
    score_threshold: float = 0.0,
    cache: EmbedCache | None = None,
) -> RetrievalResult:
    """Grade retrieval alone. With a populated cache this makes zero API calls."""
    if cache is not None and cache.get(case.question) is not None:
        q_emb = cache.require(case.question)
    else:
        q_emb = embed(client or OpenAI(), case.question, cache=cache)

    hits = store.search(q_emb, k=k, score_threshold=score_threshold)
    retrieved = [c.id for c in hits]
    return RetrievalResult(
        question=case.question,
        expected=list(case.expected_clause_ids),
        retrieved=retrieved,
        recall_at_k=recall_at_k(case.expected_clause_ids, retrieved),
        precision_at_k=precision_at_k(case.expected_clause_ids, retrieved),
        mrr=mrr(case.expected_clause_ids, retrieved),
    )
```

Thread `k`, `score_threshold` and `cache` through `run_eval`:

```python
def run_eval(
    store: ClauseStore,
    case: EvalCase,
    client: OpenAI | None = None,
    k: int = 4,
    score_threshold: float = 0.0,
    cache: EmbedCache | None = None,
) -> EvalResult:
    client = client or OpenAI()
    resp: RagResponse = ask(
        store, case.question, k=k, score_threshold=score_threshold,
        client=client, cache=cache,
    )
    ...
```

Add both F1 flavors to `AggregateReport` and `aggregate()`:

```python
@dataclass
class AggregateReport:
    n: int
    faithfulness: float
    citation_precision: float
    citation_recall: float
    citation_f1: float        # F1 of the averaged P and R (macro)
    citation_f1_mean: float   # mean of per-case F1 — lower when variance is high
    results: list[EvalResult]
```

```python
def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def aggregate(results: list[EvalResult]) -> AggregateReport:
    if not results:
        return AggregateReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    p = mean(r.citation_precision for r in results)
    r_ = mean(r.citation_recall for r in results)
    return AggregateReport(
        n=len(results),
        faithfulness=mean(1.0 if r.faithful else 0.0 for r in results),
        citation_precision=p,
        citation_recall=r_,
        citation_f1=_f1(p, r_),
        citation_f1_mean=mean(
            _f1(r.citation_precision, r.citation_recall) for r in results
        ),
        results=results,
    )
```

Update `as_markdown()` to report both and say what the difference means:

```python
    def as_markdown(self) -> str:
        gap = self.citation_f1 - self.citation_f1_mean
        lines = [
            "# ClauseLens eval report",
            "",
            f"- cases: **{self.n}**",
            f"- faithfulness: **{self.faithfulness:.2f}**",
            f"- citation precision: **{self.citation_precision:.2f}**",
            f"- citation recall: **{self.citation_recall:.2f}**",
            f"- citation F1 (macro, of the means): **{self.citation_f1:.2f}**",
            f"- citation F1 (mean of per-case): **{self.citation_f1_mean:.2f}**",
            "",
            f"> Gap between the two F1s is {gap:+.2f}. A large positive gap means a few "
            f"cases are failing badly while the averages look healthy.",
            "",
            "| # | faithful | precision | recall | question |",
            "|---|----------|-----------|--------|----------|",
        ]
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"| {i} | {'✓' if r.faithful else '✗'} | {r.citation_precision:.2f} | "
                f"{r.citation_recall:.2f} | {r.question} |"
            )
        return "\n".join(lines)
```

Add the CLI at the bottom of `evals.py`:

```python
def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClauseLens eval suite.")
    parser.add_argument("--db", default=os.environ.get("CLAUSELENS_DB", "clauselens.db"))
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--cache", default="data/embeddings.npz")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and judging. Runs offline from the cache.",
    )
    parser.add_argument("--out", default=None, help="Write the markdown report here")
    parser.add_argument("--notes", default="", help="Free-text tag for this run")
    args = parser.parse_args()

    cases = load_eval_set(args.eval_set, split=args.split)
    if not cases:
        print(f"No cases in split '{args.split}'.", file=sys.stderr)
        sys.exit(1)

    store = ClauseStore(args.db)
    if store.count() == 0:
        print("No clauses indexed. Run: python -m clauselens.seed --offline", file=sys.stderr)
        sys.exit(1)

    cache = EmbedCache(args.cache)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"<!-- run {stamp} · {_git_sha()} · split={args.split} · k={args.top_k} · "
        f"threshold={args.score_threshold} · notes={args.notes or 'none'} -->"
    )

    if args.retrieval_only:
        rows = [
            run_retrieval_eval(
                store, c, k=args.top_k, score_threshold=args.score_threshold, cache=cache
            )
            for c in cases
        ]
        body = "\n".join(
            [
                "# ClauseLens retrieval report",
                "",
                f"- cases: **{len(rows)}** (split: {args.split})",
                f"- recall@{args.top_k}: **{mean(r.recall_at_k for r in rows):.2f}**",
                f"- precision@{args.top_k}: **{mean(r.precision_at_k for r in rows):.2f}**",
                f"- MRR: **{mean(r.mrr for r in rows):.2f}**",
                "",
                "| # | recall | MRR | expected | retrieved |",
                "|---|--------|-----|----------|-----------|",
            ]
            + [
                f"| {i} | {r.recall_at_k:.2f} | {r.mrr:.2f} | "
                f"{', '.join(r.expected)} | {', '.join(r.retrieved)} |"
                for i, r in enumerate(rows, 1)
            ]
        )
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OPENAI_API_KEY is required for the full eval.\n"
                "For an offline retrieval-only run:\n"
                "  python -m clauselens.evals --retrieval-only",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OpenAI()
        results = [
            run_eval(
                store, c, client=client, k=args.top_k,
                score_threshold=args.score_threshold, cache=cache,
            )
            for c in cases
        ]
        body = aggregate(results).as_markdown()

    store.close()
    report = f"{header}\n\n{body}\n"
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Make the existing test suite split-aware**

In `tests/test_evals.py`, change the `eval_results` fixture to run the dev split only, and update the coverage assertion:

```python
@pytest.fixture(scope="module")
def eval_results(seeded_db: str) -> list[EvalResult]:
    """Run the dev split once and cache the results for all tests."""
    store = ClauseStore(seeded_db)
    client = OpenAI()
    cases = load_eval_set(EVAL_SET_PATH, split="dev")
    results = [run_eval(store, case, client=client) for case in cases]
    store.close()
    return results


@skip_no_key
def test_eval_coverage(eval_results: list[EvalResult]) -> None:
    """Sanity check: we actually ran the whole dev split."""
    assert len(eval_results) == 6
```

- [ ] **Step 6: Run the offline tests to verify they pass**

```bash
unset OPENAI_API_KEY
pytest tests/test_retrieval_offline.py tests/test_metrics.py -v
```

Expected: all pass. If `test_retrieval_recall_at_4` fails, **do not lower the threshold** — record the actual number and carry it into the README in Task 7 as the honest result.

- [ ] **Step 7: Verify the CLI works offline**

```bash
python -m clauselens.evals --retrieval-only --split all
```

Expected: a markdown retrieval report on stdout with recall@4, precision@4, MRR, and a per-case table.

- [ ] **Step 8: Commit**

```bash
ruff format clauselens/ tests/ && ruff check clauselens/ tests/
mypy clauselens/ --ignore-missing-imports
git add clauselens/evals.py data/eval_set.json tests/test_evals.py tests/test_retrieval_offline.py
git commit -m "feat(evals): runner CLI, dev/holdout split, retrieval-only mode

Adds the 'python -m clauselens.evals' entry point that Issues 1, 2 and 3 all
assume exists. Splits the 10 cases 6 dev / 4 holdout so hyperparameter tuning
stops fitting the whole set. Reports both F1-of-means and mean-of-F1s, because
the gap between them is where the variance hides."
```

---

## Task 6: Gate retrieval in CI without a secret

Addresses F5.

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `python -m clauselens.seed --offline`, `pytest tests/test_retrieval_offline.py` from Tasks 3–5.
- Produces: a CI job named `retrieval-offline` that runs on every push with no repo secret.

- [ ] **Step 1: Add the job**

Insert into `.github/workflows/ci.yml` after the `typecheck` job:

```yaml
  retrieval-offline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - name: Seed from the committed embedding cache
        run: python -m clauselens.seed --offline
      - name: Retrieval + unit tests (no API key)
        run: pytest tests/test_metrics.py tests/test_store.py tests/test_embed_cache.py tests/test_prompt_contract.py tests/test_retrieval_offline.py -v
      - name: Retrieval report
        run: python -m clauselens.evals --retrieval-only --split all
```

- [ ] **Step 2: Verify locally exactly as CI will run it**

```bash
rm -f clauselens.db
unset OPENAI_API_KEY
python -m clauselens.seed --offline
pytest tests/test_metrics.py tests/test_store.py tests/test_embed_cache.py tests/test_prompt_contract.py tests/test_retrieval_offline.py -v
python -m clauselens.evals --retrieval-only --split all
```

Expected: seed succeeds, all tests pass, report prints.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: gate retrieval quality on every push, no secret required

The existing test job skips entirely when OPENAI_API_KEY is absent, which means
forks and PRs from outside get no signal at all. The offline job runs the whole
retrieval path from the committed cache."
```

---

## Task 7: Commit real numbers and correct the README

Addresses F2, F4, F12, and the honesty items from F5/F6.

**Files:**
- Create: `results/README.md`, `results/latest.md`, `results/retrieval.md`
- Modify: `README.md`
- Modify: `docs/issues/01-tunable-retrieval-params.md`

**Interfaces:**
- Consumes: the CLI from Task 5.
- Produces: committed evidence. No code interfaces.

- [ ] **Step 1: Generate the retrieval report (offline)**

```bash
python -m clauselens.evals --retrieval-only --split all --out results/retrieval.md
```

- [ ] **Step 2: Generate the full report (needs a key, one time)**

```bash
export OPENAI_API_KEY=sk-...
python -m clauselens.evals --split all --notes "baseline, k=4" --out results/latest.md
```

Record the actual faithfulness, precision, recall, and both F1 numbers. **Use whatever they are.** If they're worse than the README's thresholds, that is the finding and it goes in the README as such.

- [ ] **Step 3: Write `results/README.md`**

```markdown
# Eval results

Committed output from `python -m clauselens.evals`. Regenerate with:

```bash
python -m clauselens.evals --retrieval-only --split all --out results/retrieval.md   # offline
python -m clauselens.evals --split all --out results/latest.md                        # needs a key
```

- `retrieval.md` — recall@k / MRR. Grades search alone. No API calls.
- `latest.md` — citation P/R/F1 + LLM-as-judge faithfulness. Grades the full pipeline.

Each report's HTML comment header records the timestamp, git SHA, split, and
retrieval params so a number can be traced back to the code that produced it.

**Read these with the corpus size in mind.** 10 clauses at k=4 means every query
retrieves 40% of the corpus, so recall is close to free and these numbers say far
more about citation behavior than about retrieval quality. Issue #2 (a CUAD subset)
is what makes them mean something.
```

- [ ] **Step 4: Fix the README's false claim**

In `README.md`, replace the image caption (line 7). The current text claims a "faithfulness-graded confidence badge"; faithfulness never runs in the request path.

```markdown
*Ask a question → get a **cited** answer with the model's own confidence, plus every
retrieved clause and its similarity score. The **CITED** clause is what the answer is
grounded in. Citation accuracy and faithfulness are scored separately — a wrong
citation and a misstated-but-cited clause are different failures — but that scoring
happens in the eval harness, not on the request path. The badge here is the model
self-reporting.*
```

- [ ] **Step 5: Add a results section to the README**

Insert after the Architecture table:

```markdown
## What it actually scores

Latest committed run (`results/latest.md`, dev + holdout, k=4):

| Metric | Value | Threshold |
|--------|-------|-----------|
| Faithfulness | _fill from results/latest.md_ | ≥ 0.80 |
| Citation F1 (macro) | _fill_ | ≥ 0.70 |
| Citation F1 (mean of per-case) | _fill_ | — |
| Recall@4 (retrieval only) | _fill from results/retrieval.md_ | ≥ 0.90 |

The two F1s differ because one averages precision and recall and then combines them,
while the other averages per-case F1. A large gap means a few cases are failing badly
behind healthy-looking averages.

**Caveat that matters more than the numbers:** the corpus is 10 clauses and `top_k`
is 4, so every query retrieves 40% of everything. Recall is nearly free. What these
numbers measure is whether the model cites the right clause from a set it was
essentially handed — not whether retrieval works. Issue #2 is the fix.
```

- [ ] **Step 6: Correct the known-limitations section**

```markdown
## Known limitations

- Vector-only retrieval; no BM25 or reranking stage (Issues #4, #5)
- Search is O(corpus) per query — one matmul over the full cached matrix. Fine into the
  low thousands of clauses; swap for pgvector before you get near six figures
- 10-clause corpus at k=4 means retrieval is barely under test (see above)
- The eval set is 10 cases, all single-hop factual extraction. No multi-hop,
  comparative, or unanswerable cases yet (Issue #7)
- No abstain path — the system always tries to answer
```

- [ ] **Step 7: Update the quickstart for offline mode**

```markdown
## Quickstart

No API key needed to see retrieval work:

```bash
pip install -r requirements.txt
python -m clauselens.seed --offline          # index from the committed embedding cache
python -m clauselens.evals --retrieval-only --split all
```

For generation (answers, citations, faithfulness) you need a key:

```bash
export OPENAI_API_KEY=sk-...
python -m clauselens.seed                    # re-embed from the API
uvicorn clauselens.app:app                   # playground at http://localhost:8000
python -m clauselens.evals --split dev       # full eval on the dev split
```
```

- [ ] **Step 8: Refresh the stale issue**

`docs/issues/01-tunable-retrieval-params.md` claims `k` is hardcoded. It isn't — `ask()`, `AskRequest`, and the playground all expose `top_k` and `score_threshold` as of the Jul 14 commit, and Task 5 added `--top-k` / `--score-threshold` to the runner. Replace the Context section:

```markdown
## Context

`top_k` and `score_threshold` are now threaded through `ask()`, the `/ask` request
model, the playground controls, and the eval runner CLI. What's still missing is a
single typed config object rather than parameters repeated at four call sites, plus
a context-length guardrail and a recorded before/after comparison.
```

And update the acceptance criteria — strike the first two boxes (done), keep `RetrievalConfig`, `max_context_tokens`, and `docs/tuning.md`.

- [ ] **Step 9: Commit**

```bash
git add results/ README.md docs/issues/01-tunable-retrieval-params.md
git commit -m "docs: commit real eval numbers, correct the confidence claim

The README described a 'faithfulness-graded confidence badge' — faithfulness
never runs on the request path; the badge is the model's self-report. Adds
committed eval and retrieval reports so the repo shows its own numbers, with
the k=4-over-10-clauses caveat stated plainly. Refreshes Issue 1, whose text
predated the tunable-params work."
```

---

## Task 8: Demo entry point and repo hygiene

Addresses F9, F11.

**Files:**
- Create: `Makefile`
- Modify: `.gitignore`
- Delete from index: `.idea/`

**Interfaces:**
- Consumes: everything above.
- Produces: `make demo`, `make eval`, `make eval-offline`, `make test`, `make check`.

- [ ] **Step 1: Write the Makefile**

```makefile
.PHONY: demo eval eval-offline test check clean

## Index from the committed cache and serve the playground. No API key needed to boot;
## /ask still requires one.
demo:
	python -m clauselens.seed --offline
	uvicorn clauselens.app:app

## Full eval on the dev split. Requires OPENAI_API_KEY.
eval:
	python -m clauselens.evals --split dev --out results/latest.md

## Retrieval-only eval across every case. Runs offline.
eval-offline:
	python -m clauselens.evals --retrieval-only --split all --out results/retrieval.md

## Everything that runs without a key.
test:
	pytest tests/test_metrics.py tests/test_store.py tests/test_embed_cache.py \
	       tests/test_prompt_contract.py tests/test_retrieval_offline.py -v

## What CI enforces.
check:
	ruff format --check .
	ruff check .
	mypy clauselens/ --ignore-missing-imports
	$(MAKE) test

clean:
	rm -f clauselens.db
	find . -name __pycache__ -type d -exec rm -rf {} +
```

- [ ] **Step 2: Untrack the JetBrains config**

```bash
git rm -r --cached .idea
printf '.idea/\nresults/*.tmp\n' >> .gitignore
```

- [ ] **Step 3: Verify a cold start works**

```bash
make clean
unset OPENAI_API_KEY
make test
make eval-offline
python -m clauselens.seed --offline && uvicorn clauselens.app:app --port 8001 &
sleep 3
curl -s localhost:8001/healthz
curl -s localhost:8001/corpus
kill %1
```

Expected: `{"status":"ok","clause_count":10}` and a contract breakdown with three contracts. This is the reviewer's first-run experience — it must work with no key.

- [ ] **Step 4: Add the one-command line to the README**

Directly under the title:

```markdown
```bash
make demo   # index from cache, serve the playground — no API key required to boot
```
```

- [ ] **Step 5: Commit**

```bash
git add Makefile .gitignore README.md
git commit -m "chore: make demo/eval/test one command each, untrack .idea

A fresh clone can now index and serve with no API key. Removes JetBrains
project files from the index."
```

---

## Self-review checklist

Run before declaring done.

- [ ] Every finding F1–F12 maps to a task. (F1→T1, F2→T7, F3→T5, F4→T7, F5→T4/T5/T6/T7, F6→T2/T7, F7→T5, F8→T5, F9→T3/T8, F10→T2, F11→T8, F12→T7.)
- [ ] `make check` is green.
- [ ] A cold clone with no `OPENAI_API_KEY` can run `make demo` and `make eval-offline` successfully.
- [ ] `results/latest.md` and `results/retrieval.md` are committed with real numbers, and the README table matches them.
- [ ] No eval threshold was lowered to make anything pass. If a number came in under target, the README says so.
- [ ] `git log --oneline` shows eight or more logical commits, not one squashed blob.
- [ ] Nothing has been pushed.

## Follow-on, deliberately deferred

Once this lands, the issue backlog is unblocked and the order that makes sense is:

1. **Issue #3** (persist runs to a results table) — the CLI now emits the metadata it needs, and without run history none of the tuning work below is readable.
2. **Issue #2** (multi-dataset + a CUAD subset) — the single change that makes every number in `results/` mean something.
3. **Issue #7** (abstain + eval taxonomy) — the split from Task 5 is the scaffolding for per-`kind` reporting.
4. **Issue #4** (hybrid BM25), then **#5** (reranker) — cases 3 and 4 are the adversarial pair built to show the difference, and Task 4's recall@k is how you'd prove it.
5. **Issue #6** (OTel) — last, because it's the only one that doesn't change a number.
