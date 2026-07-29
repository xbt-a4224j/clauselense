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
def store(tmp_path_factory: pytest.TempPathFactory):
    db = str(tmp_path_factory.mktemp("cl") / "offline.db")
    seed(db, CLAUSES, DEFAULT_CACHE, offline=True)
    s = ClauseStore(db)
    yield s
    s.close()


@pytest.fixture(scope="module")
def cache() -> EmbedCache:
    return EmbedCache(DEFAULT_CACHE)


def test_every_eval_question_is_cached(cache: EmbedCache) -> None:
    for case in load_eval_set(EVAL_SET):
        assert cache.get(case.question) is not None, f"uncached: {case.question}"


def test_splits_are_balanced() -> None:
    cases = load_eval_set(EVAL_SET)
    dev = [c for c in cases if c.split == "dev"]
    holdout = [c for c in cases if c.split == "holdout"]
    assert len(dev) == 6
    assert len(holdout) == 4


def test_retrieval_recall_at_4(store: ClauseStore, cache: EmbedCache) -> None:
    cases = load_eval_set(EVAL_SET)
    results = [run_retrieval_eval(store, c, k=4, cache=cache) for c in cases]
    recall = sum(r.recall_at_k for r in results) / len(results)
    assert recall >= 0.9, f"recall@4 {recall:.2f} below 0.90"


def test_retrieval_recall_degrades_at_k1(store: ClauseStore, cache: EmbedCache) -> None:
    """Sanity: k=1 must be strictly harder than k=4, or the metric is not measuring."""
    cases = load_eval_set(EVAL_SET)
    r4 = sum(run_retrieval_eval(store, c, k=4, cache=cache).recall_at_k for c in cases)
    r1 = sum(run_retrieval_eval(store, c, k=1, cache=cache).recall_at_k for c in cases)
    assert r1 < r4
