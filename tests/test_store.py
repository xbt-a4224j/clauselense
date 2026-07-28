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
    hits = store.search(
        np.array([1.0, 0.0], dtype=np.float32), k=2, score_threshold=0.5
    )
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
