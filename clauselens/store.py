"""
Tiny vector store backed by SQLite + numpy.

This is a toy version of what pgvector would do for you in a real deployment.
Keeping it visible here because the interesting mechanics of retrieval
(normalization, cosine similarity, top-k) get hidden behind `ORDER BY embedding <=> $1`
when you use the real thing.

Schema:
    clauses(id TEXT PRIMARY KEY, contract TEXT, text TEXT, embedding BLOB)
    embedding is a float32 numpy array serialized with .tobytes()
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clause:
    id: str
    contract: str
    text: str
    score: float = 0.0


class ClauseStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        # check_same_thread=False: FastAPI serves requests on a threadpool, so the
        # connection is used across threads. Safe here — reads dominate and writes
        # (seed/index) are serialized.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clauses (
                id TEXT PRIMARY KEY,
                contract TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self._conn.commit()
        self._invalidate()

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
            Clause(
                id=rows[i][0],
                contract=rows[i][1],
                text=rows[i][2],
                score=float(scores[i]),
            )
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
