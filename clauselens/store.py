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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
        self._conn = sqlite3.connect(self.db_path)
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

    def upsert(self, clauses: Iterable[tuple[str, str, str, np.ndarray]]) -> None:
        """clauses is an iterable of (id, contract, text, embedding)."""
        rows = [(cid, contract, text, emb.astype(np.float32).tobytes())
                for cid, contract, text, emb in clauses]
        self._conn.executemany(
            "INSERT OR REPLACE INTO clauses (id, contract, text, embedding) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def search(
        self, query_embedding: np.ndarray, k: int = 4, score_threshold: float = 0.0
    ) -> list[Clause]:
        """Return top-k clauses by cosine similarity.

        Loads all embeddings into memory. Fine for ~thousands of rows;
        swap for pgvector once you cross ~100k.
        """
        rows = self._conn.execute(
            "SELECT id, contract, text, embedding FROM clauses"
        ).fetchall()
        if not rows:
            return []

        q = query_embedding.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-12

        embs = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        # Normalize corpus vectors once per query (cheap at this scale).
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        scores = embs @ q  # cosine similarity since both sides are unit-norm
        top_idx = np.argsort(-scores)[:k]

        return [
            Clause(id=rows[i][0], contract=rows[i][1], text=rows[i][2], score=float(scores[i]))
            for i in top_idx
            if float(scores[i]) >= score_threshold
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
