"""
Seed the clause store from sample_clauses.json.

Usage:
    python -m clauselens.seed
    python -m clauselens.seed --db clauselens.db --data data/sample_clauses.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

from .rag import EMBED_MODEL
from .store import ClauseStore

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_clauses.json"
DEFAULT_DB = os.environ.get("CLAUSELENS_DB", "clauselens.db")


def _embed_batch(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of texts in a single API call."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def seed(db_path: str, data_path: str | Path) -> None:
    """Load clauses from JSON, embed them, and upsert into the store."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your environment or in a .env file:\n"
            "  export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    clauses = json.loads(data_path.read_text())
    if not clauses:
        print("Error: no clauses found in data file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    store = ClauseStore(db_path)

    print(f"Embedding {len(clauses)} clauses with {EMBED_MODEL}...")
    texts = [c["text"] for c in clauses]
    embeddings = _embed_batch(client, texts)

    rows = [
        (c["id"], c["contract"], c["text"], emb) for c, emb in zip(clauses, embeddings)
    ]
    store.upsert(rows)

    contracts = {c["contract"] for c in clauses}
    print(f"Indexed {len(clauses)} clauses from {len(contracts)} contracts.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the ClauseLens clause store.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA), help="Path to clauses JSON"
    )
    args = parser.parse_args()
    seed(args.db, args.data)


if __name__ == "__main__":
    main()
