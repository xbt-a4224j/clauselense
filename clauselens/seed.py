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

from .embed_cache import EmbedCache
from .rag import EMBED_MODEL
from .store import ClauseStore

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_clauses.json"
DEFAULT_DB = os.environ.get("CLAUSELENS_DB", "clauselens.db")
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings.npz"


def _embed_batch(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of texts in a single API call."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


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
        [
            (c["id"], c["contract"], c["text"], emb)
            for c, emb in zip(clauses, embeddings)
        ]
    )
    contracts = {c["contract"] for c in clauses}
    print(f"Indexed {len(clauses)} clauses from {len(contracts)} contracts.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the ClauseLens clause store.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA), help="Path to clauses JSON"
    )
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="Embedding cache .npz"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Embed from the cache; never call the API",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Persist fresh embeddings to the cache",
    )
    args = parser.parse_args()
    seed(
        args.db,
        args.data,
        args.cache,
        offline=args.offline,
        write_cache=args.write_cache,
    )


if __name__ == "__main__":
    main()
