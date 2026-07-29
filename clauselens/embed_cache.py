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
