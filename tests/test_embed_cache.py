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
