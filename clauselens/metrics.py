"""
Retrieval-only metrics.

These grade what the *search* returned, independent of what the LLM chose to
cite. That separation matters: citation F1 conflates a retrieval miss with a
generation mistake, and on a 10-clause corpus at k=4 it mostly measures the
latter.

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
