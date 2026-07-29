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
