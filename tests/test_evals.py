"""
End-to-end eval tests for ClauseLens.

Requires OPENAI_API_KEY to run — skips cleanly without it.
Runs the full RAG pipeline against the eval set and asserts quality thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openai import OpenAI

from clauselens.evals import EvalResult, aggregate, load_eval_set, run_eval
from clauselens.seed import seed
from clauselens.store import ClauseStore

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_clauses.json"

skip_no_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Seed a temporary database for the eval run."""
    db_path = str(tmp_path_factory.mktemp("clauselens") / "test.db")
    seed(db_path, DATA_PATH)
    return db_path


@pytest.fixture(scope="module")
def eval_results(seeded_db: str) -> list[EvalResult]:
    """Run the full eval suite once and cache the results for all tests."""
    store = ClauseStore(seeded_db)
    client = OpenAI()
    cases = load_eval_set(EVAL_SET_PATH)
    results = [run_eval(store, case, client=client) for case in cases]
    store.close()
    return results


@skip_no_key
def test_faithfulness_threshold(eval_results: list[EvalResult]) -> None:
    report = aggregate(eval_results)
    print(report.as_markdown())
    assert report.faithfulness >= 0.8, (
        f"Faithfulness {report.faithfulness:.2f} below threshold 0.80"
    )


@skip_no_key
def test_citation_f1_threshold(eval_results: list[EvalResult]) -> None:
    report = aggregate(eval_results)
    assert report.citation_f1 >= 0.7, (
        f"Citation F1 {report.citation_f1:.2f} below threshold 0.70"
    )


@skip_no_key
def test_eval_coverage(eval_results: list[EvalResult]) -> None:
    """Sanity check: we actually ran all 10 cases."""
    assert len(eval_results) == 10
