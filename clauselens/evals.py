"""
Eval harness. Two metric families:

1. Citation precision/recall (+ LLM-as-judge faithfulness)
   Grades the full pipeline against `expected_clause_ids` from the labeled eval
   set. Faithfulness is noisier than citation F1 but catches the case where the
   model cites the right clause and still misstates what it says.

2. Retrieval-only (recall@k, MRR, precision@k)
   Grades what *search* returned, before the LLM touches anything. With a
   populated embedding cache this path makes zero API calls, so it can gate
   every commit.

Threshold logic lives in the tests. CI gates on these numbers.

Runner CLI:
    python -m clauselens.evals --retrieval-only --split all      # offline
    python -m clauselens.evals --split dev --out results/latest.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from openai import OpenAI

from .embed_cache import EmbedCache
from .metrics import mrr, precision_at_k, recall_at_k
from .rag import CHAT_MODEL, RagResponse, ask, embed
from .store import ClauseStore

FAITHFULNESS_PROMPT = """You are grading a RAG answer for faithfulness.

Question: {question}
Retrieved clauses:
{context}

Answer to grade:
{answer}

Return STRICT JSON:
{{
  "faithful": true | false,
  "reason": "<1 sentence>"
}}

Rules:
- "faithful" is true ONLY if every factual claim in the answer is directly supported by
  one or more retrieved clauses.
- A correctly-cited but paraphrased answer IS faithful.
- An answer that adds facts not in the clauses is NOT faithful.
"""


@dataclass
class EvalCase:
    question: str
    expected_clause_ids: list[str]
    notes: str = ""
    split: str = "dev"


@dataclass
class EvalResult:
    question: str
    expected: list[str]
    cited: list[str]
    retrieved: list[str]
    faithful: bool
    faithful_reason: str
    citation_precision: float
    citation_recall: float


@dataclass
class RetrievalResult:
    question: str
    expected: list[str]
    retrieved: list[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float


def _prf(expected: set[str], cited: set[str]) -> tuple[float, float]:
    if not cited:
        precision = 1.0 if not expected else 0.0
    else:
        precision = len(cited & expected) / len(cited)
    if not expected:
        recall = 1.0
    else:
        recall = len(cited & expected) / len(expected)
    return precision, recall


def judge_faithfulness(
    client: OpenAI, question: str, answer: str, retrieved_text: list[str]
) -> tuple[bool, str]:
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        context="\n".join(f"- {t}" for t in retrieved_text),
        answer=answer,
    )
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(completion.choices[0].message.content or "{}")
    return bool(raw.get("faithful", False)), raw.get("reason", "")


def run_eval(
    store: ClauseStore,
    case: EvalCase,
    client: OpenAI | None = None,
    k: int = 4,
    score_threshold: float = 0.0,
    cache: EmbedCache | None = None,
) -> EvalResult:
    client = client or OpenAI()
    resp: RagResponse = ask(
        store,
        case.question,
        k=k,
        score_threshold=score_threshold,
        client=client,
        cache=cache,
    )
    expected = set(case.expected_clause_ids)
    cited = set(resp.citations)
    precision, recall = _prf(expected, cited)
    faithful, reason = judge_faithfulness(
        client, case.question, resp.answer, [c.text for c in resp.retrieved]
    )
    return EvalResult(
        question=case.question,
        expected=list(expected),
        cited=list(cited),
        retrieved=[c.id for c in resp.retrieved],
        faithful=faithful,
        faithful_reason=reason,
        citation_precision=precision,
        citation_recall=recall,
    )


def run_retrieval_eval(
    store: ClauseStore,
    case: EvalCase,
    client: OpenAI | None = None,
    k: int = 4,
    score_threshold: float = 0.0,
    cache: EmbedCache | None = None,
) -> RetrievalResult:
    """Grade retrieval alone. With a populated cache this makes zero API calls."""
    if cache is not None and cache.get(case.question) is not None:
        q_emb = cache.require(case.question)
    else:
        q_emb = embed(client or OpenAI(), case.question, cache=cache)

    hits = store.search(q_emb, k=k, score_threshold=score_threshold)
    retrieved = [c.id for c in hits]
    return RetrievalResult(
        question=case.question,
        expected=list(case.expected_clause_ids),
        retrieved=retrieved,
        recall_at_k=recall_at_k(case.expected_clause_ids, retrieved),
        precision_at_k=precision_at_k(case.expected_clause_ids, retrieved),
        mrr=mrr(case.expected_clause_ids, retrieved),
    )


def load_eval_set(path: str | Path, split: str = "all") -> list[EvalCase]:
    """Load eval cases. `split` is 'dev', 'holdout', or 'all'."""
    data = json.loads(Path(path).read_text())
    cases = [EvalCase(**row) for row in data]
    if split == "all":
        return cases
    return [c for c in cases if c.split == split]


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


@dataclass
class AggregateReport:
    n: int
    faithfulness: float
    citation_precision: float
    citation_recall: float
    citation_f1: float  # F1 of the averaged P and R (macro)
    citation_f1_mean: float  # mean of per-case F1 — lower when variance is high
    results: list[EvalResult]

    def as_markdown(self) -> str:
        gap = self.citation_f1 - self.citation_f1_mean
        lines = [
            "# ClauseLens eval report",
            "",
            f"- cases: **{self.n}**",
            f"- faithfulness: **{self.faithfulness:.2f}**",
            f"- citation precision: **{self.citation_precision:.2f}**",
            f"- citation recall: **{self.citation_recall:.2f}**",
            f"- citation F1 (macro, of the means): **{self.citation_f1:.2f}**",
            f"- citation F1 (mean of per-case): **{self.citation_f1_mean:.2f}**",
            "",
            (
                f"> Gap between the two F1s is {gap:+.2f}. A large positive gap means "
                f"a few cases are failing badly while the averages look healthy."
            ),
            "",
            "| # | faithful | precision | recall | question |",
            "|---|----------|-----------|--------|----------|",
        ]
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"| {i} | {'✓' if r.faithful else '✗'} | {r.citation_precision:.2f} | "
                f"{r.citation_recall:.2f} | {r.question} |"
            )
        return "\n".join(lines)


def aggregate(results: list[EvalResult]) -> AggregateReport:
    if not results:
        return AggregateReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    p = mean(r.citation_precision for r in results)
    r_ = mean(r.citation_recall for r in results)
    return AggregateReport(
        n=len(results),
        faithfulness=mean(1.0 if r.faithful else 0.0 for r in results),
        citation_precision=p,
        citation_recall=r_,
        citation_f1=_f1(p, r_),
        citation_f1_mean=mean(
            _f1(r.citation_precision, r.citation_recall) for r in results
        ),
        results=results,
    )


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClauseLens eval suite.")
    parser.add_argument(
        "--db", default=os.environ.get("CLAUSELENS_DB", "clauselens.db")
    )
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--cache", default="data/embeddings.npz")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and judging. Runs offline from the cache.",
    )
    parser.add_argument("--out", default=None, help="Write the markdown report here")
    parser.add_argument("--notes", default="", help="Free-text tag for this run")
    args = parser.parse_args()

    cases = load_eval_set(args.eval_set, split=args.split)
    if not cases:
        print(f"No cases in split '{args.split}'.", file=sys.stderr)
        sys.exit(1)

    store = ClauseStore(args.db)
    if store.count() == 0:
        print(
            "No clauses indexed. Run: python -m clauselens.seed --offline",
            file=sys.stderr,
        )
        sys.exit(1)

    cache = EmbedCache(args.cache)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"<!-- run {stamp} · {_git_sha()} · split={args.split} · k={args.top_k} · "
        f"threshold={args.score_threshold} · notes={args.notes or 'none'} -->"
    )

    if args.retrieval_only:
        rows = [
            run_retrieval_eval(
                store,
                c,
                k=args.top_k,
                score_threshold=args.score_threshold,
                cache=cache,
            )
            for c in cases
        ]
        body = "\n".join(
            [
                "# ClauseLens retrieval report",
                "",
                f"- cases: **{len(rows)}** (split: {args.split})",
                f"- recall@{args.top_k}: **{mean(r.recall_at_k for r in rows):.2f}**",
                f"- precision@{args.top_k}: **{mean(r.precision_at_k for r in rows):.2f}**",
                f"- MRR: **{mean(r.mrr for r in rows):.2f}**",
                "",
                "| # | recall | MRR | expected | retrieved |",
                "|---|--------|-----|----------|-----------|",
            ]
            + [
                f"| {i} | {r.recall_at_k:.2f} | {r.mrr:.2f} | "
                f"{', '.join(r.expected)} | {', '.join(r.retrieved)} |"
                for i, r in enumerate(rows, 1)
            ]
        )
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OPENAI_API_KEY is required for the full eval.\n"
                "For an offline retrieval-only run:\n"
                "  python -m clauselens.evals --retrieval-only",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OpenAI()
        results = [
            run_eval(
                store,
                c,
                client=client,
                k=args.top_k,
                score_threshold=args.score_threshold,
                cache=cache,
            )
            for c in cases
        ]
        body = aggregate(results).as_markdown()

    store.close()
    report = f"{header}\n\n{body}\n"
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
