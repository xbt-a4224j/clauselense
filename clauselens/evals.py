"""
Eval harness. Two metrics:

1. Citation precision/recall
   Compared against `expected_clause_ids` from the labeled eval set.

2. Faithfulness (LLM-as-judge)
   A separate LLM call grades whether each claim in the answer is supported by
   the retrieved clauses. This is noisier than citation F1 but catches the case
   where the model cites the right clause and still misstates what it says.

Threshold logic lives in `aggregate()`. CI gates on these numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from openai import OpenAI

from .rag import CHAT_MODEL, RagResponse, ask
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
    raw = json.loads(completion.choices[0].message.content)
    return bool(raw.get("faithful", False)), raw.get("reason", "")


def run_eval(store: ClauseStore, case: EvalCase, client: OpenAI | None = None) -> EvalResult:
    client = client or OpenAI()
    resp: RagResponse = ask(store, case.question, client=client)
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


def load_eval_set(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [EvalCase(**row) for row in data]


@dataclass
class AggregateReport:
    n: int
    faithfulness: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    results: list[EvalResult]

    def as_markdown(self) -> str:
        lines = [
            "# ClauseLens eval report",
            "",
            f"- cases: **{self.n}**",
            f"- faithfulness: **{self.faithfulness:.2f}**",
            f"- citation precision: **{self.citation_precision:.2f}**",
            f"- citation recall: **{self.citation_recall:.2f}**",
            f"- citation F1: **{self.citation_f1:.2f}**",
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
        return AggregateReport(0, 0.0, 0.0, 0.0, 0.0, [])
    p = mean(r.citation_precision for r in results)
    r_ = mean(r.citation_recall for r in results)
    f1 = 0.0 if (p + r_) == 0 else 2 * p * r_ / (p + r_)
    return AggregateReport(
        n=len(results),
        faithfulness=mean(1.0 if r.faithful else 0.0 for r in results),
        citation_precision=p,
        citation_recall=r_,
        citation_f1=f1,
        results=results,
    )
