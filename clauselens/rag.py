"""
Retrieval + generation. Deliberately simple.

Flow:
    question -> embed -> top-k clauses -> prompt LLM with cited context -> structured answer

The prompt asks the model to return JSON with an explicit citations list, so
downstream evals can grade citation accuracy separately from answer faithfulness.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI

from .store import Clause, ClauseStore

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a careful legal research assistant. Answer the user's \
question using ONLY the numbered clauses provided. Do not invent facts.

Return STRICT JSON with this shape:
{
  "answer": "<concise natural-language answer>",
  "citations": ["<clause id>", ...],
  "confidence": "high" | "medium" | "low"
}

Rules:
- Every factual claim in "answer" must be supported by at least one cited clause.
- If the clauses do not contain the answer, say so in "answer" and set citations to [].
- Do not cite a clause you did not use.
"""


@dataclass
class RagResponse:
    answer: str
    citations: list[str]
    confidence: str
    retrieved: list[Clause] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def embed(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def ask(
    store: ClauseStore,
    question: str,
    k: int = 4,
    score_threshold: float = 0.0,
    client: OpenAI | None = None,
) -> RagResponse:
    client = client or OpenAI()

    # 1. retrieve
    q_emb = embed(client, question)
    retrieved = store.search(q_emb, k=k, score_threshold=score_threshold)

    # 2. build cited context
    context_lines = [f"[{c.id}] (from {c.contract}): {c.text}" for c in retrieved]
    user_prompt = (
        "Clauses:\n"
        + "\n\n".join(context_lines)
        + f"\n\nQuestion: {question}"
    )

    # 3. generate
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(completion.choices[0].message.content)

    return RagResponse(
        answer=raw.get("answer", ""),
        citations=list(raw.get("citations", [])),
        confidence=raw.get("confidence", "low"),
        retrieved=retrieved,
        raw=raw,
    )
