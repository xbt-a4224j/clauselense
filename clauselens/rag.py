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

from .embed_cache import EmbedCache
from .store import Clause, ClauseStore

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a careful legal research assistant. Answer the user's \
question using ONLY the ID-tagged clauses provided. Do not invent facts.

Each clause is supplied as: [<clause id>] (from <contract>): <text>

Return STRICT JSON with this shape:
{
  "answer": "<concise natural-language answer>",
  "citations": ["<clause id>", ...],
  "confidence": "high" | "medium" | "low"
}

Rules:
- Every factual claim in "answer" must be supported by at least one cited clause.
- "citations" must contain clause IDs copied verbatim from the brackets, e.g. "MSA-01".
  Never use ordinal positions, list numbers, or contract names as citations.
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


def embed(client: OpenAI, text: str, cache: EmbedCache | None = None) -> np.ndarray:
    """Embed a single string, consulting `cache` first when one is supplied."""
    if cache is not None:
        hit = cache.get(text)
        if hit is not None:
            return hit
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    if cache is not None:
        cache.put(text, vec)
    return vec


def ask(
    store: ClauseStore,
    question: str,
    k: int = 4,
    score_threshold: float = 0.0,
    client: OpenAI | None = None,
    cache: EmbedCache | None = None,
) -> RagResponse:
    client = client or OpenAI()

    # 1. retrieve
    q_emb = embed(client, question, cache=cache)
    retrieved = store.search(q_emb, k=k, score_threshold=score_threshold)

    # 2. build cited context
    context_lines = [f"[{c.id}] (from {c.contract}): {c.text}" for c in retrieved]
    user_prompt = (
        "Clauses:\n" + "\n\n".join(context_lines) + f"\n\nQuestion: {question}"
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
    raw = json.loads(completion.choices[0].message.content or "{}")

    return RagResponse(
        answer=raw.get("answer", ""),
        citations=list(raw.get("citations", [])),
        confidence=raw.get("confidence", "low"),
        retrieved=retrieved,
        raw=raw,
    )
