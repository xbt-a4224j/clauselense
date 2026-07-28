"""Prompt/context contract tests. No API key required — the OpenAI client is stubbed."""

from __future__ import annotations

import json
import types

import numpy as np

from clauselens.rag import SYSTEM_PROMPT, ask
from clauselens.store import ClauseStore

DIM = 8


class StubClient:
    """Minimal OpenAI stand-in. Records the chat kwargs, returns a canned JSON payload."""

    def __init__(self, payload: dict | None = None) -> None:
        self.captured: dict = {}
        self._payload = payload or {
            "answer": "Sixty days.",
            "citations": ["MSA-01"],
            "confidence": "high",
        }
        self.embeddings = types.SimpleNamespace(create=self._embed)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._chat)
        )

    def _embed(self, model: str, input: str | list[str]):
        n = 1 if isinstance(input, str) else len(input)
        data = [types.SimpleNamespace(embedding=[1.0] * DIM) for _ in range(n)]
        return types.SimpleNamespace(data=data)

    def _chat(self, **kwargs):
        self.captured = kwargs
        msg = types.SimpleNamespace(content=json.dumps(self._payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _seeded_store(tmp_path) -> ClauseStore:
    store = ClauseStore(tmp_path / "t.db")
    store.upsert(
        [
            (
                "MSA-01",
                "SaaS MSA",
                "Either party may terminate on 60 days notice.",
                np.ones(DIM, dtype=np.float32),
            ),
            (
                "NDA-01",
                "Acme NDA",
                "The Receiving Party shall not disclose.",
                np.full(DIM, 0.5, dtype=np.float32),
            ),
        ]
    )
    return store


def test_system_prompt_does_not_say_numbered() -> None:
    assert "numbered" not in SYSTEM_PROMPT.lower()


def test_system_prompt_names_the_id_format() -> None:
    assert "ID-tagged" in SYSTEM_PROMPT
    assert "MSA-01" in SYSTEM_PROMPT


def test_context_lines_are_id_tagged(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    client = StubClient()
    ask(store, "What is the termination period?", client=client)
    user_msg = client.captured["messages"][1]["content"]
    assert "[MSA-01]" in user_msg
    assert "[NDA-01]" in user_msg
    store.close()


def test_citations_pass_through_verbatim(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    client = StubClient(
        {"answer": "a", "citations": ["MSA-01"], "confidence": "medium"}
    )
    resp = ask(store, "What is the termination period?", client=client)
    assert resp.citations == ["MSA-01"]
    assert resp.confidence == "medium"
    store.close()
