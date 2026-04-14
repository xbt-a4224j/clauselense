"""
FastAPI wrapper around the RAG pipeline.

Run:
    uvicorn clauselens.app:app --reload
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .rag import RagResponse, ask
from .store import ClauseStore

DB_PATH = os.environ.get("CLAUSELENS_DB", "clauselens.db")

app = FastAPI(title="ClauseLens", version="0.1.0")
_store = ClauseStore(DB_PATH)

_STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)


class RetrievedClause(BaseModel):
    id: str
    contract: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: str
    retrieved: list[RetrievedClause]


@app.get("/", response_class=HTMLResponse)
def playground() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "playground.html").read_text())


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "clause_count": _store.count()}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if _store.count() == 0:
        raise HTTPException(status_code=503, detail="no clauses indexed; run seed.py first")

    resp: RagResponse = ask(_store, req.question, k=req.k, score_threshold=req.score_threshold)
    return AskResponse(
        answer=resp.answer,
        citations=resp.citations,
        confidence=resp.confidence,
        retrieved=[
            RetrievedClause(id=c.id, contract=c.contract, text=c.text, score=c.score)
            for c in resp.retrieved
        ],
    )
