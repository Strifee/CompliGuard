"""
api.py — FastAPI REST layer.

POST /ask  → full RAG answer
GET  /     → health check

Usage:
    uvicorn api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from llm import answer

app = FastAPI(title="CompliGuard API", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    model: str = "mistral"
    top_k: int = 5
    livre: Optional[str] = None


class ChunkOut(BaseModel):
    livre: Optional[str]
    document: Optional[str]
    titre: Optional[str]
    chapitre: Optional[str]
    article_ref: Optional[str]
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: str
    chunks: list[ChunkOut]


@app.get("/")
def health():
    return {"status": "ok", "service": "CompliGuard"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = answer(
            question=req.question,
            model=req.model,
            top_k=req.top_k,
            livre=req.livre,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        chunks=[
            ChunkOut(
                livre=c.livre,
                document=c.document,
                titre=c.titre,
                chapitre=c.chapitre,
                article_ref=c.article_ref,
                page=c.page,
                text=c.text,
            )
            for c in result["chunks"]
        ],
    )
