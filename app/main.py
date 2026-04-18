"""
Enterprise RAG Pipeline — FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

from rag.retrieval.hybrid_search import HybridRetriever
from rag.generation.llm import generate_answer
from rag.evaluation.metrics import score_response

app = FastAPI(
    title="Enterprise RAG Pipeline",
    description="Production-grade RAG system with hybrid search and evaluation",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    evaluate: bool = False


class Source(BaseModel):
    source: str
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: float
    tokens_used: Optional[int] = None
    evaluation: Optional[dict] = None


retriever: Optional[HybridRetriever] = None


@app.on_event("startup")
async def startup():
    """Load vectorstore and build retriever on startup."""
    global retriever
    # Retriever is loaded from persisted ChromaDB on startup
    # See ingest.py to build the index first
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "retriever_loaded": retriever is not None}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not loaded. Run ingest.py first.")

    start = time.time()

    results = retriever.retrieve(request.question, top_k=request.top_k)
    context_chunks = [chunk for chunk, _ in results]

    answer, tokens = generate_answer(request.question, context_chunks)

    latency = (time.time() - start) * 1000

    sources = [
        Source(
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            score=round(score, 4),
            excerpt=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
        )
        for chunk, score in results
    ]

    evaluation = None
    if request.evaluate:
        evaluation = score_response(request.question, answer, context_chunks)

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(latency, 2),
        tokens_used=tokens,
        evaluation=evaluation,
    )
