"""
Hybrid retrieval combining dense vector search with BM25 sparse retrieval.
Consistently outperforms single-method retrieval in enterprise benchmarks.
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag.ingestion.chunker import Chunk


class HybridRetriever:
    """
    Combines dense semantic search with BM25 sparse retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge result rankings.
    """

    def __init__(self, vectorstore: Chroma, chunks: List[Chunk], alpha: float = 0.6):
        """
        Args:
            vectorstore: ChromaDB vector store with embedded documents
            chunks: Original chunks for BM25 indexing
            alpha: Weight for dense retrieval (1-alpha for sparse). 0.6 = 60% dense.
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.alpha = alpha
        self._build_bm25_index()

    def _build_bm25_index(self):
        tokenized = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _rrf_score(self, rank: int, k: int = 60) -> float:
        """Reciprocal Rank Fusion score."""
        return 1.0 / (k + rank)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top_k chunks using hybrid search.
        Returns list of (chunk, score) tuples sorted by relevance.
        """
        dense_results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
        dense_ids = {doc.metadata.get("chunk_id"): rank for rank, (doc, _) in enumerate(dense_results)}

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        sparse_ids = {self.chunks[idx].chunk_id: rank for rank, (idx, _) in enumerate(bm25_ranked[:top_k * 2])}

        all_ids = set(dense_ids.keys()) | set(sparse_ids.keys())
        fused_scores = {}
        for chunk_id in all_ids:
            dense_rank = dense_ids.get(chunk_id, top_k * 2)
            sparse_rank = sparse_ids.get(chunk_id, top_k * 2)
            fused_scores[chunk_id] = (
                self.alpha * self._rrf_score(dense_rank) +
                (1 - self.alpha) * self._rrf_score(sparse_rank)
            )

        top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        chunk_map = {c.chunk_id: c for c in self.chunks}

        return [(chunk_map[cid], fused_scores[cid]) for cid in top_ids if cid in chunk_map]
