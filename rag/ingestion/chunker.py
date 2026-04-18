"""
Smart chunking strategies for enterprise document processing.
Supports semantic, recursive, and fixed-size chunking.
"""

from typing import List, Optional
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SemanticChunker,
)
from langchain_openai import OpenAIEmbeddings


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str
    source: str
    page: Optional[int] = None


class SmartChunker:
    """
    Enterprise-grade document chunker with multiple strategies.
    Semantic chunking preserves document meaning better than fixed-size splits.
    """

    STRATEGIES = ["semantic", "recursive", "fixed"]

    def __init__(self, strategy: str = "semantic", chunk_size: int = 512, overlap: int = 64):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        if self.strategy == "semantic":
            return SemanticChunker(
                embeddings=OpenAIEmbeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
        elif self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )

    def chunk(self, text: str, source: str, metadata: dict = None) -> List[Chunk]:
        """Split text into chunks with metadata preserved."""
        metadata = metadata or {}
        raw_chunks = self._splitter.split_text(text)

        return [
            Chunk(
                text=chunk,
                metadata={**metadata, "strategy": self.strategy},
                chunk_id=f"{source}_{i}",
                source=source,
            )
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]

    def chunk_documents(self, documents: list) -> List[Chunk]:
        """Process a list of LangChain Document objects."""
        all_chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunks = self.chunk(doc.page_content, source, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks
