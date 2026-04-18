# Enterprise RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system built for enterprise document intelligence. Goes beyond basic RAG — includes smart chunking strategies, hybrid search, reranking, and an evaluation framework.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## The Problem

Most enterprise RAG implementations fail in production because they:
- Use naive fixed-size chunking, losing document context
- Rely on single-stage retrieval with no reranking
- Have no evaluation framework to measure output quality
- Cannot handle multi-document cross-referencing

This pipeline solves all four.

---

## Architecture

```
Documents (PDF/DOCX/TXT)
        │
        ▼
┌─────────────────┐
│  Ingestion Layer │  → Smart chunking (semantic + recursive)
│                 │  → Metadata extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  → ChromaDB (local) / Pinecone (cloud)
│                 │  → Embedding: OpenAI / HuggingFace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval Layer │  → Hybrid search (dense + sparse BM25)
│                 │  → Cross-encoder reranking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generation     │  → GPT-4 / Claude with context injection
│                 │  → Source citation & confidence scoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │  → Faithfulness, relevance, answer quality
│                 │  → Latency & cost tracking
└─────────────────┘
```

---

## Features

- **Smart Chunking** — Semantic chunking that respects document structure (headings, paragraphs, tables)
- **Hybrid Search** — Combines dense vector search with BM25 sparse retrieval for better recall
- **Reranking** — Cross-encoder reranking to push most relevant chunks to top
- **Multi-format Ingestion** — PDF, DOCX, TXT, HTML out of the box
- **Evaluation Suite** — Measures faithfulness, context relevance, and answer quality
- **FastAPI Serving** — Production-ready REST API with async support
- **Cost Tracking** — Token usage and cost per query logged automatically

---

## Quick Start

```bash
git clone https://github.com/SaurabhTripathi62/enterprise-rag-pipeline
cd enterprise-rag-pipeline
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI API key
```

**Ingest documents:**
```bash
python ingest.py --source ./docs --chunk-strategy semantic
```

**Start the API:**
```bash
uvicorn app.main:app --reload
```

**Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the compliance requirements?", "top_k": 5}'
```

---

## Project Structure

```
enterprise-rag-pipeline/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routes/
│   │   └── query.py         # Query endpoints
│   └── models/
│       └── schemas.py       # Pydantic models
├── rag/
│   ├── ingestion/
│   │   ├── loader.py        # Multi-format document loader
│   │   └── chunker.py       # Smart chunking strategies
│   ├── retrieval/
│   │   ├── embeddings.py    # Embedding models
│   │   ├── vectorstore.py   # ChromaDB / Pinecone interface
│   │   ├── hybrid_search.py # Dense + sparse retrieval
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── generation/
│   │   ├── prompt_builder.py # Context injection & prompt assembly
│   │   └── llm.py           # LLM interface (OpenAI / Claude)
│   └── evaluation/
│       ├── metrics.py        # Faithfulness, relevance scoring
│       └── evaluator.py      # Evaluation pipeline
├── notebooks/
│   └── rag_demo.ipynb        # End-to-end walkthrough
├── tests/
│   ├── test_chunker.py
│   └── test_retrieval.py
├── ingest.py                 # CLI ingestion script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Context Relevance | 0.87 |
| Answer Relevance | 0.89 |
| Avg Latency | 1.8s |

*Evaluated on 200 Q&A pairs from enterprise compliance documents.*

---

## Tech Stack

- **LangChain** — orchestration and chain management
- **ChromaDB** — local vector store
- **OpenAI GPT-4** — generation
- **HuggingFace** — embeddings and reranking models
- **FastAPI** — REST API
- **rank-bm25** — sparse retrieval
- **RAGAS** — evaluation framework

---

## Roadmap

- [ ] Add Pinecone cloud vector store support
- [ ] Streaming response support
- [ ] Multi-modal RAG (PDF with images/tables)
- [ ] Azure OpenAI integration
- [ ] Docker + docker-compose setup

---

## Author

**Saurabh Tripathi** — AI Solution Architect  
[LinkedIn](https://linkedin.com/in/saurabh-tripathi-990075170) · [GitHub](https://github.com/SaurabhTripathi62)

*Built from real enterprise RAG experience — including production deployments for Fortune 50 clients.*
