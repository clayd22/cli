"""
memory/ - RAG-based context management for AstroAgent

This module provides semantic retrieval of relevant context:
- Schema information (tables, columns, relationships)
- Past queries and their results
- Observations and learned patterns

Architecture:
    embedder.py  → Generate embeddings via OpenAI
    store.py     → ChromaDB vector storage
    retriever.py → Query-time context retrieval
"""

from .embedder import Embedder
from .store import MemoryStore
from .retriever import ContextRetriever, RetrievalResult

__all__ = ["Embedder", "MemoryStore", "ContextRetriever", "RetrievalResult"]
