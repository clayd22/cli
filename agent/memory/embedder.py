"""
embedder.py - Generate embeddings using OpenAI's API

Handles all embedding generation for the RAG system.
Uses text-embedding-3-small for cost efficiency.
"""

from openai import OpenAI
from typing import Union

from ..config import get_api_key


# --- Configuration ---
# Using OpenAI's small embedding model - good balance of quality and cost
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Output dimensions for this model


class Embedder:
    """
    Generates embeddings for text using OpenAI's embedding API.

    Offloads embedding computation to OpenAI for simplicity.
    Supports both single texts and batches.
    """

    def __init__(self):
        # --- Initialize OpenAI client ---
        api_key = get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key required for embeddings")
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # --- Call OpenAI embedding API ---
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in one API call.

        More efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        # --- Batch embed via OpenAI ---
        # OpenAI handles batching internally, more efficient than multiple calls
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # --- Return embeddings in original order ---
        # Response may not be in order, so sort by index
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def embed_with_metadata(self, text: str, metadata: dict) -> dict:
        """
        Generate embedding and package with metadata for storage.

        Convenience method for preparing items for ChromaDB.

        Args:
            text: The text to embed
            metadata: Additional metadata to store

        Returns:
            Dict with 'embedding', 'text', and 'metadata' keys
        """
        return {
            "embedding": self.embed(text),
            "text": text,
            "metadata": metadata,
        }
