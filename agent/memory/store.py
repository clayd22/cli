"""
store.py - ChromaDB vector storage for RAG

Manages persistent storage of embeddings with metadata.
Organizes items into collections by type (schema, queries, observations).
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
from datetime import datetime

from .embedder import Embedder


# --- Storage location ---
MEMORY_DIR = Path.home() / ".astroagent" / "memory"

# --- Collection names for different content types ---
COLLECTIONS = {
    "schema": "schema_items",      # Tables, columns, relationships
    "queries": "query_history",    # Past questions + successful queries
    "observations": "observations", # Insights and patterns learned
}


class MemoryStore:
    """
    Vector store backed by ChromaDB.

    Provides:
    - Persistent storage of embeddings
    - Separate collections for schema/queries/observations
    - Similarity search with metadata filtering
    """

    def __init__(self):
        # --- Ensure storage directory exists ---
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # --- Initialize ChromaDB with persistent storage ---
        self.client = chromadb.PersistentClient(
            path=str(MEMORY_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )

        # --- Initialize embedder for adding new items ---
        self.embedder = Embedder()

        # --- Get or create collections ---
        self.collections = {}
        for key, name in COLLECTIONS.items():
            self.collections[key] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

    # =========================================================================
    # SCHEMA INDEXING
    # =========================================================================

    def index_table(self, table_name: str, columns: list[dict], sample_data: str = None):
        """
        Index a database table and its columns.

        Creates a rich text representation for embedding that captures
        the table's purpose and structure.

        Args:
            table_name: Full table name (e.g., 'marts.fct_orders')
            columns: List of column dicts with 'name' and 'type'
            sample_data: Optional sample rows as string
        """
        # --- Build text representation of table ---
        col_descriptions = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
        text = f"Table: {table_name}\nColumns: {col_descriptions}"
        if sample_data:
            text += f"\nSample data:\n{sample_data}"

        # --- Store with metadata ---
        self._add_item(
            collection="schema",
            item_id=f"table_{table_name}",
            text=text,
            metadata={
                "type": "table",
                "table_name": table_name,
                "column_names": ",".join([c["name"] for c in columns]),
                "indexed_at": datetime.now().isoformat(),
            },
        )

    def index_column(self, table_name: str, column_name: str, column_type: str, sample_values: list = None):
        """
        Index a specific column for fine-grained retrieval.

        Useful when questions target specific columns.

        Args:
            table_name: Parent table name
            column_name: Column name
            column_type: Data type
            sample_values: Optional sample values
        """
        # --- Build column-specific text ---
        text = f"Column: {table_name}.{column_name} (type: {column_type})"
        if sample_values:
            text += f"\nSample values: {', '.join(str(v) for v in sample_values[:5])}"

        self._add_item(
            collection="schema",
            item_id=f"col_{table_name}_{column_name}",
            text=text,
            metadata={
                "type": "column",
                "table_name": table_name,
                "column_name": column_name,
                "column_type": column_type,
                "indexed_at": datetime.now().isoformat(),
            },
        )

    # =========================================================================
    # QUERY HISTORY INDEXING
    # =========================================================================

    def index_query(
        self,
        question: str,
        sql: str,
        result_summary: str,
        session_id: str = None,
    ):
        """
        Index a successful question-query-result triplet.

        This enables finding similar past questions and their solutions.

        Args:
            question: The user's original question
            sql: The SQL query that answered it
            result_summary: Brief summary of the result
            session_id: Optional session identifier
        """
        # --- Combine question and SQL for richer embedding ---
        text = f"Question: {question}\nSQL: {sql}\nResult: {result_summary}"

        # --- Generate unique ID from timestamp ---
        item_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self._add_item(
            collection="queries",
            item_id=item_id,
            text=text,
            metadata={
                "type": "query",
                "question": question,
                "sql": sql,
                "result_summary": result_summary[:500],  # Truncate long results
                "session_id": session_id or "",
                "indexed_at": datetime.now().isoformat(),
            },
        )

    # =========================================================================
    # OBSERVATION INDEXING
    # =========================================================================

    def index_observation(self, observation: str, topic: str = None, session_id: str = None):
        """
        Index an observation or insight about the data.

        Args:
            observation: The observation text
            topic: Optional topic/category
            session_id: Optional session identifier
        """
        item_id = f"obs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self._add_item(
            collection="observations",
            item_id=item_id,
            text=observation,
            metadata={
                "type": "observation",
                "topic": topic or "",
                "session_id": session_id or "",
                "indexed_at": datetime.now().isoformat(),
            },
        )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def search(
        self,
        query: str,
        collection: str,
        n_results: int = 5,
        where: dict = None,
    ) -> list[dict]:
        """
        Search a collection for similar items.

        Args:
            query: The search query text
            collection: Which collection to search ('schema', 'queries', 'observations')
            n_results: Maximum results to return
            where: Optional metadata filter

        Returns:
            List of dicts with 'text', 'metadata', and 'distance' keys
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")

        coll = self.collections[collection]

        # --- Skip if collection is empty ---
        if coll.count() == 0:
            return []

        # --- Embed query and search ---
        query_embedding = self.embedder.embed(query)

        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, coll.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # --- Format results ---
        items = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                items.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

        return items

    def search_all(self, query: str, limits: dict = None) -> dict[str, list[dict]]:
        """
        Search all collections and return combined results.

        Args:
            query: The search query text
            limits: Dict of collection -> max results (default: schema=3, queries=3, observations=2)

        Returns:
            Dict mapping collection names to their results
        """
        limits = limits or {"schema": 3, "queries": 3, "observations": 2}

        results = {}
        for collection, limit in limits.items():
            if collection in self.collections:
                results[collection] = self.search(query, collection, n_results=limit)

        return results

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _add_item(self, collection: str, item_id: str, text: str, metadata: dict):
        """
        Add or update an item in a collection.

        Args:
            collection: Target collection name
            item_id: Unique identifier for the item
            text: Text content to embed
            metadata: Associated metadata
        """
        coll = self.collections[collection]

        # --- Generate embedding ---
        embedding = self.embedder.embed(text)

        # --- Upsert (add or update) ---
        coll.upsert(
            ids=[item_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def clear_collection(self, collection: str):
        """Clear all items from a collection."""
        if collection in self.collections:
            # --- Delete and recreate collection ---
            self.client.delete_collection(COLLECTIONS[collection])
            self.collections[collection] = self.client.create_collection(
                name=COLLECTIONS[collection],
                metadata={"hnsw:space": "cosine"},
            )

    def get_stats(self) -> dict:
        """Get item counts for all collections."""
        return {
            name: self.collections[name].count()
            for name in self.collections
        }

    def index_schema_from_db(self):
        """
        Index database schema from DuckDB.

        Pulls table and column info from the schema module
        and indexes it for semantic retrieval.
        """
        # --- Import here to avoid circular dependency ---
        from ..schema import get_tables, get_columns, get_sample_data

        tables = get_tables()  # Returns list of {"schema", "table", "type"}
        indexed_count = 0

        for table_info in tables:
            schema_name = table_info["schema"]
            table_name = table_info["table"]
            full_name = f"{schema_name}.{table_name}"

            # --- Get column info ---
            columns = get_columns(schema_name, table_name)  # Returns list of {"name", "type", "nullable"}
            col_list = [{"name": c["name"], "type": c["type"]} for c in columns]

            # --- Get sample data ---
            try:
                sample_records = get_sample_data(schema_name, table_name, limit=3)
                sample = str(sample_records) if sample_records else None
            except Exception:
                sample = None

            # --- Index the table ---
            self.index_table(full_name, col_list, sample)
            indexed_count += 1

            # --- Index individual columns for fine-grained retrieval ---
            for col in columns:
                self.index_column(
                    table_name=full_name,
                    column_name=col["name"],
                    column_type=col["type"],
                )

        return indexed_count

    def is_schema_indexed(self) -> bool:
        """Check if schema has been indexed."""
        return self.collections["schema"].count() > 0
