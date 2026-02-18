"""
retriever.py - Query-time context retrieval for RAG

Retrieves and formats relevant context to inject into LLM prompts.
Handles ranking, deduplication, and token budget management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .store import MemoryStore


# --- Token budget for retrieved context ---
TOKEN_BUDGET = {
    "schema": 2000,
    "queries": 1500,
    "observations": 500,
}

# --- Approximate tokens per character ---
CHARS_PER_TOKEN = 4


@dataclass
class RetrievalResult:
    """
    Container for RAG retrieval results with scores.

    Used for both injection and observability/debugging.
    """
    schema_items: list = field(default_factory=list)      # [{text, metadata, score}, ...]
    query_items: list = field(default_factory=list)
    observation_items: list = field(default_factory=list)

    @property
    def total_items(self) -> int:
        return len(self.schema_items) + len(self.query_items) + len(self.observation_items)

    @property
    def best_schema_score(self) -> float:
        return self.schema_items[0]["score"] if self.schema_items else 0.0

    @property
    def best_query_score(self) -> float:
        return self.query_items[0]["score"] if self.query_items else 0.0

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = []
        if self.schema_items:
            parts.append(f"{len(self.schema_items)} schema (best: {self.best_schema_score:.2f})")
        if self.query_items:
            parts.append(f"{len(self.query_items)} queries (best: {self.best_query_score:.2f})")
        if self.observation_items:
            parts.append(f"{len(self.observation_items)} obs")
        return ", ".join(parts) if parts else "no matches"


class ContextRetriever:
    """
    Retrieves relevant context for a user question.

    Searches across schema, past queries, and observations,
    then formats into a prompt-ready string.
    """

    def __init__(self, store: Optional[MemoryStore] = None):
        # --- Use provided store or create new one ---
        self.store = store or MemoryStore()

    def retrieve(self, question: str) -> str:
        """
        Retrieve relevant context for a question.

        Main entry point - searches all collections and formats results.

        Args:
            question: The user's question

        Returns:
            Formatted context string ready for prompt injection
        """
        result = self.retrieve_with_scores(question)
        return self.format_for_prompt(result)

    def retrieve_with_scores(self, question: str) -> RetrievalResult:
        """
        Retrieve relevant context with similarity scores.

        Returns structured results for both injection and observability.

        Args:
            question: The user's question

        Returns:
            RetrievalResult with scored items from each collection
        """
        result = RetrievalResult()

        # --- Skip if nothing is indexed ---
        stats = self.store.get_stats()
        if sum(stats.values()) == 0:
            return result

        # --- Search all collections ---
        raw_results = self.store.search_all(
            query=question,
            limits={"schema": 5, "queries": 5, "observations": 3},
        )

        # --- Convert distances to similarity scores (ChromaDB returns distances) ---
        # Cosine distance: 0 = identical, 2 = opposite. Convert to 0-1 similarity.
        def distance_to_score(distance: float) -> float:
            return max(0, 1 - (distance / 2))

        # --- Process schema results ---
        for item in raw_results.get("schema", []):
            result.schema_items.append({
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "score": distance_to_score(item.get("distance", 2)),
            })

        # --- Process query results ---
        for item in raw_results.get("queries", []):
            result.query_items.append({
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "score": distance_to_score(item.get("distance", 2)),
            })

        # --- Process observation results ---
        for item in raw_results.get("observations", []):
            result.observation_items.append({
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "score": distance_to_score(item.get("distance", 2)),
            })

        return result

    def format_for_prompt(self, result: RetrievalResult) -> str:
        """
        Format retrieval results for injection into system prompt.

        Args:
            result: RetrievalResult from retrieve_with_scores

        Returns:
            Formatted context string
        """
        sections = []

        # --- Schema context ---
        schema_context = self._format_schema(result.schema_items)
        if schema_context:
            sections.append(schema_context)

        # --- Past query examples ---
        query_context = self._format_queries(result.query_items)
        if query_context:
            sections.append(query_context)

        # --- Observations ---
        obs_context = self._format_observations(result.observation_items)
        if obs_context:
            sections.append(obs_context)

        if not sections:
            return ""

        return "## Retrieved Context (from memory)\n\n" + "\n\n".join(sections)

    def format_debug(self, result: RetrievalResult) -> str:
        """
        Format retrieval results for debug/test display.

        Shows similarity scores prominently for observability.

        Args:
            result: RetrievalResult from retrieve_with_scores

        Returns:
            Formatted debug string
        """
        lines = ["RAG Retrieval Results:", "=" * 50]

        # --- Schema matches ---
        lines.append("\nSchema matches:")
        if result.schema_items:
            for item in result.schema_items:
                meta = item["metadata"]
                score = item["score"]
                if meta.get("type") == "table":
                    cols = meta.get("column_names", "").replace(",", ", ")[:50]
                    lines.append(f"  {score:.2f}  {meta.get('table_name', '?')} ({cols})")
                else:
                    lines.append(f"  {score:.2f}  {meta.get('table_name', '?')}.{meta.get('column_name', '?')}")
        else:
            lines.append("  (none)")

        # --- Query matches ---
        lines.append("\nSimilar past queries:")
        if result.query_items:
            for item in result.query_items:
                meta = item["metadata"]
                score = item["score"]
                question = meta.get("question", "?")[:40]
                sql = meta.get("sql", "")[:50]
                lines.append(f"  {score:.2f}  \"{question}\"")
                lines.append(f"        -> {sql}")
        else:
            lines.append("  (none)")

        # --- Observation matches ---
        lines.append("\nRelevant observations:")
        if result.observation_items:
            for item in result.observation_items:
                score = item["score"]
                text = item["text"][:60]
                lines.append(f"  {score:.2f}  \"{text}\"")
        else:
            lines.append("  (none)")

        lines.append("=" * 50)
        return "\n".join(lines)

    def retrieve_for_schema(self, question: str) -> list[str]:
        """
        Retrieve just relevant table names for a question.

        Useful for focusing schema injection on relevant tables.

        Args:
            question: The user's question

        Returns:
            List of relevant table names
        """
        results = self.store.search(question, "schema", n_results=5)

        # --- Extract unique table names ---
        tables = set()
        for item in results:
            table = item.get("metadata", {}).get("table_name")
            if table:
                tables.add(table)

        return list(tables)

    # =========================================================================
    # FORMATTING HELPERS
    # =========================================================================

    def _format_schema(self, items: list[dict]) -> str:
        """Format schema search results for prompt injection."""
        if not items:
            return ""

        # --- Filter to stay within token budget ---
        items = self._apply_token_budget(items, TOKEN_BUDGET["schema"])

        lines = ["### Relevant Schema"]
        for item in items:
            meta = item.get("metadata", {})
            if meta.get("type") == "table":
                table = meta.get("table_name", "?")
                cols = meta.get("column_names", "").replace(",", ", ")
                lines.append(f"- **{table}**: {cols}")
            elif meta.get("type") == "column":
                table = meta.get("table_name", "?")
                col = meta.get("column_name", "?")
                ctype = meta.get("column_type", "?")
                lines.append(f"- {table}.{col} ({ctype})")

        return "\n".join(lines)

    def _format_queries(self, items: list[dict]) -> str:
        """Format past query search results for prompt injection."""
        if not items:
            return ""

        # --- Filter to stay within token budget ---
        items = self._apply_token_budget(items, TOKEN_BUDGET["queries"])

        lines = ["### Similar Past Queries"]
        for item in items:
            meta = item.get("metadata", {})
            question = meta.get("question", "")
            sql = meta.get("sql", "")
            result = meta.get("result_summary", "")

            if question and sql:
                lines.append(f"**Q:** {question}")
                lines.append(f"```sql\n{sql}\n```")
                if result:
                    lines.append(f"*Result: {result[:100]}*")
                lines.append("")

        return "\n".join(lines)

    def _format_observations(self, items: list[dict]) -> str:
        """Format observation search results for prompt injection."""
        if not items:
            return ""

        # --- Filter to stay within token budget ---
        items = self._apply_token_budget(items, TOKEN_BUDGET["observations"])

        lines = ["### Related Insights"]
        for item in items:
            text = item.get("text", "")
            if text:
                if len(text) > 200:
                    text = text[:197] + "..."
                lines.append(f"- {text}")

        return "\n".join(lines)

    def _apply_token_budget(self, items: list[dict], budget: int) -> list[dict]:
        """
        Filter items to fit within a token budget.

        Uses rough character-based estimation.

        Args:
            items: List of search result items
            budget: Maximum tokens allowed

        Returns:
            Filtered list of items
        """
        max_chars = budget * CHARS_PER_TOKEN
        total_chars = 0
        filtered = []

        for item in items:
            text = item.get("text", "")
            item_chars = len(text)

            if total_chars + item_chars <= max_chars:
                filtered.append(item)
                total_chars += item_chars
            else:
                # --- Budget exceeded, stop adding ---
                break

        return filtered

    # =========================================================================
    # INDEXING HELPERS (called after successful tool calls)
    # =========================================================================

    def index_successful_query(
        self,
        question: str,
        sql: str,
        result_summary: str,
        session_id: str = None,
    ):
        """
        Index a successful query for future retrieval.

        Call this after submit_result succeeds.

        Args:
            question: Original user question
            sql: SQL that answered it
            result_summary: Brief description of result
            session_id: Current session ID
        """
        self.store.index_query(
            question=question,
            sql=sql,
            result_summary=result_summary,
            session_id=session_id,
        )

    def index_observation(self, observation: str, topic: str = None, session_id: str = None):
        """
        Index an observation for future retrieval.

        Call this after submit_observation.

        Args:
            observation: The observation text
            topic: Optional topic
            session_id: Current session ID
        """
        self.store.index_observation(
            observation=observation,
            topic=topic,
            session_id=session_id,
        )
