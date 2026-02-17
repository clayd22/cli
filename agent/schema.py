import duckdb
from pathlib import Path
from typing import Any

WAREHOUSE_PATH = Path(__file__).parent.parent / "warehouse" / "data.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=True)


def get_all_schemas() -> list[str]:
    with get_connection() as conn:
        result = conn.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
        """).fetchall()
        return [row[0] for row in result]


def get_tables(schema: str = None) -> list[dict]:
    with get_connection() as conn:
        query = """
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        """
        if schema:
            query += f" AND table_schema = '{schema}'"
        query += " ORDER BY table_schema, table_name"

        result = conn.execute(query).fetchall()
        return [
            {"schema": row[0], "table": row[1], "type": row[2]}
            for row in result
        ]


def get_columns(schema: str, table: str) -> list[dict]:
    with get_connection() as conn:
        result = conn.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
        """, [schema, table]).fetchall()
        return [
            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
            for row in result
        ]


def get_sample_data(schema: str, table: str, limit: int = 5) -> list[dict]:
    with get_connection() as conn:
        result = conn.execute(f"""
            SELECT * FROM {schema}.{table} LIMIT {limit}
        """).fetchdf()
        return result.to_dict(orient="records")


def get_row_count(schema: str, table: str) -> int:
    with get_connection() as conn:
        result = conn.execute(f"""
            SELECT COUNT(*) FROM {schema}.{table}
        """).fetchone()
        return result[0]


def get_full_schema_context() -> str:
    """Returns a formatted string of the entire schema for LLM context."""
    lines = ["# Database Schema\n"]

    for schema in get_all_schemas():
        lines.append(f"## Schema: {schema}\n")

        for table_info in get_tables(schema):
            table = table_info["table"]
            row_count = get_row_count(schema, table)
            lines.append(f"### {schema}.{table} ({row_count:,} rows)\n")
            lines.append("| Column | Type | Nullable |")
            lines.append("|--------|------|----------|")

            for col in get_columns(schema, table):
                nullable = "Yes" if col["nullable"] else "No"
                lines.append(f"| {col['name']} | {col['type']} | {nullable} |")

            lines.append("")

    return "\n".join(lines)
