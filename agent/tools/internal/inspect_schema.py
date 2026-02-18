from ... import schema as schema_module

INSPECT_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_schema",
        "description": "Inspect the database schema. Can list all tables, get columns for a specific table, or get sample data.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_tables", "get_columns", "get_sample", "full_schema"],
                    "description": "What to inspect: 'list_tables' for all tables, 'get_columns' for column info, 'get_sample' for sample rows, 'full_schema' for complete schema context"
                },
                "schema": {
                    "type": "string",
                    "description": "Schema name (e.g., 'marts', 'staging', 'raw'). Required for get_columns and get_sample."
                },
                "table": {
                    "type": "string",
                    "description": "Table name. Required for get_columns and get_sample."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of sample rows to return (default 5)"
                }
            },
            "required": ["action"]
        }
    }
}


def inspect_schema(
    action: str,
    schema: str = None,
    table: str = None,
    limit: int = 5
) -> str:
    """Inspect database schema and return formatted info."""

    if action == "list_tables":
        tables = schema_module.get_tables(schema)
        if not tables:
            return "No tables found."

        lines = ["Tables in database:"]
        for t in tables:
            row_count = schema_module.get_row_count(t["schema"], t["table"])
            lines.append(f"  {t['schema']}.{t['table']} ({row_count:,} rows)")
        return "\n".join(lines)

    elif action == "get_columns":
        if not schema or not table:
            return "ERROR: 'schema' and 'table' are required for get_columns"

        columns = schema_module.get_columns(schema, table)
        if not columns:
            return f"No columns found for {schema}.{table}"

        lines = [f"Columns in {schema}.{table}:"]
        for col in columns:
            nullable = "nullable" if col["nullable"] else "not null"
            lines.append(f"  {col['name']}: {col['type']} ({nullable})")
        return "\n".join(lines)

    elif action == "get_sample":
        if not schema or not table:
            return "ERROR: 'schema' and 'table' are required for get_sample"

        samples = schema_module.get_sample_data(schema, table, limit)

        if not samples:
            return f"No data in {schema}.{table}"

        import pandas as pd
        df = pd.DataFrame(samples)
        return f"Sample data from {schema}.{table}:\n{df.to_string(index=False)}"

    elif action == "full_schema":
        return schema_module.get_full_schema_context()

    else:
        return f"ERROR: Unknown action '{action}'"
