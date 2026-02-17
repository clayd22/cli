from ...sandbox import SQLExecutor

RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Execute a SQL query against the DuckDB warehouse and return results. Use this to explore data, test queries, and understand the data before submitting final results.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute"
                }
            },
            "required": ["sql"]
        }
    }
}

MAX_ROWS_FOR_LLM = 500


def run_sql(sql: str) -> str:
    """Execute SQL and return formatted results for the agent."""
    executor = SQLExecutor()

    df, error = executor.execute(sql)

    if error:
        return f"ERROR: {error}"

    if df.empty:
        return "Query returned no results."

    row_count = len(df)

    if row_count > MAX_ROWS_FOR_LLM:
        result = df.head(MAX_ROWS_FOR_LLM).to_string(index=False)
        result += f"\n\n[TRUNCATED: showing {MAX_ROWS_FOR_LLM} of {row_count} rows. Use LIMIT in SQL or filter to see specific data.]"
    else:
        result = df.to_string(index=False)

    return result
