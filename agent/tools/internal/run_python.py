from ...sandbox import SQLExecutor, PythonExecutor

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code on SQL query results. First runs the SQL queries to get dataframes, then executes Python code with access to those dataframes. The code must define a 'result' variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "object",
                    "description": "Map of variable names to SQL queries. Each query result will be available as a pandas DataFrame with that name.",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "code": {
                    "type": "string",
                    "description": "Python code to execute. DO NOT USE IMPORTS - pd (pandas) and np (numpy) are pre-loaded. Query results available as DataFrames. Must define a 'result' variable."
                }
            },
            "required": ["queries", "code"]
        }
    }
}


def run_python(queries: dict[str, str], code: str) -> str:
    """Execute Python code on SQL results."""
    sql_executor = SQLExecutor()
    py_executor = PythonExecutor()

    dataframes = {}
    for name, sql in queries.items():
        df, error = sql_executor.execute(sql)
        if error:
            return f"ERROR executing SQL for '{name}': {error}"
        dataframes[name] = df

    result, error = py_executor.execute(code, dataframes)

    if error:
        return f"ERROR: {error}"

    return str(result)
