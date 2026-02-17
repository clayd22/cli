"""
submit_result.py

The final output tool for AstroAgent. This is the ONLY way for the agent
to produce answers that reach the user. By forcing all outputs through
this tool, we guarantee that results come from actual database execution,
not LLM hallucination.

Architecture:
    User Question → LLM thinks/explores → submit_result() → Real DB execution → User
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from ...sandbox import SQLExecutor, PythonExecutor


# OpenAI function calling schema for this tool
SUBMIT_RESULT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_result",
        "description": """Submit the final answer to the user's question.

This executes SQL queries and applies a Python function to produce the result.
The result is computed deterministically from the database.

Use this when you have figured out the correct queries and logic to answer the question.""",
        "parameters": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "object",
                    "description": "Map of input names to SQL queries. Each query result becomes a pandas DataFrame available to the function. Example: {\"orders\": \"SELECT * FROM marts.fct_orders\", \"products\": \"SELECT * FROM marts.dim_products\"}",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "function": {
                    "type": "string",
                    "description": "Python code that processes the inputs. Has access to pandas as 'pd' and each input as a DataFrame by name. Must define a 'result' variable containing the final answer."
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of what this computation does and why it answers the user's question."
                }
            },
            "required": ["inputs", "function", "explanation"]
        }
    }
}


@dataclass
class SubmitResultOutput:
    """
    Container for the output of submit_result().

    This structured output allows the display layer to show:
    - The computed result (from real DB data)
    - The SQL inputs that were used
    - The function that was applied
    - Any errors that occurred

    Attributes:
        success: Whether the computation completed without errors
        result: The final computed value (only valid if success=True)
        inputs_used: Dict mapping input names to their DataFrames
        function_code: The Python code that was executed
        explanation: The agent's explanation of what this computes
        error: Error message if success=False
    """
    success: bool
    result: Any = None
    inputs_used: dict[str, pd.DataFrame] = field(default_factory=dict)
    function_code: str = None
    explanation: str = None
    error: str = None


def submit_result(
    inputs: dict[str, str],
    function: str,
    explanation: str
) -> SubmitResultOutput:
    """
    Execute the final computation and return structured output.

    This is the ONLY pathway for data to reach the user. The flow is:
    1. Execute each SQL query in 'inputs' against DuckDB
    2. Pass resulting DataFrames to the Python function
    3. Return the 'result' variable from the function

    The LLM never directly outputs data - it only writes the queries
    and function. The actual data comes from database execution.

    Args:
        inputs: Map of variable names to SQL queries.
                Example: {"orders": "SELECT * FROM marts.fct_orders"}
        function: Python code with access to inputs as DataFrames.
                  Must define a 'result' variable.
        explanation: Human-readable explanation of the computation.

    Returns:
        SubmitResultOutput containing the result or error information.
    """
    sql_executor = SQLExecutor()
    py_executor = PythonExecutor()

    # Step 1: Execute all SQL queries to get real data
    dataframes = {}
    for name, sql in inputs.items():
        df, error = sql_executor.execute(sql)
        if error:
            return SubmitResultOutput(
                success=False,
                error=f"SQL error for input '{name}': {error}",
                function_code=function,
                explanation=explanation
            )
        dataframes[name] = df

    # Step 2: Apply the function to the real data
    result, error = py_executor.execute(function, dataframes)

    if error:
        return SubmitResultOutput(
            success=False,
            error=f"Function execution error: {error}",
            inputs_used=dataframes,
            function_code=function,
            explanation=explanation
        )

    # Step 3: Return structured output for display
    return SubmitResultOutput(
        success=True,
        result=result,
        inputs_used=dataframes,
        function_code=function,
        explanation=explanation
    )
