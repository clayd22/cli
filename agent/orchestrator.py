"""
orchestrator.py

The central control loop for AstroAgent.

This module manages the flow between:
- User input
- LLM reasoning and tool calls
- Tool execution (internal exploration vs final output)
- Result display

The key architectural guarantee is that final answers can ONLY come
through the submit_result tool, ensuring all data originates from
real database execution, not LLM generation.
"""

import json
from typing import Generator
from openai import OpenAI

from .config import get_api_key
from .schema import get_full_schema_context
from .theme import console, print_thinking, print_error, print_divider, tool_status, print_tool_call, print_tool_result_preview
from .display import display_submit_result

# Import tool definitions and implementations
from .tools.internal.run_sql import RUN_SQL_TOOL, run_sql
from .tools.internal.run_python import RUN_PYTHON_TOOL, run_python
from .tools.internal.inspect_schema import INSPECT_SCHEMA_TOOL, inspect_schema
from .tools.internal.context_tools import (
    READ_CONTEXT_TOOL, UPDATE_CONTEXT_TOOL,
    tool_read_context, tool_update_context
)
from .tools.output.submit_result import SUBMIT_RESULT_TOOL, submit_result
from .tools.output.submit_observation import SUBMIT_OBSERVATION_TOOL, submit_observation
from .tools.output.render_artifact import RENDER_ARTIFACT_TOOL, render_artifact
from .tools.output.send_message import SEND_MESSAGE_TOOL, send_message
from .display import display_observation, display_artifact


class Orchestrator:
    """
    Manages the agent loop: question → reasoning → tools → answer.

    The orchestrator ensures that:
    1. The LLM has access to schema context
    2. Internal tools (run_sql, run_python) return results to LLM for thinking
    3. The submit_result tool produces the final output to the user
    4. The LLM cannot directly output data values

    Attributes:
        client: OpenAI API client
        model: Model identifier to use
        tools: List of tool definitions for function calling
        conversation_history: Message history for context
    """

    # All available tools for the agent
    TOOLS = [
        RUN_SQL_TOOL,
        RUN_PYTHON_TOOL,
        INSPECT_SCHEMA_TOOL,
        READ_CONTEXT_TOOL,
        UPDATE_CONTEXT_TOOL,
        SUBMIT_RESULT_TOOL,
        SUBMIT_OBSERVATION_TOOL,
        RENDER_ARTIFACT_TOOL,
        SEND_MESSAGE_TOOL,
    ]

    # Maps tool names to their implementation functions
    TOOL_HANDLERS = {
        "run_sql": run_sql,
        "run_python": run_python,
        "inspect_schema": inspect_schema,
        "read_context": tool_read_context,
        "update_context": tool_update_context,
        "submit_result": submit_result,
        "submit_observation": submit_observation,
        "render_artifact": render_artifact,
        "send_message": send_message,
    }

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the orchestrator.

        Args:
            model: OpenAI model to use for reasoning
        """
        api_key = get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not configured. Run 'astro config' first.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Construct the system prompt with schema context.

        The prompt instructs the LLM to:
        - Use internal tools for exploration
        - Always use submit_result for final answers
        - Never output raw data in text responses
        """
        schema_context = get_full_schema_context()

        return f"""You are AstroAgent, a data analysis assistant.

## Your Mission
Answer questions about data by exploring the database and computing results.

## Critical Rules
1. CHECK CONTEXT FIRST: Use read_context to see what you already know about this data
2. EXPLORE AS NEEDED: Use run_sql and inspect_schema to understand the data
3. UPDATE CONTEXT: When you learn something useful, use update_context to save it
4. NEVER output data values directly in your text responses
5. ALWAYS use submit_result, submit_observation, or render_artifact for final answers
6. FORMAT RESULTS FOR HUMANS: Results should be readable, well-formatted, and directly answer the question

## Quality Standards
- Write COMPLETE, WORKING code - no placeholders or "// TODO" comments
- Test your logic with run_sql/run_python before submitting final answers
- Handle edge cases: empty data, null values, errors
- For visualizations: use proper HTML5 structure, responsive design, error handling
- For analysis: cite specific numbers and be precise
- Match the user's request exactly - if they ask for 3D, deliver 3D

## Available Tools
- read_context: Check your notes about this data platform (do this first!)
- update_context: Save useful insights for future reference (keep it concise)
- inspect_schema: View tables, columns, and sample data
- run_sql: Execute SQL queries to explore (results come back to you)
- run_python: Test Python code on query results (for complex analysis/exploration)
- submit_result: For computed answers - executes SQL + function, shows result to user
- submit_observation: For narrative answers - describe patterns, anomalies, insights
- render_artifact: For visualizations - creates charts/files and opens in viewer
- send_message: Send a message to the user (for clarifications, progress updates)

## How run_python Works (for exploration)
Use this to test complex logic before submitting. You provide:
- queries: map of names to SQL queries - each becomes a DataFrame
- code: Python code with access to pandas (pd), numpy (np), and your query results

Example:
run_python(
  queries={{"sales": "SELECT product_name, total FROM marts.fct_orders"}},
  code="result = sales.groupby('product_name')['total'].sum().head(5)"
)

## How submit_result Works (for computed answers)
You provide:
- inputs: SQL queries to fetch data (can be multiple)
- function: Python code to process the data (must set 'result' variable)
- explanation: What this computes

IMPORTANT: The 'result' variable should be HUMAN READABLE.

## How submit_observation Works (for narrative answers)
Use when the answer is qualitative: patterns, anomalies, insights, analysis.
You provide:
- observation: Your analysis in clear prose with specific data points
- supporting_queries: (optional) The SQL queries you used
- supporting_data: (optional) Key numbers or a small table

## How render_artifact Works (for visualizations/files)
Use when output is best viewed externally: interactive charts, 3D graphs, HTML reports.
You provide:
- inputs: SQL queries (results become window.DATA.yourname in JS)
- code: Complete HTML with CSS/JS (use CDN libraries like Three.js, D3, Chart.js)
- filename: Output file name (e.g., 'chart.html')
- explanation: What this shows

DATA IS INJECTED BY SYSTEM - you write visualization code, system provides real data.
Access data via window.DATA.queryname where queryname matches your inputs key.

## Database Schema
{schema_context}

## Example 1 - Simple aggregation
User: "What's the total revenue?"
submit_result(
  inputs={{"orders": "SELECT total FROM marts.fct_orders"}},
  function=\"\"\"
total = orders['total'].sum()
result = f"Total revenue: ${{total:,.2f}}"
\"\"\",
  explanation="Sum of all order totals"
)

## Example 2 - Finding pairs/relationships
User: "Which products are bought together?"
submit_result(
  inputs={{"pairs": "SELECT ... (query to find pairs)"}},
  function=\"\"\"
top_pair = pairs.iloc[0]
result = f"The most frequently bought together products are {{top_pair['product1']}} and {{top_pair['product2']}}, appearing together in {{top_pair['count']}} orders."
\"\"\",
  explanation="Finding product pairs from transaction analysis"
)

## Example 3 - Tables/lists (when appropriate)
User: "Show me top 5 products by revenue"
submit_result(
  inputs={{"products": "SELECT product_name, SUM(total) as revenue FROM ... GROUP BY ... ORDER BY revenue DESC LIMIT 5"}},
  function=\"\"\"
result = products[['product_name', 'revenue']].to_string(index=False)
\"\"\",
  explanation="Top 5 products ranked by revenue"
)

Remember: Format the result for human readability. Use f-strings to create clear, descriptive answers."""

    def process_question(self, question: str) -> None:
        """
        Process a user question through the agent loop.

        This is the main entry point. It:
        1. Adds the question to conversation history
        2. Calls the LLM with tools available
        3. Executes any tool calls
        4. Loops until submit_result is called or no more tool calls

        Args:
            question: The user's question about the data
        """
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        # Agent loop: keep going until we get a final answer
        while True:
            response = self._call_llm()

            # Check if the LLM wants to call tools
            if response.choices[0].message.tool_calls:
                should_continue = self._handle_tool_calls(
                    response.choices[0].message
                )
                if not should_continue:
                    # submit_result was called, we're done
                    break
            else:
                # No tool calls - LLM is just responding with text
                # This shouldn't contain data, just reasoning
                assistant_message = response.choices[0].message.content
                if assistant_message:
                    console.print(f"\n[thinking]{assistant_message}[/thinking]\n")

                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                break

    def _call_llm(self):
        """
        Make an API call to the LLM with current context and tools.

        Returns:
            OpenAI ChatCompletion response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.TOOLS,
            tool_choice="auto",
        )

    def _handle_tool_calls(self, assistant_message) -> bool:
        """
        Execute tool calls from the LLM response.

        For internal tools (run_sql, run_python, inspect_schema):
            - Execute the tool
            - Return results to the LLM for further reasoning
            - Return True to continue the loop

        For submit_result:
            - Execute the final computation
            - Display results to the user
            - Return False to end the loop

        Args:
            assistant_message: The message containing tool calls

        Returns:
            True if the loop should continue, False if we're done
        """
        # Add the assistant's message with tool calls to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        })

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Create a summary of the arguments for display
            args_summary = self._summarize_args(tool_name, tool_args)
            print_tool_call(tool_name, args_summary)

            # Get the handler for this tool
            handler = self.TOOL_HANDLERS.get(tool_name)
            if not handler:
                self._add_tool_result(tool_call.id, f"Unknown tool: {tool_name}")
                continue

            # Execute the tool with spinner
            try:
                with tool_status(tool_name, args_summary):
                    if tool_name in ("submit_result", "submit_observation", "render_artifact"):
                        # Final answer - execute with spinner
                        output = handler(**tool_args)

                # Display result outside spinner
                if tool_name == "submit_result":
                    display_submit_result(output)
                    self._add_tool_result(tool_call.id, "Result displayed to user.")
                    return False
                elif tool_name == "submit_observation":
                    display_observation(output)
                    self._add_tool_result(tool_call.id, "Observation displayed to user.")
                    return False
                elif tool_name == "render_artifact":
                    display_artifact(output)
                    if output.success:
                        self._add_tool_result(tool_call.id, "Artifact rendered and opened.")
                        return False
                    else:
                        # Feed error back to LLM so it can fix
                        self._add_tool_result(tool_call.id, f"Render failed: {output.error}")
                elif tool_name == "send_message":
                    # Message to user - display and continue
                    handler(**tool_args)
                    self._add_tool_result(tool_call.id, "Message sent.")
                else:
                    # Internal tool - execute and show preview
                    with tool_status(tool_name, args_summary):
                        result = handler(**tool_args)

                    # Show a preview of internal tool results
                    print_tool_result_preview(tool_name, result)
                    self._add_tool_result(tool_call.id, result)

            except Exception as e:
                # Feed error back to LLM so it can fix
                error_msg = f"Error: {type(e).__name__}: {str(e)}"
                console.print(f"  [error]✗ {error_msg}[/error]")
                self._add_tool_result(tool_call.id, error_msg)

        return True

    def _add_tool_result(self, tool_call_id: str, result: str) -> None:
        """
        Add a tool result to the conversation history.

        Args:
            tool_call_id: ID of the tool call this is responding to
            result: String result from the tool execution
        """
        self.conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })

    def clear_history(self) -> None:
        """Clear conversation history for a fresh start."""
        self.conversation_history = []

    def _summarize_args(self, tool_name: str, args: dict) -> str:
        """
        Create a brief summary of tool arguments for display.

        Args:
            tool_name: Name of the tool being called
            args: The arguments dictionary

        Returns:
            A short human-readable summary
        """
        if tool_name == "run_sql":
            sql = args.get("sql", "")
            # Extract table name if possible
            sql_lower = sql.lower()
            if "from" in sql_lower:
                parts = sql_lower.split("from")
                if len(parts) > 1:
                    table = parts[1].strip().split()[0]
                    return f"querying {table}"
            return "executing query"

        elif tool_name == "run_python":
            queries = args.get("queries", {})
            return f"processing {len(queries)} input(s)"

        elif tool_name == "inspect_schema":
            action = args.get("action", "")
            table = args.get("table", "")
            if table:
                return f"{action} on {table}"
            return action

        elif tool_name == "submit_result":
            inputs = args.get("inputs", {})
            return f"computing from {len(inputs)} input(s)"

        elif tool_name == "send_message":
            msg = args.get("message", "")
            if len(msg) > 30:
                return msg[:27] + "..."
            return msg

        elif tool_name == "read_context":
            return "checking notes"

        elif tool_name == "update_context":
            section = args.get("section", "")
            return f"updating {section}"

        elif tool_name == "submit_observation":
            obs = args.get("observation", "")
            if len(obs) > 40:
                return obs[:37] + "..."
            return obs

        elif tool_name == "render_artifact":
            filename = args.get("filename", "")
            return f"creating {filename}"

        return ""
