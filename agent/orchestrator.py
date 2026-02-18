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

from typing import Optional
from .config import get_api_key
from .settings import AgentSettings, OutputMode
from .session import SessionManager
from .memory import ContextRetriever  # --- RAG: Import retriever ---
from .theme import console, print_thinking, print_error, print_warning, print_divider, tool_status, print_tool_call, print_tool_result_preview
from .display import display_submit_result

# Import tool definitions and implementations
from .tools.internal.run_sql import RUN_SQL_TOOL, run_sql
from .tools.internal.run_python import RUN_PYTHON_TOOL, run_python
from .tools.internal.inspect_schema import INSPECT_SCHEMA_TOOL, inspect_schema
from .tools.internal.context_tools import (
    READ_CONTEXT_TOOL, UPDATE_CONTEXT_TOOL,
    tool_read_context, tool_update_context
)
from .tools.internal.inspect_platform import INSPECT_PLATFORM_TOOL, inspect_platform
from .tools.output.submit_result import SUBMIT_RESULT_TOOL, submit_result
from .tools.output.submit_observation import SUBMIT_OBSERVATION_TOOL, submit_observation
from .tools.output.send_message import SEND_MESSAGE_TOOL, send_message
from .display import display_observation


class Orchestrator:
    """
    Manages the agent loop: question → reasoning → tools → answer submission.

    The orchestrator ensures that:
    1. The LLM has access to context from the database, schema, and platform
    2. Internal tools (run_sql, run_python, inspect_schema, inspect_platform) return results to LLM for thinking
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
        INSPECT_PLATFORM_TOOL,
        READ_CONTEXT_TOOL,
        UPDATE_CONTEXT_TOOL,
        SUBMIT_RESULT_TOOL,
        SUBMIT_OBSERVATION_TOOL,
        SEND_MESSAGE_TOOL,
    ]

    # Maps tool names to their implementation functions
    TOOL_HANDLERS = {
        "run_sql": run_sql,
        "run_python": run_python,
        "inspect_schema": inspect_schema,
        "inspect_platform": inspect_platform,
        "read_context": tool_read_context,
        "update_context": tool_update_context,
        "submit_result": submit_result,
        "submit_observation": submit_observation,
        "send_message": send_message,
    }

    def __init__(self, settings: Optional[AgentSettings] = None, session_manager: Optional[SessionManager] = None):
        """
        Initialize the orchestrator.

        Args:
            settings: Agent settings for model and output mode
            session_manager: Session manager for conversation tracking
        """
        api_key = get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not configured. Run 'astro config' first.")

        self.client = OpenAI(api_key=api_key)
        self.settings = settings or AgentSettings()
        self.session_manager = session_manager or SessionManager(self.settings.model)

        # --- RAG: Initialize context retriever ---
        self._retriever: Optional[ContextRetriever] = None
        self._current_question: str = ""  # Track for indexing after success

    @property
    def retriever(self) -> ContextRetriever:
        """Lazy-load retriever to avoid startup delay if not needed."""
        if self._retriever is None:
            self._retriever = ContextRetriever()
        return self._retriever

    @property
    def conversation_history(self) -> list:
        """Get conversation history from session manager."""
        return self.session_manager.get_history()

    @conversation_history.setter
    def conversation_history(self, value: list):
        """Set conversation history in session manager."""
        self.session_manager.set_history(value)

    def _build_system_prompt(self) -> str:
        """
        Construct the system prompt.

        The prompt instructs the LLM to:
        - Use internal tools for exploration
        - Always use submit_result for final answers
        - Never output raw data in text responses
        """
        return """You are AstroAgent, a data analysis assistant.

## Your Mission
Answer questions about data by exploring the database and computing results.

## MANDATORY OUTPUT RULES (NEVER VIOLATE)
You are FORBIDDEN from outputting data values in your text responses.
You MUST use an output tool (submit_result or submit_observation) for EVERY answer.

VIOLATIONS (never do these):
- using "staging" or stale data tables
- Writing "The answer is 1,234" in your text response
- Stating specific numbers, names, or values you found from queries
- Summarizing query results in prose without using an output tool
- Using run_sql and then describing what you found in text

CORRECT BEHAVIOR:
- Use run_sql/run_python to explore and understand the data
- When ready to answer, call submit_result with the SQL + function to compute the answer
- The submit_result tool will display the query, function, AND result to the user
- This ensures all data shown to users comes from real database execution

## Workflow
1. CHECK CONTEXT FIRST: Use read_context to see what you already know
2. EXPLORE AS NEEDED: Use run_sql and inspect_schema to understand the data
3. UPDATE CONTEXT: When you learn something useful, use update_context to save it
4. SUBMIT ANSWER: Use submit_result (computed) or submit_observation (narrative)

## Quality Standards
- Write COMPLETE, WORKING code - no placeholders or "// TODO" comments
- Test your logic with run_sql/run_python before submitting final answers
- Handle edge cases: empty data, null values, errors

## Available Tools
- read_context: Check your notes about this data platform (do this first!)
- update_context: Save useful insights for future reference (keep it concise)
- inspect_schema: View tables, columns, and sample data
- inspect_platform: View Airflow DAGs, dbt models, Evidence dashboards (data lineage, transformations)
- run_sql: Execute SQL queries to explore (results come back to you)
- run_python: Test Python code on query results (for complex analysis/exploration)
- submit_result: For computed answers - executes SQL + function, shows result to user
- submit_observation: For narrative answers - describe patterns, anomalies, insights
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

Format results for human readability using f-strings.

## FINAL REMINDER
NEVER write data values in your text responses. ALWAYS use submit_result to deliver answers.
After exploring with run_sql, you MUST call submit_result - do not summarize findings in text."""

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
        # --- RAG: Track question for indexing after successful answer ---
        self._current_question = question

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

    def _get_filtered_tools(self) -> list:
        """
        Get tools filtered by current output mode.

        Returns:
            List of tool definitions allowed in current mode.
        """
        allowed_output_tools = self.settings.get_allowed_output_tools()
        output_tool_names = {"submit_result", "submit_observation"}

        filtered = []
        for tool in self.TOOLS:
            tool_name = tool["function"]["name"]
            # Always include internal tools and send_message
            if tool_name not in output_tool_names:
                filtered.append(tool)
            # Only include output tools that are allowed in current mode
            elif tool_name in allowed_output_tools:
                filtered.append(tool)

        return filtered

    def _call_llm(self):
        """
        Make an API call to the LLM with current context and tools.

        Returns:
            OpenAI ChatCompletion response
        """
        # Check for context window warning
        if self.session_manager.current_session and self.session_manager.current_session.is_context_warning():
            pct = self.session_manager.current_session.get_context_usage_percent()
            print_warning(f"Context usage at {pct:.0f}% - consider starting a new session")

        # Build system prompt with current mode instruction
        system_prompt = self._build_system_prompt() + self.settings.get_mode_instruction()

        # --- RAG: Retrieve relevant context for current question ---
        rag_context = ""
        if self._current_question:
            try:
                result = self.retriever.retrieve_with_scores(self._current_question)
                if result.total_items > 0:
                    rag_context = self.retriever.format_for_prompt(result)
                    system_prompt += f"\n\n{rag_context}"
                    # --- RAG: Log with summary ---
                    console.print(f"[dim]  ~ RAG: {result.summary()}[/dim]")
                    # --- RAG: Verbose mode shows full debug ---
                    if self.settings.rag_verbose:
                        console.print(f"[dim]{self.retriever.format_debug(result)}[/dim]")
                else:
                    console.print("[dim]  ~ RAG: no relevant context found[/dim]")
            except Exception as e:
                console.print(f"[dim]  ~ RAG: failed ({type(e).__name__})[/dim]")

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history
        ]

        # Verbose mode: print full prompt
        if self.settings.verbose:
            console.print("\n[dim]" + "=" * 60 + "[/dim]")
            console.print("[dim]SYSTEM PROMPT:[/dim]")
            console.print(f"[dim]{system_prompt}[/dim]")
            console.print("[dim]" + "-" * 60 + "[/dim]")
            console.print(f"[dim]MESSAGES: {len(self.conversation_history)}[/dim]")
            for msg in self.conversation_history:
                role = msg.get("role", "?")
                content = msg.get("content") or ""
                preview = content[:200] if len(content) > 200 else content
                suffix = "..." if len(content) > 200 else ""
                console.print(f"[dim]  [{role}] {preview}{suffix}[/dim]")
            console.print("[dim]" + "=" * 60 + "[/dim]\n")

        # Get tools filtered by output mode
        tools = self._get_filtered_tools()

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        # Track token usage
        if response.usage:
            self.session_manager.update_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

        return response

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
                    if tool_name in ("submit_result", "submit_observation"):
                        # Final answer - execute with spinner
                        output = handler(**tool_args)

                # Display result outside spinner
                if tool_name == "submit_result":
                    display_submit_result(output)
                    self._add_tool_result(tool_call.id, "Result displayed to user.")
                    # --- RAG: Index successful query for future retrieval ---
                    if output.success and self._current_question:
                        try:
                            sql = list(output.sql_queries.values())[0] if output.sql_queries else ""
                            result_summary = str(output.result)[:200]
                            session_id = self.session_manager.current_session.id if self.session_manager.current_session else None
                            self.retriever.index_successful_query(
                                question=self._current_question,
                                sql=sql,
                                result_summary=result_summary,
                                session_id=session_id,
                            )
                            console.print("[dim]  ~ RAG: indexed query[/dim]")
                        except Exception:
                            pass
                    return False
                elif tool_name == "submit_observation":
                    display_observation(output)
                    self._add_tool_result(tool_call.id, "Observation displayed to user.")
                    # --- RAG: Index observation for future retrieval ---
                    if self._current_question:
                        try:
                            session_id = self.session_manager.current_session.id if self.session_manager.current_session else None
                            self.retriever.index_observation(
                                observation=output.observation,
                                session_id=session_id,
                            )
                            console.print("[dim]  ~ RAG: indexed observation[/dim]")
                        except Exception:
                            pass
                    return False
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
        self.session_manager.clear_history()

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

        elif tool_name == "inspect_platform":
            action = args.get("action", "")
            name = args.get("name", "")
            if name:
                return f"{action}: {name}"
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

        return ""
