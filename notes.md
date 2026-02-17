# AstroAgent - Architecture Notes

## Overview
A CLI data agent for answering ad-hoc questions about a data platform. Built for a senior software engineering interview assessment at Astronomer.

## Core Architecture Principle: Structural Hallucination Prevention

The LLM **never outputs data directly**. Instead:
1. LLM writes queries/functions
2. System executes against real DuckDB database
3. Real results returned to user

```
User Question → LLM generates SQL/function → System executes → Real data to user
```

This makes hallucinations **structurally impossible** for final outputs.

---

## Anti-Hallucination Architecture Deep Dive

### The Problem
LLMs can confidently state incorrect data. If an LLM says "Revenue was $1.2M" - how do you know it didn't hallucinate that number?

### Our Solution: Separation of Query Generation and Data Transmission

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER ASKS QUESTION                           │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INTERNAL THINKING PHASE                         │
│  (LLM can see data here, but output doesn't go to user)            │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ run_sql()   │  │ run_python()│  │ inspect_    │                │
│  │             │  │             │  │ schema()    │                │
│  │ Returns     │  │ Returns     │  │             │                │
│  │ data TO LLM │  │ data TO LLM │  │ Returns     │                │
│  │ for thinking│  │ for thinking│  │ schema info │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
│  LLM explores, debugs, iterates until it understands...            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT PHASE                                   │
│  (Data flows FROM DATABASE TO USER, not through LLM text)          │
│                                                                     │
│  LLM calls ONE of:                                                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ submit_result(                                               │   │
│  │   inputs={"orders": "SELECT SUM(total) FROM fct_orders"},   │   │
│  │   function="result = f'Revenue: ${orders.iloc[0,0]:,.2f}'"  │   │
│  │ )                                                            │   │
│  │                                                              │   │
│  │ System executes SQL → Applies function → Shows to user      │   │
│  │ LLM CANNOT alter the data at this point                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  OR                                                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ submit_observation(                                          │   │
│  │   observation="Based on the data, there are anomalies...",  │   │
│  │   supporting_queries={"anomalies": "SELECT ... "},          │   │
│  │   supporting_data="Product X: 5 std devs above mean"        │   │
│  │ )                                                            │   │
│  │                                                              │   │
│  │ User sees observation + the actual queries used             │   │
│  │ User can verify by re-running queries                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  OR                                                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ render_artifact(                                             │   │
│  │   inputs={"sales": "SELECT date, amount FROM ..."},         │   │
│  │   function="fig = px.line(sales, ...); fig.write_html(...)" │   │
│  │ )                                                            │   │
│  │                                                              │   │
│  │ System executes SQL → Runs viz code → Opens file            │   │
│  │ Visualization is from REAL data, not LLM imagination        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### What Can Be Written to User vs Internal

| Category | Goes To | Example |
|----------|---------|---------|
| **Internal tool results** | LLM only | `run_sql()` returns "35980 rows" - LLM sees this to think |
| **LLM reasoning text** | Displayed dimmed | "Let me explore the schema..." |
| **submit_result output** | User (from DB) | Actual computed value from database |
| **submit_observation** | User (narrative) | Analysis text + queries used for verification |
| **render_artifact** | User (file) | HTML/chart file opened in browser |
| **send_message** | User (info) | "I need clarification on..." |

### The Key Guarantee

```
┌────────────────────────────────────────────────────────────────┐
│  DATA VALUES IN FINAL OUTPUT                                   │
│                                                                │
│  ✗ NEVER come from: LLM generating tokens like "Revenue: $X"  │
│  ✓ ALWAYS come from: Database execution of LLM-written query  │
└────────────────────────────────────────────────────────────────┘
```

### Why This Matters

Traditional approach (unsafe):
```
User: "What's total revenue?"
LLM: "Total revenue is $1,234,567"  ← Could be hallucinated!
```

Our approach (safe):
```
User: "What's total revenue?"
LLM: [calls submit_result with SQL + function]
System: [executes SQL against DB, applies function]
User sees: "$1,234,567" ← Came from actual database!
```

The LLM wrote the QUERY, not the ANSWER. The answer came from the database.

---

## Tool Call Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LOOP                             │
│                                                                  │
│  while True:                                                     │
│      response = call_llm(conversation_history, tools)            │
│                                                                  │
│      if response.has_tool_calls:                                │
│          for tool_call in response.tool_calls:                  │
│              │                                                   │
│              ├─► INTERNAL TOOL (run_sql, run_python, etc.)      │
│              │   result = execute_tool(tool_call)               │
│              │   add_to_conversation(result)  ← LLM sees this   │
│              │   continue loop                                   │
│              │                                                   │
│              ├─► OUTPUT TOOL (submit_result, etc.)              │
│              │   IF SUCCESS:                                    │
│              │       display_to_user(result) ← User sees this   │
│              │       break loop (done!)                         │
│              │   IF ERROR:                                      │
│              │       add_error_to_conversation() ← LLM can fix  │
│              │       continue loop                              │
│              │                                                   │
│              └─► send_message                                   │
│                  display_to_user(message)                       │
│                  continue loop                                   │
│                                                                  │
│      else:                                                       │
│          # LLM responded with text (reasoning)                  │
│          display_dimmed(response.text)                          │
│          break                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Error Self-Correction Flow

```
User: "Show me a 3D chart"
        │
        ▼
LLM calls render_artifact(function="import plotly...")
        │
        ▼
System: ERROR - "ImportError: __import__ not found"
        │
        ▼
Error added to conversation history
        │
        ▼
LLM sees error, retries: render_artifact(function="fig = px.scatter_3d...")
        │
        ▼
System: SUCCESS - file created and opened
        │
        ▼
Done!
```

---

## File Structure

```
agent/
├── cli.py                 # CLI interface, REPL
├── config.py              # Persistent API key storage (~/.astroagent/)
├── theme.py               # Space-themed Rich styling
├── orchestrator.py        # Main agent loop, tool dispatch
├── schema.py              # DuckDB introspection
├── context.py             # Persistent context/memory file
├── display.py             # Result formatting
│
├── sandbox/               # Execution environments
│   ├── sql_executor.py    # DuckDB query execution
│   └── python_executor.py # Safe Python execution
│
└── tools/
    ├── internal/          # Tools for agent thinking (results go back to LLM)
    │   ├── run_sql.py
    │   ├── run_python.py
    │   ├── inspect_schema.py
    │   └── context_tools.py
    │
    └── output/            # Tools for final answers (results go to user)
        ├── submit_result.py      # Computed answers (SQL + function)
        ├── submit_observation.py # Narrative/qualitative answers
        ├── render_artifact.py    # Visualizations (files opened externally)
        └── send_message.py       # Agent-to-user communication
```

## Key Design Decisions

### 1. Internal vs Output Tools
- **Internal tools**: Agent can explore, see data, debug - results return to LLM
- **Output tools**: Final answers to user - data comes from DB execution, not LLM

### 2. Persistent Context File
- `.astroagent/context.md` stores agent's notes about the data platform
- Agent checks context first, updates when learning new insights
- Survives across sessions, improves over time

### 3. Multiple Output Modalities
- `submit_result`: For computed values (revenue totals, counts, etc.)
- `submit_observation`: For narrative analysis (anomalies, patterns, insights)
- `render_artifact`: For visualizations (charts, 3D models, HTML reports)

### 4. Self-Correction via Error Feedback
- All tool errors are fed back to the LLM conversation
- Agent can retry with corrected code
- No silent failures

### 5. Human-Readable Formatting
- System prompt instructs agent to format results as readable prose
- Use f-strings, not raw data structures
- Tables when appropriate, sentences when conversational

## Tools Summary

| Tool | Type | Purpose |
|------|------|---------|
| `read_context` | Internal | Check saved notes about data |
| `update_context` | Internal | Save insights for future |
| `inspect_schema` | Internal | View tables, columns, samples |
| `run_sql` | Internal | Explore data with SQL |
| `run_python` | Internal | Test complex logic |
| `submit_result` | Output | Computed answers |
| `submit_observation` | Output | Narrative analysis |
| `render_artifact` | Output | Visualizations/files |
| `send_message` | Output | Agent messages to user |

## Frontier Features Discussed (For Future)

1. **Agentic Reasoning**: Multi-step decomposition, ReAct pattern
2. **RAG over Schema**: Embed dbt docs, semantic search
3. **Anomaly Detection**: Statistical methods built-in
4. **Confidence Scoring**: Agent reports certainty
5. **MCP Server Mode**: Expose as MCP server for other tools

## Technical Stack

- **CLI**: Click + Rich (space theme)
- **LLM**: OpenAI GPT-4o (function calling)
- **Database**: DuckDB (warehouse/data.duckdb)
- **Visualizations**: Plotly (interactive HTML charts)
- **Python Sandbox**: Restricted exec with pandas/numpy

## Configuration

- API key stored in `~/.astroagent/config.json`
- Context file at `.astroagent/context.md` (project root)
- Artifacts at `.astroagent/artifacts/`

## Commands

```bash
uv run astro              # Start REPL
uv run astro config       # Set API key
uv run astro ask "..."    # One-shot question
uv run astro reset        # Clear config
```

REPL commands: `help`, `clear`, `schema`, `context`, `exit`

## Current Limitations

1. Plotly doesn't support true 3D bar charts
2. Python sandbox is restricted (no arbitrary imports)
3. Single-model (OpenAI only currently)

## Next Steps if More Time

1. **React/HTML visualization server**: Serve interactive visualizations on local port
2. **Multi-provider support**: Anthropic, local models
3. **Streaming responses**: Show thinking in real-time
4. **Query caching**: Avoid re-running expensive queries
5. **Export to notebook**: Generate Jupyter notebooks from conversations
