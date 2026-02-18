# AstroAgent CLI

Connecting an LLM instance to a CLI is relatively simple.  Allowing it to execute SQL and read results to answer questions is more difficult, but also achievable.
However, I think these straightforward solutions have a common problem that is particularly important for data tools: The numbers come from an LLM.  While they are, say, 98 percent
accurate in reading a value, running an internal calculation, and returning it to you, the 2 percent where misunderstanding or outright hallucination occur can be devastating.

This issue is the primary focus of my AstroAgent's framework.  Through its "query" output mode, we can effectively guarantee that the values returned to a user for a given question are
completely accurate and free from hallucination, while allowing for a quick scan of the models methodology.  This allows it to, essentially, act much more like a junior data engineer
assisting the user by tackling the question as a human would.

## How It Works

Instead of our agent having a tool call Q that runs a sql query, and then the agent returning its answer based on that, we introduce two sets of tool calls

| | |
|---|---|
| **Thinking** | `run_sql`, `run_python`, `inspect_schema`, `inspect_platform` |
| **Submission** | `submit_result`, `submit_observation` |

This breaks our agents process into two parts.  When a question comes in from a user, the agent first queries data, also runs python functions on that data (or its own test data) to think of a solution.  Then in the second stage, our agent MUST submit to our submission tool call two items: a set of SQL queries that fetch the relevant data, and a Python function that processes those query results to produce the final answer.  This is then ran and if successful, returned to the user.  This effectively guarantees that the operation that obtains the answer is run directly on the database AFTER
agent thought, bypassing the step that causes confusion and hallucination.  The only caveat to this is Qualitataive analysis, for which the agent can call an "observation" tool call to
return text to ther user that is marked as such

### Example

Take this example to illustrate:

> We ask: How much in sales did we do last quarter?

The agent first explores.  It calls `run_sql` to check what tables exist, maybe runs a quick query to understand the date range of the data.  It might call `inspect_schema` to see what columns are available in fct_orders.  All of these results come back to the agent for reasoning - the user sees nothing yet.

Then, when the agent is confident it understands the problem, it calls `submit_result` with two things: a SQL query that fetches the relevant data, and a Python function that computes the final answer.

```python
submit_result(
  inputs={"orders": "SELECT total FROM marts.fct_orders WHERE transaction_date >= '2024-10-01'"},
  function="""
total = orders['total'].sum()
result = f"Total Q4 sales: ${total:,.2f}"
""",
  explanation="Sum of order totals for Q4 2024"
)
```

The system executes this directly against the database and returns the result to the user.  The LLM never touched the number - it only wrote the code that computed it.  This is the key insight: by forcing all data through a verified execution path, we eliminate the hallucination problem entirely for quantitative answers.

---

## RAG

The second concern I came across when building this tool (and running out of tokens with it) is that of memory.  This is just a demo data warehouse and it is massive for an LLM.
Fetching the schema every time, reading multiple tables, and relearning everything through each session is super costly.

My first idea was to simply track the accesses of different tables in our database and build a frequency table that we could give to the model.  The performance of this across test questions
is very low though, because there is no way to correlate ideas in the question, say "price", "cost", etc. to different tables they are related to.

This brought me to my second idea, parsing queries and connecting keywords in the queries to the table accesses.  There are three ways to do this:
  - Simply slice every question from the user into words and store
    - this will be convoluted with filler words, assigns no meaning to words
  - Ask an LLM model to give you the keywords from a question
    - This can add a lot of cost, also does not store relations between words
  - use Langchain, spaCy, or a similar library to parse
    - This is doable, but hard to include in a CLI agent

The flaws in these attempts led me to look for a direct lightweight RAG implementation.  Now, these model sessions are already doing some version of RAG internally, so why do we need it?
Because the model has no notion of our processes and schema on startup.  This is our RAG flow and how it assists the model:

On first use, we run `/rag index` to embed and store the database schema - every table and column gets vectorized and saved to ChromaDB.  Now when a question comes in, say "what's our profit margin by category?", the question gets embedded and we search for similar vectors in our schema collection.  This returns semantically related items like `fct_orders.product_cost`, `fct_orders.unit_price`, `dim_products.margin` - and these get injected into the system prompt before the LLM starts thinking.

For this first question, there are no past queries to retrieve.  The agent explores using the schema context, builds its solution, and submits.  After a successful `submit_result`, we index the triplet: the question, the SQL that answered it, and a summary of the result.  This goes into the queries collection.

Now when someone asks "how much margin do we make on electronics?", the RAG system finds that similar past question and injects the working SQL alongside the relevant schema.  The model doesn't have to rediscover the join pattern or figure out which columns to use - it has a working example right there.  This compounds: the more questions answered, the more examples available for future questions.

Storage is at `~/.astroagent/memory/chroma/` with three collections:
- **schema**: Tables and columns from the warehouse
- **queries**: Successful question-SQL-result triplets
- **observations**: Qualitative insights the agent has noted

---

```
ChromaDB (~/.astroagent/memory/chroma/)
│
├── schema_items
│   ├── [table entry]
│   │   ├── text: "Table: marts.fct_orders\nColumns: transaction_id, ..."
│   │   ├── embedding: [1536 floats]
│   │   └── metadata: {type, table_name, column_names, indexed_at}
│   │
│   └── [column entry]
│       ├── text: "Column: marts.fct_orders.total (DOUBLE)"
│       ├── embedding: [1536 floats]
│       └── metadata: {type, table_name, column_name, column_type}
│
├── query_history
│   └── [query entry]
│       ├── text: "Question: What was total revenue?\nSQL: SELECT..."
│       ├── embedding: [1536 floats]
│       └── metadata: {question, sql, result_summary, session_id}
│
└── observations
    └── [observation entry]
        ├── text: "Electronics category shows 40% higher margins..."
        ├── embedding: [1536 floats]
        └── metadata: {type, topic, session_id, indexed_at}

Retrieval: question → embed → cosine search all 3 → rank by score → inject into prompt
```

## Other Features

- **Slash commands** with tab completion: `/model`, `/output`, `/session`, `/rag`, `/status`
- **Session management**: Token tracking, context window monitoring, save/load across restarts
- **Output modes**: Force query-only or observation-only responses via `/output`
- **Platform introspection**: Agent can view Airflow DAGs, dbt models, and Evidence dashboards
- **Persistent context**: Agent notes saved to `.astroagent/context.md`


Code flow:

CLI.py - REPL conversation, interacts with settings files, passes questions to orchestrator, and returns answers from submission tool calls.

Orchestrator.py - manages the thinking and submission agent actions, has these tool calls: 
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


agent/
  ├── cli.py              # Entry point - REPL loop, slash command handling
  ├── orchestrator.py     # Core agent loop - LLM calls, tool dispatch
  ├── settings.py         # SlashCommandRegistry, AgentSettings, output modes
  ├── session.py          # Session tracking - tokens, history, save/load
  ├── config.py           # API key storage (~/.astroagent/config.json)
  ├── schema.py           # DuckDB introspection - tables, columns, samples
  ├── context.py          # Persistent notes file (.astroagent/context.md)
  ├── display.py          # Formats submit_result/observation for terminal
  ├── theme.py            # Rich console styling, spinners, tool icons
  │
  ├── sandbox/            # Isolated code execution
  │   ├── sql_executor.py    # Runs SQL against DuckDB, returns DataFrame
  │   └── python_executor.py # Runs Python with DataFrames in restricted env
  │
  ├── tools/
  │   ├── internal/       # Results return to LLM for reasoning
  │   │   ├── run_sql.py         # Execute SQL, see results
  │   │   ├── run_python.py      # Test Python logic
  │   │   ├── inspect_schema.py  # View tables/columns/samples
  │   │   ├── inspect_platform.py# View DAGs, dbt models, dashboards
  │   │   └── context_tools.py   # Read/update persistent notes
  │   │
  │   └── output/         # Results go to user, end the loop
  │       ├── submit_result.py      # SQL + Python function → computed answer
  │       ├── submit_observation.py # Narrative/qualitative answer
  │       └── send_message.py       # Mid-conversation message to user
  │
  └── memory/             # RAG system
      ├── embedder.py     # OpenAI text-embedding-3-small
      ├── store.py        # ChromaDB storage, indexing methods
      └── retriever.py    # Query-time retrieval, formatting for prompt
