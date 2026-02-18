"""
inspect_platform.py

Tool for inspecting the data platform structure beyond the warehouse.
Provides access to Airflow DAGs, dbt models, and Evidence dashboards.
"""

import re
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
AIRFLOW_DAGS_DIR = PROJECT_ROOT / "airflow" / "dags"
DBT_MODELS_DIR = PROJECT_ROOT / "dbt_project" / "models"
EVIDENCE_PAGES_DIR = PROJECT_ROOT / "evidence" / "pages"


INSPECT_PLATFORM_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_platform",
        "description": """Inspect the data platform structure: Airflow DAGs, dbt models, Evidence dashboards.

Use this to understand:
- How data pipelines work (DAGs)
- How tables are built (dbt models)
- What dashboards/metrics exist (Evidence)

Actions:
- list_dags: List all Airflow DAGs with schedules
- show_dag: Show a specific DAG's code and structure
- list_models: List all dbt models by layer
- show_model: Show a dbt model's SQL transformation
- list_dashboards: List Evidence dashboard pages
- show_dashboard: Show dashboard queries and visualizations""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_dags", "show_dag", "list_models", "show_model", "list_dashboards", "show_dashboard"],
                    "description": "What to inspect"
                },
                "name": {
                    "type": "string",
                    "description": "Name of the DAG, model, or dashboard (for show_* actions)"
                }
            },
            "required": ["action"]
        }
    }
}


def inspect_platform(action: str, name: Optional[str] = None) -> str:
    """
    Inspect data platform components.

    Args:
        action: What to inspect (list_dags, show_dag, etc.)
        name: Name of specific item for show_* actions

    Returns:
        Formatted string with requested information
    """
    if action == "list_dags":
        return _list_dags()
    elif action == "show_dag":
        if not name:
            return "Error: 'name' required for show_dag"
        return _show_dag(name)
    elif action == "list_models":
        return _list_models()
    elif action == "show_model":
        if not name:
            return "Error: 'name' required for show_model"
        return _show_model(name)
    elif action == "list_dashboards":
        return _list_dashboards()
    elif action == "show_dashboard":
        if not name:
            return "Error: 'name' required for show_dashboard"
        return _show_dashboard(name)
    else:
        return f"Error: Unknown action '{action}'"


# --- Airflow DAG inspection ---

def _list_dags() -> str:
    """List all Airflow DAGs with metadata."""
    if not AIRFLOW_DAGS_DIR.exists():
        return "Error: Airflow dags directory not found"

    dags = []
    for dag_file in AIRFLOW_DAGS_DIR.glob("*.py"):
        if dag_file.name.startswith("__"):
            continue

        content = dag_file.read_text()
        dag_info = _parse_dag_metadata(dag_file.name, content)
        if dag_info:
            dags.append(dag_info)

    if not dags:
        return "No DAGs found"

    lines = ["# Airflow DAGs", ""]
    for dag in sorted(dags, key=lambda x: x["dag_id"]):
        schedule = dag.get("schedule", "None")
        tags = ", ".join(dag.get("tags", []))
        lines.append(f"**{dag['dag_id']}**")
        lines.append(f"  Schedule: {schedule}")
        if tags:
            lines.append(f"  Tags: {tags}")
        if dag.get("description"):
            lines.append(f"  Description: {dag['description']}")
        lines.append("")

    return "\n".join(lines)


def _parse_dag_metadata(filename: str, content: str) -> Optional[dict]:
    """Extract DAG metadata from Python file."""
    # Look for @dag decorator
    dag_match = re.search(
        r'@dag\s*\(\s*\n?(.*?)\)',
        content,
        re.DOTALL
    )

    if not dag_match:
        return None

    decorator_content = dag_match.group(1)

    # Extract dag_id
    dag_id_match = re.search(r'dag_id\s*=\s*["\']([^"\']+)["\']', decorator_content)
    dag_id = dag_id_match.group(1) if dag_id_match else filename.replace(".py", "")

    # Extract schedule
    schedule_match = re.search(r'schedule\s*=\s*["\']([^"\']+)["\']', decorator_content)
    schedule = schedule_match.group(1) if schedule_match else None

    # Extract tags
    tags_match = re.search(r'tags\s*=\s*\[(.*?)\]', decorator_content)
    tags = []
    if tags_match:
        tags = re.findall(r'["\']([^"\']+)["\']', tags_match.group(1))

    # Extract docstring
    func_match = re.search(r'def\s+\w+\s*\(\s*\)\s*:\s*\n\s*"""([^"]*?)"""', content)
    description = func_match.group(1).strip() if func_match else None

    return {
        "dag_id": dag_id,
        "schedule": schedule,
        "tags": tags,
        "description": description,
        "filename": filename
    }


def _show_dag(name: str) -> str:
    """Show details of a specific DAG."""
    # Find the DAG file
    dag_file = None
    for f in AIRFLOW_DAGS_DIR.glob("*.py"):
        if name in f.name or name in f.read_text():
            dag_file = f
            break

    if not dag_file:
        return f"Error: DAG '{name}' not found"

    content = dag_file.read_text()

    # Extract task definitions
    tasks = re.findall(r'@task\s*\(\s*\)\s*\n\s*def\s+(\w+)', content)

    lines = [f"# DAG: {name}", f"File: {dag_file.name}", ""]

    if tasks:
        lines.append("## Tasks")
        for task in tasks:
            lines.append(f"  - {task}")
        lines.append("")

    lines.append("## Source Code")
    lines.append("```python")
    lines.append(content)
    lines.append("```")

    return "\n".join(lines)


# --- dbt model inspection ---

def _list_models() -> str:
    """List all dbt models organized by layer."""
    if not DBT_MODELS_DIR.exists():
        return "Error: dbt models directory not found"

    lines = ["# dbt Models", ""]

    # Get staging models
    staging_dir = DBT_MODELS_DIR / "staging"
    if staging_dir.exists():
        lines.append("## Staging Layer")
        lines.append("Views that clean raw data:")
        for model in sorted(staging_dir.glob("*.sql")):
            if not model.name.startswith("_"):
                # Extract brief description from first comment
                content = model.read_text()
                desc = _extract_sql_comment(content)
                lines.append(f"  - **{model.stem}**" + (f": {desc}" if desc else ""))
        lines.append("")

    # Get marts models
    marts_dir = DBT_MODELS_DIR / "marts"
    if marts_dir.exists():
        lines.append("## Marts Layer")
        lines.append("Analytics-ready tables:")
        for model in sorted(marts_dir.glob("*.sql")):
            if not model.name.startswith("_"):
                content = model.read_text()
                desc = _extract_sql_comment(content)
                # Also extract refs to show dependencies
                refs = re.findall(r"{{\s*ref\s*\(\s*['\"](\w+)['\"]\s*\)\s*}}", content)
                lines.append(f"  - **{model.stem}**" + (f": {desc}" if desc else ""))
                if refs:
                    lines.append(f"    Dependencies: {', '.join(refs)}")
        lines.append("")

    return "\n".join(lines)


def _extract_sql_comment(content: str) -> Optional[str]:
    """Extract first line comment from SQL."""
    match = re.match(r'^--\s*(.+)$', content.strip(), re.MULTILINE)
    return match.group(1).strip() if match else None


def _show_model(name: str) -> str:
    """Show a dbt model's SQL transformation."""
    # Search in staging and marts
    model_file = None
    for layer in ["staging", "marts"]:
        candidate = DBT_MODELS_DIR / layer / f"{name}.sql"
        if candidate.exists():
            model_file = candidate
            break
        # Also try without stg_ prefix
        if layer == "staging" and not name.startswith("stg_"):
            candidate = DBT_MODELS_DIR / layer / f"stg_{name}.sql"
            if candidate.exists():
                model_file = candidate
                break

    if not model_file:
        return f"Error: Model '{name}' not found in staging or marts"

    content = model_file.read_text()

    # Extract refs for lineage
    refs = re.findall(r"{{\s*ref\s*\(\s*['\"](\w+)['\"]\s*\)\s*}}", content)

    lines = [f"# dbt Model: {model_file.stem}"]
    lines.append(f"Layer: {model_file.parent.name}")

    if refs:
        lines.append(f"Dependencies: {', '.join(refs)}")

    lines.append("")
    lines.append("## SQL Transformation")
    lines.append("```sql")
    lines.append(content)
    lines.append("```")

    return "\n".join(lines)


# --- Evidence dashboard inspection ---

def _list_dashboards() -> str:
    """List all Evidence dashboard pages."""
    if not EVIDENCE_PAGES_DIR.exists():
        return "Error: Evidence pages directory not found"

    lines = ["# Evidence Dashboards", ""]

    for page in sorted(EVIDENCE_PAGES_DIR.glob("*.md")):
        content = page.read_text()

        # Extract title from frontmatter
        title_match = re.search(r'^---\s*\ntitle:\s*(.+?)\n', content)
        title = title_match.group(1) if title_match else page.stem.replace("_", " ").title()

        # Count queries and components
        queries = re.findall(r'```sql\s+(\w+)', content)
        components = re.findall(r'<(\w+Chart|BigValue|DataTable)', content)

        lines.append(f"**/{page.stem}** - {title}")
        lines.append(f"  Queries: {len(queries)} | Components: {len(components)}")
        if queries:
            lines.append(f"  Query names: {', '.join(queries[:5])}" + ("..." if len(queries) > 5 else ""))
        lines.append("")

    return "\n".join(lines)


def _show_dashboard(name: str) -> str:
    """Show dashboard queries and visualizations."""
    # Find the page
    page_file = EVIDENCE_PAGES_DIR / f"{name}.md"
    if not page_file.exists():
        # Try with/without extension
        page_file = EVIDENCE_PAGES_DIR / f"{name.replace('/', '')}.md"
    if not page_file.exists():
        return f"Error: Dashboard '{name}' not found"

    content = page_file.read_text()

    # Extract title
    title_match = re.search(r'^---\s*\ntitle:\s*(.+?)\n', content)
    title = title_match.group(1) if title_match else name

    lines = [f"# Dashboard: {title}", f"Path: /{page_file.stem}", ""]

    # Extract SQL queries
    queries = re.findall(r'```sql\s+(\w+)\n(.*?)```', content, re.DOTALL)
    if queries:
        lines.append("## Queries Defined")
        for query_name, query_sql in queries:
            lines.append(f"### {query_name}")
            lines.append("```sql")
            lines.append(query_sql.strip())
            lines.append("```")
            lines.append("")

    # Extract components
    components = re.findall(r'<((?:Line|Bar|Area|Pie)Chart|BigValue|DataTable)\s*(.*?)/>', content, re.DOTALL)
    if components:
        lines.append("## Visualizations")
        for comp_type, attrs in components:
            # Extract key attributes
            data_match = re.search(r'data=\{(\w+)\}', attrs)
            title_match = re.search(r'title="([^"]+)"', attrs)

            comp_desc = comp_type
            if title_match:
                comp_desc += f': "{title_match.group(1)}"'
            if data_match:
                comp_desc += f" (data: {data_match.group(1)})"

            lines.append(f"  - {comp_desc}")

    return "\n".join(lines)
