from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.status import Status
from rich.syntax import Syntax
from contextlib import contextmanager

SPACE_THEME = Theme({
    "title": "bold bright_magenta",
    "subtitle": "dim cyan",
    "prompt": "bold cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "dim white",
    "highlight": "bold bright_white on dark_blue",
    "sql": "bright_yellow",
    "result": "bright_green",
    "thinking": "dim magenta italic",
    "tool_name": "bold bright_cyan",
    "tool_arg": "dim cyan",
})

console = Console(theme=SPACE_THEME)

LOGO = """
[bright_magenta]     _        _              _                    _
    / \\   ___| |_ _ __ ___  / \\   __ _  ___ _ __ | |_
   / _ \\ / __| __| '__/ _ \\/ _ \\ / _` |/ _ \\ '_ \\| __|
  / ___ \\\\__ \\ |_| | | (_) / ___ \\ (_| |  __/ | | | |_
 /_/   \\_\\___/\\__|_|  \\___/_/   \\_\\__, |\\___|_| |_|\\__|
                                  |___/                [/bright_magenta]
[dim cyan]        Mission Control for Your Data Universe[/dim cyan]
"""

STARS = "[dim white]·  ✦  ·  ·  ✦  ·  ✦  ·  ·  ✦  ·  ·  ✦  ·  ✦  ·[/dim white]"

# Tool indicators for visual feedback
TOOL_ICONS = {
    "run_sql": ">",
    "run_python": "#",
    "inspect_schema": "?",
    "read_context": "<",
    "update_context": "+",
    "submit_result": "*",
    "submit_observation": "~",
    "render_artifact": "@",
    "send_message": "-",
}


def print_logo():
    console.print(LOGO)
    console.print(STARS)
    console.print()


def print_welcome():
    print_logo()
    console.print("[info]Type your questions about the data, or 'exit' to quit.[/info]")
    console.print("[info]Use 'help' for available commands.[/info]")
    console.print()


def print_prompt():
    console.print("[prompt]mission>[/prompt] ", end="")


def print_thinking(message: str = "Analyzing transmission..."):
    console.print(f"[thinking]{message}[/thinking]")


def print_success(message: str):
    console.print(f"[success]{message}[/success]")


def print_error(message: str):
    console.print(f"[error]ERROR: {message}[/error]")


def print_warning(message: str):
    console.print(f"[warning]{message}[/warning]")


def print_sql(sql: str):
    console.print(Panel(sql, title="[sql]Generated SQL[/sql]", border_style="yellow"))


def print_result(result: str):
    console.print(Panel(result, title="[result]Result[/result]", border_style="green"))


def print_divider():
    console.print(STARS)


@contextmanager
def tool_status(tool_name: str, description: str = None):
    """
    Context manager for showing tool execution status with a spinner.

    Usage:
        with tool_status("run_sql", "Querying orders table..."):
            result = execute_sql(...)
    """
    icon = TOOL_ICONS.get(tool_name, "⚡")
    display_name = tool_name.replace("_", " ").title()

    status_msg = f"{icon} [tool_name]{display_name}[/tool_name]"
    if description:
        status_msg += f" [tool_arg]· {description}[/tool_arg]"

    with console.status(status_msg, spinner="dots"):
        yield


def print_tool_call(tool_name: str, args_summary: str = None):
    """Print a tool call with icon and optional args summary."""
    icon = TOOL_ICONS.get(tool_name, "⚡")
    display_name = tool_name.replace("_", " ").title()

    line = f"  {icon} [tool_name]{display_name}[/tool_name]"
    if args_summary:
        # Truncate long summaries
        if len(args_summary) > 60:
            args_summary = args_summary[:57] + "..."
        line += f" [tool_arg]· {args_summary}[/tool_arg]"

    console.print(line)


def print_tool_result_preview(tool_name: str, result: str, max_lines: int = 3):
    """Print a preview of tool results."""
    lines = result.split('\n')
    if len(lines) > max_lines:
        preview = '\n'.join(lines[:max_lines])
        preview += f"\n[dim]... ({len(lines) - max_lines} more lines)[/dim]"
    else:
        preview = result

    console.print(Panel(
        preview,
        title=f"[dim]{TOOL_ICONS.get(tool_name, '⚡')} Result[/dim]",
        border_style="dim",
        padding=(0, 1)
    ))
