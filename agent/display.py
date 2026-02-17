"""
display.py

Handles all output formatting and display for AstroAgent.

This module is responsible for presenting results to the user in a clear,
space-themed format. It takes the structured output from submit_result()
and renders it using Rich for terminal formatting.
"""

import pandas as pd
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from .theme import console


class ResultDisplay:
    """
    Handles formatting and display of agent results.

    This class provides methods to display different types of outputs
    from the agent, including submit_result outputs, DataFrames, and errors.
    """

    def __init__(self):
        self.console = console

    def show_submit_result(self, output) -> None:
        """
        Display the output from a submit_result() call.

        Shows:
        - Explanation of what was computed
        - SQL inputs used (with row counts)
        - The Python function applied
        - The final result (formatted based on type)

        Args:
            output: SubmitResultOutput instance from submit_result()
        """
        # Handle errors
        if not output.success:
            self._show_error(output)
            return

        # Show explanation
        self.console.print()
        self.console.print(f"[info]{output.explanation}[/info]")
        self.console.print()

        # Show what SQL inputs were used
        self._show_inputs_summary(output.inputs_used)

        # Show the function that was applied
        self._show_function_code(output.function_code)

        # Show the result
        self.console.print()
        self._show_result(output.result)

    def _show_error(self, output) -> None:
        """Display an error from a failed submit_result."""
        self.console.print(Panel(
            f"[error]{output.error}[/error]",
            title="Execution Failed",
            border_style="red"
        ))

        if output.function_code:
            self._show_function_code(output.function_code)

    def _show_inputs_summary(self, inputs_used: dict[str, pd.DataFrame]) -> None:
        """Show a summary of the SQL inputs that were used."""
        self.console.print("[title]SQL Inputs:[/title]")
        for name, df in inputs_used.items():
            self.console.print(f"  [prompt]{name}[/prompt]: {len(df):,} rows")
        self.console.print()

    def _show_function_code(self, code: str) -> None:
        """Display the Python function code with syntax highlighting."""
        self.console.print(Panel(
            Syntax(code, "python", theme="monokai"),
            title="[sql]Function Applied[/sql]",
            border_style="yellow"
        ))

    def _show_result(self, result) -> None:
        """
        Display the final result, formatted based on its type.

        Handles:
        - DataFrame: Rich table
        - Numbers: Formatted with commas/decimals
        - Dicts: Key-value table
        - Lists of dicts: DataFrame table
        - Other: String representation
        """
        if isinstance(result, pd.DataFrame):
            self._show_dataframe(result)
        elif isinstance(result, (int, float)):
            self._show_number(result)
        elif isinstance(result, dict):
            self._show_dict(result)
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            self._show_dataframe(pd.DataFrame(result))
        else:
            self._show_generic(result)

    def _show_dataframe(self, df: pd.DataFrame) -> None:
        """Render a DataFrame as a Rich table."""
        table = Table(show_header=True, header_style="bold cyan")

        for col in df.columns:
            table.add_column(str(col))

        for _, row in df.iterrows():
            table.add_row(*[self._format_value(v) for v in row])

        self.console.print(table)
        self.console.print(f"[info]{len(df):,} rows[/info]")

    def _show_number(self, value: int | float) -> None:
        """Display a numeric result in a panel."""
        if isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = f"{value:,.2f}"

        self.console.print(Panel(
            f"[result]{formatted}[/result]",
            title="[result]Result[/result]",
            border_style="green"
        ))

    def _show_dict(self, d: dict) -> None:
        """Display a dictionary as a key-value table."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Key")
        table.add_column("Value")

        for k, v in d.items():
            table.add_row(str(k), self._format_value(v))

        self.console.print(table)

    def _show_generic(self, result) -> None:
        """Display any other result type as a string."""
        self.console.print(Panel(
            str(result),
            title="[result]Result[/result]",
            border_style="green"
        ))

    def _format_value(self, value) -> str:
        """Format a single value for table display."""
        if value is None:
            return "[dim]null[/dim]"
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)


# Module-level instance for convenience
result_display = ResultDisplay()


def display_submit_result(output) -> None:
    """
    Convenience function to display a submit_result output.

    Args:
        output: SubmitResultOutput instance from submit_result()
    """
    result_display.show_submit_result(output)


def display_artifact(output) -> None:
    """
    Display a render_artifact output.

    Args:
        output: RenderArtifactOutput instance
    """
    console.print()

    if not output.success:
        console.print(Panel(
            f"[error]{output.error}[/error]",
            title="Render Failed",
            border_style="red"
        ))
        return

    console.print(f"[info]{output.explanation}[/info]")
    console.print()
    console.print(Panel(
        f"[result]{output.url}[/result]\n\n[dim]Opened in browser. Server running in background.[/dim]",
        title="[result]Visualization Running[/result]",
        border_style="green"
    ))


def display_observation(output) -> None:
    """
    Display a submit_observation output.

    Args:
        output: SubmitObservationOutput instance
    """
    console.print()
    console.print(Panel(
        output.observation,
        title="[result]Observation[/result]",
        border_style="green",
        padding=(1, 2)
    ))

    if output.supporting_queries:
        console.print()
        queries_text = ""
        for desc, sql in output.supporting_queries.items():
            queries_text += f"[title]{desc}:[/title]\n"
            queries_text += f"[dim]{sql}[/dim]\n\n"
        console.print(Panel(
            queries_text.strip(),
            title="[sql]Queries Used[/sql]",
            border_style="yellow"
        ))

    if output.supporting_data:
        console.print()
        console.print(Panel(
            output.supporting_data,
            title="[dim]Supporting Data[/dim]",
            border_style="dim"
        ))
