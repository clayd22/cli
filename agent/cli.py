"""
cli.py

Command-line interface for AstroAgent.

Provides:
- Interactive REPL mode for conversational data exploration
- One-shot question mode via 'ask' command
- Configuration management for API keys
"""

import sys
import click

from .config import get_api_key, set_api_key, clear_config
from .context import context_exists, create_context, get_context_path
from .theme import (
    console,
    print_welcome,
    print_prompt,
    print_thinking,
    print_success,
    print_error,
    print_warning,
    print_divider,
)
from .orchestrator import Orchestrator


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """AstroAgent - Mission Control for Your Data Universe"""
    if ctx.invoked_subcommand is None:
        start_repl()


@cli.command()
@click.option("--key", prompt="Enter your OpenAI API key", hide_input=True)
def config(key: str):
    """Configure the OpenAI API key."""
    set_api_key(key)
    print_success("API key stored in ~/.astroagent/config.json")


@cli.command()
def reset():
    """Clear stored configuration."""
    clear_config()
    print_success("Configuration cleared.")


@cli.command()
@click.argument("question", nargs=-1)
def ask(question: tuple):
    """Ask a single question without entering the REPL."""
    if not question:
        print_error("Please provide a question.")
        return

    if not get_api_key():
        print_error("No API key configured. Run 'astro config' first.")
        return

    orchestrator = Orchestrator()
    query = " ".join(question)
    orchestrator.process_question(query)


def handle_command(cmd: str, orchestrator: Orchestrator) -> bool:
    """
    Handle special REPL commands.

    Args:
        cmd: The command string entered by user
        orchestrator: The current Orchestrator instance

    Returns:
        True: Command handled, continue REPL
        False: Exit command, stop REPL
        None: Not a command, process as question
    """
    cmd = cmd.strip().lower()

    if cmd in ("exit", "quit", "q"):
        print_divider()
        console.print("[info]Transmission ended. Safe travels, astronaut.[/info]")
        return False

    if cmd == "help":
        console.print("""
[title]Available Commands[/title]
  [prompt]help[/prompt]      Show this message
  [prompt]clear[/prompt]     Clear conversation history
  [prompt]schema[/prompt]    Show database schema
  [prompt]context[/prompt]   Show current context file
  [prompt]exit[/prompt]      Exit AstroAgent
        """)
        return True

    if cmd == "context":
        from .context import read_context, context_exists
        if not context_exists():
            console.print("[info]No context file exists.[/info]")
        else:
            content = read_context()
            console.print(content)
        return True

    if cmd == "clear":
        orchestrator.clear_history()
        print_success("Conversation history cleared.")
        return True

    if cmd == "schema":
        from .schema import get_full_schema_context
        console.print(get_full_schema_context())
        return True

    return None  # Not a command, treat as question


def check_context_file():
    """
    Check if context file exists, offer to create if not.

    Returns:
        True if context exists or was created, False if user declined
    """
    if context_exists():
        return True

    console.print("[info]No context file found for this data platform.[/info]")
    console.print("[info]The context file helps me remember insights about your data.[/info]")
    console.print()

    response = console.input("[prompt]Create context file? (y/n)>[/prompt] ").strip().lower()

    if response in ("y", "yes"):
        create_context()
        print_success(f"Context file created at {get_context_path()}")
        return True
    else:
        console.print("[info]Continuing without context file.[/info]")
        return False


def start_repl():
    """
    Start the interactive REPL (Read-Eval-Print Loop).

    This is the main interaction mode where users can:
    - Ask questions about their data
    - Explore the schema
    - Have multi-turn conversations
    """
    # Ensure API key is configured
    if not get_api_key():
        print_warning("No API key configured.")
        console.print("[info]Run 'astro config' or enter your OpenAI API key now:[/info]")
        key = console.input("[prompt]API Key> [/prompt]").strip()
        if key:
            set_api_key(key)
            print_success("API key saved.")
        else:
            print_error("No API key provided. Exiting.")
            sys.exit(1)

    # Check for context file
    check_context_file()
    console.print()

    print_welcome()

    # Initialize the orchestrator
    try:
        orchestrator = Orchestrator()
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    # Main REPL loop
    while True:
        try:
            print_prompt()
            user_input = input().strip()

            if not user_input:
                continue

            # Check for special commands
            command_result = handle_command(user_input, orchestrator)
            if command_result is False:
                break
            if command_result is True:
                continue

            # Process as a data question
            orchestrator.process_question(user_input)

        except KeyboardInterrupt:
            console.print()
            print_divider()
            console.print("[info]Transmission interrupted. Safe travels, astronaut.[/info]")
            break
        except Exception as e:
            print_error(str(e))


def main():
    """Entry point for the CLI."""
    cli()
