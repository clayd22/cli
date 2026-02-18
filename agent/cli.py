"""
cli.py


Provides:
- REPL mode for conversational data search
- Slash commands for settings (/model, /output, /status, /help)
"""

import sys
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from .config import get_api_key, set_api_key, clear_config
from .context import context_exists, create_context, get_context_path
from .settings import AgentSettings, SlashCommandRegistry
from .session import SessionManager
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


class SlashCompleter(Completer):
    """Autocomplete provider for slash commands."""

    def __init__(self, registry: SlashCommandRegistry):
        self.registry = registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        completions = self.registry.get_completions(text)

        for cmd, desc in completions:
            # Calculate how much to replace
            yield Completion(
                cmd,
                start_position=-len(text),
                display=cmd,
                display_meta=desc if desc else None,
            )


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """AstroAgent - Mission Control"""
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


def handle_command(
    cmd: str,
    orchestrator: Orchestrator,
    registry: SlashCommandRegistry
) -> bool:
    """
    Handle special REPL commands and slash commands.

    Args:
        cmd: The command string entered by user
        orchestrator: The current Orchestrator instance
        registry: The slash command registry

    Returns:
        True: Command handled, continue REPL
        False: Exit command, stop REPL
        None: Not a command, process as question
    """
    cmd_stripped = cmd.strip()
    cmd_lower = cmd_stripped.lower()

    # Handle slash commands
    if cmd_stripped.startswith("/"):
        success, message = registry.execute(cmd_stripped)
        if success:
            print_success(message)
        else:
            print_error(message)
        return True

    if cmd_lower in ("exit", "quit", "q"):
        print_divider()
        console.print("[info]Transmission ended. Safe travels, astronaut.[/info]")
        return False

    if cmd_lower == "help":
        console.print("""
[title]Available Commands[/title]
  [prompt]help[/prompt]      Show this message
  [prompt]clear[/prompt]     Clear conversation history
  [prompt]schema[/prompt]    Show database schema
  [prompt]context[/prompt]   Show current context file
  [prompt]exit[/prompt]      Exit AstroAgent

[title]Slash Commands[/title]  (type / to see completions)
  [prompt]/model[/prompt]    Change AI model
  [prompt]/output[/prompt]   Set output mode (auto, observation, query)
  [prompt]/session[/prompt]  Session management (new, save, load, list, clear)
  [prompt]/rag[/prompt]      RAG memory (index, stats, clear)
  [prompt]/status[/prompt]   Show current settings and session info
  [prompt]/help[/prompt]     Show slash command help
        """)
        return True

    if cmd_lower == "context":
        from .context import read_context, context_exists
        if not context_exists():
            console.print("[info]No context file exists.[/info]")
        else:
            content = read_context()
            console.print(content)
        return True

    if cmd_lower == "clear":
        orchestrator.clear_history()
        print_success("Conversation history cleared.")
        return True

    if cmd_lower == "schema":
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
    - Use slash commands for settings
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

    # Initialize settings and session manager
    settings = AgentSettings()
    session_manager = SessionManager(settings.model)

    # Initialize slash command registry with session manager
    registry = SlashCommandRegistry(settings, session_manager)

    # Initialize the orchestrator with settings and session manager
    try:
        orchestrator = Orchestrator(settings=settings, session_manager=session_manager)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    # --- RAG: Show stats only, don't block startup ---
    try:
        from .memory import MemoryStore
        store = MemoryStore()
        stats = store.get_stats()
        if stats['schema'] > 0:
            console.print(f"[dim]RAG: {stats['schema']} schema, {stats['queries']} queries, {stats['observations']} obs[/dim]")
        else:
            console.print("[dim]RAG: not indexed (run /rag index)[/dim]")
    except Exception:
        pass  # RAG init issues shouldn't block startup

    # Show session ID
    console.print(f"[dim]Session: {session_manager.current_session.id}[/dim]")
    console.print()

    # Set up prompt_toolkit with slash command completion
    prompt_style = Style.from_dict({
        "prompt": "#00d4aa bold",
    })

    session = PromptSession(
        completer=SlashCompleter(registry),
        style=prompt_style,
        complete_while_typing=True,
    )

    # Main REPL loop
    while True:
        try:
            # Use prompt_toolkit for input with completion
            user_input = session.prompt(
                HTML("<prompt>mission&gt;</prompt> "),
            ).strip()

            if not user_input:
                continue

            # Check for special commands
            command_result = handle_command(user_input, orchestrator, registry)
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
        except EOFError:
            # Handle Ctrl+D
            console.print()
            print_divider()
            console.print("[info]Transmission ended. Safe travels, astronaut.[/info]")
            break
        except Exception as e:
            print_error(str(e))


def main():
    """Entry point for the CLI."""
    cli()
