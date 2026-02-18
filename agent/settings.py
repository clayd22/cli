"""
Agent settings and slash command system.
Manages model selection and output mode constraints.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


class OutputMode(Enum):
    AUTO = "auto"
    OBSERVATION = "observation"
    QUERY = "query"


AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
]

DEFAULT_MODEL = "gpt-4o"


@dataclass
class AgentSettings:
    """Current agent settings state."""

    model: str = DEFAULT_MODEL
    output_mode: OutputMode = OutputMode.AUTO
    rag_verbose: bool = False
    verbose: bool = False  # Print full prompts sent to LLM

    def get_allowed_output_tools(self) -> list[str]:
        """Return list of allowed output tools based on current mode."""
        if self.output_mode == OutputMode.AUTO:
            return ["submit_result", "submit_observation"]
        elif self.output_mode == OutputMode.OBSERVATION:
            return ["submit_observation"]
        elif self.output_mode == OutputMode.QUERY:
            return ["submit_result"]
        return []

    def get_mode_instruction(self) -> str:
        """Return instruction to add to system prompt based on output mode."""
        if self.output_mode == OutputMode.AUTO:
            return ""
        elif self.output_mode == OutputMode.OBSERVATION:
            return "\n\n**OUTPUT MODE: OBSERVATION ONLY** - You MUST use submit_observation for your final answer."
        elif self.output_mode == OutputMode.QUERY:
            return "\n\n**OUTPUT MODE: QUERY ONLY** - You MUST use submit_result for your final answer."
        return ""


@dataclass
class SlashCommand:
    """Definition of a slash command."""

    name: str
    description: str
    subcommands: list[str] = field(default_factory=list)
    handler: Optional[Callable] = None


class SlashCommandRegistry:
    """Registry and handler for slash commands."""

    def __init__(self, settings: AgentSettings, session_manager=None):
        self.settings = settings
        self.session_manager = session_manager
        self.commands: dict[str, SlashCommand] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        """Register built-in slash commands."""

        self.commands["model"] = SlashCommand(
            name="model",
            description="Change the AI model",
            subcommands=AVAILABLE_MODELS,
        )

        self.commands["output"] = SlashCommand(
            name="output",
            description="Set output mode constraint",
            subcommands=["auto", "observation", "query"],
        )

        self.commands["session"] = SlashCommand(
            name="session",
            description="Session management",
            subcommands=["new", "save", "load", "list", "clear"],
        )

        self.commands["rag"] = SlashCommand(
            name="rag",
            description="RAG memory management",
            subcommands=["index", "stats", "clear", "test", "verbose"],
        )

        self.commands["status"] = SlashCommand(
            name="status",
            description="Show current settings and session",
            subcommands=[],
        )

        self.commands["help"] = SlashCommand(
            name="help",
            description="Show available commands",
            subcommands=[],
        )

        self.commands["verbose"] = SlashCommand(
            name="verbose",
            description="Toggle verbose mode (show full prompts)",
            subcommands=[],
        )

    def get_completions(self, partial: str) -> list[tuple[str, str]]:
        """
        Get command completions for partial input.
        Returns list of (command, description) tuples.
        """
        if not partial.startswith("/"):
            return []

        partial = partial[1:]  # Remove leading /
        parts = partial.split(" ", 1)
        cmd_name = parts[0]

        # If we have a space, we're completing subcommands
        if len(parts) > 1:
            sub_partial = parts[1]
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                return [
                    (f"/{cmd_name} {sub}", "")
                    for sub in cmd.subcommands
                    if sub.lower().startswith(sub_partial.lower())
                ]
            return []

        # Otherwise complete command names
        return [
            (f"/{name}", cmd.description)
            for name, cmd in self.commands.items()
            if name.startswith(cmd_name.lower())
        ]

    def execute(self, command_str: str) -> tuple[bool, str]:
        """
        Execute a slash command.
        Returns (success, message).
        """
        if not command_str.startswith("/"):
            return False, "Not a slash command"

        parts = command_str[1:].split(" ", 1)
        cmd_name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd_name == "model":
            return self._handle_model(arg)
        elif cmd_name == "output":
            return self._handle_output(arg)
        elif cmd_name == "session":
            return self._handle_session(arg)
        elif cmd_name == "rag":
            return self._handle_rag(arg)
        elif cmd_name == "status":
            return self._handle_status()
        elif cmd_name == "help":
            return self._handle_help()
        elif cmd_name == "verbose":
            return self._handle_verbose()
        else:
            return False, f"Unknown command: /{cmd_name}"

    def _handle_model(self, arg: Optional[str]) -> tuple[bool, str]:
        """Handle /model command."""
        if not arg:
            models = "\n".join(f"  {m}{'  [current]' if m == self.settings.model else ''}"
                              for m in AVAILABLE_MODELS)
            return True, f"Available models:\n{models}\n\nUsage: /model <model-name>"

        if arg not in AVAILABLE_MODELS:
            return False, f"Unknown model: {arg}\nAvailable: {', '.join(AVAILABLE_MODELS)}"

        self.settings.model = arg

        # Update session context limit for new model
        if self.session_manager and self.session_manager.current_session:
            from .session import MODEL_CONTEXT_LIMITS, DEFAULT_CONTEXT_LIMIT
            self.session_manager.current_session.model = arg
            self.session_manager.current_session.context_limit = MODEL_CONTEXT_LIMITS.get(arg, DEFAULT_CONTEXT_LIMIT)

        return True, f"Model set to: {arg}"

    def _handle_output(self, arg: Optional[str]) -> tuple[bool, str]:
        """Handle /output command."""
        modes = {
            "auto": OutputMode.AUTO,
            "observation": OutputMode.OBSERVATION,
            "query": OutputMode.QUERY,
        }

        if not arg:
            current = self.settings.output_mode.value
            mode_list = "\n".join(f"  {m}{'  [current]' if m == current else ''}"
                                  for m in modes.keys())
            return True, f"Output modes:\n{mode_list}\n\nUsage: /output <mode>"

        arg_lower = arg.lower()
        if arg_lower not in modes:
            return False, f"Unknown mode: {arg}\nAvailable: {', '.join(modes.keys())}"

        self.settings.output_mode = modes[arg_lower]

        descriptions = {
            "auto": "Agent chooses best output tool",
            "observation": "Agent will only use submit_observation",
            "query": "Agent will only use submit_result",
        }

        return True, f"Output mode: {arg_lower} - {descriptions[arg_lower]}"

    def _handle_session(self, arg: Optional[str]) -> tuple[bool, str]:
        """Handle /session command."""
        if not self.session_manager:
            return False, "Session manager not available"

        if not arg:
            # Show session status
            status = self.session_manager.get_status()
            if not status.get("active"):
                return True, "No active session"

            ctx = status["context"]
            tokens = status["tokens"]
            warning = " [!]" if ctx["warning"] else ""

            return True, (
                f"Session: {status['id']}"
                + (f" ({status['name']})" if status['name'] else "")
                + f"\n  Model: {status['model']}"
                + f"\n  Messages: {status['messages']}"
                + f"\n  Tokens: {tokens['total']:,} ({tokens['prompt']:,} prompt, {tokens['completion']:,} completion)"
                + f"\n  Context: {ctx['used_percent']}% of {ctx['limit']:,}{warning}"
                + f"\n  Duration: {status['duration']}"
            )

        parts = arg.split(" ", 1)
        subcmd = parts[0].lower()
        subarg = parts[1] if len(parts) > 1 else None

        if subcmd == "new":
            self.session_manager.start_session(self.settings.model)
            return True, f"New session started: {self.session_manager.current_session.id}"

        elif subcmd == "save":
            try:
                path = self.session_manager.save_session(subarg)
                return True, f"Session saved: {path.name}"
            except ValueError as e:
                return False, str(e)

        elif subcmd == "load":
            if not subarg:
                return False, "Usage: /session load <id-or-name>"
            try:
                session = self.session_manager.load_session(subarg)
                return True, f"Session loaded: {session.id}" + (f" ({session.name})" if session.name else "")
            except ValueError as e:
                return False, str(e)

        elif subcmd == "list":
            sessions = self.session_manager.list_sessions()
            if not sessions:
                return True, "No saved sessions"

            lines = ["Saved Sessions:"]
            for s in sessions[:10]:  # Limit to 10
                name_part = f" ({s['name']})" if s['name'] and s['name'] != '-' else ""
                lines.append(f"  {s['id']}{name_part} - {s['messages']} msgs, {s['tokens']:,} tokens [{s['status']}]")
            return True, "\n".join(lines)

        elif subcmd == "clear":
            self.session_manager.clear_history()
            return True, "Session history cleared"

        else:
            return False, f"Unknown session command: {subcmd}\nAvailable: new, save, load, list, clear"

    def _handle_rag(self, arg: Optional[str]) -> tuple[bool, str]:
        """Handle /rag command."""
        try:
            from .memory import MemoryStore, ContextRetriever
            store = MemoryStore()
        except Exception as e:
            return False, f"RAG init failed: {e}"

        if not arg:
            # Show stats + verbose status
            stats = store.get_stats()
            verbose_status = "on" if self.settings.rag_verbose else "off"
            return True, (
                f"RAG Memory:\n"
                f"  Schema items: {stats['schema']}\n"
                f"  Query history: {stats['queries']}\n"
                f"  Observations: {stats['observations']}\n"
                f"  Verbose mode: {verbose_status}\n\n"
                f"Commands: /rag index, /rag test <question>, /rag verbose, /rag clear"
            )

        # --- Parse subcommand and argument ---
        parts = arg.split(" ", 1)
        subcmd = parts[0].lower()
        subarg = parts[1] if len(parts) > 1 else None

        if subcmd == "index":
            if store.is_schema_indexed():
                return True, "Schema already indexed. Use /rag clear first to re-index."
            try:
                count = store.index_schema_from_db()
                return True, f"Indexed {count} tables"
            except Exception as e:
                return False, f"Indexing failed: {e}"

        elif subcmd == "stats":
            stats = store.get_stats()
            return True, (
                f"RAG Memory:\n"
                f"  Schema items: {stats['schema']}\n"
                f"  Query history: {stats['queries']}\n"
                f"  Observations: {stats['observations']}"
            )

        elif subcmd == "clear":
            store.clear_collection("schema")
            store.clear_collection("queries")
            store.clear_collection("observations")
            return True, "RAG memory cleared"

        elif subcmd == "verbose":
            # Toggle or set verbose mode
            if subarg:
                self.settings.rag_verbose = subarg.lower() in ("on", "true", "1", "yes")
            else:
                self.settings.rag_verbose = not self.settings.rag_verbose
            status = "on" if self.settings.rag_verbose else "off"
            return True, f"RAG verbose mode: {status}"

        elif subcmd == "test":
            # Test retrieval for a question
            if not subarg:
                return False, "Usage: /rag test <question>"
            try:
                retriever = ContextRetriever(store)
                result = retriever.retrieve_with_scores(subarg)
                debug_output = retriever.format_debug(result)
                return True, debug_output
            except Exception as e:
                return False, f"Retrieval failed: {e}"

        else:
            return False, f"Unknown rag command: {subcmd}\nAvailable: index, stats, test, verbose, clear"

    def _handle_status(self) -> tuple[bool, str]:
        """Handle /status command."""
        lines = [
            "Settings:",
            f"  Model: {self.settings.model}",
            f"  Output Mode: {self.settings.output_mode.value}",
        ]

        if self.session_manager:
            status = self.session_manager.get_status()
            if status.get("active"):
                ctx = status["context"]
                warning = " [!]" if ctx["warning"] else ""
                lines.extend([
                    "",
                    "Session:",
                    f"  ID: {status['id']}" + (f" ({status['name']})" if status['name'] else ""),
                    f"  Messages: {status['messages']}",
                    f"  Tokens: {status['tokens']['total']:,}",
                    f"  Context: {ctx['used_percent']}% used{warning}",
                ])

        return True, "\n".join(lines)

    def _handle_verbose(self) -> tuple[bool, str]:
        """Handle /verbose command."""
        self.settings.verbose = not self.settings.verbose
        status = "on" if self.settings.verbose else "off"
        return True, f"Verbose mode: {status}"

    def _handle_help(self) -> tuple[bool, str]:
        """Handle /help command."""
        lines = ["Available Commands:"]
        for name, cmd in self.commands.items():
            lines.append(f"  /{name} - {cmd.description}")
        return True, "\n".join(lines)

    def is_slash_command(self, text: str) -> bool:
        """Check if text is a slash command."""
        return text.strip().startswith("/")
