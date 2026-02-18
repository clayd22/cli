"""
session.py

Session management for AstroAgent conversations.

Handles:
- Session lifecycle (start, end, pause, resume)
- Token usage tracking
- Context window management
- Session persistence to disk
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


SESSIONS_DIR = Path.home() / ".astroagent" / "sessions"

# Context window limits by model (approximate)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-preview": 128000,
}

DEFAULT_CONTEXT_LIMIT = 128000


@dataclass
class TokenUsage:
    """Tracks token consumption for a session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, prompt: int, completion: int):
        """Add tokens from an API response."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsage":
        return cls(**data)


@dataclass
class Session:
    """
    Represents a conversation session with the LLM.

    Attributes:
        id: Unique session identifier
        name: Optional human-readable name
        model: The model being used
        created_at: Session start time
        updated_at: Last activity time
        status: active, paused, or ended
        token_usage: Cumulative token consumption
        context_limit: Max tokens for the model
        conversation_history: List of messages
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: Optional[str] = None
    model: str = "gpt-4o"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    conversation_history: list = field(default_factory=list)
    message_count: int = 0

    def __post_init__(self):
        if isinstance(self.token_usage, dict):
            self.token_usage = TokenUsage.from_dict(self.token_usage)

    def update_tokens(self, prompt: int, completion: int):
        """Update token usage from API response."""
        self.token_usage.update(prompt, completion)
        self.updated_at = datetime.now().isoformat()

    def add_message(self, message: dict):
        """Add a message to history."""
        self.conversation_history.append(message)
        self.message_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        self.updated_at = datetime.now().isoformat()

    def clear_history(self):
        """Clear conversation history but keep session metadata."""
        self.conversation_history = []
        self.message_count = 0
        self.updated_at = datetime.now().isoformat()

    def get_context_usage_percent(self) -> float:
        """Get percentage of context window used."""
        if self.context_limit == 0:
            return 0.0
        return (self.token_usage.total_tokens / self.context_limit) * 100

    def is_context_warning(self, threshold: float = 80.0) -> bool:
        """Check if context usage exceeds warning threshold."""
        return self.get_context_usage_percent() >= threshold

    def to_dict(self) -> dict:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "token_usage": self.token_usage.to_dict(),
            "context_limit": self.context_limit,
            "conversation_history": self.conversation_history,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Deserialize session from dictionary."""
        return cls(**data)


class SessionManager:
    """
    Manages session lifecycle and persistence.

    Provides methods to:
    - Start new sessions
    - Save/load sessions to disk
    - Track token usage
    - Monitor context window usage
    """

    def __init__(self, model: str = "gpt-4o"):
        self.current_session: Optional[Session] = None
        self._ensure_sessions_dir()
        self.start_session(model)

    def _ensure_sessions_dir(self):
        """Create sessions directory if it doesn't exist."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def start_session(self, model: str = "gpt-4o") -> Session:
        """
        Start a new session.

        Args:
            model: The model to use for this session

        Returns:
            The new Session instance
        """
        context_limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
        self.current_session = Session(
            model=model,
            context_limit=context_limit,
        )
        return self.current_session

    def end_session(self) -> Optional[dict]:
        """
        End the current session.

        Returns:
            Summary dict of the ended session, or None if no session
        """
        if not self.current_session:
            return None

        self.current_session.status = "ended"
        self.current_session.updated_at = datetime.now().isoformat()

        summary = {
            "id": self.current_session.id,
            "duration": self._calculate_duration(),
            "messages": self.current_session.message_count,
            "tokens": self.current_session.token_usage.total_tokens,
        }

        self.current_session = None
        return summary

    def _calculate_duration(self) -> str:
        """Calculate session duration as human-readable string."""
        if not self.current_session:
            return "0s"

        start = datetime.fromisoformat(self.current_session.created_at)
        end = datetime.now()
        delta = end - start

        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def save_session(self, name: Optional[str] = None) -> Path:
        """
        Save current session to disk.

        Args:
            name: Optional name for the session

        Returns:
            Path to the saved session file
        """
        if not self.current_session:
            raise ValueError("No active session to save")

        if name:
            self.current_session.name = name

        self.current_session.status = "paused"
        self.current_session.updated_at = datetime.now().isoformat()

        filename = f"{self.current_session.id}.json"
        if name:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            filename = f"{self.current_session.id}_{safe_name}.json"

        filepath = SESSIONS_DIR / filename
        filepath.write_text(json.dumps(self.current_session.to_dict(), indent=2))

        return filepath

    def load_session(self, identifier: str) -> Session:
        """
        Load a session from disk.

        Args:
            identifier: Session ID or name to load

        Returns:
            The loaded Session instance
        """
        # Find matching session file
        matching_files = list(SESSIONS_DIR.glob(f"{identifier}*.json"))

        if not matching_files:
            # Try searching by name in filename
            matching_files = list(SESSIONS_DIR.glob(f"*_{identifier}*.json"))

        if not matching_files:
            raise ValueError(f"No session found matching: {identifier}")

        if len(matching_files) > 1:
            # Use most recently modified
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        filepath = matching_files[0]
        data = json.loads(filepath.read_text())

        self.current_session = Session.from_dict(data)
        self.current_session.status = "active"
        self.current_session.updated_at = datetime.now().isoformat()

        return self.current_session

    def list_sessions(self) -> list[dict]:
        """
        List all saved sessions.

        Returns:
            List of session summary dicts
        """
        sessions = []

        for filepath in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                sessions.append({
                    "id": data.get("id", "?"),
                    "name": data.get("name", "-"),
                    "model": data.get("model", "?"),
                    "messages": data.get("message_count", 0),
                    "tokens": data.get("token_usage", {}).get("total_tokens", 0),
                    "updated": data.get("updated_at", "?")[:16],
                    "status": data.get("status", "?"),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated time, most recent first
        sessions.sort(key=lambda s: s["updated"], reverse=True)
        return sessions

    def get_status(self) -> dict:
        """
        Get current session status.

        Returns:
            Dict with session status information
        """
        if not self.current_session:
            return {"active": False}

        return {
            "active": True,
            "id": self.current_session.id,
            "name": self.current_session.name,
            "model": self.current_session.model,
            "messages": self.current_session.message_count,
            "tokens": {
                "prompt": self.current_session.token_usage.prompt_tokens,
                "completion": self.current_session.token_usage.completion_tokens,
                "total": self.current_session.token_usage.total_tokens,
            },
            "context": {
                "limit": self.current_session.context_limit,
                "used_percent": round(self.current_session.get_context_usage_percent(), 1),
                "warning": self.current_session.is_context_warning(),
            },
            "duration": self._calculate_duration(),
        }

    def clear_history(self):
        """Clear conversation history but keep session."""
        if self.current_session:
            self.current_session.clear_history()

    def update_tokens(self, prompt: int, completion: int):
        """Update token usage from API response."""
        if self.current_session:
            self.current_session.update_tokens(prompt, completion)

    def add_message(self, message: dict):
        """Add a message to current session history."""
        if self.current_session:
            self.current_session.add_message(message)

    def get_history(self) -> list:
        """Get conversation history from current session."""
        if self.current_session:
            return self.current_session.conversation_history
        return []

    def set_history(self, history: list):
        """Set conversation history for current session."""
        if self.current_session:
            self.current_session.conversation_history = history
            self.current_session.message_count = len([m for m in history if m.get("role") == "user"])
