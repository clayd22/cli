"""
context.py

Manages the persistent context file where the agent stores
notes about the data platform for future reference.

The context file helps the agent:
- Remember insights from previous explorations
- Avoid re-discovering the same information
- Maintain concise, useful notes about the data
"""

from pathlib import Path

# Context file lives in the project root
CONTEXT_PATH = Path(__file__).parent.parent / ".astroagent" / "context.md"

DEFAULT_CONTEXT = """# Data Platform Context

## Overview
<!-- Brief description of this data platform -->

## Key Tables
<!-- Important tables and what they contain -->

## Relationships
<!-- How tables relate to each other -->

## Common Patterns
<!-- Useful query patterns discovered -->

## Notes
<!-- Other important observations -->
"""


def context_exists() -> bool:
    """Check if the context file exists."""
    return CONTEXT_PATH.exists()


def ensure_context_dir():
    """Create the .astroagent directory if needed."""
    CONTEXT_PATH.parent.mkdir(exist_ok=True)


def create_context() -> None:
    """Create a new context file with default template."""
    ensure_context_dir()
    CONTEXT_PATH.write_text(DEFAULT_CONTEXT)


def read_context() -> str:
    """
    Read the current context file.

    Returns:
        The context content, or empty string if not found.
    """
    if not CONTEXT_PATH.exists():
        return ""
    return CONTEXT_PATH.read_text()


def write_context(content: str) -> None:
    """
    Write content to the context file.

    Args:
        content: The full content to write
    """
    ensure_context_dir()
    CONTEXT_PATH.write_text(content)


def get_context_path() -> Path:
    """Get the path to the context file."""
    return CONTEXT_PATH
