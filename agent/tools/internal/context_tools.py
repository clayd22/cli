"""
context_tools.py

Tools for the agent to read and update its persistent context file.
The context file stores notes about the data platform that help
the agent provide better answers without re-exploring.
"""

from ...context import read_context, write_context, context_exists


READ_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "read_context",
        "description": "Read the context file containing notes about this data platform. Check this first to see what you already know before exploring.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


UPDATE_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "update_context",
        "description": """Update the context file with new insights about the data platform.

Use this to record useful information you discover, such as:
- What key tables contain
- How tables relate to each other
- Common query patterns that work well
- Non-obvious facts (e.g., "each row is a line item, not an order")

IMPORTANT: Keep the context CONCISE. Update existing sections rather than appending duplicates. The goal is a quick reference, not a log.""",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["overview", "key_tables", "relationships", "common_patterns", "notes"],
                    "description": "Which section to update"
                },
                "content": {
                    "type": "string",
                    "description": "The new content for this section. Be concise - bullet points preferred."
                }
            },
            "required": ["section", "content"]
        }
    }
}


def tool_read_context() -> str:
    """Read and return the context file contents."""
    if not context_exists():
        return "No context file exists yet."

    content = read_context()
    if not content.strip():
        return "Context file is empty."

    return content


def tool_update_context(section: str, content: str) -> str:
    """
    Update a specific section of the context file.

    Args:
        section: Which section to update
        content: New content for that section

    Returns:
        Confirmation message
    """
    section_map = {
        "overview": "## Overview",
        "key_tables": "## Key Tables",
        "relationships": "## Relationships",
        "common_patterns": "## Common Patterns",
        "notes": "## Notes"
    }

    header = section_map.get(section)
    if not header:
        return f"Unknown section: {section}"

    current = read_context()

    # Find the section and replace its content
    lines = current.split('\n')
    new_lines = []
    in_target_section = False
    section_updated = False

    for i, line in enumerate(lines):
        if line.strip().startswith('## '):
            if line.strip() == header:
                # Start of target section
                in_target_section = True
                new_lines.append(line)
                new_lines.append(content)
                section_updated = True
                continue
            else:
                # Start of different section
                in_target_section = False

        if not in_target_section:
            new_lines.append(line)
        # Skip lines in target section (we replaced them)

    if not section_updated:
        # Section not found, append it
        new_lines.append(f"\n{header}")
        new_lines.append(content)

    write_context('\n'.join(new_lines))
    return f"Updated {section} section."
