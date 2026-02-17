"""
send_message.py

Tool for the agent to communicate with the user through the CLI.
Use this to explain reasoning, ask clarifying questions, or provide context.
"""

from ...theme import console


SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to the user. Use this to explain your reasoning, ask clarifying questions, or provide context about what you're doing.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to display to the user."
                }
            },
            "required": ["message"]
        }
    }
}


def send_message(message: str) -> None:
    """
    Display a message to the user.

    Args:
        message: The text to display
    """
    console.print(f"\n[info]{message}[/info]\n")
