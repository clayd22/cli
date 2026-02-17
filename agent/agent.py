from openai import OpenAI

from .config import get_api_key


class AstroAgent:
    def __init__(self):
        api_key = get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not configured. Run 'astro config' first.")

        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.model = "gpt-4o"

        self.system_prompt = """You are AstroAgent, a data analysis assistant with a space theme.
You help users explore and query data from a DuckDB data warehouse.
You have access to a mini data platform with e-commerce data including:
- Products, customers, transactions, campaigns, and pageviews
- The main table is marts.fct_orders which has denormalized order data

When users ask questions:
1. Understand what they're looking for
2. Generate appropriate SQL queries
3. Explain your reasoning

Be concise but helpful. Keep the space theme subtle."""

    def chat(self, user_message: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self):
        self.conversation_history = []
