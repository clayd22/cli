"""
submit_observation.py

Tool for submitting narrative/qualitative observations to the user.
Use this when the answer is an analysis, insight, or explanation
rather than a computed value.

Examples:
- "Are there any anomalies?" -> narrative describing findings
- "What patterns do you see?" -> qualitative analysis
- "Explain the data" -> descriptive output
"""

from dataclasses import dataclass


SUBMIT_OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_observation",
        "description": """Submit a narrative observation or analysis to the user.

Use this instead of submit_result when:
- The answer is qualitative (patterns, anomalies, insights)
- You're describing findings rather than computing a single value
- The question asks for analysis or explanation

Your observation should be based on data you explored with run_sql/run_python.
Include specific numbers and examples to support your observations.""",
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "Your observation/analysis in clear, readable prose. Include specific data points to support your findings."
                },
                "supporting_queries": {
                    "type": "object",
                    "description": "The SQL queries used to find this data. Map of description to SQL query.",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "supporting_data": {
                    "type": "string",
                    "description": "Optional: Key data points or a small table that supports your observation."
                }
            },
            "required": ["observation"]
        }
    }
}


@dataclass
class SubmitObservationOutput:
    """Container for observation output."""
    observation: str
    supporting_queries: dict = None
    supporting_data: str = None


def submit_observation(
    observation: str,
    supporting_queries: dict = None,
    supporting_data: str = None
) -> SubmitObservationOutput:
    """
    Submit a narrative observation to the user.

    Args:
        observation: The main observation/analysis text
        supporting_queries: SQL queries used to find the data
        supporting_data: Optional supporting data points

    Returns:
        SubmitObservationOutput for display
    """
    return SubmitObservationOutput(
        observation=observation,
        supporting_queries=supporting_queries,
        supporting_data=supporting_data
    )
