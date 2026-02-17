from .internal import run_sql, run_python, inspect_schema
from .output import submit_result

INTERNAL_TOOLS = [run_sql, run_python, inspect_schema]
OUTPUT_TOOLS = [submit_result]
ALL_TOOLS = INTERNAL_TOOLS + OUTPUT_TOOLS
