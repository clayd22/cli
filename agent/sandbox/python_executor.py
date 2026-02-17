import pandas as pd
import numpy as np
from typing import Any
import traceback


class PythonExecutor:
    """
    Executes Python code on dataframes in a restricted environment.
    """

    ALLOWED_MODULES = {
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }

    def execute(
        self,
        code: str,
        dataframes: dict[str, pd.DataFrame]
    ) -> tuple[Any, str]:
        """
        Execute Python code with access to provided dataframes.

        Args:
            code: Python code to execute (should define a 'result' variable)
            dataframes: Dict mapping names to DataFrames (e.g., {"df1": df1, "df2": df2})

        Returns:
            (result, error) - result is the value of 'result' variable, error is None on success
        """
        local_vars = {
            **self.ALLOWED_MODULES,
            **dataframes,
        }

        try:
            exec(code, {"__builtins__": self._safe_builtins()}, local_vars)

            if "result" not in local_vars:
                return None, "Code must define a 'result' variable"

            return local_vars["result"], None

        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    def _safe_builtins(self) -> dict:
        """Return a restricted set of builtins."""
        safe = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }
        return safe
