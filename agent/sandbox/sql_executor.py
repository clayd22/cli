import duckdb
import pandas as pd
from pathlib import Path
from typing import Any

WAREHOUSE_PATH = Path(__file__).parent.parent.parent / "warehouse" / "data.duckdb"


class SQLExecutor:
    def __init__(self):
        self.warehouse_path = WAREHOUSE_PATH

    def execute(self, sql: str) -> tuple[pd.DataFrame, str]:
        """
        Execute SQL and return (dataframe, error).
        If successful, error is None.
        If failed, dataframe is None and error contains the message.
        """
        try:
            with duckdb.connect(str(self.warehouse_path), read_only=True) as conn:
                result = conn.execute(sql).fetchdf()
                return result, None
        except Exception as e:
            return None, str(e)

    def execute_to_dict(self, sql: str) -> tuple[list[dict], str]:
        """Execute SQL and return results as list of dicts."""
        df, error = self.execute(sql)
        if error:
            return None, error
        return df.to_dict(orient="records"), None

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """
        Validate SQL by preparing it without executing.
        Returns (is_valid, error_message).
        """
        try:
            with duckdb.connect(str(self.warehouse_path), read_only=True) as conn:
                conn.execute(f"EXPLAIN {sql}")
                return True, None
        except Exception as e:
            return False, str(e)
