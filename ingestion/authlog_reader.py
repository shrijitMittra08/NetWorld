from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

class AuthLogReader:
    """
    Minimal auth log reader.

    Supports CSV-style auth logs or plain text logs captured as raw lines.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def parse(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Auth log file not found: {self.path}")

        if self.path.suffix.lower() == ".csv":
            return pd.read_csv(self.path)

        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append({"raw_line": line})

        return pd.DataFrame(rows)