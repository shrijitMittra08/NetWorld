from __future__ import annotations

from pathlib import Path

import pandas as pd

class NetFlowReader:
    """
    Reader for NetFlow/IPFIX-like structured exports.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def parse(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"NetFlow file not found: {self.path}")

        if self.path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(self.path, lines=self.path.suffix.lower() == ".jsonl")

        return pd.read_csv(self.path)