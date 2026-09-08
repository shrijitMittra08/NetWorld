from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import datetime, timezone

class ImmutableDiversionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: Dict[str, Any]) -> None:
        enriched = dict(record)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(enriched, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]