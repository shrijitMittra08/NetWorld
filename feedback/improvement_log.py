from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import datetime, timezone

class ImprovementLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def append(self, record: Dict[str, Any]) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        enriched = dict(record)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data.append(enriched)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read_all(self) -> List[Dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))