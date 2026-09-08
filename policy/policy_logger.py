from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import datetime, timezone

class PolicyLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("[]", encoding="utf-8")

    def append(self, record: Dict[str, Any]) -> None:
        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        enriched = dict(record)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data.append(enriched)
        self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read_all(self) -> List[Dict[str, Any]]:
        return json.loads(self.log_path.read_text(encoding="utf-8"))