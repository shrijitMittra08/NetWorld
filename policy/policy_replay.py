from __future__ import annotations

from typing import Any, Dict, List

def replay_policy_decisions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: r.get("timestamp", ""))