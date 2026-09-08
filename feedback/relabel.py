from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

def relabel_decoy_sessions(sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert decoy sessions into labelled telemetry rows.
    Anything from a decoy is malicious by construction.
    """
    rows = []

    for session in sessions:
        row = dict(session)
        row["label.class"] = "DECOY_MALICIOUS"
        row["label.is_attack"] = 1
        row["label.attack_type"] = row.get("label.attack_type", "UNKNOWN")
        row["label.attack_stage"] = row.get("label.attack_stage", "UNKNOWN")
        rows.append(row)

    return pd.DataFrame(rows)