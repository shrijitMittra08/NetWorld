from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RetrainTrigger:
    should_retrain: bool
    reason: str
    metadata: Dict[str, Any]

def evaluate_retrain_trigger(
    new_decoy_rows: int,
    drift_score: float = 0.0,
    min_rows: int = 10,
    drift_threshold: float = 0.25,
) -> RetrainTrigger:
    if new_decoy_rows >= min_rows:
        return RetrainTrigger(
            should_retrain=True,
            reason="Enough new decoy-labelled rows collected",
            metadata={"new_decoy_rows": new_decoy_rows, "drift_score": drift_score},
        )

    if drift_score >= drift_threshold:
        return RetrainTrigger(
            should_retrain=True,
            reason="Concept drift threshold exceeded",
            metadata={"new_decoy_rows": new_decoy_rows, "drift_score": drift_score},
        )

    return RetrainTrigger(
        should_retrain=False,
        reason="Not enough new data and drift is low",
        metadata={"new_decoy_rows": new_decoy_rows, "drift_score": drift_score},
    )