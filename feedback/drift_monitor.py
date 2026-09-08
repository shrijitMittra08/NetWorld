from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math

@dataclass
class DriftReport:
    drift_score: float
    drift_detected: bool
    reason: str
    metadata: Dict[str, Any]

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def compute_simple_drift_score(
    baseline_probs: List[float],
    current_probs: List[float],
) -> float:
    """
    Very small MVP drift score: mean absolute difference.
    """
    if not baseline_probs or not current_probs:
        return 0.0
    n = min(len(baseline_probs), len(current_probs))
    diffs = [abs(baseline_probs[i] - current_probs[i]) for i in range(n)]
    return _mean(diffs)

def monitor_drift(
    baseline_probs: List[float],
    current_probs: List[float],
    threshold: float = 0.25,
) -> DriftReport:
    score = compute_simple_drift_score(baseline_probs, current_probs)
    detected = score >= threshold
    return DriftReport(
        drift_score=score,
        drift_detected=detected,
        reason="Concept drift detected" if detected else "No significant drift",
        metadata={
            "baseline_count": len(baseline_probs),
            "current_count": len(current_probs),
            "threshold": threshold,
        },
    )