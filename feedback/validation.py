from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ValidationReport:
    score: float
    passed: bool
    reason: str
    metadata: Dict[str, Any]

def validate_fresh_decoy_set(
    predictions: List[float],
    labels: List[int],
    min_score: float = 0.5,
) -> ValidationReport:
    """
    Minimal held-out validation: classification accuracy.
    """
    if not predictions or not labels:
        return ValidationReport(
            score=0.0,
            passed=False,
            reason="No validation data",
            metadata={},
        )

    n = min(len(predictions), len(labels))
    preds = [1 if p >= 0.5 else 0 for p in predictions[:n]]
    correct = sum(int(p == y) for p, y in zip(preds, labels[:n]))
    score = correct / n

    return ValidationReport(
        score=score,
        passed=score >= min_score,
        reason="Validation passed" if score >= min_score else "Validation failed",
        metadata={"count": n, "min_score": min_score},
    )