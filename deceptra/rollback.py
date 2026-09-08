from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RollbackDecision:
    should_rollback: bool
    escalate_to_isolate: bool
    reason: str
    metadata: Dict[str, Any]

def evaluate_rollback(
    evasion_detected: bool,
    suspicious_behavior_score: float,
    threshold: float = 0.7,
) -> RollbackDecision:
    if evasion_detected or suspicious_behavior_score >= threshold:
        return RollbackDecision(
            should_rollback=True,
            escalate_to_isolate=True,
            reason="Decoy evasion or suspicious behavior detected",
            metadata={
                "evasion_detected": evasion_detected,
                "suspicious_behavior_score": suspicious_behavior_score,
            },
        )

    return RollbackDecision(
        should_rollback=False,
        escalate_to_isolate=False,
        reason="No rollback required",
        metadata={
            "evasion_detected": evasion_detected,
            "suspicious_behavior_score": suspicious_behavior_score,
        },
    )