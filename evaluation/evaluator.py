from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .metrics import (
    accuracy,
    precision_recall_f1,
    false_positive_rate,
    top_k_accuracy,
    mean_reciprocal_rank,
)
from .calibration import brier_score, reliability_diagram
from .warning import early_warning_time
from .policy_metrics import (
    containment_rate,
    intel_yield,
    false_divert_rate,
    analyst_override_rate,
)
from .deceptra_metrics import (
    decoy_dwell_time,
    detection_evasion_rate,
    feedback_loop_latency,
)
from .transfer import cross_dataset_transfer_evaluation

def evaluate_detection(
    binary_preds: Sequence[int],
    binary_targets: Sequence[int],
    probs: Sequence[float] | None = None,
) -> Dict[str, float]:
    result = {
        "accuracy": accuracy(binary_preds, binary_targets),
        "false_positive_rate": false_positive_rate(binary_preds, binary_targets),
        **precision_recall_f1(binary_preds, binary_targets),
    }
    if probs is not None:
        result["brier_score"] = brier_score(probs, binary_targets)
    return result

def evaluate_ranking(
    target_scores: Sequence[Sequence[float]],
    target_indices: Sequence[int],
) -> Dict[str, float]:
    return {
        "top1_target_accuracy": top_k_accuracy(target_scores, target_indices, k=1),
        "top3_target_accuracy": top_k_accuracy(target_scores, target_indices, k=3),
        "mrr": mean_reciprocal_rank(target_scores, target_indices),
    }

def evaluate_full(
    binary_preds: Sequence[int],
    binary_targets: Sequence[int],
    probs: Sequence[float],
    target_scores: Sequence[Sequence[float]],
    target_indices: Sequence[int],
    train_metrics: Dict[str, float] | None = None,
    test_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    out = {}
    out.update(evaluate_detection(binary_preds, binary_targets, probs=probs))
    out.update(evaluate_ranking(target_scores, target_indices))
    out["reliability_diagram"] = reliability_diagram(probs, binary_targets)

    if train_metrics is not None and test_metrics is not None:
        out["transfer"] = cross_dataset_transfer_evaluation(train_metrics, test_metrics)

    return out