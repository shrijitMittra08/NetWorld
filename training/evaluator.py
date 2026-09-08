from __future__ import annotations

from typing import Any, Dict, List

from evaluation.metrics import accuracy, precision_recall_f1

def evaluate_predictions(preds: List[int], targets: List[int]) -> Dict[str, float]:
    prf = precision_recall_f1(preds, targets)
    return {
        "accuracy": accuracy(preds, targets),
        **prf,
    }

def cross_dataset_evaluation(
    train_results: Dict[str, float],
    test_results: Dict[str, float],
) -> Dict[str, float]:
    """
    Minimal cross-dataset generalization summary.
    """
    return {
        "train_accuracy": train_results.get("accuracy", 0.0),
        "test_accuracy": test_results.get("accuracy", 0.0),
        "generalization_gap": train_results.get("accuracy", 0.0) - test_results.get("accuracy", 0.0),
    }