from __future__ import annotations

from typing import Dict

def cross_dataset_transfer_evaluation(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "train_auroc": train_metrics.get("auroc", 0.0),
        "test_auroc": test_metrics.get("auroc", 0.0),
        "train_auprc": train_metrics.get("auprc", 0.0),
        "test_auprc": test_metrics.get("auprc", 0.0),
        "generalization_gap_auroc": train_metrics.get("auroc", 0.0) - test_metrics.get("auroc", 0.0),
        "generalization_gap_auprc": train_metrics.get("auprc", 0.0) - test_metrics.get("auprc", 0.0),
    }