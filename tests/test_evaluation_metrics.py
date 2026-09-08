from evaluation import (
    brier_score,
    false_positive_rate,
    top_k_accuracy,
    mean_reciprocal_rank,
    containment_rate,
    intel_yield,
    decoy_dwell_time,
    cross_dataset_transfer_evaluation,
)


def test_basic_metrics():
    assert false_positive_rate([1, 0, 1], [0, 0, 1]) >= 0.0
    assert brier_score([0.9, 0.2], [1, 0]) >= 0.0
    assert top_k_accuracy([[0.1, 0.8, 0.2]], [1], k=1) == 1.0
    assert mean_reciprocal_rank([[0.1, 0.8, 0.2]], [1]) == 1.0


def test_policy_and_deceptra_metrics():
    assert containment_rate(8, 10) == 0.8
    assert intel_yield(6, 3) == 2.0
    assert decoy_dwell_time(30.0, 3) == 10.0


def test_transfer_eval():
    out = cross_dataset_transfer_evaluation({"auroc": 0.95, "auprc": 0.92}, {"auroc": 0.81, "auprc": 0.78})
    assert "generalization_gap_auroc" in out