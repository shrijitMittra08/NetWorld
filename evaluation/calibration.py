from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

def brier_score(probs: Sequence[float], targets: Sequence[int]) -> float:
    if not probs or not targets:
        return 0.0
    n = min(len(probs), len(targets))
    diffs = [(float(probs[i]) - float(targets[i])) ** 2 for i in range(n)]
    return float(sum(diffs) / n)

def reliability_diagram(
    probs: Sequence[float],
    targets: Sequence[int],
    num_bins: int = 10,
) -> Dict[str, List[float]]:
    if not probs or not targets:
        return {"bin_confidence": [], "bin_accuracy": [], "bin_count": []}

    n = min(len(probs), len(targets))
    probs = [float(p) for p in probs[:n]]
    targets = [int(t) for t in targets[:n]]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidence = []
    bin_accuracy = []
    bin_count = []

    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j, p in enumerate(probs) if (p >= lo and (p < hi or (i == num_bins - 1 and p <= hi)))]
        if not idx:
            bin_confidence.append(0.0)
            bin_accuracy.append(0.0)
            bin_count.append(0)
            continue

        bin_probs = [probs[j] for j in idx]
        bin_targets = [targets[j] for j in idx]

        bin_confidence.append(float(sum(bin_probs) / len(bin_probs)))
        bin_accuracy.append(float(sum(bin_targets) / len(bin_targets)))
        bin_count.append(len(idx))

    return {
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
    }