from __future__ import annotations

from typing import Iterable, List, Sequence
import numpy as np

def accuracy(preds: Iterable[int], targets: Iterable[int]) -> float:
    preds = list(preds)
    targets = list(targets)
    if not targets:
        return 0.0
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    return correct / len(targets)

def precision_recall_f1(preds: Iterable[int], targets: Iterable[int]) -> dict:
    preds = list(preds)
    targets = list(targets)

    tp = sum(int(p == 1 and t == 1) for p, t in zip(preds, targets))
    fp = sum(int(p == 1 and t == 0) for p, t in zip(preds, targets))
    fn = sum(int(p == 0 and t == 1) for p, t in zip(preds, targets))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def false_positive_rate(preds: Iterable[int], targets: Iterable[int]) -> float:
    preds = list(preds)
    targets = list(targets)
    fp = sum(int(p == 1 and t == 0) for p, t in zip(preds, targets))
    tn = sum(int(p == 0 and t == 0) for p, t in zip(preds, targets))
    return fp / (fp + tn) if (fp + tn) else 0.0

def _rankdata_desc(scores: Sequence[float]) -> List[int]:
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks

def top_k_accuracy(scores: Sequence[Sequence[float]], targets: Sequence[int], k: int = 1) -> float:
    if not scores or not targets:
        return 0.0
    n = min(len(scores), len(targets))
    correct = 0
    for i in range(n):
        ranked = sorted(range(len(scores[i])), key=lambda j: scores[i][j], reverse=True)
        if targets[i] in ranked[:k]:
            correct += 1
    return correct / n

def mean_reciprocal_rank(scores: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    if not scores or not targets:
        return 0.0
    n = min(len(scores), len(targets))
    total = 0.0
    for i in range(n):
        ranked = sorted(range(len(scores[i])), key=lambda j: scores[i][j], reverse=True)
        if targets[i] in ranked:
            total += 1.0 / (ranked.index(targets[i]) + 1)
    return total / n