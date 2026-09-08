from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import numpy as np


@dataclass
class LearnedPolicyDecision:
    action: str
    action_index: int
    confidence: float
    metadata: Dict[str, Any]


class LearnedPolicy:
    """A real contextual-bandit policy using per-action ridge regression + UCB."""

    ACTIONS = ["monitor", "divert", "isolate"]
    FEATURES = [
        "attack_probability", "stage_confidence", "target_score",
        "asset_criticality", "soc_load", "decoy_capacity",
    ]

    def __init__(self, mode: str = "bandit", alpha: float = 1.0, exploration: float = 0.35):
        self.mode = mode.lower()
        if self.mode not in {"bandit", "linucb"}:
            raise ValueError("LearnedPolicy supports 'bandit'/'linucb'; use RulePolicy for deterministic Tier-1 behavior")
        self.alpha = float(alpha)
        self.exploration = float(exploration)
        d = len(self.FEATURES) + 1
        self.A = [self.alpha * np.eye(d) for _ in self.ACTIONS]
        self.b = [np.zeros(d) for _ in self.ACTIONS]
        self.action_counts = np.zeros(len(self.ACTIONS), dtype=int)
        self.action_values = np.zeros(len(self.ACTIONS), dtype=float)

    def _vectorize(self, state: Dict[str, Any]) -> np.ndarray:
        return np.asarray([float(state.get(k, 0.0) or 0.0) for k in self.FEATURES] + [1.0], dtype=float)

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURES) + 1

    def decide(self, state: Dict[str, Any]) -> LearnedPolicyDecision:
        x = self._vectorize(state)
        scores = []
        for i in range(len(self.ACTIONS)):
            A_inv = np.linalg.pinv(self.A[i])
            theta = A_inv @ self.b[i]
            mean = float(theta @ x)
            bonus = self.exploration * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            scores.append(mean + bonus)
        action_index = int(np.argmax(scores))
        probs = np.exp(np.asarray(scores) - np.max(scores))
        probs /= probs.sum() if probs.sum() else 1.0
        confidence = float(probs[action_index])
        return LearnedPolicyDecision(
            action=self.ACTIONS[action_index],
            action_index=action_index,
            confidence=confidence,
            metadata={"mode": self.mode, "scores": scores, "state": dict(state)},
        )

    def update(self, action_index: int, reward: float, state: Dict[str, Any] | None = None) -> None:
        if not 0 <= int(action_index) < len(self.ACTIONS):
            raise IndexError("action_index out of range")
        x = self._vectorize(state or {})
        i = int(action_index)
        self.A[i] += np.outer(x, x)
        self.b[i] += float(reward) * x
        self.action_counts[i] += 1
        n = self.action_counts[i]
        self.action_values[i] += (float(reward) - self.action_values[i]) / n

    def fit(self, decisions: Iterable[Dict[str, Any]]) -> int:
        count = 0
        for item in decisions:
            self.update(
                int(item.get("action_index", 0)),
                float(item.get("reward", 0.0)),
                item.get("state", item),
            )
            count += 1
        return count
