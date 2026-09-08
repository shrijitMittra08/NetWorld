from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class LearnedPolicyDecision:
    action: str
    action_index: int
    confidence: float
    metadata: Dict[str, Any]

class LearnedPolicy:
    """
    Tier 2 scaffold:
    - bandit-style action scoring
    - DQN/PPO placeholders
    """

    ACTIONS = ["monitor", "divert", "isolate"]

    def __init__(self, mode: str = "bandit"):
        self.mode = mode.lower()
        self.action_values = np.zeros(len(self.ACTIONS), dtype=float)
        self.action_counts = np.zeros(len(self.ACTIONS), dtype=float)

    def decide(self, state: Dict[str, Any]) -> LearnedPolicyDecision:
        # Very small exploration-friendly policy
        if self.mode == "bandit":
            scores = self.action_values.copy()
            action_index = int(np.argmax(scores))
            confidence = float(np.max(scores)) if len(scores) else 0.0
        else:
            # DQN/PPO placeholder: deterministic fallback
            action_index = 0
            confidence = 0.0

        return LearnedPolicyDecision(
            action=self.ACTIONS[action_index],
            action_index=action_index,
            confidence=confidence,
            metadata={"mode": self.mode, "state": state},
        )

    def update(self, action_index: int, reward: float) -> None:
        self.action_counts[action_index] += 1
        n = self.action_counts[action_index]
        current = self.action_values[action_index]
        self.action_values[action_index] = current + (reward - current) / n