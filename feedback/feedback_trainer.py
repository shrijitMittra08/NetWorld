from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from training import Trainer
from policy import LearnedPolicy
from .sample_manager import SampleManager
from .drift_monitor import monitor_drift
from .validation import validate_fresh_decoy_set
from .policy_feedback import update_policy_from_feedback
from .improvement_log import ImprovementLog

@dataclass
class FeedbackTrainingResult:
    retrained: bool
    validation_score: float
    drift_score: float
    samples_used: int
    reason: str

class FeedbackTrainer:
    def __init__(
        self,
        trainer: Trainer,
        learned_policy: Optional[LearnedPolicy] = None,
        improvement_log: Optional[ImprovementLog] = None,
    ):
        self.trainer = trainer
        self.learned_policy = learned_policy or LearnedPolicy(mode="bandit")
        self.sample_manager = SampleManager()
        self.improvement_log = improvement_log

    def ingest_feedback_samples(self, samples: List[Dict[str, Any]]) -> int:
        return self.sample_manager.add_many(samples)

    def retrain(
        self,
        graphs: List[Any],
        targets_list: List[Dict[str, Any]],
        baseline_probs: List[float],
        current_probs: List[float],
        validation_predictions: List[float],
        validation_labels: List[int],
        policy_rewards: Optional[List[Dict[str, Any]]] = None,
    ) -> FeedbackTrainingResult:
        drift_report = monitor_drift(baseline_probs, current_probs)
        validation_report = validate_fresh_decoy_set(validation_predictions, validation_labels)

        retrained = False
        reason = "No retraining triggered"

        if drift_report.drift_detected or validation_report.passed:
            retrained = True
            reason = "Retrained due to drift or validation signal"

            for graphs_item, targets in zip(graphs, targets_list):
                self.trainer.train_step(graphs_item, targets)

            if policy_rewards:
                for item in policy_rewards:
                    update_policy_from_feedback(
                        self.learned_policy,
                        action_index=int(item.get("action_index", 0)),
                        reward=float(item.get("reward", 0.0)),
                        metadata=item,
                    )

        result = FeedbackTrainingResult(
            retrained=retrained,
            validation_score=validation_report.score,
            drift_score=drift_report.drift_score,
            samples_used=len(graphs),
            reason=reason,
        )

        if self.improvement_log is not None:
            self.improvement_log.append(
                {
                    "retrained": result.retrained,
                    "validation_score": result.validation_score,
                    "drift_score": result.drift_score,
                    "samples_used": result.samples_used,
                    "reason": result.reason,
                }
            )

        return result