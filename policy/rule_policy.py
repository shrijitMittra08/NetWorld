from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass
class PolicyDecision:
    action: str
    score: float
    reason: str
    metadata: Dict[str, Any]
    requires_approval: bool = False

class RulePolicy:
    """
    Tier 1: interpretable utility-based policy.
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Policy config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _get(self, section: str, key: str, default: float = 0.0) -> float:
        return float(self.config.get(section, {}).get(key, default))

    def decide(
        self,
        forecast: Dict[str, Any],
        asset_criticality: int = 1,
        stage_confidence: float = 0.5,
        soc_load: float = 0.0,
        decoy_capacity: float = 1.0,
        human_mode: str = "advisory",
    ) -> PolicyDecision:
        attack_probability = float(
            forecast.get("attack_probability", forecast.get("final_attack_probability", 0.0))
        )
        target_score = float(forecast.get("target_score", 0.0))
        stage_logits = forecast.get("stage_logits", forecast.get("final_stage_logits", []))

        thresholds = self.config.get("thresholds", {})
        weights = self.config.get("weights", {})
        intel_values = self.config.get("intel_value", {})
        costs = self.config.get("action_cost", {})

        # Simple utility model
        risk = (
            weights.get("attack_probability", 0.0) * attack_probability
            + weights.get("target_criticality", 0.0) * (asset_criticality / 5.0)
            + weights.get("stage_confidence", 0.0) * stage_confidence
            + weights.get("soc_load", 0.0) * soc_load
            + weights.get("decoy_capacity", 0.0) * (1.0 - decoy_capacity)
        )

        # Candidate action utilities
        utilities = {
            "monitor": risk + intel_values.get("monitor", 0.0) - costs.get("monitor", 0.0),
            "divert": risk + intel_values.get("divert", 0.0) - costs.get("divert", 0.0),
            "isolate": risk + intel_values.get("isolate", 0.0) - costs.get("isolate", 0.0),
        }

        # Rule overrides to stay explainable
        if attack_probability < thresholds.get("monitor_max_attack_probability", 0.35):
            action = "monitor"
            reason = "Low attack probability"
        elif attack_probability < thresholds.get("divert_max_attack_probability", 0.75):
            action = "divert"
            reason = "Medium attack probability"
        else:
            action = "isolate"
            reason = "High attack probability"

        # Re-rank based on utility but keep rule override if extremely confident
        if attack_probability < 0.9:
            action = max(utilities, key=utilities.get)
            reason = f"Selected by utility maximization: {action}"

        score = float(utilities[action])

        requires_approval = False
        if human_mode == "advisory":
            requires_approval = True
        elif human_mode == "semi_autonomous" and action == "isolate":
            requires_approval = True

        return PolicyDecision(
            action=action,
            score=score,
            reason=reason,
            metadata={
                "attack_probability": attack_probability,
                "target_score": target_score,
                "asset_criticality": asset_criticality,
                "stage_confidence": stage_confidence,
                "soc_load": soc_load,
                "decoy_capacity": decoy_capacity,
                "utilities": utilities,
                "stage_logits": stage_logits,
            },
            requires_approval=requires_approval,
        )