from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class PolicyDecision:
    action: str
    score: float
    reason: str
    metadata: Dict[str, Any]
    requires_approval: bool = False


class RulePolicy:
    """Tier-1 interpretable risk/utility policy with explicit safety gates."""

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Policy config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def decide(
        self,
        forecast: Dict[str, Any],
        asset_criticality: int = 1,
        stage_confidence: float = 0.5,
        soc_load: float = 0.0,
        decoy_capacity: float = 1.0,
        human_mode: str = "advisory",
    ) -> PolicyDecision:
        p = float(forecast.get("attack_probability", forecast.get("final_attack_probability", 0.0)))
        target_score = float(forecast.get("target_score", 0.0))
        stage_logits = forecast.get("stage_logits", forecast.get("final_stage_logits", []))
        stage = str(forecast.get("predicted_stage", forecast.get("stage", "unknown"))).lower()

        thresholds = self.config.get("thresholds", {})
        weights = self.config.get("weights", {})
        intel = self.config.get("intel_value", {})
        costs = self.config.get("action_cost", {})

        monitor_max = float(thresholds.get("monitor_max_attack_probability", 0.35))
        divert_max = float(thresholds.get("divert_max_attack_probability", 0.75))
        isolate_min = float(thresholds.get("isolate_min_attack_probability", divert_max))
        high_value = asset_criticality >= int(thresholds.get("high_value_asset_criticality", 4))
        low_confidence = stage_confidence < float(thresholds.get("minimum_stage_confidence", 0.5))
        early_stages = {s.lower() for s in thresholds.get("early_stages", ["reconnaissance"])}
        intel_stages = {s.lower() for s in thresholds.get("divert_stages", ["lateral movement", "credential access", "command and control"])}

        risk = (
            float(weights.get("attack_probability", 0.45)) * p
            + float(weights.get("target_criticality", 0.20)) * (max(1, min(5, asset_criticality)) / 5.0)
            + float(weights.get("stage_confidence", 0.10)) * stage_confidence
            + float(weights.get("soc_load", 0.10)) * max(0.0, min(1.0, soc_load))
            + float(weights.get("decoy_capacity", 0.05)) * (1.0 - max(0.0, min(1.0, decoy_capacity)))
        )
        utilities = {
            action: risk + float(intel.get(action, 0.0)) - float(costs.get(action, 0.0))
            for action in ("monitor", "divert", "isolate")
        }

        # Safety/intent gates come before utility maximisation.
        if p < monitor_max or low_confidence or stage in early_stages:
            action, reason = "monitor", "Low risk, early-stage activity, or insufficient forecast confidence"
        elif p >= isolate_min and high_value and stage not in intel_stages:
            action, reason = "isolate", "High-confidence threat against a high-value asset"
        elif p >= divert_max:
            action, reason = "isolate", "Attack probability exceeds isolation threshold"
        elif p >= monitor_max:
            action, reason = "divert", "Medium/high risk with useful intelligence value"
        else:
            action, reason = "monitor", "No active response threshold met"

        if action == "divert" and decoy_capacity <= 0:
            action, reason = "isolate", "Decoy capacity exhausted; fail closed to containment"

        requires_approval = human_mode == "advisory" or (human_mode == "semi_autonomous" and action == "isolate")
        return PolicyDecision(
            action=action,
            score=float(utilities[action]),
            reason=reason,
            metadata={
                "attack_probability": p,
                "target_score": target_score,
                "asset_criticality": asset_criticality,
                "stage_confidence": stage_confidence,
                "predicted_stage": stage,
                "soc_load": soc_load,
                "decoy_capacity": decoy_capacity,
                "utilities": utilities,
                "stage_logits": stage_logits,
            },
            requires_approval=requires_approval,
        )
