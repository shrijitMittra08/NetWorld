from __future__ import annotations

from typing import Any, Dict, List, Optional

def build_explanation(
    forecast: Dict[str, Any],
    feature_attribution: Dict[str, float],
    graph_highlight: Dict[str, Any],
    temporal_deltas: List[str],
    attention_weights: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "forecast": forecast,
        "feature_attribution": feature_attribution,
        "graph_highlight": graph_highlight,
        "temporal_explanation": temporal_deltas,
        "attention_weights": attention_weights or {},
        "summary": [
            f"Attack probability: {forecast.get('attack_probability', forecast.get('final_attack_probability', 0.0)):.3f}",
            f"Stage logits available: {'yes' if forecast.get('stage_logits') or forecast.get('final_stage_logits') else 'no'}",
            f"Explanation-driven forecast: yes",
        ],
    }