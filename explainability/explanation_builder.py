from __future__ import annotations

from typing import Any, Dict, List, Optional


def _feature_dict(features: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not features:
        return {}
    return {
        str(k): float(v)
        for k, v in features.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def build_explanation(
    forecast: Dict[str, Any],
    feature_attribution: Optional[Dict[str, float]] = None,
    graph_highlight: Optional[Dict[str, Any]] = None,
    temporal_deltas: Optional[List[str]] = None,
    attention_weights: Optional[Dict[str, Any]] = None,
    # Backward-compatible high-level inputs used by the demo/tests.
    graph: Any = None,
    features: Optional[Dict[str, Any]] = None,
    window_summaries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if graph_highlight is None:
        nodes = []
        if graph is not None:
            nodes = [{"id": str(node), "score": 0.0, "highlight": False} for node in graph.nodes()]
        graph_highlight = {"nodes": nodes, "edges": [], "gradient_scores": {}}

    if temporal_deltas is None:
        temporal_deltas = []
        if window_summaries and len(window_summaries) >= 2:
            previous = window_summaries[-2]
            current = window_summaries[-1]
            for key, value in current.items():
                old = previous.get(key)
                if isinstance(old, (int, float)) and isinstance(value, (int, float)) and value != old:
                    temporal_deltas.append(f"{key}: {old:g} → {value:g}")
        if not temporal_deltas:
            temporal_deltas = ["No significant temporal changes detected."]

    if feature_attribution is None:
        raw = _feature_dict(features)
        feature_attribution = {k: abs(v) for k, v in raw.items()}

    probability = float(forecast.get("attack_probability", forecast.get("final_attack_probability", 0.0)))
    stage = forecast.get("predicted_stage") or forecast.get("stage") or "unknown"
    target = forecast.get("likely_target") or forecast.get("target") or "unknown"

    return {
        "forecast": forecast,
        "feature_attribution": feature_attribution,
        "graph_highlight": graph_highlight,
        "graph_explanation": graph_highlight,
        "temporal_explanation": temporal_deltas,
        "attention_weights": attention_weights or {},
        "summary": [
            f"Attack probability: {probability:.3f}",
            f"Predicted stage: {stage}",
            f"Likely target: {target}",
            "Explanation-driven forecast: yes",
        ],
    }
