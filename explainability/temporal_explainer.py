from __future__ import annotations

from typing import Any, Dict, List

def explain_temporal_deltas(window_summaries: List[Dict[str, Any]]) -> List[str]:
    if len(window_summaries) < 2:
        return ["Not enough windows to compute temporal deltas."]

    explanations: List[str] = []
    prev = window_summaries[0]

    for current in window_summaries[1:]:
        for key in current.keys():
            if key in prev and isinstance(current[key], (int, float)) and isinstance(prev[key], (int, float)):
                diff = current[key] - prev[key]
                if diff != 0:
                    sign = "+" if diff > 0 else ""
                    explanations.append(f"{key} {sign}{diff}")
        prev = current

    if not explanations:
        explanations.append("No significant temporal changes detected.")

    return explanations