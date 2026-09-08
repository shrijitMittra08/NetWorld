from __future__ import annotations

from typing import Any, Dict, List

def format_attention_scores(node_ids: List[str], scores: List[float]) -> Dict[str, float]:
    return {str(node): float(score) for node, score in zip(node_ids, scores)}

def top_k_attention(node_scores: Dict[str, float], k: int = 5) -> List[Dict[str, float]]:
    items = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"node": node, "score": score} for node, score in items]