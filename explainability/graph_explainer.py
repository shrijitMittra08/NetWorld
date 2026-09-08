from __future__ import annotations

from typing import Any, Dict, List

def build_graph_highlight(
    node_attention: Dict[str, float] | None = None,
    edge_attention: Dict[str, float] | None = None,
    gradient_scores: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    node_attention = node_attention or {}
    edge_attention = edge_attention or {}
    gradient_scores = gradient_scores or {}

    nodes = []
    for node, score in node_attention.items():
        nodes.append(
            {
                "id": node,
                "score": float(score),
                "highlight": True,
            }
        )

    edges = []
    for edge, score in edge_attention.items():
        edges.append(
            {
                "id": edge,
                "score": float(score),
                "highlight": True,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "gradient_scores": gradient_scores,
    }