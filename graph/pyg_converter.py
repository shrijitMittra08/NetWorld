from __future__ import annotations

from typing import Any, Dict
import networkx as nx

def temporal_snapshot_to_pyg(snapshot: Any) -> Dict[str, Any]:
    """
    Minimal PyG-ready conversion hook.

    Returns a dictionary with:
    - edge_index
    - node_features placeholder
    - edge_features placeholder
    """
    graph = snapshot.graph if hasattr(snapshot, "graph") else snapshot
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Expected a NetworkX DiGraph or TemporalGraphSnapshot")

    nodes = list(graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}

    edge_index = []
    edge_attr = []

    for u, v, attrs in graph.edges(data=True):
        edge_index.append([node_index[u], node_index[v]])
        edge_attr.append(attrs)

    return {
        "nodes": nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_attr": [graph.nodes[n] for n in nodes],
    }