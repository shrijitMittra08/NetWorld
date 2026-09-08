from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn

from .graph_modules import GraphSAGEEncoder, GATEncoder

def _node_feature_vector(attrs: Dict[str, Any]) -> List[float]:
    return [
        1.0 if attrs.get("role") else 0.0,
        1.0 if attrs.get("subnet") else 0.0,
        1.0 if attrs.get("os") else 0.0,
        float(attrs.get("active_edges", 0) or 0),
        float(attrs.get("new_peers_sum", 0) or 0),
        float(attrs.get("new_destinations_sum", 0) or 0),
        float(attrs.get("auth_attempts_sum", 0) or 0),
        float(attrs.get("auth_failures_sum", 0) or 0),
    ]

class GraphEncoder(nn.Module):
    """
    Encodes a graph snapshot into a latent vector.
    Supports GraphSAGE or GAT.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, encoder_type: str = "sage"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "gat":
            self.encoder = GATEncoder(input_dim, hidden_dim)
        else:
            self.encoder = GraphSAGEEncoder(input_dim, hidden_dim)

    def _build_graph_tensors(self, graph):
        nodes = list(graph.nodes())
        node_index = {n: i for i, n in enumerate(nodes)}

        x = torch.tensor(
            [_node_feature_vector(graph.nodes[n]) for n in nodes],
            dtype=torch.float32,
        )

        edges = []
        for u, v in graph.edges():
            edges.append([node_index[u], node_index[v]])

        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return x, edge_index, nodes

    def forward(self, graph):
        x, edge_index, _ = self._build_graph_tensors(graph)
        if edge_index.numel() == 0:
            return x.mean(dim=0, keepdim=True)
        return self.encoder(x, edge_index)