from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .graph_modules import GraphSAGEEncoder, GATEncoder


EDGE_FEATURE_NAMES = [
    "count",
    "duration_mean",
    "bytes_per_sec_mean",
    "packets_per_sec_mean",
    "fwd_packets_sum",
    "bwd_packets_sum",
    "ttl_mean",
    "payload_size_mean",
    "auth_failures_sum",
    "connection_failures_sum",
]


def _normalize_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    """Normalize edge features while preserving semantic zero values."""

    if edge_attr is None or edge_attr.numel() == 0:
        return edge_attr

    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)

    # Preserve sign while compressing large positive/negative values.
    return torch.sign(edge_attr) * torch.log1p(torch.abs(edge_attr))


def _node_feature_vector(attrs: Dict[str, Any]) -> List[float]:
    return [
        1.0 if attrs.get("role") else 0.0,
        1.0 if attrs.get("subnet") else 0.0,
        1.0 if attrs.get("os") else 0.0,
        float(attrs.get("active_edges", 0) or 0),
        float(attrs.get("new_peers_sum", 0) or 0),
        float(attrs.get("new_destinations_sum", 0) or 0),
        float(attrs.get("unique_dest_ports_sum", 0) or 0),
        float(attrs.get("auth_attempts_sum", 0) or 0),
        float(attrs.get("auth_failures_sum", 0) or 0),
        float(attrs.get("avg_duration", 0) or 0),
        float(attrs.get("connection_attempts_sum", 0) or 0),
        float(attrs.get("connection_failures_sum", 0) or 0),
    ]


class GraphEncoder(nn.Module):
    """
    Encodes a graph snapshot into a latent vector.

    Supports GraphSAGE or GAT and explicitly incorporates
    network-flow edge features into the graph embedding.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        encoder_type: str = "sage",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "gat":
            self.encoder = GATEncoder(input_dim, hidden_dim)
        else:
            self.encoder = GraphSAGEEncoder(input_dim, hidden_dim)

        # Encode the ten canonical edge/flow features into the
        # same latent dimension as the graph encoder.
        self.edge_encoder = nn.Sequential(
            nn.Linear(len(EDGE_FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _build_graph_tensors(self, graph):
        nodes = list(graph.nodes())
        node_index = {n: i for i, n in enumerate(nodes)}

        x = torch.tensor(
            [_node_feature_vector(graph.nodes[n]) for n in nodes],
            dtype=torch.float32,
        )

        edges = []
        edge_features = []

        feature_names = [
            "count",
            "duration_mean",
            "bytes_per_sec_mean",
            "packets_per_sec_mean",
            "fwd_packets_sum",
            "bwd_packets_sum",
            "ttl_mean",
            "payload_size_mean",
            "auth_failures_sum",
            "connection_failures_sum",
        ]

        for u, v, attrs in graph.edges(data=True):
            edges.append([
                node_index[u],
                node_index[v],
            ])

            edge_features.append([
                float(attrs.get(name, 0.0) or 0.0)
                for name in feature_names
            ])

        if not edges:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.empty(
                (0, 10),
                dtype=torch.float32,
            )
        else:
            edge_index = torch.tensor(
                edges,
                dtype=torch.long,
            ).t().contiguous()

            edge_attr = torch.tensor(
                edge_features,
                dtype=torch.float32,
            )

            edge_attr = _normalize_edge_features(edge_attr)

        return x, edge_index, edge_attr, nodes

    def forward(self, graph):
        x, edge_index, edge_attr, _ = self._build_graph_tensors(graph)
    
        if edge_index.numel() == 0:
            graph_embedding = x.mean(
                dim=0,
                keepdim=True,
            )
        else:
            graph_embedding = self.encoder(
                x,
                edge_index,
            )
    
        if edge_attr.numel() > 0:
            edge_embedding = self.edge_encoder(
                edge_attr
            ).mean(
                dim=0,
                keepdim=True,
            )
    
            graph_embedding = graph_embedding + edge_embedding
    
        return graph_embedding