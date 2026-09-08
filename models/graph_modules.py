from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
except Exception:  # pragma: no cover
    SAGEConv = None
    GATConv = None
    global_mean_pool = None


class _MeanGraphConv(nn.Module):
    """Small dependency-free message-passing layer used when PyG is unavailable."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(input_dim, output_dim)
        self.neighbor_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index
            neigh = torch.zeros_like(x)
            neigh.index_add_(0, dst, x[src])
            deg = torch.zeros(n, device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            neigh = neigh / deg.clamp_min(1.0).unsqueeze(-1)
        return self.self_proj(x) + self.neighbor_proj(neigh)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.act = nn.ReLU()
        self.uses_pyg = SAGEConv is not None
        if self.uses_pyg:
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = _MeanGraphConv(input_dim, hidden_dim)
            self.conv2 = _MeanGraphConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        if batch is not None and global_mean_pool is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)


class GATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 2):
        super().__init__()
        self.act = nn.ReLU()
        self.uses_pyg = GATConv is not None
        if self.uses_pyg:
            self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        else:
            # Dependency-free fallback keeps the public GAT option runnable.
            self.conv1 = _MeanGraphConv(input_dim, hidden_dim)
            self.conv2 = _MeanGraphConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        if batch is not None and global_mean_pool is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)
