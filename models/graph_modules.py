from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
except Exception:  # pragma: no cover
    SAGEConv = None
    GATConv = None
    global_mean_pool = None

class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGEEncoder")
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        if batch is not None and global_mean_pool is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)

class GATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 2):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is required for GATEncoder")
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        if batch is not None and global_mean_pool is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)