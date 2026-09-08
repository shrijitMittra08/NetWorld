from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.5, max=10.0)
        return logits / temp

class AttackHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)
        self.scaler = TemperatureScaler()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = self.scaler(logits)
        return torch.sigmoid(logits)

class StageHead(nn.Module):
    def __init__(self, latent_dim: int = 128, num_stages: int = 14):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_stages)
        self.scaler = TemperatureScaler()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        return self.scaler(logits)

class TargetAttentionHead(nn.Module):
    """
    Pointer-style attention over current node embeddings.
    """

    def __init__(self, latent_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(latent_dim, node_dim)

    def forward(self, z: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        z: [batch, latent_dim]
        node_embeddings: [num_nodes, node_dim]
        returns: attention scores over nodes
        """
        q = self.query(z)  # [batch, node_dim]
        scores = torch.matmul(q, node_embeddings.t())  # [batch, num_nodes]
        return torch.softmax(scores, dim=-1)