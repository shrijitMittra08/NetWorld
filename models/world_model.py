from __future__ import annotations

import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """
    Rolls latent state forward for K steps.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, steps: int = 1):
        outputs = []
        current = z
        for _ in range(steps):
            current = self.transition(current)
            outputs.append(current)
        return outputs