from __future__ import annotations

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Latent transition model that predicts future network states recursively."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.transition = nn.GRUCell(latent_dim, latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)
        self.delta_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def step(self, state: torch.Tensor) -> torch.Tensor:
        # Predict a bounded state delta, then apply it to the recurrent state.
        delta = self.delta_gate(state) * torch.tanh(self.transition(state, state) - state)
        return self.state_norm(state + delta)

    def forward(self, z: torch.Tensor, steps: int = 1):
        if steps < 1:
            return []
        outputs = []
        current = z
        for _ in range(steps):
            current = self.step(current)
            outputs.append(current)
        return outputs
