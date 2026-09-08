from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return hidden[-1]