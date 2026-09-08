from __future__ import annotations

from dataclasses import dataclass

@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best_loss: float | None = None
    bad_epochs: int = 0

    def step(self, loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_loss is None or loss < (self.best_loss - self.min_delta):
            self.best_loss = loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience