from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from forecasting import ForecastEngine
from .datasets import GraphSequenceDataset, collate_identity
from .losses import rollout_loss
from .checkpointing import save_checkpoint
from .early_stopping import EarlyStopping

@dataclass
class TrainingResult:
    train_loss: float
    val_loss: float
    stopped_early: bool
    epochs_ran: int

class Trainer:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        rollout_steps: int = 3,
        checkpoint_path: str | None = None,
        patience: int = 3,
    ):
        self.engine = ForecastEngine(rollout_steps=rollout_steps)

        self.params = (
            list(self.engine.graph_encoder.parameters())
            + list(self.engine.temporal_encoder.parameters())
            + list(self.engine.world_model.parameters())
            + list(self.engine.attack_head.parameters())
            + list(self.engine.stage_head.parameters())
            + list(self.engine.target_head.parameters())
        )
        self.optimizer = torch.optim.Adam(self.params, lr=learning_rate)
        self.checkpoint_path = checkpoint_path
        self.early_stopping = EarlyStopping(patience=patience)

    def _forward_batch(self, batch_items: List[Dict[str, Any]]):
        batch_losses = []
        for item in batch_items:
            graphs = item["graphs"]
            targets = item["targets"]
            current_node_embeddings = item.get("current_node_embeddings")

            forecast = self.engine.forecast(
                graphs,
                current_node_embeddings=current_node_embeddings,
                detach=False,
            )
            rollout_preds = forecast["_rollout_tensors"]

            attack_target = torch.tensor([targets.get("attack", 0)], dtype=torch.float32)
            stage_target = torch.tensor([targets.get("stage", 0)], dtype=torch.long)
            attack_targets = [attack_target] * len(rollout_preds)
            stage_targets = [stage_target] * len(rollout_preds)
            target_indices = None
            if "target_index" in targets:
                target_indices = [torch.tensor([int(targets.get("target_index", 0))], dtype=torch.long)] * len(rollout_preds)

            # When future observed windows are available, train the transition model
            # against their encoded latent states. This makes the rollout a true
            # predictive world model rather than a classifier wrapper.
            state_targets = []
            if len(graphs) > 1:
                for i in range(len(rollout_preds)):
                    end = min(len(graphs), i + 2)
                    observed = self.engine._encode_sequence(graphs[:end])
                    state_targets.append(observed)

            loss = rollout_loss(
                rollout_preds=rollout_preds,
                attack_targets=attack_targets,
                stage_targets=stage_targets,
                target_indices=target_indices,
                state_targets=state_targets or None,
            )
            batch_losses.append(loss)

        return torch.stack(batch_losses).mean()

    def train(
        self,
        train_items: List[Dict[str, Any]],
        val_items: List[Dict[str, Any]],
        epochs: int = 10,
        batch_size: int = 4,
    ) -> TrainingResult:
        train_loader = DataLoader(
            GraphSequenceDataset(train_items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_identity,
        )
        val_loader = DataLoader(
            GraphSequenceDataset(val_items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_identity,
        )

        best_train_loss = float("inf")
        best_val_loss = float("inf")
        stopped_early = False
        epochs_ran = 0

        for epoch in range(epochs):
            self.engine.graph_encoder.train()
            self.engine.temporal_encoder.train()
            self.engine.world_model.train()
            self.engine.attack_head.train()
            self.engine.stage_head.train()
            self.engine.target_head.train()

            train_losses = []
            for batch in train_loader:
                self.optimizer.zero_grad()
                loss = self._forward_batch(batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(float(loss.item()))

            train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            best_train_loss = min(best_train_loss, train_loss)

            self.engine.graph_encoder.eval()
            self.engine.temporal_encoder.eval()
            self.engine.world_model.eval()
            self.engine.attack_head.eval()
            self.engine.stage_head.eval()
            self.engine.target_head.eval()

            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    loss = self._forward_batch(batch)
                    val_losses.append(float(loss.item()))
            val_loss = sum(val_losses) / len(val_losses) if val_losses else train_loss
            best_val_loss = min(best_val_loss, val_loss)

            epochs_ran = epoch + 1

            if self.checkpoint_path is not None:
                save_checkpoint(
                    self.checkpoint_path,
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_state": {
                            "graph_encoder": self.engine.graph_encoder.state_dict(),
                            "temporal_encoder": self.engine.temporal_encoder.state_dict(),
                            "world_model": self.engine.world_model.state_dict(),
                            "attack_head": self.engine.attack_head.state_dict(),
                            "stage_head": self.engine.stage_head.state_dict(),
                            "target_head": self.engine.target_head.state_dict(),
                        },
                        "optimizer_state": self.optimizer.state_dict(),
                    },
                )

            if self.early_stopping.step(val_loss):
                stopped_early = True
                break

        return TrainingResult(
            train_loss=best_train_loss,
            val_loss=best_val_loss,
            stopped_early=stopped_early,
            epochs_ran=epochs_ran,
        )

    def train_step(self, graphs: List[Any], targets: Dict[str, Any], current_node_embeddings: torch.Tensor | None = None) -> float:
        """
        Backward-compatible single-step method.
        """
        item = {"graphs": graphs, "targets": targets, "current_node_embeddings": current_node_embeddings}
        self.optimizer.zero_grad()
        loss = self._forward_batch([item])
        loss.backward()
        self.optimizer.step()
        return float(loss.item())