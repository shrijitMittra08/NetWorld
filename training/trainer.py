from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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
    """End-to-end gradient trainer for the NetWorld model stack."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        rollout_steps: int = 3,
        checkpoint_path: str | None = None,
        patience: int = 3,
        attack_pos_weight: float = 1.0,
    ):
        self.engine = ForecastEngine(rollout_steps=rollout_steps)
        self.rollout_steps = rollout_steps
        self.attack_pos_weight = float(max(1.0, attack_pos_weight))

        self.params = [
            p
            for module in (
                self.engine.graph_encoder,
                self.engine.temporal_encoder,
                self.engine.world_model,
                self.engine.attack_head,
                self.engine.stage_head,
                self.engine.target_head,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.checkpoint_path = checkpoint_path
        self.early_stopping = EarlyStopping(patience=patience)

    def _forward_item(self, item: Dict[str, Any]) -> torch.Tensor:
        graphs = item["graphs"]
        future = item.get("future_snapshots", [])
        targets = item["targets"]

        current_node_embeddings = self.engine.node_embeddings(graphs[-1])
        forecast = self.engine.forecast(
            graphs,
            current_node_embeddings=current_node_embeddings,
            detach=False,
        )
        rollout_preds = forecast["_rollout_tensors"]

        attack_sequence = targets.get("attack_sequence") or [targets.get("attack", 0)] * len(rollout_preds)
        stage_sequence = targets.get("stage_sequence") or [targets.get("stage", 0)] * len(rollout_preds)
        stage_masks = targets.get("stage_mask_sequence") or [1.0] * len(rollout_preds)
        target_sequence = targets.get("target_sequence") or [targets.get("target_index")]

        attack_targets = [
            torch.tensor([float(attack_sequence[i])], dtype=torch.float32)
            for i in range(len(rollout_preds))
        ]
        stage_targets = [
            torch.tensor([int(stage_sequence[i])], dtype=torch.long)
            for i in range(len(rollout_preds))
        ]
        stage_mask_tensors = [
            torch.tensor([float(stage_masks[i])], dtype=torch.float32)
            for i in range(len(rollout_preds))
        ]

        target_indices = []
        for i in range(len(rollout_preds)):
            target = target_sequence[i] if i < len(target_sequence) else None
            target_indices.append(
                torch.tensor([int(target)], dtype=torch.long)
                if target is not None else None
            )

        # Each rollout step is supervised by the corresponding future graph.
        state_targets = []
        if future:
            for i in range(len(rollout_preds)):
                if i >= len(future):
                    break
                state_targets.append(
                    self.engine._encode_sequence(
                        graphs + future[: i + 1]
                    )
                )

        return rollout_loss(
            rollout_preds=rollout_preds,
            attack_targets=attack_targets,
            stage_targets=stage_targets,
            target_indices=target_indices,
            state_targets=state_targets or None,
            stage_masks=stage_mask_tensors,
            attack_pos_weight=self.attack_pos_weight,
        )

    def _forward_batch(self, batch_items: List[Dict[str, Any]]) -> torch.Tensor:
        losses = [self._forward_item(item) for item in batch_items]
        return torch.stack(losses).mean()

    def _set_train(self, enabled: bool) -> None:
        modules = (
            self.engine.graph_encoder,
            self.engine.temporal_encoder,
            self.engine.world_model,
            self.engine.attack_head,
            self.engine.stage_head,
            self.engine.target_head,
        )
        for module in modules:
            module.train(enabled)

    def train(
        self,
        train_items: List[Dict[str, Any]],
        val_items: List[Dict[str, Any]],
        epochs: int = 10,
        batch_size: int = 4,
    ) -> TrainingResult:
        if not train_items:
            raise ValueError("Training set is empty")

        train_loader = DataLoader(
            GraphSequenceDataset(train_items),
            batch_size=batch_size,
            shuffle=True,
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
        best_state = None

        for epoch in range(epochs):
            self._set_train(True)
            train_losses = []

            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_batch(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
                self.optimizer.step()
                train_losses.append(float(loss.detach().item()))

            train_loss = sum(train_losses) / max(1, len(train_losses))
            best_train_loss = min(best_train_loss, train_loss)

            self._set_train(False)
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_losses.append(float(self._forward_batch(batch).item()))
            val_loss = sum(val_losses) / max(1, len(val_losses)) if val_losses else train_loss

            epochs_ran = epoch + 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "graph_encoder": self.engine.graph_encoder.state_dict(),
                    "temporal_encoder": self.engine.temporal_encoder.state_dict(),
                    "world_model": self.engine.world_model.state_dict(),
                    "attack_head": self.engine.attack_head.state_dict(),
                    "stage_head": self.engine.stage_head.state_dict(),
                    "target_head": self.engine.target_head.state_dict(),
                }

            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
            )

            if self.early_stopping.step(val_loss):
                stopped_early = True
                break

        if best_state is not None:
            for name, state in best_state.items():
                getattr(self.engine, name).load_state_dict(state)

        if self.checkpoint_path:
            save_checkpoint(
                self.checkpoint_path,
                {
                    "format_version": 2,
                    "epochs_ran": epochs_ran,
                    "train_loss": best_train_loss,
                    "val_loss": best_val_loss,
                    "rollout_steps": self.rollout_steps,
                    "model_state": {
                        name: module.state_dict()
                        for name, module in (
                            ("graph_encoder", self.engine.graph_encoder),
                            ("temporal_encoder", self.engine.temporal_encoder),
                            ("world_model", self.engine.world_model),
                            ("attack_head", self.engine.attack_head),
                            ("stage_head", self.engine.stage_head),
                            ("target_head", self.engine.target_head),
                        )
                    },
                    "optimizer_state": self.optimizer.state_dict(),
                },
            )

        return TrainingResult(
            train_loss=best_train_loss,
            val_loss=best_val_loss,
            stopped_early=stopped_early,
            epochs_ran=epochs_ran,
        )

    def train_step(
        self,
        graphs: List[Any],
        targets: Dict[str, Any],
        current_node_embeddings: torch.Tensor | None = None,
    ) -> float:
        item = {
            "graphs": graphs,
            "targets": targets,
            "future_snapshots": [],
        }
        self._set_train(True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward_item(item)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())
