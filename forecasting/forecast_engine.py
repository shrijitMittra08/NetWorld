from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models import (
    GraphEncoder,
    TemporalEncoder,
    WorldModel,
    AttackHead,
    StageHead,
    TargetAttentionHead,
)
from explainability import (
    gradient_feature_attribution,
    build_graph_highlight,
    explain_temporal_deltas,
    build_explanation,
)


STAGE_NAMES = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command & Control",
    "Exfiltration",
    "Impact",
]


class ExplanationProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForecastEngine:
    """NetWorld inference stack.

    The engine owns the trainable GraphSAGE/GAT encoder, temporal GRU,
    latent world model and attack/stage/target heads. A checkpoint produced by
    train_cicids2018.py can be loaded at construction time.
    """

    def __init__(
        self,
        encoder_type: str = "sage",
        num_stages: int = 14,
        rollout_steps: int = 3,
        checkpoint_path: str | Path | None = None,
    ):
        self.graph_encoder = GraphEncoder(encoder_type=encoder_type)
        self.temporal_encoder = TemporalEncoder()
        self.world_model = WorldModel()
        self.attack_head = AttackHead()
        self.stage_head = StageHead(num_stages=num_stages)
        self.target_head = TargetAttentionHead()
        self.rollout_steps = rollout_steps
        self.checkpoint_loaded = False
        self.checkpoint_path: Optional[str] = None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, strict=False)

    # ------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = False,
    ) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)

        modules = {
            "graph_encoder": self.graph_encoder,
            "temporal_encoder": self.temporal_encoder,
            "world_model": self.world_model,
            "attack_head": self.attack_head,
            "stage_head": self.stage_head,
            "target_head": self.target_head,
        }

        loaded_any = False
        for name, module in modules.items():
            module_state = state.get(name)
            if module_state is None:
                continue
            module.load_state_dict(module_state, strict=strict)
            loaded_any = True

        if loaded_any:
            self.checkpoint_loaded = True
            self.checkpoint_path = str(path)
        return loaded_any

    # ------------------------------------------------------------
    # Graph encoding
    # ------------------------------------------------------------

    @staticmethod
    def _graph_object(snapshot: Any) -> Any:
        return snapshot.graph if hasattr(snapshot, "graph") else snapshot

    def encode_snapshot(self, snapshot: Any) -> torch.Tensor:
        return self.graph_encoder(self._graph_object(snapshot))

    def node_embeddings(self, snapshot: Any) -> torch.Tensor:
        embeddings, _ = self.graph_encoder.node_embeddings(
            self._graph_object(snapshot)
        )
        return embeddings

    def node_names(self, snapshot: Any) -> List[str]:
        graph = self._graph_object(snapshot)
        return list(graph.nodes())

    def _encode_sequence(self, graphs: List[Any]) -> torch.Tensor:
        embeddings = [self.encode_snapshot(g) for g in graphs]
        if not embeddings:
            raise ValueError("No graphs provided")
        sequence = torch.cat(embeddings, dim=0).unsqueeze(0)
        return self.temporal_encoder(sequence)

    def _target_mask(self, snapshot: Any, node_count: int) -> torch.Tensor:
        """Build a victim-only attention mask aligned with graph node order."""
        graph = self._graph_object(snapshot)
        nodes = list(graph.nodes())
        allowed = []

        explicit_targets = set(snapshot.metadata.get("target_candidate_nodes") or [])
        attacker_nodes = set(snapshot.metadata.get("attacker_nodes") or [])

        if explicit_targets:
            allowed = [n in explicit_targets and n not in attacker_nodes for n in nodes]
        else:
            # Never rank an explicitly identified attacker as a victim.
            allowed = [n not in attacker_nodes for n in nodes]

        # If metadata is unavailable, fall back to node attributes.
        if not any(allowed):
            allowed = [
                not bool(graph.nodes[n].get("is_attacker", False))
                for n in nodes
            ]

        # Last-resort safety: don't return an all-masked softmax.
        if not any(allowed):
            allowed = [True] * node_count

        return torch.tensor(allowed[:node_count], dtype=torch.bool)

    @staticmethod
    def _mask_attention(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if scores is None or mask is None:
            return scores
        mask = mask.to(device=scores.device)
        if mask.numel() != scores.shape[-1]:
            mask = mask[:scores.shape[-1]]
        if not bool(mask.any()):
            return torch.softmax(scores, dim=-1)
        masked = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        return torch.softmax(masked, dim=-1)

    # ------------------------------------------------------------
    # Training-compatible forward pass
    # ------------------------------------------------------------

    def forecast_tensors(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        z = self._encode_sequence(graphs)
        rollout_states = self.world_model(
            z,
            steps=self.rollout_steps,
        )

        rollout = []
        for state in rollout_states:
            attack_probability = self.attack_head(state)
            stage_logits = self.stage_head(state)
            target_attention = (
                self.target_head(state, current_node_embeddings)
                if current_node_embeddings is not None
                else None
            )
            if target_attention is not None:
                target_attention = self._mask_attention(
                    torch.log(target_attention.clamp_min(1e-12)),
                    self._target_mask(graphs[-1], target_attention.shape[-1]),
                )
            rollout.append({
                "attack_probability": attack_probability,
                "stage_logits": stage_logits,
                "target_attention": target_attention,
                "latent_state": state,
            })

        return {"rollout": rollout}

    # ------------------------------------------------------------
    # Public forecast
    # ------------------------------------------------------------

    def forecast(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
        feature_tensor: torch.Tensor | None = None,
        feature_names: Optional[List[str]] = None,
        window_summaries: Optional[List[Dict[str, Any]]] = None,
        detach: bool = True,
    ) -> Dict[str, Any]:
        if not graphs:
            raise ValueError("No graphs provided")

        # Training calls need graph gradients. Dashboard inference is detached.
        context = torch.enable_grad() if not detach else torch.no_grad()
        with context:
            z = self._encode_sequence(graphs)
            rollout_states = self.world_model(
                z,
                steps=self.rollout_steps,
            )

            rollout_public = []
            rollout_tensors = []
            for step, state in enumerate(rollout_states, start=1):
                attack_probability = self.attack_head(state)
                stage_logits = self.stage_head(state)
                target_attention = (
                    self.target_head(state, current_node_embeddings)
                    if current_node_embeddings is not None
                    else None
                )
                if target_attention is not None:
                    target_attention = self._mask_attention(
                        torch.log(target_attention.clamp_min(1e-12)),
                        self._target_mask(graphs[-1], target_attention.shape[-1]),
                    )
                rollout_tensors.append({
                    "attack_probability": attack_probability,
                    "stage_logits": stage_logits,
                    "target_attention": target_attention,
                    "latent_state": state,
                })
                rollout_public.append({
                    "step": step,
                    "attack_probability": float(attack_probability.detach().item()),
                    "stage_logits": stage_logits.detach().cpu().tolist(),
                    "target_attention": (
                        target_attention.detach().cpu().tolist()
                        if target_attention is not None else None
                    ),
                })

        final_tensor = rollout_tensors[-1]
        final_probability = float(
            final_tensor["attack_probability"].detach().item()
        )
        final_stage_logits = (
            final_tensor["stage_logits"].detach().cpu().tolist()
        )
        final_target_attention = (
            final_tensor["target_attention"].detach().cpu().tolist()
            if final_tensor["target_attention"] is not None else None
        )

        stage_values = final_stage_logits[0] if final_stage_logits and isinstance(final_stage_logits[0], list) else final_stage_logits
        predicted_stage = (
            int(max(range(len(stage_values)), key=lambda i: stage_values[i]))
            if stage_values else None
        )

        target_attention = final_target_attention
        if target_attention is not None and target_attention and isinstance(target_attention[0], list):
            target_attention = target_attention[0]

        likely_target = None
        target_score = 0.0
        if target_attention:
            likely_target = int(max(range(len(target_attention)), key=lambda i: target_attention[i]))
            target_score = float(target_attention[likely_target])

        # Feature attribution remains explanation-only; it does not alter predictions.
        feature_attr: Dict[str, float] = {}
        if feature_tensor is not None and feature_tensor.numel() > 0:
            probe = ExplanationProbe(feature_tensor.shape[-1])
            feature_attr = gradient_feature_attribution(
                probe,
                feature_tensor.detach().clone(),
                feature_names=feature_names,
            )

        node_attention: Dict[str, float] = {}
        if target_attention is not None:
            names = self.node_names(graphs[-1])
            node_attention = {
                names[i]: float(target_attention[i])
                for i in range(min(len(names), len(target_attention)))
            }

        graph_highlight = build_graph_highlight(
            node_attention=node_attention
        )
        temporal_deltas = explain_temporal_deltas(
            window_summaries or []
        )

        target_name = None
        if likely_target is not None:
            names = self.node_names(graphs[-1])
            if 0 <= likely_target < len(names):
                target_name = names[likely_target]

        explanation = build_explanation(
            forecast={
                "attack_probability": final_probability,
                "final_attack_probability": final_probability,
                "predicted_stage": predicted_stage,
                "stage": STAGE_NAMES[predicted_stage] if predicted_stage is not None and predicted_stage < len(STAGE_NAMES) else "unknown",
                "likely_target": likely_target,
                "target": target_name or "unknown",
            },
            feature_attribution=feature_attr,
            graph_highlight=graph_highlight,
            temporal_deltas=temporal_deltas,
            attention_weights={
                "target_attention": target_attention,
            },
        )

        return {
            "rollout": rollout_public,
            "final_attack_probability": final_probability,
            "final_stage_logits": final_stage_logits,
            "final_target_attention": final_target_attention,
            "attack_probability": final_probability,
            "stage_logits": final_stage_logits,
            "predicted_stage": predicted_stage,
            "likely_target": likely_target,
            "target_score": target_score,
            "explanation": explanation,
            "_rollout_tensors": rollout_tensors if not detach else None,
        }

    # ------------------------------------------------------------
    # History evaluation
    # ------------------------------------------------------------

    def forecast_history(self, graphs: List[Any]) -> List[float]:
        """Score every observed snapshot in one temporal pass."""
        if not graphs:
            return []

        self.graph_encoder.eval()
        self.temporal_encoder.eval()
        self.world_model.eval()
        self.attack_head.eval()

        with torch.no_grad():
            encoded = [self.encode_snapshot(g) for g in graphs]
            sequence = torch.cat(encoded, dim=0).unsqueeze(0)
            temporal_outputs, _ = self.temporal_encoder.gru(sequence)

            probabilities: List[float] = []
            for hidden in temporal_outputs[0]:
                z = hidden.unsqueeze(0)
                future = self.world_model(z, steps=self.rollout_steps)[-1]
                probabilities.append(float(self.attack_head(future).item()))
            return probabilities

