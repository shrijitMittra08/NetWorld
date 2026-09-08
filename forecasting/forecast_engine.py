from __future__ import annotations

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

class ExplanationProbe(nn.Module):
    """
    Tiny tabular probe used only for feature attribution.
    It maps raw feature tensors to a scalar probability.
    """

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
    def __init__(
        self,
        encoder_type: str = "sage",
        num_stages: int = 14,
        rollout_steps: int = 3,
    ):
        self.graph_encoder = GraphEncoder(encoder_type=encoder_type)
        self.temporal_encoder = TemporalEncoder()
        self.world_model = WorldModel()
        self.attack_head = AttackHead()
        self.stage_head = StageHead(num_stages=num_stages)
        self.target_head = TargetAttentionHead()
        self.rollout_steps = rollout_steps

    def _encode_sequence(self, graphs: List[Any]) -> torch.Tensor:
        embeddings = []
        for graph in graphs:
            g = graph.graph if hasattr(graph, "graph") else graph
            emb = self.graph_encoder(g)
            embeddings.append(emb)

        sequence = torch.cat(embeddings, dim=0).unsqueeze(0)
        return self.temporal_encoder(sequence)

    def forecast(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
        feature_tensor: torch.Tensor | None = None,
        feature_names: Optional[List[str]] = None,
        window_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not graphs:
            raise ValueError("No graphs provided")

        z = self._encode_sequence(graphs)
        rollout = self.world_model(z, steps=self.rollout_steps)

        forecasts = []
        for step, z_step in enumerate(rollout, start=1):
            attack_prob = self.attack_head(z_step)
            stage_logits = self.stage_head(z_step)

            target_attention = None
            if current_node_embeddings is not None:
                target_attention = self.target_head(z_step, current_node_embeddings)

            forecasts.append(
                {
                    "step": step,
                    "latent_state": z_step.detach().cpu().tolist(),
                    "attack_probability": float(attack_prob.item()),
                    "stage_logits": stage_logits.detach().cpu().tolist(),
                    "target_attention": (
                        target_attention.detach().cpu().tolist()
                        if target_attention is not None
                        else None
                    ),
                }
            )

        final = forecasts[-1]

        feature_attr = {}
        if feature_tensor is not None:
            feature_tensor = feature_tensor.float()
            probe = ExplanationProbe(feature_tensor.shape[-1]).to(feature_tensor.device)
            feature_attr = gradient_feature_attribution(
                probe,
                feature_tensor,
                feature_names=feature_names,
            )

        graph_highlight = build_graph_highlight()
        temporal_deltas = explain_temporal_deltas(window_summaries or [])

        explanation = build_explanation(
            forecast=final,
            feature_attribution=feature_attr,
            graph_highlight=graph_highlight,
            temporal_deltas=temporal_deltas,
            attention_weights={
                "target_attention": final.get("target_attention"),
            },
        )

        return {
            "rollout": forecasts,
            "final_attack_probability": final["attack_probability"],
            "final_stage_logits": final["stage_logits"],
            "final_target_attention": final["target_attention"],
            "explanation": explanation,
        }