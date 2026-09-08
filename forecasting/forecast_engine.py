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

    def forecast_tensors(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        """
        Differentiable forecasting path for training.
    
        Unlike forecast(), this method never converts tensors to Python
        scalars/lists and never detaches them from the autograd graph.
        """
        if not graphs:
            raise ValueError("No graphs provided")
    
        z = self._encode_sequence(graphs)
    
        rollout = self.world_model(
            z,
            steps=self.rollout_steps,
        )
    
        rollout_tensors = []
    
        for z_step in rollout:
            attack_probability = self.attack_head(z_step)
            stage_logits = self.stage_head(z_step)
    
            target_attention = None
    
            if current_node_embeddings is not None:
                target_attention = self.target_head(
                    z_step,
                    current_node_embeddings,
                )
    
            rollout_tensors.append(
                {
                    "attack_probability": attack_probability,
                    "stage_logits": stage_logits,
                    "target_attention": target_attention,
                    "latent_state": z_step,
                }
            )
    
        final = rollout_tensors[-1]
    
        return {
            "rollout": rollout_tensors,
            "attack_probability": final["attack_probability"],
            "stage_logits": final["stage_logits"],
            "target_attention": final["target_attention"],
            "latent_state": final["latent_state"],
        }

    def forecast_history(self, graphs: List[Any]) -> List[float]:
        """Return one attack probability per observed graph snapshot efficiently.

        This avoids re-encoding every prefix of the sequence. The temporal GRU
        is run once over the complete sequence, then each timestep hidden state
        is passed through the same world-model rollout used by ``forecast``.
        This is intended for dashboard evaluation/history, not training.
        """
        if not graphs:
            raise ValueError("No graphs provided")

        embeddings = []
        for graph in graphs:
            g = graph.graph if hasattr(graph, "graph") else graph
            embeddings.append(self.graph_encoder(g))

        sequence = torch.cat(embeddings, dim=0).unsqueeze(0)
        temporal_outputs, _ = self.temporal_encoder.gru(sequence)

        probabilities: List[float] = []
        with torch.no_grad():
            for hidden in temporal_outputs[0]:
                state = hidden.unsqueeze(0)
                rollout = self.world_model(state, steps=self.rollout_steps)
                probability = self.attack_head(rollout[-1])
                probabilities.append(float(probability.detach().item()))

        return probabilities

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
                    "latent_state": z_step.detach().cpu().tolist() if detach else z_step,
                    "attack_probability": float(attack_prob.item()) if detach else attack_prob,
                    "stage_logits": stage_logits.detach().cpu().tolist() if detach else stage_logits,
                    "target_attention": (
                        (target_attention.detach().cpu().tolist() if detach else target_attention)
                        if target_attention is not None else None
                    ),
                }
            )

        final = forecasts[-1]
        if not detach:
            # Keep differentiable tensors for the trainer; public forecast output stays JSON-safe by default.
            final["_attack_probability_tensor"] = self.attack_head(rollout[-1])
            final["_stage_logits_tensor"] = self.stage_head(rollout[-1])
            final["_latent_tensor"] = rollout[-1]
            if current_node_embeddings is not None:
                final["_target_attention_tensor"] = self.target_head(rollout[-1], current_node_embeddings)

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

        explanation_forecast = dict(final)
        for _k in list(explanation_forecast):
            if _k.startswith("_"):
                explanation_forecast.pop(_k, None)
        if not detach:
            explanation_forecast["attack_probability"] = float(final["_attack_probability_tensor"].detach().item())
            explanation_forecast["stage_logits"] = final["_stage_logits_tensor"].detach().cpu().tolist()
            if "_target_attention_tensor" in final:
                explanation_forecast["target_attention"] = final["_target_attention_tensor"].detach().cpu().tolist()

        explanation = build_explanation(
            forecast=explanation_forecast,
            feature_attribution=feature_attr,
            graph_highlight=graph_highlight,
            temporal_deltas=temporal_deltas,
            attention_weights={
                "target_attention": final.get("target_attention"),
            },
        )

        target_attention = final.get("target_attention")
        target_score = 0.0
        likely_target = None
        if target_attention is not None:
            flat = target_attention[0] if isinstance(target_attention, list) and target_attention and isinstance(target_attention[0], list) else target_attention
            if flat:
                target_score = float(max(flat))
                likely_target = int(max(range(len(flat)), key=lambda i: flat[i]))
        stage_logits_public = explanation_forecast.get("stage_logits", [])
        stage_values = stage_logits_public[0] if stage_logits_public and isinstance(stage_logits_public[0], list) else stage_logits_public
        predicted_stage = int(max(range(len(stage_values)), key=lambda i: stage_values[i])) if stage_values else None

        result = {
            "rollout": forecasts,
            "final_attack_probability": float(final["_attack_probability_tensor"].detach().item()) if not detach else final["attack_probability"],
            "final_stage_logits": final["_stage_logits_tensor"].detach().cpu().tolist() if not detach else final["stage_logits"],
            "final_target_attention": (final["_target_attention_tensor"].detach().cpu().tolist() if not detach and "_target_attention_tensor" in final else final["target_attention"]),
            "explanation": explanation,
            # Backward-compatible aliases consumed by the policy/demo/tests.
            "attack_probability": float(final["_attack_probability_tensor"].detach().item()) if not detach else final["attack_probability"],
            "stage_logits": stage_logits_public,
            "target_score": target_score,
            "predicted_stage": predicted_stage,
            "likely_target": likely_target,
            "_rollout_tensors": [
                {
                    "attack_probability": self.attack_head(z_step),
                    "stage_logits": self.stage_head(z_step),
                    "target_attention": self.target_head(z_step, current_node_embeddings) if current_node_embeddings is not None else None,
                    "latent_state": z_step,
                }
                for z_step in rollout
            ] if not detach else None,
        }
        return result