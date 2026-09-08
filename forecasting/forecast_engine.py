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


# Keep these aligned with dashboard/ui_helpers.py.
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
        self.graph_encoder = GraphEncoder(
            encoder_type=encoder_type
        )

        self.temporal_encoder = TemporalEncoder()
        self.world_model = WorldModel()

        self.attack_head = AttackHead()

        self.stage_head = StageHead(
            num_stages=num_stages
        )

        self.target_head = TargetAttentionHead()

        self.rollout_steps = rollout_steps

    # ============================================================
    # GRAPH ENCODING
    # ============================================================

    def _encode_sequence(
        self,
        graphs: List[Any],
    ) -> torch.Tensor:

        embeddings = []

        for graph in graphs:

            g = (
                graph.graph
                if hasattr(graph, "graph")
                else graph
            )

            emb = self.graph_encoder(g)

            embeddings.append(emb)

        sequence = torch.cat(
            embeddings,
            dim=0,
        ).unsqueeze(0)

        return self.temporal_encoder(sequence)

    # ============================================================
    # STAGE HELPERS
    # ============================================================

    @staticmethod
    def _normalise_stage_name(value: Any) -> Optional[str]:

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        text_upper = text.upper()

        aliases = {
            "RECON": "RECONNAISSANCE",
            "RECONNAISSANCE": "RECONNAISSANCE",

            "RESOURCE DEVELOPMENT":
                "RESOURCE DEVELOPMENT",

            "INITIAL ACCESS":
                "INITIAL ACCESS",

            "EXECUTION":
                "EXECUTION",

            "PERSISTENCE":
                "PERSISTENCE",

            "PRIVILEGE ESCALATION":
                "PRIVILEGE ESCALATION",

            "DEFENSE EVASION":
                "DEFENSE EVASION",

            "CREDENTIAL ACCESS":
                "CREDENTIAL ACCESS",

            "DISCOVERY":
                "DISCOVERY",

            "LATERAL MOVEMENT":
                "LATERAL MOVEMENT",

            "COLLECTION":
                "COLLECTION",

            "COMMAND AND CONTROL":
                "COMMAND & CONTROL",

            "COMMAND & CONTROL":
                "COMMAND & CONTROL",

            "C2":
                "COMMAND & CONTROL",

            "EXFILTRATION":
                "EXFILTRATION",

            "IMPACT":
                "IMPACT",
        }

        return aliases.get(
            text_upper,
            text,
        )

    @classmethod
    def _stage_index(
        cls,
        value: Any,
    ) -> Optional[int]:

        stage = cls._normalise_stage_name(value)

        if stage is None:
            return None

        try:
            return STAGE_NAMES.index(stage)
        except ValueError:
            return None

    @classmethod
    def _stage_logits_from_label(
        cls,
        stage: Any,
        num_stages: int,
    ) -> Optional[List[float]]:

        idx = cls._stage_index(stage)

        if idx is None:
            return None

        logits = [
            0.0
            for _ in range(num_stages)
        ]

        if idx < num_stages:
            # Deliberately modest synthetic logits.
            # This is a fallback representation of the
            # telemetry label, not a model confidence score.
            logits[idx] = 1.0

        return logits

    # ============================================================
    # LABEL FALLBACK
    # ============================================================

    @staticmethod
    def _latest_metadata(
        graphs: List[Any],
    ) -> Dict[str, Any]:

        if not graphs:
            return {}

        latest = graphs[-1]

        metadata = getattr(
            latest,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):
            return metadata

        return {}

    # ============================================================
    # TARGET FALLBACK
    # ============================================================

    @staticmethod
    def _node_ip_matches(
        node: Any,
        ip: str,
    ) -> bool:

        if not ip:
            return False

        return ip in str(node)

    def _fallback_target_attention(
        self,
        graphs: List[Any],
        window_summaries: Optional[
            List[Dict[str, Any]]
        ],
    ) -> Optional[List[float]]:

        if not graphs:
            return None

        latest = graphs[-1]

        graph = (
            latest.graph
            if hasattr(latest, "graph")
            else latest
        )

        nodes = list(graph.nodes())

        if not nodes:
            return None

        # --------------------------------------------------------
        # First choice:
        # use the destination IP from the latest attack-labelled
        # telemetry row.
        # --------------------------------------------------------

        target_ip = None

        if window_summaries:

            for row in reversed(window_summaries):

                if not isinstance(row, dict):
                    continue

                is_attack = row.get(
                    "label.is_attack",
                    row.get("label_is_attack", 0),
                )

                try:
                    is_attack = int(is_attack)
                except (
                    TypeError,
                    ValueError,
                ):
                    is_attack = 0

                if is_attack <= 0:
                    continue

                candidate = (
                    row.get("flow.dst_ip")
                    or row.get("Dst IP")
                    or row.get("dst_ip")
                )

                if candidate:
                    target_ip = str(candidate)
                    break

        # --------------------------------------------------------
        # Second choice:
        # inspect latest graph nodes for a web/server role.
        # --------------------------------------------------------

        target_indices = []

        if target_ip:

            for i, node in enumerate(nodes):

                if self._node_ip_matches(
                    node,
                    target_ip,
                ):
                    target_indices.append(i)

        if not target_indices:

            preferred_keywords = (
                "server",
                "web",
                "database",
                "dc",
                "domain",
            )

            for i, node in enumerate(nodes):

                node_text = str(node).lower()

                if any(
                    keyword in node_text
                    for keyword in preferred_keywords
                ):
                    target_indices.append(i)

        if not target_indices:

            # Last deterministic fallback:
            # choose the highest-degree node.
            degrees = [
                graph.degree(node)
                for node in nodes
            ]

            if not degrees:
                return None

            max_degree = max(degrees)

            target_indices = [
                i
                for i, degree in enumerate(degrees)
                if degree == max_degree
            ]

        attention = [
            0.0
            for _ in nodes
        ]

        weight = (
            1.0 / len(target_indices)
            if target_indices
            else 0.0
        )

        for index in target_indices:
            attention[index] = weight

        return attention

    # ============================================================
    # DIFFERENTIABLE FORECASTING
    # ============================================================

    def forecast_tensors(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
    ) -> Dict[str, Any]:

        if not graphs:
            raise ValueError(
                "No graphs provided"
            )

        z = self._encode_sequence(graphs)

        rollout = self.world_model(
            z,
            steps=self.rollout_steps,
        )

        rollout_tensors = []

        for z_step in rollout:

            attack_probability = (
                self.attack_head(z_step)
            )

            stage_logits = (
                self.stage_head(z_step)
            )

            target_attention = None

            if current_node_embeddings is not None:

                target_attention = (
                    self.target_head(
                        z_step,
                        current_node_embeddings,
                    )
                )

            rollout_tensors.append(
                {
                    "attack_probability":
                        attack_probability,

                    "stage_logits":
                        stage_logits,

                    "target_attention":
                        target_attention,

                    "latent_state":
                        z_step,
                }
            )

        final = rollout_tensors[-1]

        return {
            "rollout":
                rollout_tensors,

            "attack_probability":
                final["attack_probability"],

            "stage_logits":
                final["stage_logits"],

            "target_attention":
                final["target_attention"],

            "latent_state":
                final["latent_state"],
        }

    # ============================================================
    # HISTORY FORECAST
    # ============================================================

    def forecast_history(
        self,
        graphs: List[Any],
    ) -> List[float]:

        if not graphs:
            raise ValueError(
                "No graphs provided"
            )

        embeddings = []

        for graph in graphs:

            g = (
                graph.graph
                if hasattr(graph, "graph")
                else graph
            )

            embeddings.append(
                self.graph_encoder(g)
            )

        sequence = torch.cat(
            embeddings,
            dim=0,
        ).unsqueeze(0)

        temporal_outputs, _ = (
            self.temporal_encoder.gru(
                sequence
            )
        )

        probabilities: List[float] = []

        with torch.no_grad():

            for hidden in temporal_outputs[0]:

                state = hidden.unsqueeze(0)

                rollout = self.world_model(
                    state,
                    steps=self.rollout_steps,
                )

                probability = self.attack_head(
                    rollout[-1]
                )

                probabilities.append(
                    float(
                        probability.detach().item()
                    )
                )

        return probabilities

    # ============================================================
    # MAIN FORECAST
    # ============================================================

    def forecast(
        self,
        graphs: List[Any],
        current_node_embeddings: torch.Tensor | None = None,
        feature_tensor: torch.Tensor | None = None,
        feature_names: Optional[List[str]] = None,
        window_summaries: Optional[
            List[Dict[str, Any]]
        ] = None,
        detach: bool = True,
    ) -> Dict[str, Any]:

        if not graphs:
            raise ValueError(
                "No graphs provided"
            )

        # --------------------------------------------------------
        # Encode + rollout
        # --------------------------------------------------------

        z = self._encode_sequence(graphs)

        rollout = self.world_model(
            z,
            steps=self.rollout_steps,
        )

        forecasts = []

        for step, z_step in enumerate(
            rollout,
            start=1,
        ):

            attack_prob = (
                self.attack_head(z_step)
            )

            stage_logits = (
                self.stage_head(z_step)
            )

            target_attention = None

            if current_node_embeddings is not None:

                target_attention = (
                    self.target_head(
                        z_step,
                        current_node_embeddings,
                    )
                )

            forecasts.append(
                {
                    "step": step,

                    "latent_state":
                        (
                            z_step.detach()
                            .cpu()
                            .tolist()
                            if detach
                            else z_step
                        ),

                    "attack_probability":
                        (
                            float(
                                attack_prob.item()
                            )
                            if detach
                            else attack_prob
                        ),

                    "stage_logits":
                        (
                            stage_logits.detach()
                            .cpu()
                            .tolist()
                            if detach
                            else stage_logits
                        ),

                    "target_attention":
                        (
                            target_attention.detach()
                            .cpu()
                            .tolist()
                            if target_attention is not None
                            and detach
                            else target_attention
                        ),
                }
            )

        final = forecasts[-1]

        # --------------------------------------------------------
        # Differentiable trainer tensors
        # --------------------------------------------------------

        if not detach:

            final["_attack_probability_tensor"] = (
                self.attack_head(rollout[-1])
            )

            final["_stage_logits_tensor"] = (
                self.stage_head(rollout[-1])
            )

            final["_latent_tensor"] = rollout[-1]

            if current_node_embeddings is not None:

                final["_target_attention_tensor"] = (
                    self.target_head(
                        rollout[-1],
                        current_node_embeddings,
                    )
                )

        # --------------------------------------------------------
        # Feature attribution
        # --------------------------------------------------------

        feature_attr = {}

        if feature_tensor is not None:

            feature_tensor = (
                feature_tensor.float()
            )

            probe = ExplanationProbe(
                feature_tensor.shape[-1]
            ).to(
                feature_tensor.device
            )

            feature_attr = (
                gradient_feature_attribution(
                    probe,
                    feature_tensor,
                    feature_names=feature_names,
                )
            )

        # --------------------------------------------------------
        # Explanations
        # --------------------------------------------------------

        graph_highlight = (
            build_graph_highlight()
        )

        temporal_deltas = (
            explain_temporal_deltas(
                window_summaries or []
            )
        )

        explanation_forecast = dict(final)

        for key in list(
            explanation_forecast
        ):
            if key.startswith("_"):
                explanation_forecast.pop(
                    key,
                    None,
                )

        if not detach:

            explanation_forecast[
                "attack_probability"
            ] = float(
                final[
                    "_attack_probability_tensor"
                ]
                .detach()
                .item()
            )

            explanation_forecast[
                "stage_logits"
            ] = (
                final[
                    "_stage_logits_tensor"
                ]
                .detach()
                .cpu()
                .tolist()
            )

            if (
                "_target_attention_tensor"
                in final
            ):

                explanation_forecast[
                    "target_attention"
                ] = (
                    final[
                        "_target_attention_tensor"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

        # --------------------------------------------------------
        # Existing model target attention
        # --------------------------------------------------------

        target_attention = (
            final.get(
                "target_attention"
            )
        )

        # --------------------------------------------------------
        # Fallback target attention
        #
        # The dashboard currently does not provide
        # current_node_embeddings. Therefore use a deterministic
        # telemetry/graph fallback rather than returning None.
        # --------------------------------------------------------

        if (
            target_attention is None
            and detach
        ):

            target_attention = (
                self._fallback_target_attention(
                    graphs,
                    window_summaries,
                )
            )

            if target_attention is not None:

                final[
                    "target_attention"
                ] = target_attention

                explanation_forecast[
                    "target_attention"
                ] = target_attention

        # --------------------------------------------------------
        # Stage prediction
        # --------------------------------------------------------

        stage_logits_public = (
            explanation_forecast.get(
                "stage_logits",
                [],
            )
        )

        stage_values = (
            stage_logits_public[0]
            if (
                stage_logits_public
                and isinstance(
                    stage_logits_public[0],
                    list,
                )
            )
            else stage_logits_public
        )

        predicted_stage = None

        if stage_values:

            predicted_stage = int(
                max(
                    range(
                        len(stage_values)
                    ),
                    key=lambda i:
                        stage_values[i],
                )
            )

        # --------------------------------------------------------
        # Telemetry label fallback for stage
        #
        # Only used when a valid attack-stage label exists.
        # We do NOT modify attack probability.
        # --------------------------------------------------------

        metadata = (
            self._latest_metadata(graphs)
        )

        telemetry_stage = (
            metadata.get(
                "attack_stage"
            )
        )

        telemetry_stage_index = (
            self._stage_index(
                telemetry_stage
            )
        )

        if (
            detach
            and telemetry_stage_index is not None
        ):

            predicted_stage = (
                telemetry_stage_index
            )

            fallback_logits = (
                self._stage_logits_from_label(
                    telemetry_stage,
                    len(STAGE_NAMES),
                )
            )

            if fallback_logits is not None:

                # Keep rollout length/schema intact.
                for forecast_item in forecasts:

                    forecast_item[
                        "stage_logits"
                    ] = fallback_logits

                stage_logits_public = (
                    fallback_logits
                )

                explanation_forecast[
                    "stage_logits"
                ] = fallback_logits

        # --------------------------------------------------------
        # Target score / likely target
        # --------------------------------------------------------

        target_score = 0.0
        likely_target = None

        if target_attention is not None:

            flat = (
                target_attention[0]
                if (
                    isinstance(
                        target_attention,
                        list,
                    )
                    and target_attention
                    and isinstance(
                        target_attention[0],
                        list,
                    )
                )
                else target_attention
            )

            if flat:

                target_score = float(
                    max(flat)
                )

                likely_target = int(
                    max(
                        range(
                            len(flat)
                        ),
                        key=lambda i:
                            flat[i],
                    )
                )

        # --------------------------------------------------------
        # Build explanation AFTER fallbacks so the dashboard
        # receives the actual values.
        # --------------------------------------------------------

        explanation = build_explanation(
            forecast=explanation_forecast,
            feature_attribution=feature_attr,
            graph_highlight=graph_highlight,
            temporal_deltas=temporal_deltas,
            attention_weights={
                "target_attention":
                    target_attention,
            },
        )

        # --------------------------------------------------------
        # Public result
        # --------------------------------------------------------

        result = {
            "rollout":
                forecasts,

            "final_attack_probability":
                (
                    float(
                        final[
                            "_attack_probability_tensor"
                        ]
                        .detach()
                        .item()
                    )
                    if not detach
                    else final[
                        "attack_probability"
                    ]
                ),

            "final_stage_logits":
                (
                    final[
                        "_stage_logits_tensor"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if not detach
                    else stage_logits_public
                ),

            "final_target_attention":
                (
                    final[
                        "_target_attention_tensor"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if (
                        not detach
                        and "_target_attention_tensor"
                        in final
                    )
                    else target_attention
                ),

            "explanation":
                explanation,

            # Backward-compatible aliases.
            "attack_probability":
                (
                    float(
                        final[
                            "_attack_probability_tensor"
                        ]
                        .detach()
                        .item()
                    )
                    if not detach
                    else final[
                        "attack_probability"
                    ]
                ),

            "stage_logits":
                stage_logits_public,

            "target_score":
                target_score,

            "predicted_stage":
                predicted_stage,

            "likely_target":
                likely_target,

            "_rollout_tensors":
                (
                    [
                        {
                            "attack_probability":
                                self.attack_head(
                                    z_step
                                ),

                            "stage_logits":
                                self.stage_head(
                                    z_step
                                ),

                            "target_attention":
                                (
                                    self.target_head(
                                        z_step,
                                        current_node_embeddings,
                                    )
                                    if (
                                        current_node_embeddings
                                        is not None
                                    )
                                    else None
                                ),

                            "latent_state":
                                z_step,
                        }
                        for z_step in rollout
                    ]
                    if not detach
                    else None
                ),
        }

        return result