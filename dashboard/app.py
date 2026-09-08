from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.feature_extractor import extractFeatures
from preprocessing.windowing import add_time_windows
from graph import build_temporal_graphs
from forecasting import ForecastEngine
from policy import RulePolicy, PolicyLogger
from deceptra import DecoyManager, divert_session, SessionCapture, ImmutableDiversionLog
from feedback import ImprovementLog
from evaluation import (
    evaluate_detection,
    reliability_diagram,
    containment_rate,
    intel_yield,
    false_divert_rate,
    analyst_override_rate,
    decoy_dwell_time,
    detection_evasion_rate,
    feedback_loop_latency,
)

from dashboard.ui_helpers import (
    show_metric_cards,
    show_attack_timeline,
    show_stage_transition,
    show_target_ranking,
    show_explainability,
    show_graph_visualization,
    show_reliability_diagram,
    show_decision_panel,
    show_session_replay,
    show_attention_details,
    show_graph_highlight_json,
)

st.set_page_config(page_title="NetWorld + Deceptra", layout="wide")
st.title("NetWorld + Deceptra Demo Dashboard")

st.sidebar.header("Controls")
window_seconds = st.sidebar.selectbox("Window size (seconds)", [1, 5, 10, 30], index=1)
human_mode = st.sidebar.selectbox("Policy mode", ["advisory", "semi_autonomous", "autonomous"], index=0)
show_overlay_json = st.sidebar.checkbox("Show raw graph highlight JSON", value=False)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

policy = RulePolicy("policy/policy_config.yaml")
policy_logger = PolicyLogger("logs/policy_log.json")
improvement_log = ImprovementLog("logs/improvement_log.json")
diversion_log = ImmutableDiversionLog("logs/diversion.log")
capture = SessionCapture()
decoy_manager = DecoyManager()
forecast_engine = ForecastEngine(rollout_steps=3)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    raw_df = pd.read_csv(tmp_path)
    st.subheader("Raw Data")
    st.dataframe(raw_df.head(20), use_container_width=True)

    canonical_df = extractFeatures(
        raw_df,
        source_file=Path(uploaded.name).name,
        dataset_name="CIC-IDS2018",
    )
    windowed_df = add_time_windows(canonical_df, window_seconds=window_seconds)
    sequence = build_temporal_graphs(windowed_df)

    st.subheader("Canonical Data")
    st.dataframe(canonical_df.head(20), use_container_width=True)

    st.subheader("Windowed Data")
    st.dataframe(windowed_df.head(20), use_container_width=True)

    st.subheader("Temporal Graphs")
    st.write(f"Snapshots: {len(sequence)}")
    latest = sequence.latest()

    feature_tensor = None
    feature_names = None
    if len(windowed_df.columns) >= 8:
        preferred_features = [
            "flow.duration",
            "flow.total_fwd_packets",
            "flow.total_bwd_packets",
            "flow.flow_bytes_per_sec",
            "flow.flow_packets_per_sec",
            "packet.ttl_mean",
            "packet.payload_size_mean",
            "behaviour.auth_failures",
        ]

        available = [c for c in preferred_features if c in windowed_df.columns]
        if available:
            feature_names = available
            feature_tensor = torch.tensor(
                windowed_df[available].fillna(0).iloc[0:1].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
        else:
            feature_names = []
            feature_tensor = None

    forecast = forecast_engine.forecast(
        [snap for snap in sequence.snapshots],
        feature_tensor=feature_tensor,
        feature_names=feature_names,
        window_summaries=windowed_df.tail(3).to_dict(orient="records"),
    )

    explanation = forecast["explanation"]
    graph_highlight = explanation.get("graph_highlight", {})
    attention_weights = explanation.get("attention_weights", {})
    target_attention = attention_weights.get("target_attention")

    node_attention = {}
    if target_attention and latest is not None:
        # target_attention may be nested [batch][nodes]
        if isinstance(target_attention, list) and target_attention:
            maybe_first = target_attention[0]
            if isinstance(maybe_first, list):
                target_attention = maybe_first
        if latest is not None and isinstance(target_attention, list):
            nodes = list(latest.graph.nodes())
            node_attention = {
                nodes[i]: float(target_attention[i])
                for i in range(min(len(nodes), len(target_attention)))
            }

    if latest is not None:
        show_graph_visualization(
            latest.graph,
            node_attention=node_attention,
            graph_highlight=graph_highlight,
        )
        show_attention_details(node_attention, top_k=10)
        if show_overlay_json:
            show_graph_highlight_json(graph_highlight)

    rollout = forecast["rollout"]
    attack_timeline = [
        {"timestamp": f"step-{p['step']}", "attack_probability": p["attack_probability"]}
        for p in rollout
    ]
    stage_events = [{"step": p["step"], "stage_logits": p["stage_logits"]} for p in rollout]

    st.subheader("Forecast")
    show_metric_cards({
        "attack_probability": forecast["final_attack_probability"],
        "auroc": 0.0,
        "auprc": 0.0,
        "fpr": 0.0,
    })

    show_explainability(explanation)
    show_attack_timeline(attack_timeline)
    show_stage_transition(stage_events)

    if target_attention is not None and latest is not None:
        show_target_ranking(
            target_attention[0] if isinstance(target_attention, list) and target_attention and isinstance(target_attention[0], list) else target_attention,
            list(latest.graph.nodes()),
        )

    rel = reliability_diagram([forecast["final_attack_probability"]], [1])
    show_reliability_diagram(rel)

    asset_criticality = int(windowed_df["host.asset_criticality"].fillna(1).iloc[-1]) if "host.asset_criticality" in windowed_df.columns else 1
    stage_confidence = 0.7
    soc_load = 0.3
    decoy_capacity = 1.0

    decision = policy.decide(
        {
            "attack_probability": forecast["final_attack_probability"],
            "stage_logits": forecast["final_stage_logits"],
            "target_score": 0.0,
        },
        asset_criticality=asset_criticality,
        stage_confidence=stage_confidence,
        soc_load=soc_load,
        decoy_capacity=decoy_capacity,
        human_mode=human_mode,
    )

    approved = show_decision_panel(
        {"action": decision.action, "reason": decision.reason},
        require_approval=decision.requires_approval,
    )

    policy_logger.append({
        "action": decision.action,
        "score": decision.score,
        "reason": decision.reason,
        "approved": approved,
        "metadata": decision.metadata,
    })

    if approved and decision.action == "divert":
        decoy = decoy_manager.register_decoy("server-03-clone", "server", "10.0.9.0/24", "Linux")
        diversion = divert_session(
            source=str(windowed_df.iloc[-1].get("flow.src_ip", "unknown")),
            original_target="real-target",
            decoy_target=decoy.decoy_id,
            reason=decision.reason,
            method="iptables",
            source_ip=str(windowed_df.iloc[-1].get("flow.src_ip", "10.0.0.1")),
            original_target_ip="10.0.0.2",
            decoy_target_ip="10.0.9.2",
            dst_port=443,
        )

        diversion_log.append({
            "action": "divert",
            "source": diversion.source,
            "original_target": diversion.original_target,
            "decoy_target": diversion.decoy_target,
            "reason": diversion.reason,
            "method": diversion.method,
            "commands": diversion.commands,
        })

        sess = capture.start_session("sess-1", diversion.source, diversion.decoy_target)
        capture.record_event("sess-1", {"event": "diverted"})
        capture.record_event("sess-1", {"event": "session_started"})
        show_session_replay(sess.events)

    improvement_log.append({
        "attack_probability": forecast["final_attack_probability"],
        "decision": decision.action,
        "approved": approved,
        "reliability": rel,
        "containment_rate": containment_rate(1 if approved and decision.action in {"divert", "isolate"} else 0, 1),
        "intel_yield": intel_yield(1 if decision.action == "divert" else 0, 1 if decision.action == "divert" else 0),
        "false_divert_rate": false_divert_rate(0, 1),
        "analyst_override_rate": analyst_override_rate(0 if approved else 1, 1),
        "decoy_dwell_time": decoy_dwell_time(10.0, 1),
        "detection_evasion_rate": detection_evasion_rate(0, 1),
        "feedback_loop_latency": feedback_loop_latency(0.0),
    })
else:
    st.info("Upload a CSV file to run the pipeline.")