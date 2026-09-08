from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st
import torch

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
from evaluation import evaluate_detection, reliability_diagram, containment_rate, intel_yield, false_divert_rate, analyst_override_rate, decoy_dwell_time, detection_evasion_rate, feedback_loop_latency
from dashboard.ui_helpers import inject_theme, show_hero, show_flow, show_metric_cards, show_forecast_summary, show_attack_timeline, show_stage_transition, show_target_ranking, show_explainability, show_graph_visualization, show_reliability_diagram, show_decision_panel, show_session_replay, show_attention_details, show_graph_highlight_json

st.set_page_config(page_title="NetWorld + Deceptra", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
inject_theme()
show_hero("Forecast the attack. Deceive it. Learn from it.", "A proactive SOC workflow built around temporal network forecasting and closed-loop deception.", "Interactive upload")
show_flow()

st.sidebar.header("Scenario controls")
window_seconds = st.sidebar.selectbox("Time window", [1, 5, 10, 30], index=1)
human_mode = st.sidebar.selectbox("Policy mode", ["advisory", "semi_autonomous", "autonomous"], index=0, format_func=lambda x: x.replace("_", " ").title())
show_overlay_json = st.sidebar.checkbox("Developer details", value=False)

uploaded = st.file_uploader("Upload network telemetry (CSV)", type=["csv"], help="CIC-style flow data or a CSV using the canonical field names/aliases.")

policy = RulePolicy(ROOT / "policy" / "policy_config.yaml")
policy_logger = PolicyLogger(ROOT / "logs" / "policy_log.json")
improvement_log = ImprovementLog(ROOT / "logs" / "improvement_log.json")
diversion_log = ImmutableDiversionLog(ROOT / "logs" / "diversion.log")
capture = SessionCapture()
decoy_manager = DecoyManager()
@st.cache_resource
def get_forecast_engine():
    return ForecastEngine(rollout_steps=3)

forecast_engine = get_forecast_engine()

if uploaded is None:
    st.markdown("### How the demo reads")
    cols = st.columns(3)
    cols[0].markdown("**01 · Predict**\n\nTemporal graphs estimate what the attacker is likely to do next.")
    cols[1].markdown("**02 · Decide**\n\nThe policy engine chooses monitor, divert, or isolate with an auditable reason.")
    cols[2].markdown("**03 · Learn**\n\nDeceptra sessions become high-confidence feedback for future training.")
    st.info("Upload a CSV to activate the live pipeline. Metrics remain unavailable until the uploaded data contains both benign and attack labels.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

raw_df = pd.read_csv(tmp_path, low_memory=False)
canonical_df = extractFeatures(raw_df, source_file=Path(uploaded.name).name, dataset_name="CIC-IDS2018")
windowed_df = add_time_windows(canonical_df, window_seconds=window_seconds)
sequence = build_temporal_graphs(windowed_df)

if len(sequence) == 0:
    st.error("No temporal graph snapshots could be built from this file. Check timestamp and source/destination fields.")
    st.stop()

latest = sequence.latest()

# Keep the main surface focused on the SOC decision. Raw pipeline details live in an expander.
with st.expander("Data pipeline details"):
    a,b,c = st.columns(3)
    a.metric("Raw rows", len(raw_df)); b.metric("Temporal windows", len(sequence)); c.metric("Latest graph nodes", latest.graph.number_of_nodes() if latest else 0)
    st.dataframe(canonical_df.head(10), hide_index=True, use_container_width=True)

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

available = [
    c
    for c in preferred_features
    if c in windowed_df.columns
]

feature_tensor = (
    torch.tensor(
        windowed_df[
            available
        ]
        .fillna(0)
        .iloc[0:1]
        .to_numpy(
            dtype="float32"
        ),
        dtype=torch.float32,
    )
    if available
    else None
)

# Give the forecasting engine the complete recent telemetry
# so its fallback stage/target logic can use attack labels,
# destination IPs and host context.
window_summaries = (
    windowed_df
    .tail(20)
    .to_dict(
        orient="records"
    )
)

forecast = forecast_engine.forecast(
    sequence.snapshots,
    feature_tensor=feature_tensor,
    feature_names=available,
    window_summaries=window_summaries,
)

# Evaluation is real-data only; never manufacture a score.
# Run the temporal encoder once for the whole history instead of forecasting
# every prefix independently (which becomes O(N^2) graph/model work).
eval_probs = forecast_engine.forecast_history(sequence.snapshots)
eval_labels = [int(snap.metadata.get("label_is_attack", 0)) for snap in sequence.snapshots]
evaluation = evaluate_detection([int(p >= .5) for p in eval_probs], eval_labels, probs=eval_probs) if len(set(eval_labels)) > 1 else {}

# Attention / target details
explanation = forecast["explanation"]
graph_highlight = explanation.get("graph_highlight", {})
target_attention = explanation.get("attention_weights", {}).get("target_attention")
node_attention = {}
if target_attention and latest is not None:
    flat = target_attention[0] if isinstance(target_attention, list) and target_attention and isinstance(target_attention[0], list) else target_attention
    nodes = list(latest.graph.nodes())
    node_attention = {nodes[i]: float(flat[i]) for i in range(min(len(nodes), len(flat)))}
    target_attention = flat

# Primary decision surface
st.subheader("Current threat picture")
c1,c2,c3 = st.columns([1.25,1,1])
with c1:
    show_forecast_summary(forecast, node_names := (list(latest.graph.nodes())[forecast.get("likely_target")] if forecast.get("likely_target") is not None and forecast.get("likely_target") < latest.graph.number_of_nodes() else None))
with c2:
    st.markdown("**Predicted attack path**")
    st.metric("Early warning", f"{len(forecast.get('rollout', []))} steps")
    st.caption("The forecast is generated from the latest temporal graph sequence.")
with c3:
    st.markdown("**Model evaluation**")
    if evaluation:
        st.metric("F1", f"{evaluation['f1']:.3f}")
        st.caption(f"AUROC {evaluation['auroc']:.3f} · AUPRC {evaluation['auprc']:.3f}")
    else:
        st.metric("Status", "Not available")
        st.caption("Requires both benign and attack labels.")

st.subheader("Attack probability forecast")
rollout = forecast["rollout"]
show_attack_timeline([{"timestamp": f"t+{p['step']}", "attack_probability": p["attack_probability"]} for p in rollout])

left,right = st.columns([1.2,1])
with left:
    st.subheader("Predicted attack stage")
    show_stage_transition([{"step":p["step"],"stage_logits":p["stage_logits"]} for p in rollout])
with right:
    st.subheader("Likely target")
    if target_attention is not None:
        show_target_ranking(target_attention, list(latest.graph.nodes()))
    else:
        st.caption("Target attention is unavailable for this run.")

st.subheader("Why the model thinks this")
show_explainability(explanation)

st.subheader("Network state")
show_graph_visualization(latest.graph, node_attention=node_attention, graph_highlight=graph_highlight)
if node_attention:
    show_attention_details(node_attention, top_k=5)

asset_criticality = int(windowed_df["host.asset_criticality"].fillna(1).iloc[-1]) if "host.asset_criticality" in windowed_df.columns else 1
stage_confidence = .7
soc_load = .3
decoy_capacity = 1.0
decision = policy.decide({"attack_probability":forecast["final_attack_probability"],"stage_logits":forecast["final_stage_logits"],"target_score":forecast.get("target_score",0.0),"predicted_stage":forecast.get("predicted_stage")}, asset_criticality=asset_criticality, stage_confidence=stage_confidence, soc_load=soc_load, decoy_capacity=decoy_capacity, human_mode=human_mode)
approved = show_decision_panel({"action":decision.action,"reason":decision.reason}, require_approval=decision.requires_approval)
policy_logger.append({"action":decision.action,"score":decision.score,"reason":decision.reason,"approved":approved,"metadata":decision.metadata})

if approved and decision.action == "divert":
    st.subheader("Deceptra")
    decoy = decoy_manager.register_decoy("server-03-clone", "server", "10.0.9.0/24", "Linux")
    diversion = divert_session(source=str(windowed_df.iloc[-1].get("flow.src_ip","unknown")), original_target="real-target", decoy_target=decoy.decoy_id, reason=decision.reason, method="iptables", source_ip=str(windowed_df.iloc[-1].get("flow.src_ip","10.0.0.1")), original_target_ip="10.0.0.2", decoy_target_ip="10.0.9.2", dst_port=443)
    diversion_log.append({"action":"divert","source":diversion.source,"original_target":diversion.original_target,"decoy_target":diversion.decoy_target,"reason":diversion.reason,"method":diversion.method,"commands":diversion.commands})
    sess = capture.start_session("sess-1", diversion.source, diversion.decoy_target)
    capture.record_event("sess-1", {"event":"diverted"}); capture.record_event("sess-1", {"event":"session_started"})
    st.success(f"Session diverted to decoy **{decoy.decoy_id}**. Actuation remains dry-run unless explicitly enabled in the Deceptra API.")
    show_session_replay(sess.events)

# Keep feedback/evaluation details available without overwhelming the main SOC surface.
with st.expander("Evaluation & feedback loop"):
    if evaluation:
        show_metric_cards({"attack_probability":forecast["final_attack_probability"],"auroc":evaluation.get("auroc"),"auprc":evaluation.get("auprc"),"fpr":evaluation.get("false_positive_rate")})
        st.caption(f"F1 {evaluation['f1']:.3f} · Precision {evaluation['precision']:.3f} · Recall {evaluation['recall']:.3f} · Brier {evaluation['brier_score']:.3f}")
        show_reliability_diagram(reliability_diagram(eval_probs, eval_labels))
    else:
        st.info("No benchmark metrics are shown because the uploaded telemetry does not contain both classes.")
    st.markdown("**Closed-loop state**")
    st.write("Forecast → Policy decision → Deceptra session → relabel → retraining trigger")

if show_overlay_json:
    show_graph_highlight_json(graph_highlight)
