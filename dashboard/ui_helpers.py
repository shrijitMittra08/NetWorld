from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


STAGE_NAMES = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution",
    "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Discovery", "Lateral Movement", "Collection", "Command & Control",
    "Exfiltration", "Impact",
]


def _metric_text(value: Any, percent: bool = False) -> str:
    if isinstance(value, str):
        return value
    try:
        v = float(value)
        return f"{v * 100:.0f}%" if percent else f"{v:.3f}"
    except (TypeError, ValueError):
        return "N/A"


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Streamlit metric cards */
[data-testid="stMetric"] {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px 16px;
    background: #ffffff;
    color: #111827;
}

[data-testid="stMetric"] label {
    color: #6b7280 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #374151 !important;
}

/* Hero */
.hero {
    padding: 22px 26px;
    border-radius: 16px;
    background: linear-gradient(135deg,#111827,#1f2937);
    color: white;
    margin-bottom: 18px;
}

.hero h1 {
    margin: 0 0 5px 0;
    font-size: 2.05rem;
    color: #ffffff;
}

.hero p {
    margin: 0;
    color: #d1d5db;
    font-size: 1rem;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .72rem;
    color: #9ca3af;
    font-weight: 700;
}

.status {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: .78rem;
}

.status.good {
    background: #dcfce7;
    color: #166534 !important;
}

.status.warn {
    background: #fef3c7;
    color: #92400e !important;
}

.status.bad {
    background: #fee2e2;
    color: #991b1b !important;
}

/* White action cards need explicit dark text */
.action-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    background: #ffffff;
    color: #111827;
}

.action-card h3 {
    margin: 0 0 5px 0;
    color: #111827 !important;
}

.action-card p {
    margin: 4px 0;
    color: #4b5563 !important;
}

.action-card b {
    color: #111827 !important;
}

.flow {
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 8px 0 18px 0;
    flex-wrap: wrap;
}

.flow-step {
    padding: 8px 12px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    background: #f9fafb;
    color: #111827;
    font-weight: 650;
    font-size: .85rem;
}

.flow-arrow {
    color: #9ca3af;
}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_hero(title: str, subtitle: str, mode: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">NETWORLD + DECEPTRA · CLOSED-LOOP DEFENSE</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div style="margin-top:12px"><span class="status {'warn' if mode == 'Demo scenario' else 'good'}">{mode}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_flow() -> None:
    labels = ["Observe", "Understand", "Predict", "Decide", "Deceive", "Learn"]
    html = '<div class="flow">'
    for i, label in enumerate(labels):
        html += f'<span class="flow-step">{label}</span>'
        if i < len(labels) - 1:
            html += '<span class="flow-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def show_metric_cards(metrics: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Attack probability", _metric_text(metrics.get("attack_probability"), percent=True))
    cols[1].metric("AUROC", _metric_text(metrics.get("auroc")))
    cols[2].metric("AUPRC", _metric_text(metrics.get("auprc")))
    cols[3].metric("False-positive rate", _metric_text(metrics.get("fpr")))


def show_forecast_summary(forecast: Dict[str, Any], target_name: str | None = None) -> None:
    p = float(forecast.get("final_attack_probability", 0.0))
    stage_idx = forecast.get("predicted_stage")
    stage = STAGE_NAMES[int(stage_idx)] if isinstance(stage_idx, int) and 0 <= stage_idx < len(STAGE_NAMES) else "Unknown stage"
    target = target_name or (f"Node {forecast.get('likely_target')}" if forecast.get('likely_target') is not None else "Unknown target")
    if p >= .75: cls, label = "bad", "HIGH RISK"
    elif p >= .35: cls, label = "warn", "ELEVATED"
    else: cls, label = "good", "LOW RISK"
    st.markdown(
        f"""
        <div class="action-card">
          <div class="eyebrow">FORECAST</div>
          <h3>{p*100:.0f}% · {label} <span class="status {cls}" style="float:right">{stage}</span></h3>
          <p><b>Likely target:</b> {target}</p>
          <p><b>Forecast horizon:</b> {len(forecast.get('rollout', []))} future steps</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_attack_timeline(timeline: List[Dict[str, Any]]) -> None:
    if not timeline:
        st.info("No attack timeline data.")
        return
    df = pd.DataFrame(timeline)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["attack_probability"], mode="lines+markers",
        name="Attack probability", line=dict(width=3), fill="tozeroy",
    ))
    fig.add_hline(y=.35, line_dash="dash", annotation_text="Monitor threshold")
    fig.add_hline(y=.75, line_dash="dash", annotation_text="Isolate threshold")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Probability")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=25,b=10), template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def show_stage_transition(stage_events: List[Dict[str, Any]]) -> None:
    if not stage_events:
        return
    rows = []
    for event in stage_events:
        logits = event.get("stage_logits", [])
        if logits and isinstance(logits[0], list): logits = logits[0]
        if logits:
            idx = max(range(len(logits)), key=lambda i: logits[i])
            rows.append({"Step": event["step"], "Predicted stage": STAGE_NAMES[idx] if idx < len(STAGE_NAMES) else f"Stage {idx}", "Confidence": float(pd.Series(logits).max())})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def show_target_ranking(target_attention: List[float], node_names: List[str]) -> None:
    if not target_attention:
        return
    top = pd.DataFrame({"Target": node_names[:len(target_attention)], "Score": target_attention}).sort_values("Score", ascending=False).head(5)
    top["Score"] = top["Score"].map(lambda x: f"{float(x):.3f}")
    st.dataframe(top, hide_index=True, use_container_width=True)


def show_explainability(explanation: Dict[str, Any]) -> None:
    st.subheader("Why this forecast?")
    summary = explanation.get("summary")
    if summary:
        st.info(summary)
    attr = explanation.get("feature_attribution", {}) or {}
    deltas = explanation.get("temporal_deltas", []) or []
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top contributing features**")
        if attr:
            rows = sorted(attr.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
            st.dataframe(pd.DataFrame(rows, columns=["Feature", "Attribution"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Feature attribution unavailable for this run.")
    with c2:
        st.markdown("**Recent behavioural changes**")
        if deltas:
            st.dataframe(pd.DataFrame(deltas).head(5), hide_index=True, use_container_width=True)
        else:
            st.caption("No temporal deltas available.")


def show_graph_visualization(graph, node_attention: Dict[str, float] | None = None, graph_highlight: Dict[str, Any] | None = None) -> None:
    if graph is None:
        st.info("No graph available.")
        return
    try:
        node_attention = node_attention or {}
        highlighted = {n.get("id") for n in (graph_highlight or {}).get("nodes", [])}
        pos = nx.spring_layout(graph, seed=42)
        edge_x, edge_y = [], []
        for u,v in graph.edges():
            edge_x += [pos[u][0], pos[v][0], None]; edge_y += [pos[u][1], pos[v][1], None]
        node_x, node_y, sizes, texts = [], [], [], []
        vals = list(node_attention.values()) or [0]
        lo, hi = min(vals), max(vals)
        for n in graph.nodes():
            x,y = pos[n]; a=float(node_attention.get(n,0)); ratio=0 if hi<=lo else (a-lo)/(hi-lo)
            node_x.append(x); node_y.append(y); sizes.append(18 + 26*ratio + (8 if n in highlighted else 0)); texts.append(f"{n}<br>attention={a:.3f}")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=1.5),hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=[str(n) for n in graph.nodes()],textposition="top center",marker=dict(size=sizes,line=dict(width=1)),hovertext=texts,hoverinfo="text"))
        fig.update_layout(height=480,showlegend=False,template="plotly_white",margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as e:
        st.warning(f"Graph visualization unavailable: {e}")


def show_reliability_diagram(rel: Dict[str, Any]) -> None:
    if not rel: return
    st.caption("Calibration data")
    st.dataframe(pd.DataFrame(rel), hide_index=True, use_container_width=True)


def show_decision_panel(decision: Dict[str, Any], require_approval: bool = False) -> bool:
    action = str(decision.get("action", "unknown")).upper()
    st.subheader("Recommended defense action")
    st.markdown(f'<div class="action-card"><div class="eyebrow">POLICY ENGINE</div><h3>{action}</h3><p>{decision.get("reason", "N/A")}</p></div>', unsafe_allow_html=True)
    approved = True
    if require_approval:
        approved = st.radio("Analyst approval", ["Approve", "Deny"], horizontal=True) == "Approve"
    return approved


def show_session_replay(events: List[Dict[str, Any]]) -> None:
    if not events: return
    st.subheader("Deceptra session")
    st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)


def show_attention_details(node_attention: Dict[str, float], top_k: int = 10) -> None:
    if not node_attention: return
    st.caption("Top graph attention")
    df=pd.DataFrame([{"Node":k,"Attention":v} for k,v in node_attention.items()]).sort_values("Attention",ascending=False).head(top_k)
    st.dataframe(df, hide_index=True, use_container_width=True)


def show_graph_highlight_json(graph_highlight: Dict[str, Any]) -> None:
    with st.expander("Developer details"):
        st.json(graph_highlight)
