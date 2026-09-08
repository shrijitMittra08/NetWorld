from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import networkx as nx


def _node_color(value: float, min_v: float, max_v: float) -> str:
    if max_v <= min_v:
        return "rgba(31, 119, 180, 0.85)"
    ratio = (value - min_v) / (max_v - min_v)
    # blue -> red scale
    r = int(31 + ratio * (214 - 31))
    g = int(119 - ratio * 80)
    b = int(180 - ratio * 140)
    return f"rgba({r}, {g}, {b}, 0.9)"


def show_metric_cards(metrics: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Attack Prob", f"{metrics.get('attack_probability', 0.0):.3f}")
    cols[1].metric("AUROC", f"{metrics.get('auroc', 0.0):.3f}")
    cols[2].metric("AUPRC", f"{metrics.get('auprc', 0.0):.3f}")
    cols[3].metric("FPR", f"{metrics.get('fpr', 0.0):.3f}")


def show_attack_timeline(timeline: List[Dict[str, Any]]) -> None:
    if not timeline:
        st.info("No attack timeline data.")
        return

    df = pd.DataFrame(timeline)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    fig = go.Figure()
    if "attack_probability" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"] if "timestamp" in df.columns else df.index,
            y=df["attack_probability"],
            mode="lines+markers",
            name="Attack Probability",
            line=dict(color="#d62728", width=3),
        ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_stage_transition(stage_events: List[Dict[str, Any]]) -> None:
    if not stage_events:
        st.info("No stage events.")
        return
    df = pd.DataFrame(stage_events)
    st.dataframe(df, use_container_width=True)


def show_target_ranking(target_attention: List[float], node_names: List[str]) -> None:
    if not target_attention:
        st.info("No target attention available.")
        return
    top = pd.DataFrame(
        {"node": node_names[:len(target_attention)], "score": target_attention}
    ).sort_values("score", ascending=False)
    st.dataframe(top, use_container_width=True)


def show_explainability(explanation: Dict[str, Any]) -> None:
    st.subheader("Explainability")
    st.json(explanation)


def show_graph_visualization(
    graph,
    node_attention: Dict[str, float] | None = None,
    graph_highlight: Dict[str, Any] | None = None,
) -> None:
    st.subheader("Graph Visualization")
    if graph is None:
        st.info("No graph available.")
        return

    try:
        node_attention = node_attention or {}
        highlighted_nodes = set()
        highlighted_edges = set()

        if graph_highlight:
            for n in graph_highlight.get("nodes", []):
                highlighted_nodes.add(n.get("id"))
            for e in graph_highlight.get("edges", []):
                highlighted_edges.add(e.get("id"))

        pos = nx.spring_layout(graph, seed=42)
        edge_x, edge_y = [], []

        for u, v, attrs in graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        if node_attention:
            vals = list(node_attention.values())
            min_v, max_v = min(vals), max(vals)
        else:
            min_v, max_v = 0.0, 1.0

        for n in graph.nodes():
            x, y = pos[n]
            attn = float(node_attention.get(n, 0.0))
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{n}<br>attention={attn:.3f}")
            node_color.append(_node_color(attn, min_v, max_v))
            size = 14 + attn * 25
            if n in highlighted_nodes:
                size += 8
            node_size.append(size)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="rgba(120,120,120,0.5)"),
            hoverinfo="none",
            name="Edges",
        ))
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in graph.nodes()],
            textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#333")),
            hovertext=node_text,
            hoverinfo="text",
            name="Nodes",
        ))
        fig.update_layout(
            height=550,
            showlegend=False,
            template="plotly_white",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Graph visualization unavailable: {e}")


def show_reliability_diagram(rel: Dict[str, Any]) -> None:
    if not rel:
        return
    st.subheader("Reliability Diagram")
    df = pd.DataFrame(rel)
    st.dataframe(df, use_container_width=True)


def show_decision_panel(decision: Dict[str, Any], require_approval: bool = False) -> bool:
    st.subheader("Action Control Panel")
    st.write(f"**Action:** {decision.get('action', 'unknown')}")
    st.write(f"**Reason:** {decision.get('reason', 'N/A')}")
    approved = True
    if require_approval:
        approved = st.radio("Analyst approval", ["Approve", "Deny"], horizontal=True) == "Approve"
    return approved


def show_session_replay(events: List[Dict[str, Any]]) -> None:
    st.subheader("Live Session Replay")
    if not events:
        st.info("No session events.")
        return
    st.dataframe(pd.DataFrame(events), use_container_width=True)


def show_attention_details(node_attention: Dict[str, float], top_k: int = 10) -> None:
    st.subheader("Node Attention")
    if not node_attention:
        st.info("No node attention available.")
        return

    df = pd.DataFrame(
        [{"node": k, "attention": v} for k, v in node_attention.items()]
    ).sort_values("attention", ascending=False).head(top_k)
    st.dataframe(df, use_container_width=True)


def show_graph_highlight_json(graph_highlight: Dict[str, Any]) -> None:
    st.subheader("Graph Highlight Overlay")
    st.json(graph_highlight)