
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .feature_aggregation import aggregate_edge_features, aggregate_node_features
from .temporal_graph import TemporalGraphSequence, TemporalGraphSnapshot

def _host_identity(row: pd.Series, side: str) -> str:
    """
    Build a true host identity from host context when available.
    Falls back to IP only, not IP:port.
    """
    role = row.get(f"host.{side}_role")
    subnet = row.get(f"host.{side}_subnet")
    os_name = row.get(f"host.{side}_os")
    ip = row.get(f"flow.{side}_ip")

    parts = []
    if pd.notna(role):
        parts.append(str(role))
    if pd.notna(subnet):
        parts.append(str(subnet))
    if pd.notna(os_name):
        parts.append(str(os_name))
    if pd.notna(ip):
        parts.append(str(ip))

    if not parts:
        return f"unknown_{side}"

    return "|".join(parts)

def _ensure_window_columns(df: pd.DataFrame) -> None:
    required = ["network.window_id", "network.window_start", "network.window_end"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

def build_temporal_graphs(df: pd.DataFrame) -> TemporalGraphSequence:
    """
    Build an explicit temporal graph sequence G(t-k)...G(t).
    """
    _ensure_window_columns(df)

    sequence = TemporalGraphSequence()

    for window_id, window_df in df.groupby("network.window_id", dropna=False):
        if pd.isna(window_id):
            continue

        first_row = window_df.iloc[0]
        snapshot = TemporalGraphSnapshot(
            window_id=int(window_id),
            window_start=first_row["network.window_start"],
            window_end=first_row["network.window_end"],
        )

        edge_groups: Dict[Tuple[str, str], pd.DataFrame] = {}

        for _, row in window_df.iterrows():
            src = _host_identity(row, "src")
            dst = _host_identity(row, "dst")

            key = (src, dst)
            if key not in edge_groups:
                edge_groups[key] = pd.DataFrame(columns=window_df.columns)
            edge_groups[key] = pd.concat([edge_groups[key], row.to_frame().T], ignore_index=True)

        # Add nodes with aggregated features
        node_sources: Dict[str, pd.DataFrame] = {}
        for _, row in window_df.iterrows():
            src = _host_identity(row, "src")
            dst = _host_identity(row, "dst")

            node_sources.setdefault(src, pd.DataFrame(columns=window_df.columns))
            node_sources.setdefault(dst, pd.DataFrame(columns=window_df.columns))

            node_sources[src] = pd.concat([node_sources[src], row.to_frame().T], ignore_index=True)
            node_sources[dst] = pd.concat([node_sources[dst], row.to_frame().T], ignore_index=True)

        for node_id, node_df in node_sources.items():
            snapshot.graph.add_node(
                node_id,
                **aggregate_node_features(node_df),
            )

        # Add aggregated edges
        for (src, dst), edge_df in edge_groups.items():
            snapshot.graph.add_edge(
                src,
                dst,
                **aggregate_edge_features(edge_df),
            )

        snapshot.metadata["window_row_count"] = int(len(window_df))
        snapshot.metadata["node_count"] = snapshot.graph.number_of_nodes()
        snapshot.metadata["edge_count"] = snapshot.graph.number_of_edges()

        sequence.append(snapshot)

    return sequence