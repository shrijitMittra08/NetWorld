from __future__ import annotations

from typing import Any, Dict
import pandas as pd

def safe_mean(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return None
    return float(numeric.mean())

def safe_sum(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return None
    return float(numeric.sum())

def aggregate_edge_features(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "count": int(len(df)),
        "duration_mean": safe_mean(df.get("flow.duration", pd.Series(dtype=float))),
        "bytes_per_sec_mean": safe_mean(df.get("flow.flow_bytes_per_sec", pd.Series(dtype=float))),
        "packets_per_sec_mean": safe_mean(df.get("flow.flow_packets_per_sec", pd.Series(dtype=float))),
        "fwd_packets_sum": safe_sum(df.get("flow.total_fwd_packets", pd.Series(dtype=float))),
        "bwd_packets_sum": safe_sum(df.get("flow.total_bwd_packets", pd.Series(dtype=float))),
        "ttl_mean": safe_mean(df.get("packet.ttl_mean", pd.Series(dtype=float))),
        "payload_size_mean": safe_mean(df.get("packet.payload_size_mean", pd.Series(dtype=float))),
        "auth_failures_sum": safe_sum(df.get("behaviour.auth_failures", pd.Series(dtype=float))),
        "connection_failures_sum": safe_sum(df.get("behaviour.connection_failures", pd.Series(dtype=float))),
    }

def aggregate_node_features(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "active_edges": int(len(df)),
        "new_peers_sum": safe_sum(df.get("behaviour.new_peers", pd.Series(dtype=float))),
        "new_destinations_sum": safe_sum(df.get("behaviour.new_destinations", pd.Series(dtype=float))),
        "unique_dest_ports_sum": safe_sum(df.get("behaviour.unique_dest_ports", pd.Series(dtype=float))),
        "auth_attempts_sum": safe_sum(df.get("behaviour.auth_attempts", pd.Series(dtype=float))),
        "auth_failures_sum": safe_sum(df.get("behaviour.auth_failures", pd.Series(dtype=float))),
        "avg_duration": safe_mean(df.get("flow.duration", pd.Series(dtype=float))),
        "connection_attempts_sum": safe_sum(df.get("behaviour.connection_attempts", pd.Series(dtype=float))),
        "connection_failures_sum": safe_sum(df.get("behaviour.connection_failures", pd.Series(dtype=float))),
    }