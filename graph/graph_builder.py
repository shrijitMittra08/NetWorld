from __future__ import annotations

from typing import Dict, List, Tuple, Set
import pandas as pd

from .feature_aggregation import aggregate_edge_features, aggregate_node_features
from .temporal_graph import TemporalGraphSequence, TemporalGraphSnapshot


def _host_identity(row: pd.Series, side: str) -> str:
    """Build a stable host identity; the IP is always the final component."""
    role = row.get(f"host.{side}_role")
    subnet = row.get(f"host.{side}_subnet")
    os_name = row.get(f"host.{side}_os")
    ip = row.get(f"flow.{side}_ip")

    parts = []
    for value in (role, subnet, os_name, ip):
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() not in {"nan", "none"}:
            parts.append(str(value).strip())
    return "|".join(parts) if parts else f"unknown_{side}"


def _ip(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _node_for_ip(nodes: List[str], ip: str | None) -> str | None:
    if not ip:
        return None
    for node in nodes:
        if str(node).split("|")[-1] == ip:
            return node
    return None


def _ensure_window_columns(df: pd.DataFrame) -> None:
    required = ["network.window_id", "network.window_start", "network.window_end"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def build_temporal_graphs(df: pd.DataFrame) -> TemporalGraphSequence:
    """
    Build temporal graphs and explicit attacker/target metadata.

    Target semantics:
      * attacker = source endpoint of an attack-labelled flow
      * target   = destination endpoint of an attack-labelled flow

    For unlabeled telemetry, the graph records directed source/destination
    candidates so inference can still exclude obvious source-only attacker nodes.
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
        node_sources: Dict[str, pd.DataFrame] = {}
        src_counts: Dict[str, int] = {}
        dst_counts: Dict[str, int] = {}
        attack_src_ips: Set[str] = set()
        attack_dst_ips: Set[str] = set()

        for _, row in window_df.iterrows():
            src = _host_identity(row, "src")
            dst = _host_identity(row, "dst")
            key = (src, dst)
            edge_groups.setdefault(key, pd.DataFrame(columns=window_df.columns))
            edge_groups[key] = pd.concat([edge_groups[key], row.to_frame().T], ignore_index=True)

            node_sources.setdefault(src, pd.DataFrame(columns=window_df.columns))
            node_sources.setdefault(dst, pd.DataFrame(columns=window_df.columns))
            node_sources[src] = pd.concat([node_sources[src], row.to_frame().T], ignore_index=True)
            node_sources[dst] = pd.concat([node_sources[dst], row.to_frame().T], ignore_index=True)

            src_ip = _ip(row.get("flow.src_ip"))
            dst_ip = _ip(row.get("flow.dst_ip"))
            if src_ip:
                src_counts[src_ip] = src_counts.get(src_ip, 0) + 1
            if dst_ip:
                dst_counts[dst_ip] = dst_counts.get(dst_ip, 0) + 1

            attack = pd.to_numeric(pd.Series([row.get("label.is_attack", 0)]), errors="coerce").fillna(0).iloc[0] > 0
            if attack:
                if src_ip:
                    attack_src_ips.add(src_ip)
                if dst_ip:
                    attack_dst_ips.add(dst_ip)

        nodes = list(node_sources.keys())

        # If explicit labels exist, they are authoritative.
        attacker_ips = set(attack_src_ips)
        target_ips = set(attack_dst_ips)

        # Unlabelled fallback: source-only endpoints are likely initiators.
        # Prefer endpoints with unusually high outbound activity when there
        # is no explicit attack source.
        if not attacker_ips:
            source_only = set(src_counts) - set(dst_counts)
            if source_only:
                attacker_ips = source_only

        # Candidate targets are destinations, excluding known attackers.
        if not target_ips:
            target_ips = set(dst_counts) - attacker_ips

        attacker_nodes = {
            n for ip in attacker_ips
            for n in [_node_for_ip(nodes, ip)]
            if n is not None
        }
        target_nodes = {
            n for ip in target_ips
            for n in [_node_for_ip(nodes, ip)]
            if n is not None and n not in attacker_nodes
        }

        for node_id, node_df in node_sources.items():
            is_attacker = node_id in attacker_nodes
            is_target = node_id in target_nodes
            snapshot.graph.add_node(
                node_id,
                **aggregate_node_features(node_df),
                is_attacker=bool(is_attacker),
                is_target_candidate=bool(is_target),
            )

        for (src, dst), edge_df in edge_groups.items():
            snapshot.graph.add_edge(
                src,
                dst,
                **aggregate_edge_features(edge_df),
            )

        labels = pd.to_numeric(
            window_df.get("label.is_attack", pd.Series(0, index=window_df.index)),
            errors="coerce",
        ).fillna(0)
        label_is_attack = int(labels.max() > 0)

        snapshot.metadata.update({
            "window_row_count": int(len(window_df)),
            "node_count": snapshot.graph.number_of_nodes(),
            "edge_count": snapshot.graph.number_of_edges(),
            "label_is_attack": label_is_attack,
            "attacker_ips": sorted(attacker_ips),
            "target_ips": sorted(target_ips),
            "attacker_nodes": sorted(attacker_nodes),
            "target_candidate_nodes": sorted(target_nodes),
            "source_ip_counts": src_counts,
            "destination_ip_counts": dst_counts,
        })

        if "label.attack_stage" in window_df.columns:
            stages = [
                str(v).strip()
                for v in window_df["label.attack_stage"].dropna().tolist()
                if str(v).strip() and str(v).strip().lower() not in {"nan", "none", "unknown"}
            ]
            snapshot.metadata["attack_stage"] = stages[-1] if stages else "unknown"

        sequence.append(snapshot)

    return sequence
