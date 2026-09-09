from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from preprocessing.feature_extractor import extractFeatures
from preprocessing.windowing import add_time_windows
from graph import build_temporal_graphs

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


def stage_index(stage: Any) -> Optional[int]:
    if stage is None:
        return None
    text = str(stage).strip().upper()
    aliases = {
        "RECON": "RECONNAISSANCE",
        "RECONNAISSANCE": "RECONNAISSANCE",
        "COMMAND AND CONTROL": "COMMAND & CONTROL",
        "C2": "COMMAND & CONTROL",
    }
    text = aliases.get(text, text)
    try:
        return STAGE_NAMES.index(text.title() if text != "COMMAND & CONTROL" else text)
    except ValueError:
        return None


def _target_index(snapshot: Any, destination_ip: Optional[str]) -> Optional[int]:
    if not destination_ip:
        return None
    nodes = list(snapshot.graph.nodes())
    needle = str(destination_ip)
    for i, node in enumerate(nodes):
        if needle in str(node):
            return i
    return None


def snapshot_target(snapshot: Any) -> Optional[int]:
    """Return a supervised victim index, never the attacker index."""
    nodes = list(snapshot.graph.nodes())
    if not nodes:
        return None

    target_nodes = snapshot.metadata.get("target_candidate_nodes") or []
    attacker_nodes = set(snapshot.metadata.get("attacker_nodes") or [])
    target_ips = snapshot.metadata.get("target_ips") or []
    destination_counts = snapshot.metadata.get("destination_ip_counts") or {}

    # Prefer the most frequently attacked destination in the window.
    ordered_ips = sorted(
        target_ips,
        key=lambda ip: (-int(destination_counts.get(ip, 0)), str(ip)),
    )
    for ip in ordered_ips:
        for i, node in enumerate(nodes):
            if str(node).split("|")[-1] == str(ip) and node not in attacker_nodes:
                return i

    for target_node in target_nodes:
        if target_node in nodes and target_node not in attacker_nodes:
            return nodes.index(target_node)

    # Labelled attack fallback: destination-derived target IPs.
    target_ips = snapshot.metadata.get("target_ips") or []
    for ip in target_ips:
        for i, node in enumerate(nodes):
            if str(node).split("|")[-1] == str(ip) and node not in attacker_nodes:
                return i

    return None

def build_items_from_sequence(
    sequence: Any,
    sequence_length: int = 8,
    rollout_steps: int = 3,
    max_sequences: Optional[int] = 5000,
) -> List[Dict[str, Any]]:
    snapshots = list(sequence.snapshots)
    if len(snapshots) < sequence_length + rollout_steps:
        return []

    items: List[Dict[str, Any]] = []
    last_start = len(snapshots) - rollout_steps
    for end in range(sequence_length, last_start + 1):
        context = snapshots[end - sequence_length:end]
        future = snapshots[end:end + rollout_steps]
        context_nodes = list(context[-1].graph.nodes())

        future_targets: List[Optional[int]] = []
        future_attacks: List[int] = []
        future_stages: List[int] = []
        future_stage_masks: List[float] = []

        for snap in future:
            attack = int(snap.metadata.get("label_is_attack", 0))
            stage = stage_index(snap.metadata.get("attack_stage"))
            target = snapshot_target(snap)

            # Target attention is over the current/context node set. Only supervise
            # it when the selected target is present in that same node set.
            if target is not None:
                future_nodes = list(snap.graph.nodes())
                target_node = future_nodes[target]
                try:
                    target = context_nodes.index(target_node)
                except ValueError:
                    target = None

            future_attacks.append(attack)
            future_stages.append(stage if stage is not None else 0)
            future_stage_masks.append(1.0 if attack and stage is not None else 0.0)
            future_targets.append(target)

        items.append({
            "graphs": context,
            "future_snapshots": future,
            "targets": {
                "attack_sequence": future_attacks,
                "stage_sequence": future_stages,
                "stage_mask_sequence": future_stage_masks,
                "target_sequence": future_targets,
                # Backward-compatible aliases for the first rollout step.
                "attack": future_attacks[0],
                "stage": future_stages[0],
                "target_index": future_targets[0],
            },
        })

    if max_sequences is not None and len(items) > max_sequences:
        # Keep temporal coverage across the file instead of taking only the head.
        indices = [round(i * (len(items) - 1) / (max_sequences - 1)) for i in range(max_sequences)] if max_sequences > 1 else [0]
        items = [items[i] for i in indices]
    return items


def load_cicids_csv(
    path: Path,
    window_seconds: int = 5,
    max_rows: Optional[int] = None,
) -> Any:
    read_kwargs: Dict[str, Any] = {"low_memory": False}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    raw = pd.read_csv(path, **read_kwargs)
    canonical = extractFeatures(
        raw,
        source_file=path.name,
        dataset_name="CIC-IDS2018",
    )
    # Remove rows that cannot participate in a graph.
    canonical = canonical.dropna(
        subset=["network.timestamp", "flow.src_ip", "flow.dst_ip"]
    ).copy()
    if canonical.empty:
        return None
    windowed = add_time_windows(
        canonical,
        window_seconds=window_seconds,
    )
    return build_temporal_graphs(windowed)


def discover_cicids_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {root.resolve()}"
        )
    return files
