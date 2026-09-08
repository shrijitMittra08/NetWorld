
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from .temporal_graph import TemporalGraphSequence, TemporalGraphSnapshot

def _node_link_graph(graph: nx.DiGraph) -> Dict[str, Any]:
    return nx.node_link_data(graph)

def _graph_from_node_link(data: Dict[str, Any]) -> nx.DiGraph:
    return nx.node_link_graph(data, directed=True)

def save_temporal_sequence(sequence: TemporalGraphSequence, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "snapshots": [
            {
                "window_id": snap.window_id,
                "window_start": str(snap.window_start),
                "window_end": str(snap.window_end),
                "metadata": snap.metadata,
                "graph": _node_link_graph(snap.graph),
            }
            for snap in sequence.snapshots
        ]
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def load_temporal_sequence(path: str | Path) -> TemporalGraphSequence:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    sequence = TemporalGraphSequence()
    for item in payload.get("snapshots", []):
        snapshot = TemporalGraphSnapshot(
            window_id=int(item["window_id"]),
            window_start=item.get("window_start"),
            window_end=item.get("window_end"),
            metadata=item.get("metadata", {}),
            graph=_graph_from_node_link(item["graph"]),
        )
        sequence.append(snapshot)

    return sequence