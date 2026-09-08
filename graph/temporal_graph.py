from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import networkx as nx

@dataclass
class TemporalGraphSnapshot:
    window_id: int
    window_start: Any
    window_end: Any
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalGraphSequence:
    """
    Explicit G(t-k)...G(t) sequence object.
    """
    snapshots: List[TemporalGraphSnapshot] = field(default_factory=list)

    def append(self, snapshot: TemporalGraphSnapshot) -> None:
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.window_id)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> TemporalGraphSnapshot:
        return self.snapshots[idx]

    def latest(self) -> Optional[TemporalGraphSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    def window_ids(self) -> List[int]:
        return [s.window_id for s in self.snapshots]