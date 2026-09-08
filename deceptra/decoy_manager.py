from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

@dataclass
class Decoy:
    decoy_id: str
    role: str
    subnet: str
    os: str
    active: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class DecoyManager:
    decoys: Dict[str, Decoy] = field(default_factory=dict)

    def register_decoy(self, decoy_id: str, role: str, subnet: str, os: str, metadata: Optional[Dict[str, str]] = None) -> Decoy:
        decoy = Decoy(
            decoy_id=decoy_id,
            role=role,
            subnet=subnet,
            os=os,
            active=True,
            metadata=metadata or {},
        )
        self.decoys[decoy_id] = decoy
        return decoy

    def list_active_decoys(self) -> List[Decoy]:
        return [d for d in self.decoys.values() if d.active]

    def get_decoy(self, decoy_id: str) -> Decoy:
        if decoy_id not in self.decoys:
            raise KeyError(f"Decoy not found: {decoy_id}")
        return self.decoys[decoy_id]

    def deactivate_decoy(self, decoy_id: str) -> None:
        self.get_decoy(decoy_id).active = False