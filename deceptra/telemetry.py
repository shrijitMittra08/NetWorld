from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class TelemetryExport:
    source: str
    destination: str
    payload: Dict[str, Any]
    one_way: bool = True

def export_one_way_telemetry(source: str, destination: str, payload: Dict[str, Any]) -> TelemetryExport:
    """
    Contract for one-way telemetry export from decoy network to analysis pipeline.
    """
    return TelemetryExport(
        source=source,
        destination=destination,
        payload=payload,
        one_way=True,
    )