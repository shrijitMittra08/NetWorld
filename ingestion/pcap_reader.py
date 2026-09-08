from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

class PcapReader:
    """
    Minimal PCAP reader interface.

    This class intentionally returns a canonical-friendly DataFrame shape even
    when actual packet parsing is not yet implemented.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def parse(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(f"PCAP file not found: {self.path}")

        # Placeholder parser contract for future Scapy/PyShark implementation.
        rows: List[Dict[str, Any]] = []
        return pd.DataFrame(
            rows,
            columns=[
                "network.timestamp",
                "flow.src_ip",
                "flow.dst_ip",
                "flow.src_port",
                "flow.dst_port",
                "flow.protocol",
                "packet.ttl_mean",
                "packet.ttl_var",
                "packet.tcp_window_mean",
                "packet.tcp_window_var",
                "packet.fragmentation_flags",
                "packet.payload_size_mean",
                "packet.payload_size_var",
                "packet.retransmissions",
                "packet.availability",
                "label.class",
            ],
        )