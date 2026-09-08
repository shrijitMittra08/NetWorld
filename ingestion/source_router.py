from __future__ import annotations

from pathlib import Path

def detect_source_type(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".pcap", ".pcapng"}:
        return "pcap"
    if suffix == ".csv":
        return "csv"
    if suffix in {".log", ".txt"}:
        return "authlog"
    if suffix in {".json", ".jsonl"}:
        return "netflow"

    return "unknown"