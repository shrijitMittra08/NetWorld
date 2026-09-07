from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .schema import CANONICAL_SCHEMA, canonicalFieldNames, emptyCanonicalRow

COLUMN_MAP: Dict[str, str] = {
    # network / source metadata
    "Timestamp": "network.timestamp",

    # flow.*
    "Src IP": "flow.src_ip",
    "Dst IP": "flow.dst_ip",
    "Src Port": "flow.src_port",
    "Dst Port": "flow.dst_port",
    "Protocol": "flow.protocol",
    "Flow Duration": "flow.duration",
    "Tot Fwd Pkts": "flow.total_fwd_packets",
    "Tot Bwd Pkts": "flow.total_bwd_packets",
    "Total Length of Fwd Packets": "flow.total_length_fwd_packets",
    "Total Length of Bwd Packets": "flow.total_length_bwd_packets",
    "Flow Bytes/s": "flow.flow_bytes_per_sec",
    "Flow Packets/s": "flow.flow_packets_per_sec",
    "IAT Mean": "flow.iat_mean",
    "IAT Std": "flow.iat_std",
    "TCP Flags": "flow.tcp_flags",

    # labels
    "Label": "label.class",
}


NUMERIC_INT_COLUMNS = {
    "flow.src_port",
    "flow.dst_port",
    "flow.total_fwd_packets",
    "flow.total_bwd_packets",
    "packet.fragmentation_flags",
    "packet.retransmissions",
    "behaviour.new_peers",
    "behaviour.new_destinations",
    "behaviour.unique_dest_ports",
    "behaviour.connection_attempts",
    "behaviour.connection_failures",
    "behaviour.auth_attempts",
    "behaviour.auth_failures",
    "host.asset_criticality",
    "label.is_attack",
}

NUMERIC_FLOAT_COLUMNS = {
    "flow.duration",
    "flow.total_length_fwd_packets",
    "flow.total_length_bwd_packets",
    "flow.flow_bytes_per_sec",
    "flow.flow_packets_per_sec",
    "flow.iat_mean",
    "flow.iat_std",
    "packet.ttl_mean",
    "packet.ttl_var",
    "packet.tcp_window_mean",
    "packet.tcp_window_var",
    "packet.payload_size_mean",
    "packet.payload_size_var",
}

def cleanValue(value: Any) -> Any:
    if pd.isna(value): return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null", "inf", "-inf", "infinity"}: return None
        return stripped
    return value

def safeToInt(value: int) -> Optional[int]:
    value = cleanValue(value)
    if value is None: return None
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value): return None
            return int(value)
        return int(float(str(value)))
    except ValueError:
        return None

def safeToFloat(value: Any) -> Optional[float]:
    value = cleanValue(value)
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            return float(value)
        if isinstance(value, (int, np.integer)):
            return float(value)
        return float(str(value))
    except (ValueError, TypeError):
        return None

def normalizeLabelClass(label: Any) -> str:
    value = cleanValue(label)
    if value is None:
        return "UNKNOWN"
    return str(value).strip()

def isAttack(label_class: str) -> int:
    return 0 if label_class.upper() == "BENIGN" else 1

def normalizeDataset(df: pd.DataFrame, sourceFile: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        empty = pd.DataFrame(columns=canonicalFieldNames())
        return empty

    out = df.rename(columns=COLUMN_MAP).copy()

    if sourceFile is not None:
        out["network.source_file"] = sourceFile
    elif "network.source_file" not in out.columns:
        out["network.source_file"] = None

    out["network.dataset"] = "CIC-IDS2018"

    if "network.timestamp" in out.columns:
        out["network.timestamp"] = out["network.timestamp"].map(cleanValue)
    else:
        out["network.timestamp"] = None

    for col in NUMERIC_INT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(safeToInt)

    for col in NUMERIC_FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(safeToFloat)

    if "label.class" not in out.columns:
        out["label.class"] = "UNKNOWN"
    out["label.class"] = out["label.class"].map(normalizeLabelClass)
    out["label.is_attack"] = out["label.class"].map(isAttack).astype("int64")

    packet_placeholder_cols = [
        "packet.ttl_mean",
        "packet.ttl_var",
        "packet.tcp_window_mean",
        "packet.tcp_window_var",
        "packet.fragmentation_flags",
        "packet.payload_size_mean",
        "packet.payload_size_var",
        "packet.retransmissions",
    ]

    behaviour_placeholder_cols = [
        "behaviour.new_peers",
        "behaviour.new_destinations",
        "behaviour.unique_dest_ports",
        "behaviour.connection_attempts",
        "behaviour.connection_failures",
        "behaviour.auth_attempts",
        "behaviour.auth_failures",
    ]

    host_placeholder_cols = [
        "host.src_role",
        "host.dst_role",
        "host.asset_criticality",
        "host.src_subnet",
        "host.dst_subnet",
        "host.src_os",
        "host.dst_os",
    ]

    for col in packet_placeholder_cols + behaviour_placeholder_cols + host_placeholder_cols:
        if col in out.columns:
            out[col] = None

    out["packet.availability"] = pd.Series([False] * len(out), index=out.index, dtype="bool")

    if "label.attack_type" not in out.columns:
        out["label.attack_type"] = None
    if "label.attack_stage" not in out.columns:
        out["label.attack_stage"] = None

    for col in ["network.window_start", "network.window_end"]:
        if col not in out.columns:
            out[col] = None

    # Ensure every canonical field exists.
    for col in canonicalFieldNames():
        if col not in out.columns:
            out[col] = None

    # Reorder to canonical schema first, then any extra columns.
    canonical_cols = canonicalFieldNames()
    extra_cols = [c for c in out.columns if c not in canonical_cols]
    out = out[canonical_cols + extra_cols]

    return out

def validateCanonicalDataFrame(df: pd.DataFrame) -> None:
    missing = [col for col in canonicalFieldNames() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    for col, spec in CANONICAL_SCHEMA.items():
        if not spec["nullable"] and df[col].isna().any():
            raise ValueError(f"Non-nullable column contains nulls: {col}")