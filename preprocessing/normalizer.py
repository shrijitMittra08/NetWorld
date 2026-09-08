from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ingestion.dataset_adapter import get_dataset_field_map
from .schema import CANONICAL_SCHEMA, canonicalFieldNames

GENERIC_COLUMN_ALIASES: Dict[str, str] = {
    "Timestamp": "network.timestamp",
    "timestamp": "network.timestamp",
    "time": "network.timestamp",
    "datetime": "network.timestamp",

    "Src IP": "flow.src_ip",
    "src_ip": "flow.src_ip",
    "srcip": "flow.src_ip",
    "source_ip": "flow.src_ip",

    "Dst IP": "flow.dst_ip",
    "dst_ip": "flow.dst_ip",
    "dstip": "flow.dst_ip",
    "destination_ip": "flow.dst_ip",

    "Src Port": "flow.src_port",
    "src_port": "flow.src_port",
    "sport": "flow.src_port",

    "Dst Port": "flow.dst_port",
    "dst_port": "flow.dst_port",
    "dsport": "flow.dst_port",

    "Protocol": "flow.protocol",
    "proto": "flow.protocol",
    "protocol": "flow.protocol",

    "Flow Duration": "flow.duration",
    "duration": "flow.duration",

    "Tot Fwd Pkts": "flow.total_fwd_packets",
    "Tot Bwd Pkts": "flow.total_bwd_packets",
    "Total Length of Fwd Packets": "flow.total_length_fwd_packets",
    "Total Length of Bwd Packets": "flow.total_length_bwd_packets",
    "Flow Bytes/s": "flow.flow_bytes_per_sec",
    "Flow Packets/s": "flow.flow_packets_per_sec",
    "Flow IAT Mean": "flow.iat_mean",
    "Flow IAT Std": "flow.iat_std",
    "TCP Flags": "flow.tcp_flags",

    "TTL Mean": "packet.ttl_mean",
    "TTL Var": "packet.ttl_var",
    "TCP Window Mean": "packet.tcp_window_mean",
    "TCP Window Var": "packet.tcp_window_var",
    "Fragmentation Flags": "packet.fragmentation_flags",
    "Payload Size Mean": "packet.payload_size_mean",
    "Payload Size Var": "packet.payload_size_var",
    "Retransmissions": "packet.retransmissions",

    "New Peers": "behaviour.new_peers",
    "New Destinations": "behaviour.new_destinations",
    "Unique Destination Ports": "behaviour.unique_dest_ports",
    "Connection Attempts": "behaviour.connection_attempts",
    "Connection Failures": "behaviour.connection_failures",
    "Auth Attempts": "behaviour.auth_attempts",
    "Auth Failures": "behaviour.auth_failures",

    "Src Role": "host.src_role",
    "Dst Role": "host.dst_role",
    "Asset Criticality": "host.asset_criticality",
    "Src Subnet": "host.src_subnet",
    "Dst Subnet": "host.dst_subnet",
    "Src OS": "host.src_os",
    "Dst OS": "host.dst_os",

    "Label": "label.class",
    "label": "label.class",
    "Class": "label.class",
    "Attack Type": "label.attack_type",
    "Attack Stage": "label.attack_stage",
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
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null", "inf", "-inf", "infinity"}:
            return None
        return stripped
    return value

def safeToInt(value: Any) -> Optional[int]:
    value = cleanValue(value)
    if value is None:
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            return int(value)
        return int(float(str(value)))
    except (ValueError, TypeError):
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
    return 0 if str(label_class).upper() == "BENIGN" else 1

def _resolve_column_map(dataset_name: Optional[str]) -> Dict[str, str]:
    resolved = dict(GENERIC_COLUMN_ALIASES)
    if dataset_name:
        resolved.update(get_dataset_field_map(dataset_name))
    return resolved

def normalizeDataset(
    df: pd.DataFrame,
    sourceFile: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=canonicalFieldNames())

    column_map = _resolve_column_map(dataset_name)
    out = df.rename(columns=column_map).copy()

    if sourceFile is not None:
        out["network.source_file"] = sourceFile
    elif "network.source_file" not in out.columns:
        out["network.source_file"] = None

    if dataset_name is not None:
        out["network.dataset"] = dataset_name
    elif "network.dataset" not in out.columns:
        out["network.dataset"] = "UNKNOWN"

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

    if "label.attack_type" not in out.columns:
        out["label.attack_type"] = None
    if "label.attack_stage" not in out.columns:
        out["label.attack_stage"] = None

    if "packet.availability" not in out.columns:
        out["packet.availability"] = False
    else:
        out["packet.availability"] = out["packet.availability"].fillna(False).astype("bool")

    for col in canonicalFieldNames():
        if col not in out.columns:
            out[col] = None

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