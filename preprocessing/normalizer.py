from __future__ import annotations

from typing import Any, Dict, Optional
import ipaddress
import re

import numpy as np
import pandas as pd

from ingestion.dataset_adapter import get_dataset_field_map
from .schema import CANONICAL_SCHEMA, canonicalFieldNames


# CIC-IDS2018 has a few spelling/abbreviation variants across releases.
# These aliases are deliberately kept here so the raw files can be consumed
# without manually editing the dataset.
GENERIC_COLUMN_ALIASES: Dict[str, str] = {
    "Timestamp": "network.timestamp",
    "timestamp": "network.timestamp",
    "Time": "network.timestamp",
    "time": "network.timestamp",
    "datetime": "network.timestamp",

    "Src IP": "flow.src_ip",
    "Source IP": "flow.src_ip",
    "src_ip": "flow.src_ip",
    "srcip": "flow.src_ip",
    "source_ip": "flow.src_ip",

    "Dst IP": "flow.dst_ip",
    "Destination IP": "flow.dst_ip",
    "dst_ip": "flow.dst_ip",
    "dstip": "flow.dst_ip",
    "destination_ip": "flow.dst_ip",

    "Src Port": "flow.src_port",
    "Source Port": "flow.src_port",
    "src_port": "flow.src_port",
    "sport": "flow.src_port",

    "Dst Port": "flow.dst_port",
    "Destination Port": "flow.dst_port",
    "dst_port": "flow.dst_port",
    "dsport": "flow.dst_port",

    "Protocol": "flow.protocol",
    "proto": "flow.protocol",
    "protocol": "flow.protocol",

    "Flow Duration": "flow.duration",
    "duration": "flow.duration",

    "Tot Fwd Pkts": "flow.total_fwd_packets",
    "Total Fwd Packets": "flow.total_fwd_packets",
    "Tot Bwd Pkts": "flow.total_bwd_packets",
    "Total Bwd Packets": "flow.total_bwd_packets",

    "TotLen Fwd Pkts": "flow.total_length_fwd_packets",
    "Total Length of Fwd Packets": "flow.total_length_fwd_packets",
    "TotLen Bwd Pkts": "flow.total_length_bwd_packets",
    "Total Length of Bwd Packets": "flow.total_length_bwd_packets",

    "Flow Byts/s": "flow.flow_bytes_per_sec",
    "Flow Bytes/s": "flow.flow_bytes_per_sec",
    "Flow Pkts/s": "flow.flow_packets_per_sec",
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
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {
            "nan", "none", "null", "inf", "-inf", "infinity", "-infinity"
        }:
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
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except (ValueError, TypeError):
        return None


def normalizeLabelClass(label: Any) -> str:
    value = cleanValue(label)
    if value is None:
        return "UNKNOWN"
    return str(value).strip()


def isAttack(label_class: str) -> int:
    return 0 if str(label_class).strip().upper() == "BENIGN" else 1


def infer_cic_attack_stage(label: Any) -> Optional[str]:
    """Heuristic ATT&CK-style stage labels for CIC-IDS2018 attack names.

    CIC-IDS2018 does not provide a native MITRE ATT&CK stage column. These
    mappings make stage supervision possible while preserving the original
    CIC label in label.class / label.attack_type.
    """
    text = str(cleanValue(label) or "").strip().lower()
    if not text or text == "benign":
        return None

    if any(x in text for x in ("brute force", "ssh-bruteforce", "ftp-bruteforce", "password")):
        return "Credential Access"
    if "sql injection" in text or "xss" in text or "web" in text:
        return "Initial Access"
    if "infilteration" in text or "infiltration" in text:
        return "Lateral Movement"
    if "bot" in text:
        return "Command & Control"
    if "heartbleed" in text:
        return "Credential Access"
    if any(x in text for x in ("ddos", "dos attacks", "dos ", "slowloris", "slowhttp", "hulk", "goldeneye")):
        return "Impact"
    return "Execution"


def _subnet(value: Any) -> Optional[str]:
    value = cleanValue(value)
    if value is None:
        return None
    try:
        ip = ipaddress.ip_address(str(value))
        if ip.version == 4:
            return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except ValueError:
        pass
    return None


def _role_for_port(port: Any, source: bool) -> str:
    p = safeToInt(port)
    if p is None:
        return "workstation" if source else "server"
    if p in {22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 8080, 8443}:
        return "client" if source else "server"
    return "workstation" if source else "server"


def _derive_context(out: pd.DataFrame) -> None:
    """Populate host/behaviour features that CIC flow CSVs do not expose."""
    if "host.src_subnet" in out.columns:
        out["host.src_subnet"] = out["host.src_subnet"].where(
            out["host.src_subnet"].notna(),
            out.get("flow.src_ip", pd.Series(index=out.index)).map(_subnet),
        )
    if "host.dst_subnet" in out.columns:
        out["host.dst_subnet"] = out["host.dst_subnet"].where(
            out["host.dst_subnet"].notna(),
            out.get("flow.dst_ip", pd.Series(index=out.index)).map(_subnet),
        )

    if "host.src_role" in out.columns:
        out["host.src_role"] = out["host.src_role"].where(
            out["host.src_role"].notna(),
            out.get("flow.dst_port", pd.Series(index=out.index)).map(lambda p: _role_for_port(p, True)),
        )
    if "host.dst_role" in out.columns:
        out["host.dst_role"] = out["host.dst_role"].where(
            out["host.dst_role"].notna(),
            out.get("flow.dst_port", pd.Series(index=out.index)).map(lambda p: _role_for_port(p, False)),
        )

    if "host.asset_criticality" in out.columns:
        def criticality(port: Any) -> int:
            p = safeToInt(port)
            if p in {22, 445, 3389}:
                return 4
            if p in {443, 3306, 5432}:
                return 3
            if p in {80, 53, 25, 110, 143}:
                return 2
            return 1
        out["host.asset_criticality"] = out["host.asset_criticality"].where(
            out["host.asset_criticality"].notna(),
            out.get("flow.dst_port", pd.Series(index=out.index)).map(criticality),
        )

    # One row is one flow record; use conservative derived behaviour counts.
    defaults = {
        "behaviour.new_peers": 1,
        "behaviour.new_destinations": 1,
        "behaviour.unique_dest_ports": 1,
        "behaviour.connection_attempts": 1,
        "behaviour.auth_attempts": 0,
        "behaviour.auth_failures": 0,
    }
    for col, default in defaults.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    if "behaviour.connection_failures" in out.columns:
        if "flow.tcp_flags" in out.columns:
            rst = out["flow.tcp_flags"].astype(str).str.contains("0x0004|0x0006|0x0014|RST", case=False, regex=True, na=False)
        else:
            rst = pd.Series(False, index=out.index)
        out["behaviour.connection_failures"] = out["behaviour.connection_failures"].fillna(rst.astype(int))


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

    # Strip accidental whitespace in headers before applying aliases.
    cleaned_columns = {c: str(c).strip() for c in df.columns}
    out = df.rename(columns=cleaned_columns).rename(
        columns=_resolve_column_map(dataset_name)
    ).copy()

    out["network.source_file"] = sourceFile if sourceFile is not None else out.get("network.source_file", None)
    out["network.dataset"] = dataset_name if dataset_name is not None else out.get("network.dataset", "UNKNOWN")

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
        out["label.attack_type"] = out["label.class"].where(out["label.is_attack"].eq(1), None)
    else:
        out["label.attack_type"] = out["label.attack_type"].where(
            out["label.attack_type"].notna(),
            out["label.class"].where(out["label.is_attack"].eq(1), None),
        )

    if "label.attack_stage" not in out.columns:
        out["label.attack_stage"] = None

    if dataset_name == "CIC-IDS2018":
        inferred = out["label.class"].map(infer_cic_attack_stage)
        out["label.attack_stage"] = out["label.attack_stage"].where(
            out["label.attack_stage"].notna(), inferred
        )

    if "packet.availability" not in out.columns:
        out["packet.availability"] = False
    else:
        out["packet.availability"] = out["packet.availability"].fillna(False).astype("bool")

    _derive_context(out)

    for col in canonicalFieldNames():
        if col not in out.columns:
            out[col] = None

    canonical_cols = canonicalFieldNames()
    extra_cols = [c for c in out.columns if c not in canonical_cols]
    return out[canonical_cols + extra_cols]


def validateCanonicalDataFrame(df: pd.DataFrame) -> None:
    missing = [col for col in canonicalFieldNames() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    for col, spec in CANONICAL_SCHEMA.items():
        if not spec["nullable"] and df[col].isna().any():
            raise ValueError(f"Non-nullable column contains nulls: {col}")
