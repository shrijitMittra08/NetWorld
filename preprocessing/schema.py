from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    nullable: bool = True

CANONICAL_FIELDS: List[FieldSpec] = [
    # -------------------------
    # network.*
    # -------------------------
    FieldSpec("network.dataset", "string", False),
    FieldSpec("network.source_file", "string", False),
    FieldSpec("network.timestamp", "timestamp", True),
    FieldSpec("network.window_start", "timestamp", True),
    FieldSpec("network.window_end", "timestamp", True),

    # -------------------------
    # flow.*
    # -------------------------
    FieldSpec("flow.src_ip", "string", True),
    FieldSpec("flow.dst_ip", "string", True),
    FieldSpec("flow.src_port", "int64", True),
    FieldSpec("flow.dst_port", "int64", True),
    FieldSpec("flow.protocol", "string", True),
    FieldSpec("flow.duration", "float64", True),
    FieldSpec("flow.total_fwd_packets", "int64", True),
    FieldSpec("flow.total_bwd_packets", "int64", True),
    FieldSpec("flow.total_length_fwd_packets", "float64", True),
    FieldSpec("flow.total_length_bwd_packets", "float64", True),
    FieldSpec("flow.flow_bytes_per_sec", "float64", True),
    FieldSpec("flow.flow_packets_per_sec", "float64", True),
    FieldSpec("flow.iat_mean", "float64", True),
    FieldSpec("flow.iat_std", "float64", True),
    FieldSpec("flow.tcp_flags", "string", True),

    # -------------------------
    # packet.* (CSV-only placeholder)
    # -------------------------
    FieldSpec("packet.ttl_mean", "float64", True),
    FieldSpec("packet.ttl_var", "float64", True),
    FieldSpec("packet.tcp_window_mean", "float64", True),
    FieldSpec("packet.tcp_window_var", "float64", True),
    FieldSpec("packet.fragmentation_flags", "int64", True),
    FieldSpec("packet.payload_size_mean", "float64", True),
    FieldSpec("packet.payload_size_var", "float64", True),
    FieldSpec("packet.retransmissions", "int64", True),
    FieldSpec("packet.availability", "boolean", False),

    # -------------------------
    # behaviour.* (CSV-only placeholder)
    # -------------------------
    FieldSpec("behaviour.new_peers", "int64", True),
    FieldSpec("behaviour.new_destinations", "int64", True),
    FieldSpec("behaviour.unique_dest_ports", "int64", True),
    FieldSpec("behaviour.connection_attempts", "int64", True),
    FieldSpec("behaviour.connection_failures", "int64", True),
    FieldSpec("behaviour.auth_attempts", "int64", True),
    FieldSpec("behaviour.auth_failures", "int64", True),

    # -------------------------
    # host.* (CSV-only placeholder)
    # -------------------------
    FieldSpec("host.src_role", "string", True),
    FieldSpec("host.dst_role", "string", True),
    FieldSpec("host.asset_criticality", "int64", True),
    FieldSpec("host.src_subnet", "string", True),
    FieldSpec("host.dst_subnet", "string", True),
    FieldSpec("host.src_os", "string", True),
    FieldSpec("host.dst_os", "string", True),

    # -------------------------
    # label.*
    # -------------------------
    FieldSpec("label.class", "string", False),
    FieldSpec("label.is_attack", "int64", False),
    FieldSpec("label.attack_type", "string", True),
    FieldSpec("label.attack_stage", "string", True),
]


CANONICAL_SCHEMA: Dict[str, Dict[str, Any]] = {
    field.name: {"dtype": field.dtype, "nullable": field.nullable}
    for field in CANONICAL_FIELDS
}

def canonicalFieldNames() -> List[str]:
    return [field.name for field in CANONICAL_FIELDS]

def requiredFieldNames() -> List[str]:
    return [field.name for field in CANONICAL_FIELDS if not field.nullable]

def optionalFieldNames() -> List[str]:
    return [field.name for field in CANONICAL_FIELDS if field.nullable]

def getFieldSpec(name: str) -> FieldSpec:
    for field in CANONICAL_FIELDS:
        if field.name == name:
            return field
    raise KeyError("Unknown field name: {name}")

def validateSchemaColumns(columns: List[str]) -> None:
    missing = [name for name in canonicalFieldNames() if name not in columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

def emptyCanonicalRow() -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in CANONICAL_FIELDS:
        row[field.name] = None
    return row