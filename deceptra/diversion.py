from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class DiversionResult:
    diverted: bool
    source: str
    original_target: str
    decoy_target: str
    reason: str
    method: str = "simulation"
    commands: List[str] = None
    evasion_detected: bool = False

def build_iptables_dnat_rule(
    src_ip: str,
    original_target_ip: str,
    decoy_target_ip: str,
    dst_port: Optional[int] = None,
) -> List[str]:
    cmd = [
        "iptables",
        "-t",
        "nat",
        "-A",
        "PREROUTING",
        "-s",
        src_ip,
        "-d",
        original_target_ip,
    ]
    if dst_port is not None:
        cmd.extend(["-p", "tcp", "--dport", str(dst_port)])
    cmd.extend(["-j", "DNAT", "--to-destination", decoy_target_ip])
    return [" ".join(cmd)]

def build_ovs_flow_rule(
    in_port: str,
    original_target_ip: str,
    decoy_target_ip: str,
) -> str:
    return (
        f"ovs-ofctl add-flow br0 "
        f"\"in_port={in_port},ip,nw_dst={original_target_ip},actions=set_field:{decoy_target_ip}->ip_dst,normal\""
    )

def detect_evasion_signals(session_events: List[Dict[str, Any]]) -> bool:
    suspicious_markers = {"decoy", "honeypot", "sandbox", "virtual", "test"}
    for ev in session_events:
        text = " ".join(str(v).lower() for v in ev.values())
        if any(marker in text for marker in suspicious_markers):
            return True
    return False

def divert_session(
    source: str,
    original_target: str,
    decoy_target: str,
    reason: str = "policy_divert",
    method: str = "iptables",
    source_ip: Optional[str] = None,
    original_target_ip: Optional[str] = None,
    decoy_target_ip: Optional[str] = None,
    dst_port: Optional[int] = None,
) -> DiversionResult:
    commands: List[str] = []

    if method == "iptables" and source_ip and original_target_ip and decoy_target_ip:
        commands = build_iptables_dnat_rule(source_ip, original_target_ip, decoy_target_ip, dst_port=dst_port)
    elif method == "ovs" and source_ip and original_target_ip and decoy_target_ip:
        commands = [build_ovs_flow_rule(source_ip, original_target_ip, decoy_target_ip)]

    return DiversionResult(
        diverted=True,
        source=source,
        original_target=original_target,
        decoy_target=decoy_target,
        reason=reason,
        method=method,
        commands=commands,
        evasion_detected=False,
    )