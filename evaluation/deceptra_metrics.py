from __future__ import annotations

from typing import Iterable

def decoy_dwell_time(total_seconds: float, session_count: int) -> float:
    return total_seconds / session_count if session_count else 0.0

def detection_evasion_rate(evasion_events: int, total_decoy_sessions: int) -> float:
    return evasion_events / total_decoy_sessions if total_decoy_sessions else 0.0

def feedback_loop_latency(event_to_retrain_seconds: float) -> float:
    return float(event_to_retrain_seconds)