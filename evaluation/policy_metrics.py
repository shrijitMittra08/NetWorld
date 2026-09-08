from __future__ import annotations

from typing import Iterable, Sequence

def containment_rate(real_assets_protected: int, total_risky_events: int) -> float:
    return real_assets_protected / total_risky_events if total_risky_events else 0.0

def intel_yield(new_techniques_observed: int, diverted_sessions: int) -> float:
    return new_techniques_observed / diverted_sessions if diverted_sessions else 0.0

def false_divert_rate(false_diverts: int, total_diverts: int) -> float:
    return false_diverts / total_diverts if total_diverts else 0.0

def analyst_override_rate(overrides: int, total_decisions: int) -> float:
    return overrides / total_decisions if total_decisions else 0.0