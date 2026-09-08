from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime, timezone

@dataclass
class CapturedSession:
    session_id: str
    source: str
    target: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    evasion_detected: bool = False
    closed: bool = False

@dataclass
class SessionCapture:
    sessions: Dict[str, CapturedSession] = field(default_factory=dict)

    def start_session(self, session_id: str, source: str, target: str) -> CapturedSession:
        session = CapturedSession(session_id=session_id, source=source, target=target)
        self.sessions[session_id] = session
        return session

    def record_event(self, session_id: str, event: Dict[str, Any]) -> None:
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")
        ev = dict(event)
        ev.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.sessions[session_id].events.append(ev)

    def mark_evasion(self, session_id: str) -> None:
        self.get_session(session_id).evasion_detected = True

    def close_session(self, session_id: str) -> None:
        self.get_session(session_id).closed = True

    def get_session(self, session_id: str) -> CapturedSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self.sessions[session_id]