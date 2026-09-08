from __future__ import annotations
from typing import Any, Dict


def update_policy_from_feedback(
    policy: Any,
    action_index: int,
    reward: float,
    metadata: Dict[str, Any] | None = None,
) -> None:
    metadata = metadata or {}
    state = metadata.get("state", metadata)
    policy.update(action_index=action_index, reward=reward, state=state)
