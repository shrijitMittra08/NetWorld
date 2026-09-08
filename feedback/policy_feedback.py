from __future__ import annotations

from typing import Any, Dict, List, Optional

from policy import LearnedPolicy

def update_policy_from_feedback(
    learned_policy: LearnedPolicy,
    action_index: int,
    reward: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Feed reward back into the learned policy.
    """
    learned_policy.update(action_index=action_index, reward=reward)