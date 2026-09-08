from __future__ import annotations

from typing import Iterable, Optional, Sequence

def early_warning_time(
    prediction_times: Sequence[float],
    event_time: float,
    threshold_cross_time: Optional[float],
) -> float:
    """
    Positive means predicted before the event.
    """
    if threshold_cross_time is None:
        return 0.0
    return float(event_time - threshold_cross_time)