from __future__ import annotations

from typing import Any, List, Sequence, Tuple

def temporal_split(
    items: Sequence[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[list[Any], list[Any], list[Any]]:
    """
    Split items in time order. Assumes items are already sorted temporally.
    """
    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = list(items[:train_end])
    val = list(items[train_end:val_end])
    test = list(items[val_end:])
    return train, val, test

def scenario_split(
    items: Sequence[Any],
    scenario_key_fn,
    train_scenarios: set,
    val_scenarios: set,
    test_scenarios: set,
):
    train, val, test = [], [], []
    for item in items:
        key = scenario_key_fn(item)
        if key in train_scenarios:
            train.append(item)
        elif key in val_scenarios:
            val.append(item)
        elif key in test_scenarios:
            test.append(item)
    return train, val, test