from __future__ import annotations

from typing import Any, Dict, List
from torch.utils.data import Dataset

class TemporalGraphSequenceDataset(Dataset):
    """
    Expects items with:
    - graphs: list of temporal snapshots
    - targets: dict with attack/stage/target labels
    """

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]