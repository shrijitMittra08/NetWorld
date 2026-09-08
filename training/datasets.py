from __future__ import annotations

from typing import Any, Dict, List

from torch.utils.data import Dataset

class GraphSequenceDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_identity(batch):
    return batch