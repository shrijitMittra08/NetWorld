from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import hashlib
import json

def _sample_hash(sample: Dict[str, Any]) -> str:
    normalized = json.dumps(sample, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

@dataclass
class SampleManager:
    seen_hashes: set[str] = field(default_factory=set)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def add_sample(self, sample: Dict[str, Any]) -> bool:
        h = _sample_hash(sample)
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)
        self.samples.append(sample)
        return True

    def add_many(self, samples: List[Dict[str, Any]]) -> int:
        added = 0
        for sample in samples:
            if self.add_sample(sample):
                added += 1
        return added

    def all_samples(self) -> List[Dict[str, Any]]:
        return list(self.samples)