from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from preprocessing.feature_extractor import extractFeatures
from preprocessing.windowing import add_time_windows
from graph.graph_builder import build_temporal_graphs
from forecasting import ForecastEngine
from evaluation import evaluate_detection


def evaluate_csv(csv_path: str, window_seconds: int = 5) -> dict:
    raw = pd.read_csv(csv_path)
    canonical = extractFeatures(raw, source_file=Path(csv_path).name)
    windowed = add_time_windows(canonical, window_seconds=window_seconds)
    sequence = build_temporal_graphs(windowed)
    engine = ForecastEngine()

    probs, labels = [], []
    for i, snap in enumerate(sequence.snapshots):
        result = engine.forecast(sequence.snapshots[: i + 1])
        probs.append(float(result["final_attack_probability"]))
        labels.append(int(snap.metadata.get("label_is_attack", 0)))

    if len(set(labels)) < 2:
        return {"windows": len(labels), "labelled": False, "reason": "Need both benign and attack windows"}

    metrics = evaluate_detection([int(p >= 0.5) for p in probs], labels, probs=probs)
    return {"windows": len(labels), "labelled": True, **metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NetWorld on a labelled CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--window-seconds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate_csv(args.csv_path, args.window_seconds), indent=2))
