#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

import torch

from training.cicids import (
    discover_cicids_files,
    load_cicids_csv,
    build_items_from_sequence,
)
from training.trainer import Trainer


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "raw" / "cic-ids2018"
DEFAULT_CHECKPOINT = ROOT / "models" / "networld_cicids2018.pt"


def split_by_file(file_items: Dict[Path, List[Dict[str, Any]]]):
    """Prefer scenario/file separation; fall back to deterministic class-balanced
    window splitting when a file-level split would leave the training set with
    only benign or only attack windows."""
    files = list(file_items.keys())

    def label(item):
        return int(item["targets"]["attack_sequence"][0])

    def stratified(items):
        positives = [x for x in items if label(x) == 1]
        negatives = [x for x in items if label(x) == 0]
        if not positives or not negatives:
            n = len(items)
            a = max(1, int(n * 0.70))
            b = min(n, a + max(1, int(n * 0.15)))
            return items[:a], items[a:b], items[b:]

        def cut(group):
            n = len(group)
            a = max(1, int(n * 0.70))
            b = min(n, a + max(1, int(n * 0.15)))
            return group[:a], group[a:b], group[b:]

        pt, pv, pe = cut(positives)
        nt, nv, ne = cut(negatives)
        return pt + nt, pv + nv, pe + ne

    if len(files) >= 3:
        n = len(files)
        train_end = max(1, int(n * 0.70))
        val_end = min(n - 1, train_end + max(1, int(n * 0.15)))
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        train = [item for f in train_files for item in file_items[f]]
        val = [item for f in val_files for item in file_items[f]]
        test = [item for f in test_files for item in file_items[f]]
        if train and {label(x) for x in train} == {0, 1}:
            return train, val, test

    all_items = [item for f in files for item in file_items[f]]
    return stratified(all_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NetWorld on CIC-IDS2018 CSV files.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-rows-per-file", type=int, default=250000)
    parser.add_argument("--max-sequences-per-file", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = discover_cicids_files(args.data)
    print(f"Found {len(files)} CIC-IDS2018 CSV files under {args.data.resolve()}")

    file_items: Dict[Path, List[Dict[str, Any]]] = {}
    attack_windows = 0
    total_windows = 0

    for path in files:
        print(f"\nLoading {path.name} ...")
        sequence = load_cicids_csv(
            path,
            window_seconds=args.window_seconds,
            max_rows=args.max_rows_per_file,
        )
        if sequence is None:
            print("  skipped: no usable timestamp/source/destination rows")
            continue

        items = build_items_from_sequence(
            sequence,
            sequence_length=args.sequence_length,
            rollout_steps=args.rollout_steps,
            max_sequences=args.max_sequences_per_file,
        )
        file_items[path] = items
        total_windows += len(sequence)
        attack_windows += sum(
            int(s.metadata.get("label_is_attack", 0))
            for s in sequence.snapshots
        )
        print(f"  snapshots={len(sequence)} sequences={len(items)}")

    if not file_items:
        raise RuntimeError("No trainable CIC-IDS2018 sequences were built.")

    train_items, val_items, test_items = split_by_file(file_items)

    if not train_items:
        raise RuntimeError("Training split is empty. Increase --max-sequences-per-file or reduce --sequence-length.")

    positive = sum(
        int(item["targets"]["attack_sequence"][0])
        for item in train_items
    )
    negative = max(0, len(train_items) - positive)
    pos_weight = negative / max(1, positive)

    print("\nDataset summary")
    print(f"  total snapshots: {total_windows}")
    print(f"  attack snapshots: {attack_windows}")
    print(f"  train sequences: {len(train_items)}")
    print(f"  validation sequences: {len(val_items)}")
    print(f"  test sequences: {len(test_items)}")
    print(f"  attack positive weight: {pos_weight:.3f}")

    trainer = Trainer(
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        checkpoint_path=str(args.checkpoint),
        patience=args.patience,
        attack_pos_weight=pos_weight,
    )

    result = trainer.train(
        train_items=train_items,
        val_items=val_items,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\nTraining complete")
    print(f"  epochs: {result.epochs_ran}")
    print(f"  best train loss: {result.train_loss:.6f}")
    print(f"  best validation loss: {result.val_loss:.6f}")
    print(f"  early stopped: {result.stopped_early}")
    print(f"  checkpoint: {args.checkpoint.resolve()}")
    print("\nThe Streamlit dashboard will load this checkpoint automatically.")


if __name__ == "__main__":
    main()
