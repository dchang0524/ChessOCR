"""CLI for training the square classifier.

Example:
    python scripts/train_model.py \
        --metadata data/processed/synthetic_v1/metadata.csv \
        --checkpoint models/square_classifier.pt --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.square_dataset import (  # noqa: E402
    SquareDataset,
    build_eval_transforms,
    build_train_transforms,
)
from chess_ocr.models.square_classifier import SquareClassifier  # noqa: E402
from chess_ocr.training.trainer import Trainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chess square classifier.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root that image_path entries are relative to (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/square_classifier.pt"),
        help="Where to write the best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to train on; auto-detected when omitted",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Optional path for writing the training history as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    train_dataset = SquareDataset(
        metadata_csv=args.metadata,
        data_root=args.data_root,
        split="train",
        transform=build_train_transforms(args.input_size),
    )
    val_dataset = SquareDataset(
        metadata_csv=args.metadata,
        data_root=args.data_root,
        split="val",
        transform=build_eval_transforms(args.input_size),
    )
    print(f"train squares: {len(train_dataset)} | val squares: {len(val_dataset)}")
    print(f"train class distribution: {train_dataset.class_distribution()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=SquareClassifier(),
        checkpoint_path=args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        input_size=args.input_size,
    )
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    if args.history_json is not None:
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "best_epoch": history.best_epoch,
            "best_val_accuracy": history.best_val_accuracy,
            "checkpoint_path": str(history.checkpoint_path),
        }
        args.history_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"History written to {args.history_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
