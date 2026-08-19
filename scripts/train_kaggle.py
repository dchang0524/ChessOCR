"""Fine-tune the square CNN on the Kaggle 80/10/10 board split.

Each JPEG is decoded once per epoch. All occupied squares and a randomized
subset of empty squares are used for training; validation still evaluates all
64 squares from every board.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import (  # noqa: E402
    KaggleBoardDataset,
    collate_kaggle_boards,
)
from chess_ocr.data.labels import CLASS_NAMES  # noqa: E402
from chess_ocr.models.square_classifier import SquareClassifier  # noqa: E402
from chess_ocr.training.trainer import Trainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the CNN on Kaggle full boards.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/kaggle_80_10_10.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/square_classifier_kaggle.pt"),
    )
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--history-json", type=Path, default=Path("outputs/training_history_kaggle.json")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--board-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--empty-samples", type=int, default=16)
    parser.add_argument("--crop-jitter-pixels", type=int, default=5)
    parser.add_argument("--crop-jitter-probability", type=float, default=0.8)
    parser.add_argument("--color-jitter", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--max-train-boards", type=int, default=None)
    parser.add_argument("--max-val-boards", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    return parser.parse_args(argv)


class BoardProgressLoader:
    """Loader wrapper that reports progress in full-board units."""

    def __init__(
        self,
        loader: DataLoader,
        phase: str,
        board_count: int,
        board_batch_size: int,
        progress_every: int,
    ) -> None:
        self.loader = loader
        self.phase = phase
        self.board_count = board_count
        self.board_batch_size = board_batch_size
        self.progress_every = progress_every

    def __len__(self) -> int:
        """Return the underlying loader length."""
        return len(self.loader)

    def __iter__(self) -> Any:
        """Yield batches while reporting board throughput."""
        started = time.monotonic()
        last_reported = 0
        for batch_index, batch in enumerate(self.loader, start=1):
            yield batch
            completed = min(batch_index * self.board_batch_size, self.board_count)
            if completed == self.board_count or completed - last_reported >= self.progress_every:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"{self.phase}: {completed:,}/{self.board_count:,} boards "
                    f"({rate:.1f} boards/s)",
                    flush=True,
                )
                last_reported = completed


def load_initial_model(path: Path | None) -> tuple[SquareClassifier, str | None]:
    """Create a fresh model or load weights from an existing checkpoint."""
    model = SquareClassifier()
    if path is None:
        return model, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    if class_names != CLASS_NAMES:
        raise ValueError("Initial checkpoint class ordering does not match this project")
    if int(checkpoint.get("input_size", 64)) != 64:
        raise ValueError("Initial checkpoint input size is not 64")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, str(path)


def main(argv: list[str] | None = None) -> int:
    """Train and checkpoint the Kaggle-fine-tuned classifier."""
    args = parse_args(argv)
    if args.board_batch_size <= 0:
        raise ValueError("board-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    torch.manual_seed(args.seed)
    train_dataset = KaggleBoardDataset.from_manifest(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        split="train",
        input_size=64,
        max_boards=args.max_train_boards,
        augment=True,
        empty_samples=args.empty_samples,
        crop_jitter_pixels=args.crop_jitter_pixels,
        crop_jitter_probability=args.crop_jitter_probability,
        color_jitter=args.color_jitter,
    )
    val_dataset = KaggleBoardDataset.from_manifest(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        split="val",
        input_size=64,
        max_boards=args.max_val_boards,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.board_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_kaggle_boards,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.board_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_kaggle_boards,
        persistent_workers=args.num_workers > 0,
    )
    train_progress = BoardProgressLoader(
        train_loader,
        phase="train",
        board_count=len(train_dataset),
        board_batch_size=args.board_batch_size,
        progress_every=args.progress_every,
    )
    val_progress = BoardProgressLoader(
        val_loader,
        phase="val",
        board_count=len(val_dataset),
        board_batch_size=args.board_batch_size,
        progress_every=args.progress_every,
    )

    model, initial_checkpoint = load_initial_model(args.initial_checkpoint)
    print(
        f"Kaggle boards: train {len(train_dataset):,} | val {len(val_dataset):,} | "
        f"sampled empty squares/board {args.empty_samples}",
        flush=True,
    )
    trainer = Trainer(
        model=model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        checkpoint_metadata={
            "training_dataset": "koryakinp/chess-positions",
            "split_manifest": str(args.manifest),
            "split_seed": args.seed,
            "train_board_count": len(train_dataset),
            "validation_board_count": len(val_dataset),
            "empty_samples_per_board": args.empty_samples,
            "initial_checkpoint": initial_checkpoint,
        },
    )
    history = trainer.fit(train_progress, val_progress, epochs=args.epochs)

    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs": [asdict(epoch) for epoch in history.epochs],
        "best_epoch": history.best_epoch,
        "best_val_accuracy": history.best_val_accuracy,
        "checkpoint_path": str(history.checkpoint_path),
        "train_board_count": len(train_dataset),
        "validation_board_count": len(val_dataset),
        "split_manifest": str(args.manifest),
        "seed": args.seed,
    }
    args.history_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"History written to {args.history_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
