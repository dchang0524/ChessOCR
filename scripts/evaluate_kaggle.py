"""Evaluate a checkpoint on the Kaggle chess-positions test boards.

The dataset is read lazily as full boards; no square crops are saved.

Example:
    python scripts/evaluate_kaggle.py \
        --checkpoint models/square_classifier_2d.pt \
        --image-dir data/raw/kaggle_chess_positions/test
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
    NUM_SQUARES,
    KaggleBoardDataset,
    collate_kaggle_boards,
)
from chess_ocr.inference.board_predictor import resolve_device  # noqa: E402
from chess_ocr.models.square_classifier import SquareClassifier  # noqa: E402
from chess_ocr.training.evaluator import EvaluationReport, Evaluator  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a square classifier on Kaggle full-board images."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/square_classifier_2d.pt"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions/test"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional split manifest; when set, --image-dir is used as its data root",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Manifest split to evaluate",
    )
    parser.add_argument("--board-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-boards", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/kaggle_evaluation.json"))
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path("outputs/confusion_matrices/kaggle_test.png"),
    )
    return parser.parse_args(argv)


class ProgressLoader:
    """Data-loader wrapper that prints board-level progress and throughput."""

    def __init__(
        self,
        loader: DataLoader,
        board_count: int,
        board_batch_size: int,
        progress_every: int,
    ) -> None:
        self.loader = loader
        self.board_count = board_count
        self.board_batch_size = board_batch_size
        self.progress_every = progress_every

    def __len__(self) -> int:
        """Return the underlying loader length."""
        return len(self.loader)

    def __iter__(self) -> Any:
        """Yield square batches while periodically printing progress."""
        started = time.monotonic()
        last_reported = 0
        for batch_index, batch in enumerate(self.loader, start=1):
            yield batch
            completed = min(batch_index * self.board_batch_size, self.board_count)
            if completed == self.board_count or completed - last_reported >= self.progress_every:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"Evaluated {completed:,}/{self.board_count:,} boards ({rate:.1f} boards/s)",
                    flush=True,
                )
                last_reported = completed


def report_to_dict(
    report: EvaluationReport,
    checkpoint: Path,
    image_dir: Path,
    elapsed_seconds: float,
    manifest: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Convert an evaluation report into serializable benchmark metadata."""
    return {
        "checkpoint": str(checkpoint),
        "image_dir": str(image_dir),
        "manifest": str(manifest) if manifest is not None else None,
        "split": split if manifest is not None else None,
        "elapsed_seconds": elapsed_seconds,
        "overall_accuracy": report.overall_accuracy,
        "empty_accuracy": report.empty_accuracy,
        "occupied_accuracy": report.occupied_accuracy,
        "per_class": [asdict(metrics) for metrics in report.per_class],
        "board_metrics": asdict(report.board_metrics) if report.board_metrics else None,
        "confusion_matrix": report.confusion.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the full Kaggle evaluation."""
    args = parse_args(argv)
    if args.board_batch_size <= 0:
        raise ValueError("board-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint.get("class_names", []))
    input_size = int(checkpoint.get("input_size", 64))
    model = SquareClassifier(num_classes=len(class_names) or 13)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.manifest is None:
        dataset = KaggleBoardDataset(
            image_dir=args.image_dir,
            input_size=input_size,
            max_boards=args.max_boards,
        )
    else:
        dataset = KaggleBoardDataset.from_manifest(
            manifest_csv=args.manifest,
            data_root=args.image_dir,
            split=args.split,
            input_size=input_size,
            max_boards=args.max_boards,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.board_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_kaggle_boards,
        persistent_workers=args.num_workers > 0,
    )
    progress_loader = ProgressLoader(
        loader=loader,
        board_count=len(dataset),
        board_batch_size=args.board_batch_size,
        progress_every=args.progress_every,
    )
    position_ids = [path.stem for path in dataset.paths for _ in range(NUM_SQUARES)]

    print(
        f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}, "
        f"validation accuracy {checkpoint.get('validation_accuracy')})"
    )
    print(
        f"Kaggle test boards: {len(dataset):,} | squares: "
        f"{len(dataset) * NUM_SQUARES:,} | device: {device}",
        flush=True,
    )
    started = time.monotonic()
    evaluator = Evaluator(model=model, device=device, class_names=class_names or None)
    report = evaluator.evaluate(progress_loader, position_ids=position_ids)
    elapsed_seconds = time.monotonic() - started

    print()
    print(report.to_text())
    print(f"\nElapsed: {elapsed_seconds:.1f} seconds")

    csv_path = report.save_confusion_matrix(args.confusion_matrix)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_dict(
        report=report,
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        elapsed_seconds=elapsed_seconds,
        manifest=args.manifest,
        split=args.split,
    )
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {args.output_json}")
    print(f"Confusion matrix written to {args.confusion_matrix} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
