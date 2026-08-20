"""CLI for evaluating a trained checkpoint on a held-out split.

Example:
    python scripts/evaluate_model.py \
        --checkpoint models/square_classifier.pt \
        --metadata data/processed/synthetic_v1/metadata.csv --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.square_dataset import SquareDataset, build_eval_transforms  # noqa: E402
from chess_ocr.inference.board_predictor import resolve_device  # noqa: E402
from chess_ocr.models.square_classifier import square_classifier_from_checkpoint  # noqa: E402
from chess_ocr.training.evaluator import Evaluator  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained square classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path("outputs/confusion_matrices/confusion_matrix.png"),
        help="Where to save the confusion matrix image (a CSV is saved alongside it)",
    )
    parser.add_argument(
        "--failure-cases",
        type=Path,
        default=Path("outputs/failure_cases/failures.csv"),
        help="Where to save the list of misclassified squares",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint.get("class_names", []))
    input_size = int(checkpoint.get("input_size", 64))

    model = square_classifier_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = SquareDataset(
        metadata_csv=args.metadata,
        data_root=args.data_root,
        split=args.split,
        transform=build_eval_transforms(input_size),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    metadata = dataset.metadata
    position_ids: list[str] | None = None
    if "position_id" in metadata.columns:
        # One board = one position rendered with one theme.
        keys = metadata["position_id"].astype(str)
        if "theme" in metadata.columns:
            keys = keys + "|" + metadata["theme"].astype(str)
        position_ids = keys.tolist()

    evaluator = Evaluator(model=model, device=device, class_names=class_names or None)
    report = evaluator.evaluate(loader, position_ids=position_ids)

    print(
        f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}, "
        f"val acc {checkpoint.get('validation_accuracy')})"
    )
    print(f"Split: {args.split} | squares: {len(dataset)} | device: {device}")
    print()
    print(report.to_text())

    csv_path = report.save_confusion_matrix(args.confusion_matrix)
    print(f"\nConfusion matrix written to {args.confusion_matrix} and {csv_path}")

    failures = report.misclassified_indices()
    if failures:
        args.failure_cases.parent.mkdir(parents=True, exist_ok=True)
        frame = metadata.iloc[failures].copy()
        frame["predicted_label"] = [
            report.class_names[report.predictions[index]] for index in failures
        ]
        frame.to_csv(args.failure_cases, index=False)
        print(f"{len(failures)} misclassified squares written to {args.failure_cases}")
    else:
        print("No misclassified squares in this split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
