"""Train the Siamese square-similarity model from generated square metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.similarity_pair_dataset import SimilarityPairDataset  # noqa: E402
from chess_ocr.models.similarity_classifier import SimilarityClassifier  # noqa: E402
from chess_ocr.training.similarity_trainer import SimilarityTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the square-similarity model.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/similarity_generated.pt")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-pairs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_dataset = SimilarityPairDataset(
        args.metadata,
        args.data_root,
        split="train",
        pairs_per_epoch=args.pairs_per_epoch,
        augment=True,
        seed=0,
    )
    val_dataset = SimilarityPairDataset(
        args.metadata,
        args.data_root,
        split="val",
        pairs_per_epoch=args.validation_pairs,
        augment=False,
        seed=1_000_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    trainer = SimilarityTrainer(
        SimilarityClassifier(),
        args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        checkpoint_metadata={
            "training_metadata": str(args.metadata),
            "trained_from_scratch": True,
            "includes_empty_class": True,
            "background_normalization": (
                "four-corner residual and neutral-gray compositing"
            ),
        },
    )
    history = trainer.fit(train_loader, val_loader, args.epochs)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibrated_model = SimilarityClassifier(int(checkpoint.get("embedding_size", 64)))
    calibrated_model.load_state_dict(checkpoint["model_state_dict"])
    calibrated_model.eval()
    similarities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for square_a, square_b, batch_targets in val_loader:
            embedding_a = calibrated_model.encode(square_a)
            embedding_b = calibrated_model.encode(square_b)
            similarities.append((embedding_a * embedding_b).sum(dim=1))
            targets.append(batch_targets)
    similarity_values = torch.cat(similarities)
    target_values = torch.cat(targets)
    negative_values = similarity_values[target_values < 0.5]
    # False merges are costlier than false splits. Select a conservative
    # threshold at the 99.5th percentile of held-out negative pairs.
    threshold = float(torch.quantile(negative_values, 0.995).clamp(-1, 1))
    checkpoint["similarity_threshold"] = threshold
    checkpoint["calibration_negative_false_positive_target"] = 0.005
    checkpoint["calibration_positive_recall"] = float(
        (similarity_values[target_values >= 0.5] >= threshold).float().mean()
    )
    checkpoint["model_state_dict"]["similarity_threshold"] = torch.tensor(threshold)
    torch.save(checkpoint, args.checkpoint)
    print(
        f"Calibrated clustering threshold {threshold:.4f} at <=0.5% validation "
        f"negative-pair false merges; positive recall "
        f"{checkpoint['calibration_positive_recall']:.4f}"
    )
    if args.history_json:
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(
            json.dumps(
                {
                    "epochs": [asdict(epoch) for epoch in history.epochs],
                    "best_epoch": history.best_epoch,
                    "best_val_accuracy": history.best_val_accuracy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
