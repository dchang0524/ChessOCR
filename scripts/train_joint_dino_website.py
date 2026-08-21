"""Fine-tune the Kaggle joint DINO model on the exact website preview theme.

Kaggle train-split pairs are replayed alongside the generated website-theme
pairs to reduce catastrophic forgetting. The reserved Kaggle test split is
never used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Dataset  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import KaggleBoardPairDataset  # noqa: E402
from chess_ocr.data.similarity_pair_dataset import SimilarityPairDataset  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    DINO_ARCHITECTURE,
    dino_joint_model_from_checkpoint,
)
from chess_ocr.training.joint_trainer import JointTrainer  # noqa: E402


class EpochConcatDataset(ConcatDataset):
    """ConcatDataset that forwards the epoch to deterministic pair samplers."""

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(epoch)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--website-metadata",
        type=Path,
        default=Path("data/processed/website_preview_v1/metadata.csv"),
    )
    parser.add_argument(
        "--kaggle-manifest",
        type=Path,
        default=Path("data/metadata/kaggle_all_90_10.csv"),
    )
    parser.add_argument(
        "--kaggle-data-root",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("models/joint_dinov2_vits14_kaggle90.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/joint_dinov2_vits14_kaggle90_website.pt"),
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=Path("outputs/training_history_joint_dinov2_vits14_kaggle90_website.json"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--website-pairs-per-epoch", type=int, default=2_000)
    parser.add_argument("--website-validation-pairs", type=int, default=500)
    parser.add_argument("--kaggle-replay-boards", type=int, default=2_000)
    parser.add_argument("--kaggle-calibration-boards", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--deployment-similarity-threshold",
        type=float,
        default=0.95,
        help=(
            "Complete-linkage cutoff stored in the checkpoint after pair calibration. "
            "The production grouped pipeline uses 0.95 to avoid board-level false merges."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--similarity-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--cross-background-positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-probability", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate_threshold(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    similarities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            square_a, square_b, target = batch[:3]
            embedding_a = model.encode(square_a.to(device))
            embedding_b = model.encode(square_b.to(device))
            similarities.append((embedding_a * embedding_b).sum(dim=1).cpu())
            targets.append(target.float())
    values = torch.cat(similarities)
    labels = torch.cat(targets)
    negatives = values[labels < 0.5]
    threshold = float(torch.quantile(negatives, 0.995).clamp(-1, 1))
    return {
        "similarity_threshold": threshold,
        "calibration_positive_recall": float(
            (values[labels >= 0.5] >= threshold).float().mean()
        ),
        "calibration_negative_false_positive_rate": float(
            (negatives >= threshold).float().mean()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.initial_checkpoint.resolve() == args.checkpoint.resolve():
        raise ValueError("Output checkpoint must differ from the preserved initial checkpoint")
    positive_counts = (
        args.epochs,
        args.website_pairs_per_epoch,
        args.website_validation_pairs,
        args.kaggle_replay_boards,
        args.kaggle_calibration_boards,
        args.batch_size,
    )
    if any(value <= 0 for value in positive_counts) or args.num_workers < 0:
        raise ValueError("Epoch, sample and batch counts must be positive")
    if not -1.0 <= args.deployment_similarity_threshold <= 1.0:
        raise ValueError("deployment-similarity-threshold must be in [-1, 1]")
    torch.manual_seed(args.seed)

    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("architecture") != DINO_ARCHITECTURE:
        raise ValueError("Initial checkpoint is not the joint DINOv2 architecture")
    model = dino_joint_model_from_checkpoint(initial)
    model.load_state_dict(initial["model_state_dict"])
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    input_size = int(initial["input_size"])

    website_options = {
        "metadata_csv": args.website_metadata,
        "input_size": input_size,
        "cross_background_positive_weight": args.cross_background_positive_weight,
        "include_class_labels": True,
        "include_cross_background_flag": True,
        "hard_negative_probability": args.hard_negative_probability,
    }
    website_train = SimilarityPairDataset(
        **website_options,
        split="train",
        pairs_per_epoch=args.website_pairs_per_epoch,
        augment=True,
        seed=args.seed,
    )
    website_validation = SimilarityPairDataset(
        **website_options,
        split="val",
        pairs_per_epoch=args.website_validation_pairs,
        augment=False,
        seed=args.seed + 1_000_000,
    )
    kaggle_options = {
        "manifest_csv": args.kaggle_manifest,
        "data_root": args.kaggle_data_root,
        "split": "train",
        "input_size": input_size,
        "cross_background_positive_weight": args.cross_background_positive_weight,
        "hard_negative_probability": args.hard_negative_probability,
        "seed": args.seed,
    }
    kaggle_replay = KaggleBoardPairDataset.from_manifest(
        **kaggle_options,
        max_boards=args.kaggle_replay_boards,
        augment=True,
        crop_jitter_pixels=4,
        crop_jitter_probability=0.8,
    )
    kaggle_calibration = KaggleBoardPairDataset.from_manifest(
        **kaggle_options,
        max_boards=args.kaggle_calibration_boards,
        augment=False,
    )
    train_dataset: Dataset = EpochConcatDataset([website_train, kaggle_replay])
    validation_dataset: Dataset = EpochConcatDataset(
        [website_validation, kaggle_calibration]
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"Mixed pairs: train {len(train_dataset):,} "
        f"({len(website_train):,} website + {len(kaggle_replay):,} Kaggle); "
        f"validation {len(validation_dataset):,}",
        flush=True,
    )
    trainer = JointTrainer(
        model=model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        similarity_loss_weight=args.similarity_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        checkpoint_metadata={
            "training_data_source": "generated website preview plus Kaggle train replay",
            "website_training_metadata": str(args.website_metadata),
            "website_training_metadata_sha256": file_sha256(args.website_metadata),
            "kaggle_training_manifest": str(args.kaggle_manifest),
            "kaggle_training_manifest_sha256": file_sha256(args.kaggle_manifest),
            "initial_checkpoint": str(args.initial_checkpoint),
            "website_pairs_per_epoch": len(website_train),
            "kaggle_replay_boards": len(kaggle_replay),
            "reserved_kaggle_test_used_for_training": False,
            "pretrained_backbone": initial.get("pretrained_backbone"),
            "backbone_frozen": True,
            "background_normalization": "disabled",
            "cross_background_positive_weight": args.cross_background_positive_weight,
            "hard_negative_probability": args.hard_negative_probability,
            "similarity_loss_weight": args.similarity_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
        },
    )
    history = trainer.fit(
        train_loader,
        validation_loader,
        epochs=args.epochs,
        start_epoch=int(initial.get("epoch", 0)),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(trainer.device)
    calibration = calibrate_threshold(model, validation_loader, trainer.device)
    checkpoint.update(calibration)
    checkpoint["pair_calibrated_similarity_threshold"] = calibration[
        "similarity_threshold"
    ]
    checkpoint["similarity_threshold"] = args.deployment_similarity_threshold
    checkpoint["calibration_negative_false_positive_target"] = 0.005
    checkpoint["threshold_calibration_source"] = (
        "website validation plus Kaggle training-split calibration"
    )
    torch.save(checkpoint, args.checkpoint)

    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(
        json.dumps(
            {
                "epochs": [asdict(epoch) for epoch in history.epochs],
                "best_epoch": history.best_epoch,
                "best_validation_class_accuracy": history.best_val_class_accuracy,
                "checkpoint_path": str(args.checkpoint),
                "initial_checkpoint": str(args.initial_checkpoint),
                "website_pairs_per_epoch": len(website_train),
                "kaggle_replay_boards": len(kaggle_replay),
                "website_validation_pairs": len(website_validation),
                "kaggle_calibration_boards": len(kaggle_calibration),
                "reserved_kaggle_test_used_for_training": False,
                "deployment_similarity_threshold": args.deployment_similarity_threshold,
                **calibration,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {args.checkpoint}; calibrated threshold "
        f"{calibration['similarity_threshold']:.6f}; deployment threshold "
        f"{args.deployment_similarity_threshold:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
