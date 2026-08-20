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
from chess_ocr.models.similarity_classifier import (  # noqa: E402
    SIMILARITY_ARCHITECTURES,
    SimilarityClassifier,
    similarity_model_from_checkpoint,
)
from chess_ocr.training.similarity_trainer import SimilarityTrainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the square-similarity model.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/similarity_generated.pt")
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional similarity checkpoint whose encoder weights are fine-tuned",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-pairs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument(
        "--architecture",
        choices=SIMILARITY_ARCHITECTURES,
        default="compact",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize a transfer architecture with torchvision ImageNet weights",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the projection/pair-decision head during this run",
    )
    parser.add_argument(
        "--trainable-backbone-blocks",
        type=int,
        default=None,
        help="Freeze the backbone except for its final N feature blocks",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cross-background-positive-weight",
        type=float,
        default=3.0,
        help=(
            "Loss multiplier for same-piece positive pairs sampled from opposite "
            "light/dark square backgrounds"
        ),
    )
    parser.add_argument(
        "--hard-negative-probability",
        type=float,
        default=0.0,
        help="Probability that a negative pair shares piece colour or piece type",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    return parser.parse_args(argv)


def load_initial_model(
    path: Path | None,
    architecture: str = "compact",
    embedding_size: int = 64,
    input_size: int = 64,
    pretrained_backbone: bool = False,
) -> tuple[SimilarityClassifier, int]:
    """Create a fresh model or restore encoder weights for continued training.

    The pair-decision threshold is deliberately reset. Older checkpoints store
    the separately calibrated clustering cutoff in that state-dict entry, so
    reusing it as the trainable BCE boundary would make the first resumed epoch
    start from an artificially conservative decision rule.
    """
    if path is None:
        return SimilarityClassifier(
            embedding_size,
            architecture,
            pretrained_backbone,
            input_size,
        ), 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = similarity_model_from_checkpoint(checkpoint)
    state_dict = dict(checkpoint["model_state_dict"])
    saved_boundary = state_dict.get("similarity_threshold")
    exported_boundary = checkpoint.get("similarity_threshold")
    boundary_is_calibrated = (
        saved_boundary is None
        or exported_boundary is None
        or torch.isclose(
            saved_boundary.detach().float().cpu(),
            torch.tensor(float(exported_boundary)),
        ).item()
    )
    if boundary_is_calibrated:
        state_dict.pop("similarity_threshold", None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = ["similarity_threshold"] if boundary_is_calibrated else []
    if incompatible.missing_keys != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Initial similarity checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model, int(checkpoint.get("epoch", 0))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_dataset = SimilarityPairDataset(
        args.metadata,
        args.data_root,
        split="train",
        pairs_per_epoch=args.pairs_per_epoch,
        augment=True,
        seed=0,
        cross_background_positive_weight=args.cross_background_positive_weight,
        input_size=args.input_size,
        hard_negative_probability=args.hard_negative_probability,
    )
    val_dataset = SimilarityPairDataset(
        args.metadata,
        args.data_root,
        split="val",
        pairs_per_epoch=args.validation_pairs,
        augment=False,
        seed=1_000_000,
        cross_background_positive_weight=1.0,
        input_size=args.input_size,
        hard_negative_probability=1.0,
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
    model, start_epoch = load_initial_model(
        args.initial_checkpoint,
        args.architecture,
        args.embedding_size,
        args.input_size,
        args.pretrained_backbone,
    )
    if model.input_size != args.input_size:
        raise ValueError(
            f"Initial checkpoint input size is {model.input_size}, not {args.input_size}"
        )
    if args.freeze_backbone and args.trainable_backbone_blocks is not None:
        raise ValueError(
            "--freeze-backbone and --trainable-backbone-blocks are mutually exclusive"
        )
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    elif args.trainable_backbone_blocks is not None:
        block_count = len(model.features)
        if not 0 <= args.trainable_backbone_blocks <= block_count:
            raise ValueError(
                f"--trainable-backbone-blocks must be between 0 and {block_count}"
            )
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        if args.trainable_backbone_blocks:
            for block in model.features[-args.trainable_backbone_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
    trainer = SimilarityTrainer(
        model,
        args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_metadata={
            "training_metadata": str(args.metadata),
            "trained_from_scratch": (
                args.initial_checkpoint is None and not args.pretrained_backbone
            ),
            "training_data_source": "generated-only",
            "initial_checkpoint": (
                str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
            ),
            "continued_from_epoch": start_epoch,
            "includes_empty_class": True,
            "background_normalization": "disabled",
            "background_variation": "generator-only, before piece compositing",
            "cross_background_positive_weight": args.cross_background_positive_weight,
            "hard_negative_probability": args.hard_negative_probability,
            "state_dict_similarity_threshold": "learned",
            "pretrained_backbone": (
                args.pretrained_backbone if args.initial_checkpoint is None else None
            ),
            "backbone_frozen": args.freeze_backbone,
            "trainable_backbone_blocks": args.trainable_backbone_blocks,
        },
    )
    history = trainer.fit(train_loader, val_loader, args.epochs, start_epoch=start_epoch)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibrated_model = similarity_model_from_checkpoint(checkpoint)
    calibrated_model.load_state_dict(checkpoint["model_state_dict"])
    calibrated_model.eval()
    similarities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for square_a, square_b, batch_targets, _ in val_loader:
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
    checkpoint["calibration_negative_false_positive_rate"] = float(
        (negative_values >= threshold).float().mean()
    )
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
                    "similarity_threshold": threshold,
                    "calibration_positive_recall": checkpoint[
                        "calibration_positive_recall"
                    ],
                    "calibration_negative_false_positive_rate": checkpoint[
                        "calibration_negative_false_positive_rate"
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
