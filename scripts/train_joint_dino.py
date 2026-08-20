"""Train one DINOv2 backbone for square classification and appearance grouping."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.similarity_pair_dataset import SimilarityPairDataset  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    DINO_EMBEDDING_SIZE,
    DINO_INPUT_SIZE,
    DinoJointClassifier,
    dino_joint_model_from_checkpoint,
)
from chess_ocr.training.joint_trainer import JointTrainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/joint_dinov2_vits14.pt")
    )
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=20_000)
    parser.add_argument("--validation-pairs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=DINO_INPUT_SIZE)
    parser.add_argument("--embedding-size", type=int, default=DINO_EMBEDDING_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--similarity-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--cross-background-positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-probability", type=float, default=0.75)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--trainable-backbone-blocks",
        type=int,
        default=None,
        help="Freeze DINO except for its final N transformer blocks",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument(
        "--validation-theme-families",
        nargs="*",
        default=None,
        help=(
            "Hold out complete sprite families for validation, for example "
            "--validation-theme-families spatial"
        ),
    )
    return parser.parse_args(argv)


def _configure_backbone(
    model: DinoJointClassifier,
    freeze_backbone: bool,
    trainable_backbone_blocks: int | None,
) -> None:
    if freeze_backbone and trainable_backbone_blocks is not None:
        raise ValueError(
            "--freeze-backbone and --trainable-backbone-blocks are mutually exclusive"
        )
    if not freeze_backbone and trainable_backbone_blocks is None:
        return
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if freeze_backbone:
        return
    blocks = model.features.blocks
    if not 0 <= trainable_backbone_blocks <= len(blocks):
        raise ValueError(
            f"--trainable-backbone-blocks must be between 0 and {len(blocks)}"
        )
    if trainable_backbone_blocks:
        for block in blocks[-trainable_backbone_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    for parameter in model.features.norm.parameters():
        parameter.requires_grad = True


def _split_theme_families(
    metadata: Path, held_out_families: list[str] | None
) -> tuple[set[str] | None, set[str] | None]:
    """Return train/validation themes with complete families held out."""
    if not held_out_families:
        return None, None
    frame = pd.read_csv(metadata, usecols=["theme"])
    themes = {str(theme) for theme in frame["theme"].unique()}
    held_out = set(held_out_families)

    def family(theme: str) -> str:
        return theme.rsplit("_", maxsplit=1)[0]

    available_families = {family(theme) for theme in themes}
    unknown = held_out - available_families
    if unknown:
        raise ValueError(f"Unknown validation theme families: {sorted(unknown)}")
    validation_themes = {theme for theme in themes if family(theme) in held_out}
    train_themes = themes - validation_themes
    if not train_themes or not validation_themes:
        raise ValueError("Theme-family holdout must leave training and validation themes")
    return train_themes, validation_themes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_themes, validation_themes = _split_theme_families(
        args.metadata, args.validation_theme_families
    )
    dataset_options = {
        "metadata_csv": args.metadata,
        "data_root": args.data_root,
        "input_size": args.input_size,
        "cross_background_positive_weight": args.cross_background_positive_weight,
        "include_class_labels": True,
        "include_cross_background_flag": True,
    }
    train_dataset = SimilarityPairDataset(
        **dataset_options,
        split="train",
        themes=train_themes,
        pairs_per_epoch=args.pairs_per_epoch,
        augment=True,
        seed=0,
        hard_negative_probability=args.hard_negative_probability,
    )
    val_dataset = SimilarityPairDataset(
        **dataset_options,
        split="val",
        themes=validation_themes,
        pairs_per_epoch=args.validation_pairs,
        augment=False,
        seed=1_000_000,
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

    start_epoch = 0
    if args.initial_checkpoint is None:
        model = DinoJointClassifier(
            embedding_size=args.embedding_size,
            input_size=args.input_size,
            pretrained_backbone=True,
        )
    else:
        checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        model = dino_joint_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        if model.input_size != args.input_size:
            raise ValueError("Initial checkpoint input size does not match --input-size")
        if model.embedding_size != args.embedding_size:
            raise ValueError("Initial checkpoint embedding size does not match")
    _configure_backbone(model, args.freeze_backbone, args.trainable_backbone_blocks)

    trainer = JointTrainer(
        model,
        args.checkpoint,
        device=args.device,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        similarity_loss_weight=args.similarity_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        checkpoint_metadata={
            "training_metadata": str(args.metadata),
            "training_data_source": "generated-only",
            "pretrained_backbone": "DINOv2 ViT-S/14 LVD-142M",
            "shape_combiner": "learned-query six-head patch attention plus CLS token",
            "background_normalization": "disabled",
            "background_invariance": "paired cross-background consistency loss",
            "cross_background_positive_weight": args.cross_background_positive_weight,
            "hard_negative_probability": args.hard_negative_probability,
            "similarity_loss_weight": args.similarity_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "backbone_frozen": args.freeze_backbone,
            "trainable_backbone_blocks": args.trainable_backbone_blocks,
            "validation_theme_families": args.validation_theme_families,
        },
    )
    history = trainer.fit(
        train_loader, val_loader, args.epochs, start_epoch=start_epoch
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(trainer.device).eval()
    similarities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            square_a, square_b, pair_targets = batch[:3]
            square_a = square_a.to(trainer.device)
            square_b = square_b.to(trainer.device)
            embeddings_a = model.encode(square_a)
            embeddings_b = model.encode(square_b)
            similarities.append((embeddings_a * embeddings_b).sum(dim=1).cpu())
            targets.append(pair_targets.float().cpu())
    similarity_values = torch.cat(similarities)
    target_values = torch.cat(targets)
    negative_values = similarity_values[target_values < 0.5]
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
        f"Calibrated threshold {threshold:.4f}; positive recall "
        f"{checkpoint['calibration_positive_recall']:.4f}; negative false-positive "
        f"rate {checkpoint['calibration_negative_false_positive_rate']:.4f}"
    )
    if args.history_json is not None:
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(
            json.dumps(
                {
                    "epochs": [asdict(epoch) for epoch in history.epochs],
                    "best_epoch": history.best_epoch,
                    "best_val_class_accuracy": history.best_val_class_accuracy,
                    "checkpoint_path": str(history.checkpoint_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
