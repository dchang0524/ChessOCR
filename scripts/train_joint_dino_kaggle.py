"""Fine-tune the current joint DINO model on a Kaggle 90/10 board manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import KaggleBoardPairDataset  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    DINO_ARCHITECTURE,
    dino_joint_model_from_checkpoint,
)
from chess_ocr.training.joint_trainer import JointTrainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/kaggle_all_90_10.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("models/joint_dinov2_vits14_head.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/joint_dinov2_vits14_kaggle90.pt"),
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=Path("outputs/training_history_joint_dinov2_vits14_kaggle90.json"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--similarity-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--cross-background-positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-probability", type=float, default=0.75)
    parser.add_argument("--crop-jitter-pixels", type=int, default=4)
    parser.add_argument("--crop-jitter-probability", type=float, default=0.8)
    parser.add_argument("--calibration-boards", type=int, default=5_000)
    parser.add_argument("--max-train-boards", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    return parser.parse_args(argv)


class PairProgressLoader:
    """DataLoader wrapper that emits bounded progress updates in board units."""

    def __init__(self, loader: DataLoader, phase: str, progress_every: int) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.phase = phase
        self.progress_every = progress_every

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Any:
        started = time.monotonic()
        completed = 0
        last_reported = 0
        for batch in self.loader:
            completed += int(batch[2].shape[0])
            yield batch
            if completed == len(self.dataset) or completed - last_reported >= self.progress_every:
                elapsed = time.monotonic() - started
                print(
                    f"{self.phase}: {completed:,}/{len(self.dataset):,} boards "
                    f"({completed / max(elapsed, 1e-9):.1f} boards/s)",
                    flush=True,
                )
                last_reported = completed


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibrate_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    similarities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            square_a, square_b, pair_targets = batch[:3]
            square_a = square_a.to(device)
            square_b = square_b.to(device)
            embedding_a = model.encode(square_a)
            embedding_b = model.encode(square_b)
            similarities.append((embedding_a * embedding_b).sum(dim=1).cpu())
            targets.append(pair_targets.float().cpu())
    similarity_values = torch.cat(similarities)
    target_values = torch.cat(targets)
    negative_values = similarity_values[target_values < 0.5]
    threshold = float(torch.quantile(negative_values, 0.995).clamp(-1, 1))
    return {
        "similarity_threshold": threshold,
        "calibration_positive_recall": float(
            (similarity_values[target_values >= 0.5] >= threshold).float().mean()
        ),
        "calibration_negative_false_positive_rate": float(
            (negative_values >= threshold).float().mean()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.initial_checkpoint.resolve() == args.checkpoint.resolve():
        raise ValueError("Output checkpoint must differ from the preserved initial checkpoint")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.calibration_boards <= 0 or args.progress_every <= 0:
        raise ValueError("calibration-boards and progress-every must be positive")
    torch.manual_seed(args.seed)

    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("architecture") != DINO_ARCHITECTURE:
        raise ValueError("Initial checkpoint is not the joint DINOv2 architecture")
    model = dino_joint_model_from_checkpoint(initial)
    model.load_state_dict(initial["model_state_dict"])
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    dataset_options = {
        "manifest_csv": args.manifest,
        "data_root": args.data_root,
        "split": "train",
        "input_size": int(initial["input_size"]),
        "cross_background_positive_weight": args.cross_background_positive_weight,
        "hard_negative_probability": args.hard_negative_probability,
        "seed": args.seed,
    }
    train_dataset = KaggleBoardPairDataset.from_manifest(
        **dataset_options,
        max_boards=args.max_train_boards,
        augment=True,
        crop_jitter_pixels=args.crop_jitter_pixels,
        crop_jitter_probability=args.crop_jitter_probability,
    )
    calibration_dataset = KaggleBoardPairDataset.from_manifest(
        **dataset_options,
        max_boards=min(args.calibration_boards, len(train_dataset)),
        augment=False,
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
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    train_progress = PairProgressLoader(train_loader, "train", args.progress_every)
    calibration_progress = PairProgressLoader(
        calibration_loader, "train-calibration", args.progress_every
    )
    print(
        f"Kaggle pair boards: train {len(train_dataset):,} | "
        f"training calibration {len(calibration_dataset):,}",
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
            "training_data_source": "generated pretraining plus Kaggle 90% fine-tuning",
            "training_dataset": "koryakinp/chess-positions",
            "training_manifest": str(args.manifest),
            "training_manifest_sha256": _manifest_sha256(args.manifest),
            "split_seed": args.seed,
            "train_board_count": len(train_dataset),
            "reserved_test_fraction": 0.1,
            "initial_checkpoint": str(args.initial_checkpoint),
            "pretrained_backbone": initial.get("pretrained_backbone"),
            "backbone_frozen": True,
            "pair_sampling": "one within-board pair per board per epoch, balanced by index",
            "background_normalization": "disabled",
            "crop_jitter_pixels": args.crop_jitter_pixels,
            "crop_jitter_probability": args.crop_jitter_probability,
            "cross_background_positive_weight": args.cross_background_positive_weight,
            "hard_negative_probability": args.hard_negative_probability,
            "similarity_loss_weight": args.similarity_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "validation_source": "training-split calibration subset; reserved test untouched",
        },
    )
    history = trainer.fit(
        train_progress,
        calibration_progress,
        epochs=args.epochs,
        start_epoch=int(initial.get("epoch", 0)),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(trainer.device)
    calibration = _calibrate_threshold(model, calibration_loader, trainer.device)
    checkpoint.update(calibration)
    checkpoint["calibration_negative_false_positive_target"] = 0.005
    checkpoint["threshold_calibration_source"] = "Kaggle 90% training split"
    checkpoint["threshold_calibration_board_count"] = len(calibration_dataset)
    torch.save(checkpoint, args.checkpoint)

    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(
        json.dumps(
            {
                "epochs": [asdict(epoch) for epoch in history.epochs],
                "best_epoch": history.best_epoch,
                "best_calibration_class_accuracy": history.best_val_class_accuracy,
                "checkpoint_path": str(args.checkpoint),
                "initial_checkpoint": str(args.initial_checkpoint),
                "train_board_count": len(train_dataset),
                "training_calibration_board_count": len(calibration_dataset),
                "manifest": str(args.manifest),
                "manifest_sha256": _manifest_sha256(args.manifest),
                "seed": args.seed,
                **calibration,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {args.checkpoint}; calibrated similarity threshold "
        f"{calibration['similarity_threshold']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
