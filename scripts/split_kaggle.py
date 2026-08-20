"""Create a reproducible manifest for Kaggle chess positions.

The downloaded dataset already provides 80,000 boards in ``train`` and 20,000
in ``test``. The default strategy keeps the original 80,000 training boards
unchanged and divides the original holdout evenly into validation and final-test
sets. The ``all-90-10`` strategy combines both source folders, then reserves a
deterministic 10% test split. No images are copied or moved.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png"}
FIELDS = ("image_path", "split", "board_fen")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create the Kaggle 80/10/10 split manifest.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/kaggle_80_10_10.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=("original-80-10-10", "all-90-10"),
        default="original-80-10-10",
        help="Whether to preserve the source split or reshuffle all boards into 90/10",
    )
    return parser.parse_args(argv)


def image_paths(directory: Path) -> list[Path]:
    """Return supported image files in deterministic filename order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No board images found in {directory}")
    return paths


def build_split_rows(dataset_root: Path, seed: int) -> list[dict[str, str]]:
    """Build manifest rows using the original train/holdout boundary."""
    train_paths = image_paths(dataset_root / "train")
    holdout_paths = image_paths(dataset_root / "test")
    random.Random(seed).shuffle(holdout_paths)
    midpoint = len(holdout_paths) // 2
    if midpoint == 0 or midpoint == len(holdout_paths):
        raise ValueError("The original test folder is too small to divide into val and test")

    assignments = (
        ((path, "train") for path in train_paths),
        ((path, "val") for path in holdout_paths[:midpoint]),
        ((path, "test") for path in holdout_paths[midpoint:]),
    )
    rows: list[dict[str, str]] = []
    for group in assignments:
        for path, split in group:
            rows.append(
                {
                    "image_path": path.relative_to(dataset_root).as_posix(),
                    "split": split,
                    "board_fen": path.stem.replace("-", "/"),
                }
            )
    return rows


def build_all_90_10_rows(dataset_root: Path, seed: int) -> list[dict[str, str]]:
    """Combine both source folders and reserve a deterministic 10% test split."""
    paths = image_paths(dataset_root / "train") + image_paths(dataset_root / "test")
    random.Random(seed).shuffle(paths)
    test_count = max(1, round(len(paths) * 0.1))
    if test_count >= len(paths):
        raise ValueError("The dataset is too small to divide into 90/10 splits")
    test_paths = set(paths[:test_count])
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.append(
            {
                "image_path": path.relative_to(dataset_root).as_posix(),
                "split": "test" if path in test_paths else "train",
                "board_fen": path.stem.replace("-", "/"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write the split manifest and print its counts."""
    args = parse_args(argv)
    rows = (
        build_all_90_10_rows(args.dataset_root, args.seed)
        if args.strategy == "all-90-10"
        else build_split_rows(args.dataset_root, args.seed)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    split_names = ("train", "test") if args.strategy == "all-90-10" else (
        "train",
        "val",
        "test",
    )
    counts = {split: sum(row["split"] == split for row in rows) for split in split_names}
    total = len(rows)
    print(f"Wrote {args.output} with seed {args.seed} using {args.strategy}")
    for split in split_names:
        print(f"  {split:<5}: {counts[split]:>6,} boards ({counts[split] / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
