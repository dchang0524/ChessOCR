from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from chess_ocr.data.similarity_pair_dataset import SimilarityPairDataset


def make_metadata(tmp_path: Path) -> Path:
    rows = []
    for label, value in (("empty", 10), ("white_pawn", 50), ("white_bishop", 200)):
        for color in ("light", "dark"):
            path = tmp_path / f"{label}_{color}.png"
            Image.new("RGB", (64, 64), color=(value, value, value)).save(path)
            rows.append(
                {
                    "image_path": path.name,
                    "label": label,
                    "theme": "theme_a",
                    "square_color": color,
                    "split": "train",
                }
            )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def test_pair_dataset_alternates_balanced_targets(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path), pairs_per_epoch=6, augment=False
    )

    targets = [dataset[index][2] for index in range(len(dataset))]

    assert targets == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_pair_dataset_returns_square_tensors(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path), pairs_per_epoch=2, augment=False
    )

    first, second, target = dataset[0]

    assert first.shape == (3, 64, 64)
    assert second.shape == (3, 64, 64)
    assert target == 1.0


def test_pair_dataset_includes_empty_as_an_appearance_label(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path), pairs_per_epoch=30, augment=False, seed=4
    )

    sampled_values = {
        round(float(square.mean()), 2)
        for index in range(len(dataset))
        for square in dataset[index][:2]
    }

    # Pixel value 10 maps to roughly -0.92 after [-1, 1] normalization.
    assert -0.92 in sampled_values
