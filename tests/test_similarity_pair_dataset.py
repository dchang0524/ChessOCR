from __future__ import annotations

import csv
from pathlib import Path

import pytest
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

    first, second, target, pair_weight = dataset[0]

    assert first.shape == (3, 64, 64)
    assert second.shape == (3, 64, 64)
    assert target == 1.0
    assert pair_weight == 1.0


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


def test_pair_dataset_resamples_deterministically_each_epoch(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path), pairs_per_epoch=12, augment=False, seed=8
    )
    first_epoch = [
        (float(dataset[index][0].mean()), float(dataset[index][1].mean()))
        for index in range(len(dataset))
    ]
    dataset.set_epoch(1)
    second_epoch = [
        (float(dataset[index][0].mean()), float(dataset[index][1].mean()))
        for index in range(len(dataset))
    ]
    dataset.set_epoch(0)
    repeated_first_epoch = [
        (float(dataset[index][0].mean()), float(dataset[index][1].mean()))
        for index in range(len(dataset))
    ]

    assert second_epoch != first_epoch
    assert repeated_first_epoch == first_epoch


def test_opposite_background_positive_pairs_receive_extra_weight(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path),
        pairs_per_epoch=2,
        augment=False,
        cross_background_positive_weight=3.0,
    )

    assert dataset[0][2:] == (1.0, 3.0)
    assert dataset[1][2:] == (0.0, 1.0)


def test_cross_background_weight_cannot_downweight_positive_pairs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SimilarityPairDataset(
            make_metadata(tmp_path),
            pairs_per_epoch=2,
            cross_background_positive_weight=0.5,
        )


def test_joint_training_fields_include_labels_and_background_flag(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path),
        split="train",
        pairs_per_epoch=2,
        augment=False,
        include_class_labels=True,
        include_cross_background_flag=True,
    )
    positive = dataset[0]
    negative = dataset[1]
    assert len(positive) == 7
    assert positive[2] == 1.0
    assert positive[4] == positive[5]
    assert positive[6] == 1.0
    assert len(negative) == 7
    assert negative[2] == 0.0
    assert negative[4] != negative[5]
    assert negative[6] == 0.0


def test_hard_negatives_avoid_empty_when_confusable_pieces_exist(tmp_path: Path) -> None:
    dataset = SimilarityPairDataset(
        make_metadata(tmp_path),
        pairs_per_epoch=20,
        augment=False,
        hard_negative_probability=1.0,
    )

    negative_means = [
        (float(dataset[index][0].mean()), float(dataset[index][1].mean()))
        for index in range(1, len(dataset), 2)
    ]

    # Empty pixels have normalized mean around -0.92. All forced hard-negative
    # pairs should instead be the white pawn/white bishop pair.
    assert all(first > -0.7 and second > -0.7 for first, second in negative_means)


def test_hard_negative_probability_must_be_valid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hard_negative_probability"):
        SimilarityPairDataset(
            make_metadata(tmp_path),
            pairs_per_epoch=2,
            hard_negative_probability=1.1,
        )
