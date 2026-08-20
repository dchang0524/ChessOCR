"""Balanced on-demand pair sampling for Siamese similarity training."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from chess_ocr.data.labels import CLASS_NAME_TO_ID
from chess_ocr.data.square_dataset import build_eval_transforms, build_train_transforms


class SimilarityPairDataset(Dataset):
    """Sample same/different appearance pairs from a square metadata CSV.

    Both members always come from the same rendered theme. Positive pairs use
    the same semantic label and prefer opposite square colours; negative pairs
    use different labels. Empty is deliberately included as the thirteenth
    appearance class, so the learned embedding can form an empty-square group
    instead of depending on the baseline classifier's occupancy decision.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        data_root: str | Path | None = None,
        split: str | None = None,
        themes: set[str] | None = None,
        pairs_per_epoch: int = 50_000,
        input_size: int = 64,
        augment: bool = True,
        seed: int = 0,
        cross_background_positive_weight: float = 1.0,
        hard_negative_probability: float = 0.0,
        include_class_labels: bool = False,
        include_cross_background_flag: bool = False,
    ) -> None:
        csv_path = Path(metadata_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
        if pairs_per_epoch <= 0:
            raise ValueError("pairs_per_epoch must be positive")
        if cross_background_positive_weight < 1.0:
            raise ValueError("cross_background_positive_weight must be at least 1")
        if not 0.0 <= hard_negative_probability <= 1.0:
            raise ValueError("hard_negative_probability must be in [0, 1]")

        frame = pd.read_csv(csv_path)
        required = {"image_path", "label", "theme", "square_color"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {missing}")
        if split is not None:
            if "split" not in frame.columns:
                raise ValueError("Metadata CSV has no 'split' column")
            frame = frame[frame["split"] == split]
        if themes is not None:
            frame = frame[frame["theme"].isin(themes)]
        frame = frame.reset_index(drop=True)
        if frame.empty:
            raise ValueError("No square samples remain after filtering")

        self.data_root = Path(data_root) if data_root is not None else csv_path.parent
        self.frame = frame
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        self.epoch = 0
        self.cross_background_positive_weight = cross_background_positive_weight
        self.hard_negative_probability = hard_negative_probability
        self.include_class_labels = include_class_labels
        self.include_cross_background_flag = include_cross_background_flag
        self.transform = (
            build_train_transforms(input_size) if augment else build_eval_transforms(input_size)
        )

        by_theme_label: dict[tuple[str, str], list[int]] = defaultdict(list)
        by_theme_label_color: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        labels_by_theme: dict[str, set[str]] = defaultdict(set)
        for index, row in frame.iterrows():
            theme = str(row["theme"])
            label = str(row["label"])
            color = str(row["square_color"])
            by_theme_label[(theme, label)].append(index)
            by_theme_label_color[(theme, label, color)].append(index)
            labels_by_theme[theme].add(label)

        self.by_theme_label = dict(by_theme_label)
        self.by_theme_label_color = dict(by_theme_label_color)
        self.labels_by_theme = {
            theme: sorted(labels) for theme, labels in labels_by_theme.items() if len(labels) >= 2
        }
        self.themes = sorted(self.labels_by_theme)
        if not self.themes:
            raise ValueError("Need at least one theme containing two labels")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic pair sample for a training epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _load(self, index: int) -> torch.Tensor:
        path = self.data_root / str(self.frame.iloc[index]["image_path"])
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor | float | int, ...]:
        rng = random.Random(self.seed + self.epoch * self.pairs_per_epoch + index)
        theme = rng.choice(self.themes)
        labels = self.labels_by_theme[theme]
        positive = index % 2 == 0

        if positive:
            label_a = rng.choice(labels)
            label_b = label_a
            light = self.by_theme_label_color.get((theme, label_a, "light"), [])
            dark = self.by_theme_label_color.get((theme, label_a, "dark"), [])
            if light and dark:
                first, second = rng.choice(light), rng.choice(dark)
                pair_weight = self.cross_background_positive_weight
                cross_background = True
            else:
                candidates = self.by_theme_label[(theme, label_a)]
                first = rng.choice(candidates)
                second = rng.choice(candidates)
                if len(candidates) > 1:
                    while second == first:
                        second = rng.choice(candidates)
                first_color = str(self.frame.iloc[first]["square_color"])
                second_color = str(self.frame.iloc[second]["square_color"])
                pair_weight = (
                    self.cross_background_positive_weight
                    if first_color != second_color
                    else 1.0
                )
                cross_background = first_color != second_color
            target = 1.0
        else:
            hard_anchors = [label for label in labels if label != "empty"]
            use_hard_negative = bool(hard_anchors) and (
                rng.random() < self.hard_negative_probability
            )
            label_a = rng.choice(hard_anchors if use_hard_negative else labels)
            hard_labels: list[str] = []
            if use_hard_negative:
                side_a, piece_a = label_a.split("_", maxsplit=1)
                hard_labels = [
                    candidate
                    for candidate in labels
                    if candidate != "empty"
                    and candidate != label_a
                    and (
                        candidate.split("_", maxsplit=1)[0] == side_a
                        or candidate.split("_", maxsplit=1)[1] == piece_a
                    )
                ]
            if hard_labels:
                label_b = rng.choice(hard_labels)
            else:
                label_b = rng.choice([candidate for candidate in labels if candidate != label_a])
            first = rng.choice(self.by_theme_label[(theme, label_a)])
            second = rng.choice(self.by_theme_label[(theme, label_b)])
            target = 0.0
            pair_weight = 1.0
            cross_background = False

        result: tuple[torch.Tensor | float | int, ...] = (
            self._load(first),
            self._load(second),
            target,
            pair_weight,
        )
        if self.include_class_labels:
            result += (CLASS_NAME_TO_ID[label_a], CLASS_NAME_TO_ID[label_b])
        if self.include_cross_background_flag:
            result += (float(cross_background),)
        return result
