"""Balanced on-demand pair sampling for Siamese similarity training."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
    ) -> None:
        csv_path = Path(metadata_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
        if pairs_per_epoch <= 0:
            raise ValueError("pairs_per_epoch must be positive")

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

    def _load(self, index: int) -> torch.Tensor:
        path = self.data_root / str(self.frame.iloc[index]["image_path"])
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        rng = random.Random(self.seed + index)
        theme = rng.choice(self.themes)
        labels = self.labels_by_theme[theme]
        positive = index % 2 == 0

        if positive:
            label = rng.choice(labels)
            light = self.by_theme_label_color.get((theme, label, "light"), [])
            dark = self.by_theme_label_color.get((theme, label, "dark"), [])
            if light and dark:
                first, second = rng.choice(light), rng.choice(dark)
            else:
                candidates = self.by_theme_label[(theme, label)]
                first = rng.choice(candidates)
                second = rng.choice(candidates)
                if len(candidates) > 1:
                    while second == first:
                        second = rng.choice(candidates)
            target = 1.0
        else:
            label_a, label_b = rng.sample(labels, 2)
            first = rng.choice(self.by_theme_label[(theme, label_a)])
            second = rng.choice(self.by_theme_label[(theme, label_b)])
            target = 0.0

        return self._load(first), self._load(second), target
