"""Lazy loading of the Kaggle chess-positions full-board dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Self

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from chess_ocr.chess.fen_builder import board_fen_to_class_ids
from chess_ocr.data.dataset_generator import jitter_board_crop
from chess_ocr.data.labels import CLASS_NAMES
from chess_ocr.data.square_dataset import NORMALIZATION_MEAN, NORMALIZATION_STD
from chess_ocr.data.square_dataset import build_eval_transforms

BOARD_SIDE = 8
NUM_SQUARES = BOARD_SIDE * BOARD_SIDE
SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png"}


def fen_from_kaggle_filename(path: str | Path) -> str:
    """Decode the board-FEN stored in a Kaggle image filename.

    The dataset replaces each FEN ``/`` with ``-`` because slashes cannot occur
    in a filename.

    Args:
        path: Image path such as ``8-8-8-8-8-8-4K3-4k3.jpeg``.

    Returns:
        A validated board-placement FEN string.

    Raises:
        ValueError: If the filename does not contain a valid 64-square FEN.
    """
    board_fen = Path(path).stem.replace("-", "/")
    try:
        board_fen_to_class_ids(board_fen)
    except ValueError as error:
        raise ValueError(f"Invalid board FEN in Kaggle filename {Path(path).name!r}") from error
    return board_fen


class KaggleBoardDataset(Dataset):
    """Return one full Kaggle board as 64 normalized square tensors.

    Images are decoded once per board. The entire board is first resized to
    ``8 * input_size`` with bicubic interpolation, matching
    :class:`~chess_ocr.preprocessing.board_normalizer.BoardNormalizer`, and is
    then rearranged into 64 tensors without writing square crops to disk.
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        input_size: int = 64,
        max_boards: int | None = None,
        *,
        image_paths: list[str | Path] | None = None,
        augment: bool = False,
        empty_samples: int | None = None,
        crop_jitter_pixels: int = 0,
        crop_jitter_probability: float = 0.0,
        color_jitter: float = 0.0,
    ) -> None:
        """Index and validate a directory of Kaggle board images.

        Args:
            image_dir: Directory containing the JPEG board images.
            input_size: Model input side length for one square.
            max_boards: Optional deterministic prefix used for smoke tests.
            image_paths: Explicit image paths, used by manifest-backed splits.
            augment: Whether to apply randomized training augmentation.
            empty_samples: When set, retain every occupied square and sample at
                most this many empty squares from each training board.
            crop_jitter_pixels: Maximum full-board crop error in source pixels.
            crop_jitter_probability: Probability of applying crop jitter.
            color_jitter: Symmetric brightness, contrast, and saturation jitter.

        Raises:
            FileNotFoundError: If ``image_dir`` does not exist.
            ValueError: If an argument is invalid or no images are found.
        """
        if (image_dir is None) == (image_paths is None):
            raise ValueError("Provide exactly one of image_dir or image_paths")
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if max_boards is not None and max_boards <= 0:
            raise ValueError(f"max_boards must be positive, got {max_boards}")
        if empty_samples is not None and empty_samples < 0:
            raise ValueError(f"empty_samples must be non-negative, got {empty_samples}")
        if crop_jitter_pixels < 0:
            raise ValueError("crop_jitter_pixels must be non-negative")
        if not 0.0 <= crop_jitter_probability <= 1.0:
            raise ValueError("crop_jitter_probability must be in [0, 1]")
        if color_jitter < 0:
            raise ValueError("color_jitter must be non-negative")

        self.input_size = input_size
        self.augment = augment
        self.empty_samples = empty_samples
        self.crop_jitter_pixels = crop_jitter_pixels
        self.crop_jitter_probability = crop_jitter_probability
        self.color_transform = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
            if augment and color_jitter > 0
            else None
        )

        if image_paths is not None:
            self.image_dir = None
            self.paths = [Path(path) for path in image_paths]
        else:
            self.image_dir = Path(image_dir)  # type: ignore[arg-type]
            if not self.image_dir.is_dir():
                raise FileNotFoundError(f"Kaggle image directory not found: {self.image_dir}")
            self.paths = sorted(
                path
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
        if max_boards is not None:
            self.paths = self.paths[:max_boards]
        if not self.paths:
            location = self.image_dir if self.image_dir is not None else "the provided path list"
            raise ValueError(f"No supported board images found in {location}")

        # Validate filenames before a long evaluation run starts.
        self.board_fens = [fen_from_kaggle_filename(path) for path in self.paths]

    @classmethod
    def from_manifest(
        cls,
        manifest_csv: str | Path,
        data_root: str | Path,
        split: str,
        **kwargs: object,
    ) -> Self:
        """Create a dataset from one split in a generated manifest CSV.

        Args:
            manifest_csv: CSV containing ``image_path`` and ``split`` columns.
            data_root: Directory that relative image paths resolve against.
            split: Split name to select, normally ``train``, ``val`` or ``test``.
            **kwargs: Remaining arguments forwarded to the constructor.
        """
        manifest_path = Path(manifest_csv)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Kaggle split manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "split"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest must contain columns {sorted(required)}")
            relative_paths = [row["image_path"] for row in reader if row["split"] == split]
        if not relative_paths:
            raise ValueError(f"Manifest split {split!r} contains no images")
        root = Path(data_root)
        return cls(image_paths=[root / path for path in relative_paths], **kwargs)

    def __len__(self) -> int:
        """Return the number of full boards."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return ``(64 square tensors, 64 labels, board id)``."""
        path = self.paths[index]
        board_size = BOARD_SIDE * self.input_size
        with Image.open(path) as image:
            board = image.convert("RGB")
        if (
            self.augment
            and self.crop_jitter_pixels > 0
            and random.random() < self.crop_jitter_probability
        ):
            board = jitter_board_crop(board, self.crop_jitter_pixels, random)  # type: ignore[arg-type]
        if self.color_transform is not None:
            board = self.color_transform(board)
        board = board.resize((board_size, board_size), resample=Image.Resampling.BICUBIC)

        tensor = transform_functional.pil_to_tensor(board).float().div_(255.0)
        tensor = transform_functional.normalize(tensor, NORMALIZATION_MEAN, NORMALIZATION_STD)
        squares = (
            tensor.reshape(3, BOARD_SIDE, self.input_size, BOARD_SIDE, self.input_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(NUM_SQUARES, 3, self.input_size, self.input_size)
        )
        labels = torch.tensor(board_fen_to_class_ids(self.board_fens[index]), dtype=torch.long)
        if self.empty_samples is not None:
            occupied = torch.nonzero(labels != 0, as_tuple=False).flatten()
            empty = torch.nonzero(labels == 0, as_tuple=False).flatten()
            sample_count = min(self.empty_samples, empty.numel())
            sampled_empty = empty[torch.randperm(empty.numel())[:sample_count]]
            selected = torch.cat((occupied, sampled_empty)).sort().values
            squares = squares[selected]
            labels = labels[selected]
        return squares, labels, path.stem


def collate_kaggle_boards(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a board batch into the square batch expected by the model."""
    squares, labels, _ = zip(*batch, strict=True)
    return torch.cat(squares, dim=0), torch.cat(labels, dim=0)


class KaggleBoardPairDataset(Dataset):
    """Sample one deterministic same/different square pair from every board.

    Pair members always come from the same rendered board, so the embedding
    learns within-theme appearance similarity. Across one epoch every manifest
    board is decoded exactly once. Even dataset indices produce positive pairs
    and odd indices produce negative pairs, giving a balanced pair objective
    after the manifest has been deterministically shuffled.
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        input_size: int = 224,
        *,
        augment: bool = False,
        crop_jitter_pixels: int = 0,
        crop_jitter_probability: float = 0.0,
        cross_background_positive_weight: float = 3.0,
        hard_negative_probability: float = 0.75,
        seed: int = 42,
    ) -> None:
        if not image_paths:
            raise ValueError("Kaggle pair dataset requires at least one board")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if crop_jitter_pixels < 0:
            raise ValueError("crop_jitter_pixels must be non-negative")
        if not 0.0 <= crop_jitter_probability <= 1.0:
            raise ValueError("crop_jitter_probability must be in [0, 1]")
        if cross_background_positive_weight < 1.0:
            raise ValueError("cross-background weight must be at least one")
        if not 0.0 <= hard_negative_probability <= 1.0:
            raise ValueError("hard-negative probability must be in [0, 1]")
        self.paths = [Path(path) for path in image_paths]
        self.board_labels = [
            tuple(board_fen_to_class_ids(fen_from_kaggle_filename(path)))
            for path in self.paths
        ]
        self.input_size = input_size
        self.augment = augment
        self.crop_jitter_pixels = crop_jitter_pixels
        self.crop_jitter_probability = crop_jitter_probability
        self.cross_background_positive_weight = cross_background_positive_weight
        self.hard_negative_probability = hard_negative_probability
        self.seed = seed
        self.epoch = 0
        self.transform = build_eval_transforms(input_size)

    @classmethod
    def from_manifest(
        cls,
        manifest_csv: str | Path,
        data_root: str | Path,
        split: str,
        max_boards: int | None = None,
        **kwargs: object,
    ) -> Self:
        """Construct one pair sample for every board in a manifest split."""
        manifest_path = Path(manifest_csv)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Kaggle split manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "split"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest must contain columns {sorted(required)}")
            relative_paths = [row["image_path"] for row in reader if row["split"] == split]
        if max_boards is not None:
            if max_boards <= 0:
                raise ValueError("max_boards must be positive")
            relative_paths = relative_paths[:max_boards]
        root = Path(data_root)
        return cls([root / path for path in relative_paths], **kwargs)

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic sampling choices without changing board coverage."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    @staticmethod
    def _is_cross_background(first: int, second: int) -> bool:
        return (first // 8 + first % 8) % 2 != (second // 8 + second % 8) % 2

    def _select_pair(
        self, index: int, rng: random.Random
    ) -> tuple[int, int, float, float, float]:
        labels = self.board_labels[index]
        by_label: dict[int, list[int]] = {}
        for square, label in enumerate(labels):
            by_label.setdefault(label, []).append(square)
        positive = index % 2 == 0
        if positive:
            repeatable = [label for label, squares in by_label.items() if len(squares) >= 2]
            label = rng.choice(repeatable)
            candidates = by_label[label]
            opposite = [
                (first, second)
                for first in candidates
                for second in candidates
                if first < second and self._is_cross_background(first, second)
            ]
            if opposite:
                first, second = rng.choice(opposite)
                cross_background = 1.0
                pair_weight = self.cross_background_positive_weight
            else:
                first, second = rng.sample(candidates, 2)
                cross_background = float(self._is_cross_background(first, second))
                pair_weight = (
                    self.cross_background_positive_weight if cross_background else 1.0
                )
            return first, second, 1.0, pair_weight, cross_background

        available = list(by_label)
        label_a = rng.choice(available)
        label_b_candidates = [label for label in available if label != label_a]
        use_hard_negative = label_a != 0 and rng.random() < self.hard_negative_probability
        hard_labels: list[int] = []
        if use_hard_negative:
            name_a = CLASS_NAMES[label_a]
            side_a, piece_a = name_a.split("_", maxsplit=1)
            for label in label_b_candidates:
                if label == 0:
                    continue
                side_b, piece_b = CLASS_NAMES[label].split("_", maxsplit=1)
                if side_a == side_b or piece_a == piece_b:
                    hard_labels.append(label)
        label_b = rng.choice(hard_labels or label_b_candidates)
        return (
            rng.choice(by_label[label_a]),
            rng.choice(by_label[label_b]),
            0.0,
            1.0,
            0.0,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor | float | int, ...]:
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        first, second, target, pair_weight, cross_background = self._select_pair(index, rng)
        with Image.open(self.paths[index]) as image:
            board = image.convert("RGB")
        if (
            self.augment
            and self.crop_jitter_pixels > 0
            and rng.random() < self.crop_jitter_probability
        ):
            board = jitter_board_crop(board, self.crop_jitter_pixels, rng)
        x_edges = [round(column * board.width / BOARD_SIDE) for column in range(BOARD_SIDE + 1)]
        y_edges = [round(row * board.height / BOARD_SIDE) for row in range(BOARD_SIDE + 1)]

        def load_square(square: int) -> torch.Tensor:
            row, column = divmod(square, BOARD_SIDE)
            crop = board.crop(
                (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
            )
            return self.transform(crop)

        labels = self.board_labels[index]
        return (
            load_square(first),
            load_square(second),
            target,
            pair_weight,
            labels[first],
            labels[second],
            cross_background,
        )
