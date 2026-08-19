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
from chess_ocr.data.square_dataset import NORMALIZATION_MEAN, NORMALIZATION_STD

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
