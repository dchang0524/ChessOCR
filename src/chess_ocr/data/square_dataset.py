"""PyTorch dataset over the generated square images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from chess_ocr.data.labels import CLASS_NAME_TO_ID

INPUT_SIZE = 64
NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]

REQUIRED_COLUMNS = ("image_path", "label")


def build_train_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Return the training augmentation pipeline.

    Augmentations are deliberately mild: chess pieces are always upright and
    axis-aligned in digital screenshots, so rotations and flips would teach the
    model the wrong invariances (a flipped pawn is still a pawn, but a rotated
    board changes which colour sits where).

    Args:
        input_size: Side length of the tensors handed to the model.

    Returns:
        A ``torchvision`` transform pipeline producing normalised tensors.
    """
    padding = max(2, input_size // 16)
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomCrop(input_size, padding=padding, padding_mode="edge"),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


def build_eval_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Return the deterministic evaluation/inference pipeline.

    Args:
        input_size: Side length of the tensors handed to the model.

    Returns:
        A ``torchvision`` transform pipeline producing normalised tensors.
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


class SquareDataset(Dataset):
    """Dataset of labelled chessboard squares described by a metadata CSV.

    The CSV must contain at least ``image_path`` and ``label`` columns; the
    generator also writes ``position_id``, ``square``, ``theme``,
    ``square_color`` and ``split``, which are used for stratified reporting and
    for split-by-position.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        data_root: str | Path | None = None,
        split: str | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        """Load the metadata and prepare the sample list.

        Args:
            metadata_csv: Path to the metadata CSV.
            data_root: Directory that ``image_path`` entries are relative to.
                Defaults to the directory containing the CSV.
            split: Optional split name (``"train"``, ``"val"``, ``"test"``) used
                to filter rows. Requires a ``split`` column.
            transform: Transform applied to each loaded image. Defaults to the
                deterministic evaluation pipeline.

        Raises:
            FileNotFoundError: If the metadata CSV does not exist.
            ValueError: If required columns are missing, a label is unknown, or
                the selected split is empty.
        """
        csv_path = Path(metadata_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {missing}")

        if split is not None:
            if "split" not in frame.columns:
                raise ValueError("Metadata CSV has no 'split' column")
            frame = frame[frame["split"] == split].reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"Split {split!r} contains no rows")

        unknown_labels = sorted(set(frame["label"]) - set(CLASS_NAME_TO_ID))
        if unknown_labels:
            raise ValueError(f"Metadata contains unknown labels: {unknown_labels}")

        self.data_root = Path(data_root) if data_root is not None else csv_path.parent
        self.metadata = frame
        self.split = split
        self.transform = transform if transform is not None else build_eval_transforms()
        self.image_paths: list[Path] = [self.data_root / str(path) for path in frame["image_path"]]
        self.labels: list[int] = [CLASS_NAME_TO_ID[name] for name in frame["label"]]

    def __len__(self) -> int:
        """Return the number of squares in this split."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the transformed image tensor and integer class id at ``index``.

        Args:
            index: Sample index.

        Returns:
            A tuple ``(image_tensor, class_id)``.
        """
        with Image.open(self.image_paths[index]) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image), self.labels[index]

    def class_distribution(self) -> dict[str, int]:
        """Return the number of samples per class name in this split."""
        counts = self.metadata["label"].value_counts().to_dict()
        return {str(name): int(count) for name, count in counts.items()}
