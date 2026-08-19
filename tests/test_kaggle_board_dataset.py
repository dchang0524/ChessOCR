from pathlib import Path

import pytest
import torch
from PIL import Image

from chess_ocr.data.kaggle_board_dataset import (
    KaggleBoardDataset,
    collate_kaggle_boards,
    fen_from_kaggle_filename,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_fen_from_kaggle_filename() -> None:
    filename = STARTING_FEN.replace("/", "-") + ".jpeg"
    assert fen_from_kaggle_filename(filename) == STARTING_FEN


def test_kaggle_board_dataset_loads_and_splits_once(tmp_path: Path) -> None:
    filename = STARTING_FEN.replace("/", "-") + ".png"
    Image.new("RGB", (400, 400), color=(255, 128, 0)).save(tmp_path / filename)

    dataset = KaggleBoardDataset(tmp_path, input_size=64)
    squares, labels, board_id = dataset[0]

    assert squares.shape == (64, 3, 64, 64)
    assert labels.shape == (64,)
    assert board_id == Path(filename).stem
    assert labels[:8].tolist() == [10, 8, 9, 11, 12, 9, 8, 10]
    assert torch.allclose(squares[0, :, 0, 0], torch.tensor([1.0, 1 / 255, -1.0]), atol=1e-6)


def test_collate_kaggle_boards_flattens_board_batch(tmp_path: Path) -> None:
    filename = STARTING_FEN.replace("/", "-") + ".png"
    Image.new("RGB", (400, 400), color="white").save(tmp_path / filename)
    sample = KaggleBoardDataset(tmp_path)[0]

    squares, labels = collate_kaggle_boards([sample, sample])

    assert squares.shape == (128, 3, 64, 64)
    assert labels.shape == (128,)


def test_invalid_kaggle_filename_is_rejected(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10), color="black").save(tmp_path / "not-a-fen.png")
    with pytest.raises(ValueError, match="Invalid board FEN"):
        KaggleBoardDataset(tmp_path)
