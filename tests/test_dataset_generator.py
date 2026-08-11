"""Tests for dataset-level board augmentation."""

from __future__ import annotations

import random

import pytest
from PIL import Image, ImageChops

from chess_ocr.data.dataset_generator import GenerationConfig, jitter_board_crop


def make_border_test_board(size: int = 64) -> Image.Image:
    """Create a board-like image with visually distinct edges and centre."""
    image = Image.new("RGB", (size, size), color=(240, 220, 180))
    for coordinate in range(size):
        image.putpixel((coordinate, 0), (255, 0, 0))
        image.putpixel((coordinate, size - 1), (0, 255, 0))
        image.putpixel((0, coordinate), (0, 0, 255))
        image.putpixel((size - 1, coordinate), (255, 255, 0))
    return image


def test_zero_crop_jitter_preserves_pixels_and_size() -> None:
    board = make_border_test_board()

    result = jitter_board_crop(board, max_pixels=0, rng=random.Random(0))

    assert result is not board
    assert result.size == board.size
    assert ImageChops.difference(result, board).getbbox() is None


def test_crop_jitter_is_seeded_and_changes_the_board() -> None:
    board = make_border_test_board()

    first = jitter_board_crop(board, max_pixels=6, rng=random.Random(7))
    second = jitter_board_crop(board, max_pixels=6, rng=random.Random(7))

    assert first.size == board.size
    assert ImageChops.difference(first, second).getbbox() is None
    assert ImageChops.difference(first, board).getbbox() is not None


def test_negative_crop_jitter_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        jitter_board_crop(make_border_test_board(), max_pixels=-1, rng=random.Random(0))


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_invalid_crop_jitter_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="crop_jitter_probability"):
        GenerationConfig(
            output_dir="unused",
            crop_jitter_probability=probability,
        )
