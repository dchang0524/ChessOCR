"""Tests for dataset-level board augmentation."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from torchvision import transforms

from chess_ocr.data.dataset_generator import (
    GenerationConfig,
    ImageAssetBoardTheme,
    jitter_board_crop,
    render_square_backgrounds,
)
from chess_ocr.data.square_dataset import build_train_transforms


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


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_invalid_background_variation_probability_is_rejected(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="background_variation_probability"):
        GenerationConfig(
            output_dir="unused",
            background_variation_probability=probability,
        )


def test_background_variation_is_seeded_and_changes_the_palette() -> None:
    first = render_square_backgrounds(
        64, (240, 217, 181), (181, 136, 99), random.Random(5), 0.75
    )
    second = render_square_backgrounds(
        64, (240, 217, 181), (181, 136, 99), random.Random(5), 0.75
    )
    different = render_square_backgrounds(
        64, (240, 217, 181), (181, 136, 99), random.Random(6), 0.75
    )

    assert ImageChops.difference(first, second).getbbox() is None
    assert ImageChops.difference(first, different).getbbox() is not None


def test_background_variation_does_not_modify_opaque_sprite_pixels(tmp_path: Path) -> None:
    sprite_dir = tmp_path / "sprites"
    sprite_dir.mkdir()
    Image.new("RGBA", (32, 32), (231, 17, 93, 255)).save(sprite_dir / "wR.png")
    theme = ImageAssetBoardTheme(
        name="test", asset_dir=sprite_dir, piece_scale=0.5
    )
    board_fen = "8/8/8/8/8/8/8/R7"

    first = theme.render_board(board_fen, 64, random.Random(1), 1.0)
    second = theme.render_board(board_fen, 64, random.Random(2), 1.0)

    # a1 occupies x=0..7, y=56..63; the 4x4 opaque sprite is centered there.
    assert first.getpixel((3, 59)) == (231, 17, 93)
    assert second.getpixel((3, 59)) == (231, 17, 93)
    assert ImageChops.difference(first, second).getbbox() is not None


def test_training_transforms_do_not_colour_jitter_composited_pieces() -> None:
    pipeline = build_train_transforms()

    assert not any(isinstance(transform, transforms.ColorJitter) for transform in pipeline.transforms)
