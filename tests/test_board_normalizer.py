"""Tests for BoardNormalizer."""

from __future__ import annotations

import pytest
from PIL import Image

from chess_ocr.preprocessing.board_normalizer import BoardNormalizer


def test_output_is_rgb_and_resized() -> None:
    normalizer = BoardNormalizer(output_size=512)
    image = Image.new("L", (137, 291), color=200)

    result = normalizer.normalize(image)

    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_rgba_is_converted_to_rgb() -> None:
    normalizer = BoardNormalizer(output_size=64)
    image = Image.new("RGBA", (32, 32), color=(10, 20, 30, 128))

    result = normalizer.normalize(image)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (10, 20, 30)


def test_custom_output_size_is_respected() -> None:
    normalizer = BoardNormalizer(output_size=128)

    result = normalizer.normalize(Image.new("RGB", (300, 300)))

    assert result.size == (128, 128)


def test_source_image_is_not_mutated() -> None:
    normalizer = BoardNormalizer(output_size=64)
    image = Image.new("RGB", (16, 16), color=(1, 2, 3))

    normalizer.normalize(image)

    assert image.size == (16, 16)


def test_zero_dimension_image_is_rejected() -> None:
    normalizer = BoardNormalizer()

    with pytest.raises(ValueError, match="nonzero"):
        normalizer.normalize(Image.new("RGB", (0, 100)))


def test_non_image_input_is_rejected() -> None:
    normalizer = BoardNormalizer()

    with pytest.raises(TypeError):
        normalizer.normalize("not-an-image")  # type: ignore[arg-type]


def test_invalid_output_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="output_size"):
        BoardNormalizer(output_size=0)
