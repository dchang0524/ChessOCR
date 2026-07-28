"""Tests for BoardSplitter."""

from __future__ import annotations

import pytest
from PIL import Image

from chess_ocr.preprocessing.board_splitter import BoardSplitter


def make_indexed_board(square_size: int = 1) -> Image.Image:
    """Build a board whose raster-order square ``i`` is filled with colour ``(i, i, i)``."""
    size = 8 * square_size
    board = Image.new("RGB", (size, size))
    for row in range(8):
        for column in range(8):
            index = row * 8 + column
            patch = Image.new("RGB", (square_size, square_size), (index, index, index))
            board.paste(patch, (column * square_size, row * square_size))
    return board


def test_returns_sixty_four_squares_of_equal_size() -> None:
    squares = BoardSplitter().split(Image.new("RGB", (512, 512)))

    assert len(squares) == 64
    assert {square.size for square in squares} == {(64, 64)}


def test_white_at_bottom_returns_raster_order() -> None:
    squares = BoardSplitter().split(make_indexed_board(), white_at_bottom=True)

    values = [square.getpixel((0, 0))[0] for square in squares]

    assert values == list(range(64))


def test_black_at_bottom_reverses_the_order() -> None:
    squares = BoardSplitter().split(make_indexed_board(), white_at_bottom=False)

    values = [square.getpixel((0, 0))[0] for square in squares]

    assert values == list(reversed(range(64)))


def test_fen_order_maps_a8_first_and_h1_last() -> None:
    board = make_indexed_board(square_size=8)
    squares = BoardSplitter().split(board, white_at_bottom=True)

    # a8 is the top-left square of the image when White is at the bottom.
    assert squares[0].getpixel((0, 0)) == (0, 0, 0)
    # h1 is the bottom-right square.
    assert squares[63].getpixel((0, 0)) == (63, 63, 63)


def test_black_orientation_places_a8_at_bottom_right() -> None:
    board = make_indexed_board(square_size=8)
    squares = BoardSplitter().split(board, white_at_bottom=False)

    # With Black at the bottom the image's bottom-right square is a8.
    assert squares[0].getpixel((0, 0)) == (63, 63, 63)
    # ...and the image's top-left square is h1.
    assert squares[63].getpixel((0, 0)) == (0, 0, 0)


def test_no_pixels_are_lost_for_sizes_not_divisible_by_eight() -> None:
    size = 515
    squares = BoardSplitter().split(Image.new("RGB", (size, size)))

    widths = [square.size[0] for square in squares[:8]]
    heights = [squares[row * 8].size[1] for row in range(8)]

    assert sum(widths) == size
    assert sum(heights) == size
    assert all(w in (size // 8, size // 8 + 1) for w in widths)


def test_non_square_board_is_rejected() -> None:
    with pytest.raises(ValueError, match="square"):
        BoardSplitter().split(Image.new("RGB", (512, 256)))


def test_tiny_board_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        BoardSplitter().split(Image.new("RGB", (4, 4)))


def test_non_image_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        BoardSplitter().split(object())  # type: ignore[arg-type]
