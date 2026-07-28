"""Deterministic splitting of a normalised board into its 64 squares."""

from __future__ import annotations

from PIL import Image

BOARD_SIDE = 8
NUM_SQUARES = BOARD_SIDE * BOARD_SIDE


def _boundaries(size: int) -> list[int]:
    """Return the 9 pixel boundaries that divide ``size`` into 8 parts.

    Rounding is done on the cumulative coordinate rather than on a per-square
    width, so no pixel column or row is dropped or duplicated even when ``size``
    is not a multiple of eight.

    Args:
        size: Side length of the board image in pixels.

    Returns:
        A list of nine increasing pixel coordinates starting at 0 and ending at
        ``size``.
    """
    return [round(index * size / BOARD_SIDE) for index in range(BOARD_SIDE + 1)]


class BoardSplitter:
    """Split a normalised, axis-aligned board image into 64 square crops.

    The splitter assumes it receives a top-down, square, tightly-cropped board
    (the output of :class:`~chess_ocr.preprocessing.board_normalizer.BoardNormalizer`).
    Squares are located by arithmetic, never by line detection or a neural
    network.
    """

    def split(
        self,
        board: Image.Image,
        white_at_bottom: bool = True,
    ) -> list[Image.Image]:
        """Split ``board`` into 64 crops ordered a8, b8, ..., h8, ..., a1, ..., h1.

        Args:
            board: A square board image, normally 512x512 RGB.
            white_at_bottom: ``True`` when the board is shown from White's
                point of view (a1 in the bottom-left of the image). When
                ``False`` the board is shown from Black's point of view, so the
                raster scan order is exactly reversed. Digital boards draw the
                pieces upright in both orientations, so the square images
                themselves are only reordered, never rotated.

        Returns:
            A list of exactly 64 Pillow images in FEN order.

        Raises:
            TypeError: If ``board`` is not a Pillow image.
            ValueError: If ``board`` is not square, or is smaller than 8x8.
        """
        if not isinstance(board, Image.Image):
            raise TypeError(f"Expected a PIL.Image.Image, got {type(board)!r}")

        width, height = board.size
        if width != height:
            raise ValueError(f"Board image must be square, got {width}x{height}")
        if width < BOARD_SIDE:
            raise ValueError(
                f"Board image must be at least {BOARD_SIDE}x{BOARD_SIDE}, got {width}x{height}"
            )

        edges = _boundaries(width)
        raster: list[Image.Image] = []
        for row in range(BOARD_SIDE):
            for column in range(BOARD_SIDE):
                box = (edges[column], edges[row], edges[column + 1], edges[row + 1])
                raster.append(board.crop(box))

        if white_at_bottom:
            return raster
        return list(reversed(raster))
