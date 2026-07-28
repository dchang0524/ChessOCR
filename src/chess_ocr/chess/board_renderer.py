"""Rendering of predicted positions as SVG using ``python-chess``."""

from __future__ import annotations

import chess
import chess.svg

DEFAULT_SIZE = 400


class BoardRenderer:
    """Render a board-FEN as an SVG string suitable for inline display."""

    def __init__(self, size: int = DEFAULT_SIZE) -> None:
        """Initialise the renderer.

        Args:
            size: Width and height of the rendered SVG in pixels.

        Raises:
            ValueError: If ``size`` is not positive.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = size

    def render(
        self,
        board_fen: str,
        white_at_bottom: bool = True,
        highlight_squares: list[str] | None = None,
    ) -> str:
        """Render ``board_fen`` as an SVG string.

        Args:
            board_fen: The board-placement field of a FEN string.
            white_at_bottom: Orientation of the rendered board.
            highlight_squares: Optional algebraic square names to highlight, for
                example the low-confidence squares.

        Returns:
            An SVG document as a string.

        Raises:
            ValueError: If ``board_fen`` cannot be parsed by ``python-chess``.
        """
        board = chess.Board(None)
        try:
            board.set_board_fen(board_fen)
        except ValueError as error:
            raise ValueError(f"Unparseable board FEN: {board_fen!r}") from error

        fill: dict[int, str] = {}
        for name in highlight_squares or []:
            try:
                square = chess.parse_square(name)
            except ValueError as error:
                raise ValueError(f"Unknown square name: {name!r}") from error
            fill[square] = "#e6a23c80"

        return chess.svg.board(
            board,
            size=self.size,
            orientation=chess.WHITE if white_at_bottom else chess.BLACK,
            fill=fill,
            coordinates=True,
        )
