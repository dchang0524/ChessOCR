"""Lightweight sanity checks on a predicted position.

The validator never modifies a prediction. It only reports warnings so the user
can judge whether a recognition result is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

MAX_PAWNS_PER_SIDE = 8


@dataclass
class ValidationResult:
    """Outcome of validating a predicted board-FEN.

    Attributes:
        is_parseable: Whether ``python-chess`` could parse the board-FEN.
        warnings: Human-readable descriptions of every rule that was violated.
            An empty list means the position passed all checks.
    """

    is_parseable: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def is_plausible(self) -> bool:
        """``True`` when the position parses and produced no warnings."""
        return self.is_parseable and not self.warnings


class PositionValidator:
    """Check a predicted board-FEN against basic chess constraints."""

    def validate(self, board_fen: str) -> ValidationResult:
        """Validate ``board_fen`` and return warnings.

        Checks performed:

        * the board-FEN is parseable by ``python-chess``,
        * exactly one white king and one black king,
        * at most eight pawns per side,
        * no pawns on the first or eighth rank.

        Args:
            board_fen: The board-placement field of a FEN string.

        Returns:
            A :class:`ValidationResult`. When the FEN cannot be parsed, the
            remaining checks are skipped.
        """
        board = chess.Board(None)
        try:
            board.set_board_fen(board_fen)
        except ValueError as error:
            return ValidationResult(
                is_parseable=False,
                warnings=[f"FEN could not be parsed by python-chess: {error}"],
            )

        warnings: list[str] = []
        for color, color_name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
            king_count = len(board.pieces(chess.KING, color))
            if king_count != 1:
                warnings.append(
                    f"{color_name} has {king_count} kings; a legal position has exactly one."
                )

            pawn_squares = board.pieces(chess.PAWN, color)
            if len(pawn_squares) > MAX_PAWNS_PER_SIDE:
                warnings.append(
                    f"{color_name} has {len(pawn_squares)} pawns; the maximum is "
                    f"{MAX_PAWNS_PER_SIDE}."
                )

        back_rank_pawns = [
            chess.square_name(square)
            for square in board.pieces(chess.PAWN, chess.WHITE)
            | board.pieces(chess.PAWN, chess.BLACK)
            if chess.square_rank(square) in (0, 7)
        ]
        if back_rank_pawns:
            warnings.append(
                "Pawns cannot stand on the first or eighth rank: "
                + ", ".join(sorted(back_rank_pawns))
            )

        return ValidationResult(is_parseable=True, warnings=warnings)
