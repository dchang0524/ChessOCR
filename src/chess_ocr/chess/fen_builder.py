"""Conversion of 64 per-square predictions into FEN notation.

This module is intentionally free of any PyTorch dependency: it operates on
plain class ids or FEN symbols so it can be unit tested and reused without a
model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chess_ocr.data.labels import CLASS_TO_FEN, NUM_CLASSES, class_id_to_fen

BOARD_SIDE = 8
NUM_SQUARES = BOARD_SIDE * BOARD_SIDE

#: Fields the classifier cannot infer from a still image. They are assumed.
DEFAULT_CASTLING = "-"
DEFAULT_EN_PASSANT = "-"
DEFAULT_HALFMOVE_CLOCK = 0
DEFAULT_FULLMOVE_NUMBER = 1
VALID_SIDES = ("w", "b")


@dataclass(frozen=True)
class AssumedFenFields:
    """The FEN fields that are assumed rather than detected.

    Attributes:
        side_to_move: ``"w"`` or ``"b"``; chosen by the user in the UI.
        castling: Castling availability field, assumed ``"-"``.
        en_passant: En passant target square field, assumed ``"-"``.
        halfmove_clock: Halfmove clock, assumed ``0``.
        fullmove_number: Fullmove number, assumed ``1``.
    """

    side_to_move: str = "w"
    castling: str = DEFAULT_CASTLING
    en_passant: str = DEFAULT_EN_PASSANT
    halfmove_clock: int = DEFAULT_HALFMOVE_CLOCK
    fullmove_number: int = DEFAULT_FULLMOVE_NUMBER

    def __post_init__(self) -> None:
        if self.side_to_move not in VALID_SIDES:
            raise ValueError(
                f"side_to_move must be one of {VALID_SIDES}, got {self.side_to_move!r}"
            )


class FenBuilder:
    """Build board-FEN (and optionally full FEN) from per-square predictions."""

    def build_board_fen(self, class_ids: Sequence[int]) -> str:
        """Convert 64 class ids in FEN order into the board part of a FEN.

        Args:
            class_ids: Exactly 64 integer class ids ordered a8, b8, ..., h1.

        Returns:
            The board-placement field of a FEN string, for example
            ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"``.

        Raises:
            ValueError: If the number of predictions is not 64, or a class id is
                outside the valid range.
        """
        if len(class_ids) != NUM_SQUARES:
            raise ValueError(f"Expected {NUM_SQUARES} predictions, got {len(class_ids)}")
        symbols: list[str] = []
        for position, class_id in enumerate(class_ids):
            if not isinstance(class_id, (int,)) or isinstance(class_id, bool):
                raise ValueError(
                    f"Class id at position {position} must be an int, got {class_id!r}"
                )
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Class id at position {position} must be in "
                    f"[0, {NUM_CLASSES - 1}], got {class_id}"
                )
            symbols.append(class_id_to_fen(class_id))
        return self.build_board_fen_from_symbols(symbols)

    def build_board_fen_from_symbols(self, symbols: Sequence[str]) -> str:
        """Convert 64 FEN symbols in FEN order into the board part of a FEN.

        Args:
            symbols: Exactly 64 FEN piece symbols ordered a8, b8, ..., h1. An
                empty string denotes an empty square.

        Returns:
            The board-placement field of a FEN string.

        Raises:
            ValueError: If the number of symbols is not 64, or a symbol is not a
                recognised FEN piece symbol.
        """
        if len(symbols) != NUM_SQUARES:
            raise ValueError(f"Expected {NUM_SQUARES} symbols, got {len(symbols)}")

        valid_symbols = set(CLASS_TO_FEN.values())
        ranks: list[str] = []
        for rank_index in range(BOARD_SIDE):
            rank_symbols = symbols[rank_index * BOARD_SIDE : (rank_index + 1) * BOARD_SIDE]
            ranks.append(self._encode_rank(rank_symbols, valid_symbols))
        return "/".join(ranks)

    def build_full_fen(
        self,
        board_fen: str,
        side_to_move: str = "w",
        castling: str = DEFAULT_CASTLING,
        en_passant: str = DEFAULT_EN_PASSANT,
        halfmove_clock: int = DEFAULT_HALFMOVE_CLOCK,
        fullmove_number: int = DEFAULT_FULLMOVE_NUMBER,
    ) -> str:
        """Combine a board-FEN with the assumed fields into a complete FEN.

        The classifier detects piece placement only. Every field other than
        ``board_fen`` is an assumption and should be labelled as such in the UI.

        Args:
            board_fen: The board-placement field.
            side_to_move: ``"w"`` or ``"b"``.
            castling: Castling availability field.
            en_passant: En passant target square field.
            halfmove_clock: Halfmove clock.
            fullmove_number: Fullmove number.

        Returns:
            A complete six-field FEN string.

        Raises:
            ValueError: If ``side_to_move`` is not ``"w"`` or ``"b"``.
        """
        fields = AssumedFenFields(
            side_to_move=side_to_move,
            castling=castling,
            en_passant=en_passant,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )
        return (
            f"{board_fen} {fields.side_to_move} {fields.castling} "
            f"{fields.en_passant} {fields.halfmove_clock} {fields.fullmove_number}"
        )

    @staticmethod
    def _encode_rank(rank_symbols: Sequence[str], valid_symbols: set[str]) -> str:
        """Encode one rank, compressing consecutive empty squares into digits."""
        encoded: list[str] = []
        empty_run = 0
        for symbol in rank_symbols:
            if symbol not in valid_symbols:
                raise ValueError(f"Unknown FEN symbol: {symbol!r}")
            if symbol == "":
                empty_run += 1
                continue
            if empty_run:
                encoded.append(str(empty_run))
                empty_run = 0
            encoded.append(symbol)
        if empty_run:
            encoded.append(str(empty_run))
        return "".join(encoded)


def board_fen_to_class_ids(board_fen: str) -> list[int]:
    """Expand a board-FEN into 64 class ids in FEN order.

    This is the inverse of :meth:`FenBuilder.build_board_fen` and is used when
    generating labelled training data from known positions.

    Args:
        board_fen: The board-placement field of a FEN string.

    Returns:
        A list of 64 class ids ordered a8, b8, ..., h1.

    Raises:
        ValueError: If the board-FEN is malformed.
    """
    from chess_ocr.data.labels import fen_to_class_id

    ranks = board_fen.split("/")
    if len(ranks) != BOARD_SIDE:
        raise ValueError(f"Board FEN must have {BOARD_SIDE} ranks, got {len(ranks)}")

    class_ids: list[int] = []
    for rank in ranks:
        rank_ids: list[int] = []
        for character in rank:
            if character.isdigit():
                rank_ids.extend([fen_to_class_id("")] * int(character))
            else:
                rank_ids.append(fen_to_class_id(character))
        if len(rank_ids) != BOARD_SIDE:
            raise ValueError(
                f"Rank {rank!r} describes {len(rank_ids)} squares, expected {BOARD_SIDE}"
            )
        class_ids.extend(rank_ids)
    return class_ids
