"""Tests for FenBuilder and the FEN <-> class-id round trip."""

from __future__ import annotations

import pytest

from chess_ocr.chess.fen_builder import FenBuilder, board_fen_to_class_ids
from chess_ocr.data.labels import CLASS_NAME_TO_ID, fen_to_class_id

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
EMPTY = CLASS_NAME_TO_ID["empty"]


def symbols_to_ids(symbols: list[str]) -> list[int]:
    return [fen_to_class_id(symbol) for symbol in symbols]


def test_starting_position_board_fen() -> None:
    symbols = list("rnbqkbnr") + list("pppppppp") + [""] * 32 + list("PPPPPPPP") + list("RNBQKBNR")

    assert FenBuilder().build_board_fen(symbols_to_ids(symbols)) == STARTING_BOARD_FEN


def test_empty_board() -> None:
    assert FenBuilder().build_board_fen([EMPTY] * 64) == "8/8/8/8/8/8/8/8"


def test_single_rank_of_pieces_compresses_to_eight_characters() -> None:
    assert (
        FenBuilder().build_board_fen_from_symbols(list("rnbqkbnr") + [""] * 56)
        == "rnbqkbnr/8/8/8/8/8/8/8"
    )


def test_mixed_empty_runs_within_a_rank() -> None:
    rank = ["", "", "R", "", "k", "", "", ""]
    symbols = rank + [""] * 56

    assert FenBuilder().build_board_fen_from_symbols(symbols) == "2R1k3/8/8/8/8/8/8/8"


def test_leading_and_trailing_pieces_on_a_rank() -> None:
    rank = ["K", "", "", "", "", "", "", "q"]
    symbols = [""] * 56 + rank

    assert FenBuilder().build_board_fen_from_symbols(symbols) == "8/8/8/8/8/8/8/K6q"


def test_wrong_number_of_predictions_is_rejected() -> None:
    builder = FenBuilder()

    with pytest.raises(ValueError, match="Expected 64"):
        builder.build_board_fen([EMPTY] * 63)
    with pytest.raises(ValueError, match="Expected 64"):
        builder.build_board_fen([EMPTY] * 65)


def test_out_of_range_class_id_is_rejected() -> None:
    predictions = [EMPTY] * 64
    predictions[7] = 99

    with pytest.raises(ValueError, match="position 7"):
        FenBuilder().build_board_fen(predictions)


def test_negative_class_id_is_rejected() -> None:
    predictions = [EMPTY] * 64
    predictions[0] = -1

    with pytest.raises(ValueError, match="position 0"):
        FenBuilder().build_board_fen(predictions)


def test_unknown_symbol_is_rejected() -> None:
    symbols = [""] * 64
    symbols[3] = "X"

    with pytest.raises(ValueError, match="Unknown FEN symbol"):
        FenBuilder().build_board_fen_from_symbols(symbols)


def test_full_fen_uses_assumed_defaults() -> None:
    full_fen = FenBuilder().build_full_fen(STARTING_BOARD_FEN, side_to_move="w")

    assert full_fen == f"{STARTING_BOARD_FEN} w - - 0 1"


def test_full_fen_with_black_to_move() -> None:
    full_fen = FenBuilder().build_full_fen(STARTING_BOARD_FEN, side_to_move="b")

    assert full_fen.split()[1] == "b"


def test_full_fen_rejects_invalid_side_to_move() -> None:
    with pytest.raises(ValueError, match="side_to_move"):
        FenBuilder().build_full_fen(STARTING_BOARD_FEN, side_to_move="white")


def test_board_fen_to_class_ids_round_trip() -> None:
    class_ids = board_fen_to_class_ids(STARTING_BOARD_FEN)

    assert len(class_ids) == 64
    assert FenBuilder().build_board_fen(class_ids) == STARTING_BOARD_FEN


def test_board_fen_to_class_ids_rejects_short_rank() -> None:
    with pytest.raises(ValueError, match="describes"):
        board_fen_to_class_ids("rnbqkbn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")


def test_board_fen_to_class_ids_rejects_wrong_rank_count() -> None:
    with pytest.raises(ValueError, match="8 ranks"):
        board_fen_to_class_ids("8/8/8")
