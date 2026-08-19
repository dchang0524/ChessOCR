"""Tests for BoardPredictor using a mocked model (no trained checkpoint needed)."""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch import nn

from chess_ocr.chess.fen_builder import board_fen_to_class_ids
from chess_ocr.data.labels import CLASS_NAME_TO_ID
from chess_ocr.inference.board_predictor import BoardPredictor
from chess_ocr.preprocessing.board_normalizer import BoardNormalizer
from chess_ocr.preprocessing.board_splitter import BoardSplitter

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
EMPTY = CLASS_NAME_TO_ID["empty"]


class ScriptedModel(nn.Module):
    """A stand-in model that returns fixed logits for a known class sequence."""

    def __init__(self, class_ids: list[int], logit_scale: float = 10.0) -> None:
        super().__init__()
        self.class_ids = class_ids
        self.logit_scale = logit_scale
        self.calls: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(x.shape))
        logits = torch.zeros(x.shape[0], len(CLASS_NAME_TO_ID))
        for row, class_id in enumerate(self.class_ids[: x.shape[0]]):
            logits[row, class_id] = self.logit_scale
        return logits


class ScriptedSimilarity(nn.Module):
    """Return identical embeddings for scripted semantic classes."""

    def __init__(self, class_ids: list[int]) -> None:
        super().__init__()
        self.class_ids = class_ids

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.tensor(self.class_ids[: x.shape[0]]), num_classes=13
        ).float()


class RecordingNormalizer(BoardNormalizer):
    def __init__(self, output_size: int = 512) -> None:
        super().__init__(output_size)
        self.call_count = 0

    def normalize(self, image: Image.Image) -> Image.Image:
        self.call_count += 1
        return super().normalize(image)


class RecordingSplitter(BoardSplitter):
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def split(self, board: Image.Image, white_at_bottom: bool = True) -> list[Image.Image]:
        self.calls.append(white_at_bottom)
        return super().split(board, white_at_bottom=white_at_bottom)


def build_predictor(
    class_ids: list[int],
    logit_scale: float = 10.0,
    threshold: float = 0.80,
) -> tuple[BoardPredictor, ScriptedModel, RecordingNormalizer, RecordingSplitter]:
    model = ScriptedModel(class_ids, logit_scale=logit_scale)
    normalizer = RecordingNormalizer()
    splitter = RecordingSplitter()
    predictor = BoardPredictor(
        model=model,
        normalizer=normalizer,
        splitter=splitter,
        device=torch.device("cpu"),
        low_confidence_threshold=threshold,
    )
    return predictor, model, normalizer, splitter


def dummy_image() -> Image.Image:
    return Image.new("RGB", (240, 240), color=(120, 130, 140))


def test_normalizer_and_splitter_are_called() -> None:
    predictor, _, normalizer, splitter = build_predictor([EMPTY] * 64)

    predictor.predict(dummy_image(), white_at_bottom=False)

    assert normalizer.call_count == 1
    assert splitter.calls == [False]


def test_model_receives_one_batch_of_sixty_four_squares() -> None:
    predictor, model, _, _ = build_predictor([EMPTY] * 64)

    predictor.predict(dummy_image())

    assert model.calls == [(64, 3, 64, 64)]


def test_starting_position_produces_the_expected_board_fen() -> None:
    class_ids = board_fen_to_class_ids(STARTING_BOARD_FEN)
    predictor, _, _, _ = build_predictor(class_ids)

    result = predictor.predict(dummy_image())

    assert result.board_fen == STARTING_BOARD_FEN


def test_predictions_land_on_the_correct_squares() -> None:
    class_ids = [EMPTY] * 64
    class_ids[0] = CLASS_NAME_TO_ID["black_rook"]  # a8
    class_ids[60] = CLASS_NAME_TO_ID["white_king"]  # e1
    predictor, _, _, _ = build_predictor(class_ids)

    result = predictor.predict(dummy_image())
    by_square = {prediction.square: prediction for prediction in result.squares}

    assert [prediction.square for prediction in result.squares][:3] == ["a8", "b8", "c8"]
    assert result.squares[-1].square == "h1"
    assert by_square["a8"].class_name == "black_rook"
    assert by_square["a8"].fen_symbol == "r"
    assert by_square["e1"].class_name == "white_king"
    assert by_square["e1"].fen_symbol == "K"
    assert by_square["d4"].class_name == "empty"
    assert by_square["d4"].fen_symbol == ""
    assert result.board_fen == "r7/8/8/8/8/8/8/4K3"


def test_confidence_matches_softmax_of_the_mocked_logits() -> None:
    predictor, _, _, _ = build_predictor([EMPTY] * 64, logit_scale=2.0)
    expected = torch.softmax(torch.tensor([2.0] + [0.0] * 12), dim=0).max().item()

    result = predictor.predict(dummy_image())

    assert result.squares[0].confidence == pytest.approx(expected, abs=1e-6)
    assert result.mean_confidence == pytest.approx(expected, abs=1e-6)
    assert result.minimum_confidence == pytest.approx(expected, abs=1e-6)
    assert len(result.squares[0].probabilities) == 13
    assert sum(result.squares[0].probabilities) == pytest.approx(1.0, abs=1e-5)


def test_low_confidence_squares_are_flagged_and_sorted() -> None:
    class_ids = [EMPTY] * 64
    predictor, model, _, _ = build_predictor(class_ids, threshold=0.5)

    # Make a8 the least confident square and b8 the second least confident.
    original_forward = model.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        logits = original_forward(x)
        logits[0] = torch.zeros(13)  # uniform -> confidence 1/13
        logits[1, EMPTY] = 0.5  # low but higher than uniform
        return logits

    model.forward = forward  # type: ignore[method-assign]

    result = predictor.predict(dummy_image())

    assert result.low_confidence_squares == ["a8", "b8"]
    assert result.minimum_confidence == pytest.approx(1 / 13, abs=1e-6)


def test_no_low_confidence_squares_when_model_is_certain() -> None:
    predictor, _, _, _ = build_predictor([EMPTY] * 64, logit_scale=50.0)

    result = predictor.predict(dummy_image())

    assert result.low_confidence_squares == []
    assert result.mean_confidence == pytest.approx(1.0, abs=1e-6)


def test_full_fen_is_none_without_a_side_to_move() -> None:
    predictor, _, _, _ = build_predictor([EMPTY] * 64)

    result = predictor.predict(dummy_image())

    assert result.full_fen is None


def test_full_fen_uses_assumed_fields() -> None:
    class_ids = board_fen_to_class_ids(STARTING_BOARD_FEN)
    predictor, _, _, _ = build_predictor(class_ids)

    result = predictor.predict(dummy_image(), side_to_move="b")

    assert result.full_fen == f"{STARTING_BOARD_FEN} b - - 0 1"


def test_black_orientation_reverses_square_assignment() -> None:
    class_ids = [EMPTY] * 64
    class_ids[0] = CLASS_NAME_TO_ID["white_queen"]
    predictor, _, _, _ = build_predictor(class_ids)

    result = predictor.predict(dummy_image(), white_at_bottom=False)

    # The scripted model always labels the first tensor in the batch, and with
    # Black at the bottom the first square in FEN order is still a8.
    assert result.squares[0].square == "a8"
    assert result.squares[0].class_name == "white_queen"


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="low_confidence_threshold"):
        BoardPredictor(
            model=ScriptedModel([EMPTY] * 64),
            normalizer=BoardNormalizer(),
            splitter=BoardSplitter(),
            device=torch.device("cpu"),
            low_confidence_threshold=1.5,
        )


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        BoardPredictor.from_checkpoint("models/does_not_exist.pt", device="cpu")


def test_to_rows_exposes_table_columns() -> None:
    predictor, _, _, _ = build_predictor([EMPTY] * 64)

    rows = predictor.predict(dummy_image()).to_rows()

    assert len(rows) == 64
    assert set(rows[0]) == {"square", "predicted_class", "fen_symbol", "confidence"}
    assert rows[0]["square"] == "a8"


def test_similarity_groups_same_piece_squares_and_assigns_labels() -> None:
    class_ids = board_fen_to_class_ids(STARTING_BOARD_FEN)
    predictor, _, _, _ = build_predictor(class_ids)
    predictor.similarity_model = ScriptedSimilarity(class_ids)  # type: ignore[assignment]
    predictor.clusterer = predictor.clusterer.__class__(0.99)

    result = predictor.predict(dummy_image())

    white_pawn_groups = [group for group in result.groups if group.class_name == "white_pawn"]
    assert len(white_pawn_groups) == 1
    assert set(white_pawn_groups[0].squares) == {
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"
    }
    assert result.board_fen == STARTING_BOARD_FEN


def test_group_correction_propagates_to_every_group_member() -> None:
    pawn = CLASS_NAME_TO_ID["white_pawn"]
    bishop = CLASS_NAME_TO_ID["white_bishop"]
    class_ids = [EMPTY] * 64
    class_ids[48] = pawn
    class_ids[49] = pawn
    predictor, _, _, _ = build_predictor(class_ids)
    predictor.similarity_model = ScriptedSimilarity(class_ids)  # type: ignore[assignment]
    predictor.clusterer = predictor.clusterer.__class__(0.99)
    result = predictor.predict(dummy_image())
    pawn_group = next(group for group in result.groups if group.class_id == pawn)

    corrected = predictor.apply_group_correction(result, pawn_group.group_id, bishop)

    assert corrected.squares[48].class_id == bishop
    assert corrected.squares[49].class_id == bishop
    corrected_group = next(
        group for group in corrected.groups if group.group_id == pawn_group.group_id
    )
    assert corrected_group.user_corrected


def test_empty_squares_form_a_group_and_can_be_corrected_to_empty() -> None:
    pawn = CLASS_NAME_TO_ID["white_pawn"]
    class_ids = [EMPTY] * 64
    class_ids[0] = pawn
    predictor, _, _, _ = build_predictor(class_ids)
    predictor.similarity_model = ScriptedSimilarity(class_ids)  # type: ignore[assignment]
    predictor.clusterer = predictor.clusterer.__class__(0.99)

    result = predictor.predict(dummy_image())
    empty_group = next(group for group in result.groups if group.class_id == EMPTY)
    assert len(empty_group.squares) == 63

    corrected = predictor.apply_group_correction(result, empty_group.group_id, EMPTY)
    corrected_group = next(
        group for group in corrected.groups if group.group_id == empty_group.group_id
    )
    assert corrected_group.class_id == EMPTY
    assert corrected_group.user_corrected
