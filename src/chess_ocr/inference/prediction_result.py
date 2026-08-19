"""Dataclasses describing the result of a board recognition."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.80


@dataclass
class SquarePrediction:
    """Prediction for a single square.

    Attributes:
        square: Algebraic square name, for example ``"e4"``.
        class_id: Predicted class id.
        class_name: Predicted class name, for example ``"white_queen"``.
        fen_symbol: FEN symbol for the predicted class (``""`` when empty).
        confidence: Softmax probability of the predicted class.
        probabilities: Full softmax distribution over all classes.
    """

    square: str
    class_id: int
    class_name: str
    fen_symbol: str
    confidence: float
    probabilities: list[float]
    group_id: int | None = None
    raw_class_id: int | None = None
    raw_class_name: str | None = None
    raw_logits: list[float] = field(default_factory=list)


@dataclass
class PieceGroupPrediction:
    """Prediction and membership information for one appearance group."""

    group_id: int
    squares: list[str]
    class_id: int
    class_name: str
    confidence: float
    clustering_confidence: float
    class_probabilities: list[float]
    user_corrected: bool = False


@dataclass
class BoardPrediction:
    """Prediction for a whole board.

    Attributes:
        board_fen: Detected board-placement field of a FEN string.
        full_fen: Complete FEN built from the detected placement plus assumed
            fields, or ``None`` when the user did not choose a side to move.
        squares: The 64 per-square predictions in FEN order (a8 first).
        mean_confidence: Mean confidence across all 64 squares.
        minimum_confidence: Lowest confidence across all 64 squares.
        low_confidence_squares: Names of squares whose confidence fell below the
            configured threshold, ordered from least to most confident.
    """

    board_fen: str
    full_fen: str | None
    squares: list[SquarePrediction]
    mean_confidence: float
    minimum_confidence: float
    low_confidence_squares: list[str]
    groups: list[PieceGroupPrediction] = field(default_factory=list)

    def to_rows(self) -> list[dict[str, object]]:
        """Return the per-square predictions as table rows for display.

        Returns:
            A list of dictionaries with ``square``, ``predicted_class``,
            ``fen_symbol`` and ``confidence`` keys.
        """
        return [
            {
                "square": prediction.square,
                "predicted_class": prediction.class_name,
                "fen_symbol": prediction.fen_symbol or "-",
                "confidence": round(prediction.confidence, 4),
            }
            for prediction in self.squares
        ]
