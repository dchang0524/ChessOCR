"""High-level orchestration of board recognition.

This module contains no UI code: it takes a cropped Pillow image and returns a
:class:`~chess_ocr.inference.prediction_result.BoardPrediction`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn

from chess_ocr.chess.fen_builder import FenBuilder
from chess_ocr.data.labels import CLASS_NAMES, class_id_to_fen, class_id_to_name, square_name
from chess_ocr.data.square_dataset import INPUT_SIZE, build_eval_transforms
from chess_ocr.inference.prediction_result import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    BoardPrediction,
    SquarePrediction,
)
from chess_ocr.models.square_classifier import SquareClassifier
from chess_ocr.preprocessing.board_normalizer import BoardNormalizer
from chess_ocr.preprocessing.board_splitter import BoardSplitter

NUM_SQUARES = 64


def resolve_device(device: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        device: Explicit device string such as ``"cpu"``, ``"cuda"`` or
            ``"mps"``. When ``None``, CUDA is preferred, then Apple MPS, then
            CPU.

    Returns:
        A :class:`torch.device`.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BoardPredictor:
    """Normalise, split, classify, and convert a board image into a FEN."""

    def __init__(
        self,
        model: nn.Module,
        normalizer: BoardNormalizer,
        splitter: BoardSplitter,
        device: torch.device,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        class_names: list[str] | None = None,
        input_size: int = INPUT_SIZE,
    ) -> None:
        """Initialise the predictor.

        Args:
            model: A trained classifier returning raw logits.
            normalizer: Converts the cropped image into a fixed-size RGB board.
            splitter: Cuts the normalised board into 64 square images.
            device: Device the model runs on.
            low_confidence_threshold: Squares below this confidence are flagged.
            class_names: Class ordering the model was trained with. Defaults to
                :data:`chess_ocr.data.labels.CLASS_NAMES`.
            input_size: Side length of the square tensors fed to the model.

        Raises:
            ValueError: If ``low_confidence_threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError(
                f"low_confidence_threshold must be in [0, 1], got {low_confidence_threshold}"
            )
        self.model = model
        self.normalizer = normalizer
        self.splitter = splitter
        self.device = device
        self.low_confidence_threshold = low_confidence_threshold
        self.class_names = class_names or list(CLASS_NAMES)
        self.input_size = input_size
        self.transform = build_eval_transforms(input_size)
        self.fen_builder = FenBuilder()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    ) -> BoardPredictor:
        """Build a predictor from a training checkpoint.

        Args:
            checkpoint_path: Path to a checkpoint written by
                :class:`~chess_ocr.training.trainer.Trainer`.
            device: Explicit device string, or ``None`` to auto-detect.
            low_confidence_threshold: Squares below this confidence are flagged.

        Returns:
            A ready-to-use :class:`BoardPredictor` in evaluation mode.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            KeyError: If the checkpoint has no ``model_state_dict``.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        torch_device = resolve_device(device)
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {path} does not contain a 'model_state_dict' entry")

        class_names = list(checkpoint.get("class_names", CLASS_NAMES))
        input_size = int(checkpoint.get("input_size", INPUT_SIZE))
        model = SquareClassifier(num_classes=len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch_device)
        model.eval()

        return cls(
            model=model,
            normalizer=BoardNormalizer(),
            splitter=BoardSplitter(),
            device=torch_device,
            low_confidence_threshold=low_confidence_threshold,
            class_names=class_names,
            input_size=input_size,
        )

    def predict(
        self,
        image: Image.Image,
        white_at_bottom: bool = True,
        side_to_move: str | None = None,
    ) -> BoardPrediction:
        """Recognise the position in a cropped board image.

        All 64 squares are classified in a single forward pass.

        Args:
            image: An already-cropped board image.
            white_at_bottom: Orientation of the board in the image.
            side_to_move: ``"w"`` or ``"b"`` to also build a complete FEN using
                assumed values for the remaining fields; ``None`` to skip it.

        Returns:
            A :class:`BoardPrediction`.

        Raises:
            ValueError: If the image cannot be normalised or split, or if
                ``side_to_move`` is invalid, or the model output has an
                unexpected shape.
        """
        board = self.normalizer.normalize(image)
        squares = self.splitter.split(board, white_at_bottom=white_at_bottom)
        if len(squares) != NUM_SQUARES:
            raise ValueError(f"Expected {NUM_SQUARES} squares, got {len(squares)}")

        batch = torch.stack([self.transform(square) for square in squares]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
        if logits.dim() != 2 or logits.shape[0] != NUM_SQUARES:
            raise ValueError(
                f"Model returned logits of shape {tuple(logits.shape)}; "
                f"expected ({NUM_SQUARES}, num_classes)"
            )

        probabilities = torch.softmax(logits.float(), dim=1).cpu()
        confidences, class_ids = probabilities.max(dim=1)

        predictions: list[SquarePrediction] = []
        for index in range(NUM_SQUARES):
            class_id = int(class_ids[index])
            predictions.append(
                SquarePrediction(
                    square=square_name(index),
                    class_id=class_id,
                    class_name=self._class_name(class_id),
                    fen_symbol=class_id_to_fen(class_id),
                    confidence=float(confidences[index]),
                    probabilities=[float(value) for value in probabilities[index]],
                )
            )

        board_fen = self.fen_builder.build_board_fen(
            [prediction.class_id for prediction in predictions]
        )
        full_fen = (
            self.fen_builder.build_full_fen(board_fen, side_to_move=side_to_move)
            if side_to_move is not None
            else None
        )

        low_confidence = sorted(
            (p for p in predictions if p.confidence < self.low_confidence_threshold),
            key=lambda prediction: prediction.confidence,
        )
        confidence_values = [prediction.confidence for prediction in predictions]

        return BoardPrediction(
            board_fen=board_fen,
            full_fen=full_fen,
            squares=predictions,
            mean_confidence=sum(confidence_values) / NUM_SQUARES,
            minimum_confidence=min(confidence_values),
            low_confidence_squares=[prediction.square for prediction in low_confidence],
        )

    def _class_name(self, class_id: int) -> str:
        """Return the class name for ``class_id`` using the checkpoint ordering."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return class_id_to_name(class_id)
