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
from chess_ocr.data.labels import (
    CLASS_NAMES,
    CLASS_NAME_TO_ID,
    SQUARE_NAME_TO_INDEX,
    class_id_to_fen,
    class_id_to_name,
    square_name,
)
from chess_ocr.data.square_dataset import INPUT_SIZE, build_eval_transforms
from chess_ocr.inference.group_label_assigner import GroupLabelAssigner
from chess_ocr.inference.piece_clusterer import PieceCluster, PieceClusterer
from chess_ocr.inference.prediction_result import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    BoardPrediction,
    PieceGroupPrediction,
    SquarePrediction,
)
from chess_ocr.models.similarity_classifier import SimilarityClassifier
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
        similarity_model: SimilarityClassifier | None = None,
        similarity_threshold: float = 0.5,
        duplicate_penalty: float = 1.5,
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
        self.similarity_model = similarity_model
        self.clusterer = PieceClusterer(similarity_threshold)
        self.assigner = GroupLabelAssigner(duplicate_penalty, self.class_names)
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

    @classmethod
    def from_checkpoints(
        cls,
        classifier_checkpoint_path: str | Path,
        similarity_checkpoint_path: str | Path,
        device: str | None = None,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        duplicate_penalty: float = 1.5,
    ) -> BoardPredictor:
        """Build grouped inference from classifier and similarity checkpoints."""
        predictor = cls.from_checkpoint(
            classifier_checkpoint_path,
            device=device,
            low_confidence_threshold=low_confidence_threshold,
        )
        path = Path(similarity_checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Similarity checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=predictor.device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {path} does not contain a 'model_state_dict' entry")
        similarity_model = SimilarityClassifier(int(checkpoint.get("embedding_size", 64)))
        similarity_model.load_state_dict(checkpoint["model_state_dict"])
        similarity_model.to(predictor.device).eval()
        predictor.similarity_model = similarity_model
        predictor.clusterer = PieceClusterer(
            float(checkpoint.get("similarity_threshold", 0.5))
        )
        predictor.assigner = GroupLabelAssigner(duplicate_penalty, predictor.class_names)
        return predictor

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

        cpu_logits = logits.float().cpu()
        probabilities = torch.softmax(cpu_logits, dim=1)
        raw_confidences, raw_class_ids = probabilities.max(dim=1)
        final_class_ids = [int(value) for value in raw_class_ids]
        group_by_square: dict[int, int] = {}
        group_confidences: dict[int, float] = {}
        piece_groups: list[PieceGroupPrediction] = []

        if self.similarity_model is not None:
            self.similarity_model.eval()
            with torch.no_grad():
                embeddings = self.similarity_model.encode(batch).cpu()
            # Cluster every square, including empty squares. Occupancy is now a
            # group label inferred jointly with the twelve piece labels.
            clustering = self.clusterer.cluster(embeddings, list(range(NUM_SQUARES)))
            assignments = self.assigner.assign(cpu_logits, clustering.clusters)
            final_class_ids = self.assigner.apply(
                final_class_ids, clustering.clusters, assignments
            )
            assignment_by_group = {assignment.group_id: assignment for assignment in assignments}
            for cluster in clustering.clusters:
                assignment = assignment_by_group[cluster.group_id]
                group_probabilities = torch.softmax(
                    torch.tensor(assignment.mean_logits), dim=0
                ).tolist()
                piece_groups.append(
                    PieceGroupPrediction(
                        group_id=cluster.group_id,
                        squares=[square_name(index) for index in cluster.square_indices],
                        class_id=assignment.class_id,
                        class_name=self._class_name(assignment.class_id),
                        confidence=assignment.confidence,
                        clustering_confidence=cluster.confidence,
                        class_probabilities=[float(value) for value in group_probabilities],
                    )
                )
                for index in cluster.square_indices:
                    group_by_square[index] = cluster.group_id
                    group_confidences[index] = assignment.confidence

        predictions: list[SquarePrediction] = []
        for index in range(NUM_SQUARES):
            class_id = final_class_ids[index]
            raw_class_id = int(raw_class_ids[index])
            predictions.append(
                SquarePrediction(
                    square=square_name(index),
                    class_id=class_id,
                    class_name=self._class_name(class_id),
                    fen_symbol=class_id_to_fen(class_id),
                    confidence=group_confidences.get(index, float(raw_confidences[index])),
                    probabilities=[float(value) for value in probabilities[index]],
                    group_id=group_by_square.get(index),
                    raw_class_id=raw_class_id,
                    raw_class_name=self._class_name(raw_class_id),
                    raw_logits=[float(value) for value in cpu_logits[index]],
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
            groups=piece_groups,
        )

    def apply_group_correction(
        self,
        prediction: BoardPrediction,
        group_id: int,
        class_id: int,
    ) -> BoardPrediction:
        """Fix one group label, reassign other groups, and rebuild the FEN."""
        if not 0 <= class_id < len(self.class_names):
            raise ValueError(f"class_id must be in [0, {len(self.class_names) - 1}]")
        groups_by_id = {group.group_id: group for group in prediction.groups}
        if group_id not in groups_by_id:
            raise ValueError(f"Unknown group id: {group_id}")
        if any(len(square.raw_logits) != len(self.class_names) for square in prediction.squares):
            raise ValueError("Prediction does not retain the raw logits needed for reassignment")

        clusters = tuple(
            PieceCluster(
                group.group_id,
                tuple(SQUARE_NAME_TO_INDEX[name] for name in group.squares),
                group.clustering_confidence,
            )
            for group in prediction.groups
        )
        fixed = {
            group.group_id: group.class_id
            for group in prediction.groups
            if group.user_corrected
        }
        fixed[group_id] = class_id
        logits = torch.tensor([square.raw_logits for square in prediction.squares])
        assignments = self.assigner.assign(logits, clusters, fixed)
        raw_ids = [
            square.raw_class_id if square.raw_class_id is not None else square.class_id
            for square in prediction.squares
        ]
        final_ids = self.assigner.apply(raw_ids, clusters, assignments)
        assignments_by_group = {assignment.group_id: assignment for assignment in assignments}

        for square_index, square in enumerate(prediction.squares):
            square.class_id = final_ids[square_index]
            square.class_name = self._class_name(square.class_id)
            square.fen_symbol = class_id_to_fen(square.class_id)
            if square.group_id is not None:
                square.confidence = assignments_by_group[square.group_id].confidence
        for group in prediction.groups:
            assignment = assignments_by_group[group.group_id]
            group.class_id = assignment.class_id
            group.class_name = self._class_name(assignment.class_id)
            group.confidence = assignment.confidence
            group.class_probabilities = [
                float(value)
                for value in torch.softmax(torch.tensor(assignment.mean_logits), dim=0)
            ]
            group.user_corrected = group.group_id in fixed

        prediction.board_fen = self.fen_builder.build_board_fen(final_ids)
        if prediction.full_fen is not None:
            side_to_move = prediction.full_fen.split()[1]
            prediction.full_fen = self.fen_builder.build_full_fen(
                prediction.board_fen, side_to_move=side_to_move
            )
        low_confidence = sorted(
            (p for p in prediction.squares if p.confidence < self.low_confidence_threshold),
            key=lambda item: item.confidence,
        )
        confidences = [square.confidence for square in prediction.squares]
        prediction.mean_confidence = sum(confidences) / NUM_SQUARES
        prediction.minimum_confidence = min(confidences)
        prediction.low_confidence_squares = [square.square for square in low_confidence]
        return prediction

    def _class_name(self, class_id: int) -> str:
        """Return the class name for ``class_id`` using the checkpoint ordering."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return class_id_to_name(class_id)
