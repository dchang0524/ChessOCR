"""Evaluation metrics for the square classifier.

Overall square accuracy is a misleading headline number: roughly half of the
squares on a typical board are empty, so a model that only learned "empty"
already scores well. Every report therefore breaks the number down by
occupancy, by class, and — when board metadata is available — by whole board.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from chess_ocr.data.labels import CLASS_NAME_TO_ID, CLASS_NAMES

EMPTY_CLASS_ID = CLASS_NAME_TO_ID["empty"]


@dataclass
class ClassMetrics:
    """Per-class metrics.

    Attributes:
        class_name: Name of the class.
        support: Number of ground-truth samples of this class.
        precision: Precision for this class.
        recall: Recall for this class.
        accuracy: One-vs-rest accuracy for this class.
    """

    class_name: str
    support: int
    precision: float
    recall: float
    accuracy: float


@dataclass
class BoardMetrics:
    """Board-level metrics, available when position ids are supplied.

    Attributes:
        board_count: Number of complete boards evaluated.
        exact_board_accuracy: Fraction of boards predicted with zero errors.
        mean_incorrect_squares: Average number of wrong squares per board.
        error_distribution: Fraction of boards with 0, 1, 2, and 3+ errors,
            keyed by ``"0"``, ``"1"``, ``"2"`` and ``"3+"``.
    """

    board_count: int
    exact_board_accuracy: float
    mean_incorrect_squares: float
    error_distribution: dict[str, float]


@dataclass
class EvaluationReport:
    """Full evaluation result.

    Attributes:
        class_names: Class ordering used for the metrics.
        overall_accuracy: Accuracy over every square.
        empty_accuracy: Accuracy over ground-truth empty squares.
        occupied_accuracy: Accuracy over ground-truth occupied squares.
        per_class: Per-class metrics in class order.
        confusion: Confusion matrix with true labels on rows.
        targets: Ground-truth class ids in dataset order.
        predictions: Predicted class ids in dataset order.
        board_metrics: Board-level metrics, or ``None`` when position ids were
            not supplied.
    """

    class_names: list[str]
    overall_accuracy: float
    empty_accuracy: float
    occupied_accuracy: float
    per_class: list[ClassMetrics] = field(default_factory=list)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    predictions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    board_metrics: BoardMetrics | None = None

    def misclassified_indices(self) -> list[int]:
        """Return dataset indices whose prediction was wrong."""
        return np.flatnonzero(self.targets != self.predictions).tolist()

    def to_text(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            "Square-level metrics",
            f"  overall accuracy   : {self.overall_accuracy:.4f}",
            f"  empty accuracy     : {self.empty_accuracy:.4f}",
            f"  occupied accuracy  : {self.occupied_accuracy:.4f}",
            "",
            f"{'class':<14}{'support':>9}{'precision':>11}{'recall':>9}{'accuracy':>10}",
        ]
        for metrics in self.per_class:
            lines.append(
                f"{metrics.class_name:<14}{metrics.support:>9}"
                f"{metrics.precision:>11.4f}{metrics.recall:>9.4f}{metrics.accuracy:>10.4f}"
            )
        if self.board_metrics is not None:
            board = self.board_metrics
            lines += [
                "",
                "Board-level metrics",
                f"  boards evaluated        : {board.board_count}",
                f"  exact-board accuracy    : {board.exact_board_accuracy:.4f}",
                f"  mean incorrect squares  : {board.mean_incorrect_squares:.4f}",
                "  boards by error count   : "
                + ", ".join(
                    f"{key} err {value:.4f}" for key, value in board.error_distribution.items()
                ),
            ]
        return "\n".join(lines)

    def save_confusion_matrix(self, output_path: str | Path, normalize: bool = True) -> Path:
        """Save the confusion matrix as a CSV and, if matplotlib is available, a PNG.

        Args:
            output_path: Destination path. The suffix determines the image
                format; a CSV is always written alongside it.
            normalize: Normalise rows to sum to one in the plotted image.

        Returns:
            The path of the written CSV file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = path.with_suffix(".csv")

        header = "," + ",".join(self.class_names)
        rows = [
            self.class_names[index] + "," + ",".join(str(int(value)) for value in row)
            for index, row in enumerate(self.confusion)
        ]
        csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - matplotlib is optional
            return csv_path

        matrix = self.confusion.astype(float)
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

        figure, axes = plt.subplots(figsize=(9, 8))
        image = axes.imshow(matrix, cmap="Blues", vmin=0.0)
        axes.set_xticks(range(len(self.class_names)), self.class_names, rotation=90)
        axes.set_yticks(range(len(self.class_names)), self.class_names)
        axes.set_xlabel("predicted")
        axes.set_ylabel("true")
        axes.set_title("Square classifier confusion matrix")
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return csv_path


class Evaluator:
    """Run a model over a data loader and compute detailed metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise the evaluator.

        Args:
            model: Trained classifier returning raw logits.
            device: Device to run inference on.
            class_names: Class ordering; defaults to
                :data:`chess_ocr.data.labels.CLASS_NAMES`.
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = list(class_names or CLASS_NAMES)

    def evaluate(
        self,
        loader: DataLoader,
        position_ids: Sequence[str] | None = None,
    ) -> EvaluationReport:
        """Evaluate ``loader`` and return a full report.

        Args:
            loader: Loader over the evaluation split. It must **not** shuffle,
                otherwise ``position_ids`` cannot be aligned with predictions.
            position_ids: Optional per-sample board id, in dataset order,
                enabling board-level metrics. When the same position is rendered
                with several themes, combine position and theme into the id so
                each rendered board is counted separately.

        Returns:
            An :class:`EvaluationReport`.

        Raises:
            ValueError: If the loader is empty or ``position_ids`` has the wrong
                length.
        """
        targets, predictions = self._collect_predictions(loader)
        if targets.size == 0:
            raise ValueError("Data loader produced no samples")

        correct = targets == predictions
        empty_mask = targets == EMPTY_CLASS_ID
        occupied_mask = ~empty_mask

        report = EvaluationReport(
            class_names=self.class_names,
            overall_accuracy=float(correct.mean()),
            empty_accuracy=float(correct[empty_mask].mean()) if empty_mask.any() else float("nan"),
            occupied_accuracy=(
                float(correct[occupied_mask].mean()) if occupied_mask.any() else float("nan")
            ),
            confusion=confusion_matrix(
                targets, predictions, labels=list(range(len(self.class_names)))
            ),
            targets=targets,
            predictions=predictions,
        )
        report.per_class = self._per_class_metrics(targets, predictions)

        if position_ids is not None:
            if len(position_ids) != targets.size:
                raise ValueError(
                    f"position_ids has {len(position_ids)} entries but the loader "
                    f"produced {targets.size} samples"
                )
            report.board_metrics = self._board_metrics(targets, predictions, position_ids)
        return report

    def _collect_predictions(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run the model over ``loader`` and return ``(targets, predictions)``."""
        self.model.eval()
        target_batches: list[np.ndarray] = []
        prediction_batches: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                logits = self.model(images.to(self.device))
                prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
                target_batches.append(labels.numpy())

        if not target_batches:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return np.concatenate(target_batches), np.concatenate(prediction_batches)

    def _per_class_metrics(
        self, targets: np.ndarray, predictions: np.ndarray
    ) -> list[ClassMetrics]:
        """Compute precision, recall, one-vs-rest accuracy and support per class."""
        labels = list(range(len(self.class_names)))
        precision, recall, _, support = precision_recall_fscore_support(
            targets, predictions, labels=labels, zero_division=0
        )
        total = targets.size

        metrics: list[ClassMetrics] = []
        for class_id, class_name in enumerate(self.class_names):
            true_positive = int(((targets == class_id) & (predictions == class_id)).sum())
            true_negative = int(((targets != class_id) & (predictions != class_id)).sum())
            metrics.append(
                ClassMetrics(
                    class_name=class_name,
                    support=int(support[class_id]),
                    precision=float(precision[class_id]),
                    recall=float(recall[class_id]),
                    accuracy=(true_positive + true_negative) / total if total else float("nan"),
                )
            )
        return metrics

    @staticmethod
    def _board_metrics(
        targets: np.ndarray,
        predictions: np.ndarray,
        position_ids: Sequence[str],
    ) -> BoardMetrics:
        """Aggregate square errors into per-board statistics."""
        errors: dict[str, int] = {}
        for position_id, target, prediction in zip(position_ids, targets, predictions, strict=True):
            errors.setdefault(position_id, 0)
            if target != prediction:
                errors[position_id] += 1

        counts = np.array(list(errors.values()), dtype=float)
        board_count = counts.size
        distribution = {
            "0": float((counts == 0).mean()),
            "1": float((counts == 1).mean()),
            "2": float((counts == 2).mean()),
            "3+": float((counts >= 3).mean()),
        }
        return BoardMetrics(
            board_count=board_count,
            exact_board_accuracy=float((counts == 0).mean()),
            mean_incorrect_squares=float(counts.mean()),
            error_distribution=distribution,
        )
