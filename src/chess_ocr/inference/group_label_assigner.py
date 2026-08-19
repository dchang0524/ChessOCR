"""Globally assign semantic chess labels to appearance clusters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from chess_ocr.data.labels import CLASS_NAMES, CLASS_NAME_TO_ID
from chess_ocr.inference.piece_clusterer import PieceCluster

EMPTY_CLASS_ID = CLASS_NAME_TO_ID["empty"]
KING_CLASS_IDS = {
    CLASS_NAME_TO_ID["white_king"],
    CLASS_NAME_TO_ID["black_king"],
}


@dataclass(frozen=True)
class GroupLabelAssignment:
    """Final label and classifier evidence for one appearance group."""

    group_id: int
    class_id: int
    confidence: float
    mean_logits: tuple[float, ...]
    duplicate_number: int


class GroupLabelAssigner:
    """Maximum-score assignment with soft duplicate-label penalties."""

    def __init__(
        self,
        duplicate_penalty: float = 1.5,
        class_names: list[str] | None = None,
    ) -> None:
        if duplicate_penalty < 0:
            raise ValueError("duplicate_penalty must be non-negative")
        self.duplicate_penalty = duplicate_penalty
        self.class_names = list(class_names or CLASS_NAMES)

    def assign(
        self,
        square_logits: torch.Tensor,
        clusters: tuple[PieceCluster, ...] | list[PieceCluster],
        fixed_labels: dict[int, int] | None = None,
    ) -> tuple[GroupLabelAssignment, ...]:
        """Assign every cluster a label while allowing penalised repeats."""
        if square_logits.dim() != 2:
            raise ValueError("square_logits must have shape (squares, classes)")
        if square_logits.shape[1] != len(self.class_names):
            raise ValueError("square_logits class dimension does not match class_names")
        if not clusters:
            return ()
        fixed = dict(fixed_labels or {})
        group_ids = {cluster.group_id for cluster in clusters}
        if unknown := set(fixed) - group_ids:
            raise ValueError(f"Fixed labels reference unknown group ids: {sorted(unknown)}")

        group_logits = torch.stack(
            [
                square_logits[list(cluster.square_indices)].float().mean(dim=0)
                for cluster in clusters
            ]
        )
        log_probabilities = torch.log_softmax(group_logits, dim=1).cpu().numpy()
        group_count = len(clusters)

        # Each non-king label receives enough duplicate slots for every group.
        # Empty duplicates are not penalized: if appearance clustering splits
        # the background into several groups, all of them may still be empty.
        # Kings are unique on a legal board and therefore get one slot each.
        slots: list[tuple[int, int, float]] = []
        for class_id in range(len(self.class_names)):
            copies = 1 if class_id in KING_CLASS_IDS else group_count
            for duplicate_number in range(copies):
                penalty = (
                    0.0
                    if class_id == EMPTY_CLASS_ID
                    else duplicate_number * self.duplicate_penalty
                )
                slots.append(
                    (
                        class_id,
                        duplicate_number,
                        penalty,
                    )
                )

        scores = np.empty((group_count, len(slots)), dtype=np.float64)
        impossible = -1e9
        for row, cluster in enumerate(clusters):
            required_class = fixed.get(cluster.group_id)
            for column, (class_id, _, penalty) in enumerate(slots):
                if required_class is not None and class_id != required_class:
                    scores[row, column] = impossible
                else:
                    scores[row, column] = log_probabilities[row, class_id] - penalty

        rows, columns = linear_sum_assignment(-scores)
        selected = {int(row): int(column) for row, column in zip(rows, columns, strict=True)}
        assignments: list[GroupLabelAssignment] = []
        probabilities = torch.softmax(group_logits, dim=1)
        for row, cluster in enumerate(clusters):
            column = selected[row]
            class_id, duplicate_number, _ = slots[column]
            assignments.append(
                GroupLabelAssignment(
                    group_id=cluster.group_id,
                    class_id=class_id,
                    confidence=float(probabilities[row, class_id]),
                    mean_logits=tuple(float(value) for value in group_logits[row]),
                    duplicate_number=duplicate_number,
                )
            )
        return tuple(assignments)

    @staticmethod
    def apply(
        baseline_class_ids: list[int],
        clusters: tuple[PieceCluster, ...] | list[PieceCluster],
        assignments: tuple[GroupLabelAssignment, ...] | list[GroupLabelAssignment],
    ) -> list[int]:
        """Return class ids with every group assignment propagated to its members."""
        result = list(baseline_class_ids)
        by_group = {assignment.group_id: assignment for assignment in assignments}
        for cluster in clusters:
            assignment = by_group[cluster.group_id]
            for square_index in cluster.square_indices:
                result[square_index] = assignment.class_id
        return result
