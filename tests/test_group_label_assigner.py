from __future__ import annotations

import torch

from chess_ocr.data.labels import CLASS_NAME_TO_ID
from chess_ocr.inference.group_label_assigner import GroupLabelAssigner
from chess_ocr.inference.piece_clusterer import PieceCluster


def cluster(group_id: int, *indices: int) -> PieceCluster:
    return PieceCluster(group_id, tuple(indices), 0.95)


def test_joint_assignment_gives_weaker_group_its_second_choice() -> None:
    pawn = CLASS_NAME_TO_ID["white_pawn"]
    bishop = CLASS_NAME_TO_ID["white_bishop"]
    logits = torch.full((3, 13), -10.0)
    logits[0, pawn], logits[0, bishop] = 4.0, 1.0
    logits[1, pawn], logits[1, bishop] = 3.0, 2.8
    logits[2, pawn], logits[2, bishop] = 3.0, 2.8

    assignments = GroupLabelAssigner(duplicate_penalty=2.0).assign(
        logits, [cluster(0, 0), cluster(1, 1, 2)]
    )

    assert [assignment.class_id for assignment in assignments] == [pawn, bishop]


def test_strong_duplicate_evidence_can_overcome_penalty() -> None:
    pawn = CLASS_NAME_TO_ID["white_pawn"]
    logits = torch.full((2, 13), -20.0)
    logits[:, pawn] = 10.0

    assignments = GroupLabelAssigner(duplicate_penalty=1.0).assign(
        logits, [cluster(0, 0), cluster(1, 1)]
    )

    assert [assignment.class_id for assignment in assignments] == [pawn, pawn]


def test_fixed_user_label_is_respected() -> None:
    pawn = CLASS_NAME_TO_ID["white_pawn"]
    bishop = CLASS_NAME_TO_ID["white_bishop"]
    logits = torch.full((2, 13), -5.0)
    logits[:, pawn] = 5.0

    assignments = GroupLabelAssigner().assign(
        logits,
        [cluster(0, 0), cluster(1, 1)],
        fixed_labels={1: bishop},
    )

    assert assignments[0].class_id == pawn
    assert assignments[1].class_id == bishop


def test_empty_is_a_valid_repeatable_group_label() -> None:
    empty = CLASS_NAME_TO_ID["empty"]
    logits = torch.full((4, 13), -10.0)
    logits[:, empty] = 10.0

    assignments = GroupLabelAssigner().assign(
        logits, [cluster(0, 0, 1), cluster(1, 2, 3)]
    )

    assert [assignment.class_id for assignment in assignments] == [empty, empty]
