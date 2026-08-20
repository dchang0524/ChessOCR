from __future__ import annotations

import torch

from chess_ocr.models.square_classifier import (
    SquareClassifier,
    square_classifier_from_checkpoint,
)


def test_transfer_classifier_supports_256px_inputs() -> None:
    model = SquareClassifier(
        architecture="mobilenet_v3_small",
        pretrained_backbone=False,
        input_size=256,
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 256, 256))

    assert logits.shape == (2, 13)


def test_classifier_checkpoint_factory_preserves_architecture() -> None:
    model = square_classifier_from_checkpoint(
        {
            "architecture": "mobilenet_v3_small",
            "input_size": 128,
            "class_names": ["empty", "white_pawn"],
        }
    )

    assert model.architecture == "mobilenet_v3_small"
    assert model.input_size == 128
    assert model.num_classes == 2
