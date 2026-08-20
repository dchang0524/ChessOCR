from __future__ import annotations

import torch

from chess_ocr.models.background_normalizer import SquareBackgroundNormalizer
from chess_ocr.models.similarity_classifier import (
    SimilarityClassifier,
    similarity_model_from_checkpoint,
)


def test_background_normalizer_maps_two_square_colours_to_same_neutral_background() -> None:
    normalizer = SquareBackgroundNormalizer()
    light = torch.full((1, 3, 64, 64), 0.7)
    dark = torch.full((1, 3, 64, 64), -0.6)
    light[:, :, 20:44, 28:36] = -0.8
    dark[:, :, 20:44, 28:36] = -0.8

    result = normalizer(torch.cat((light, dark)))

    assert torch.allclose(result[:, :, 0, 0], torch.zeros(2, 3), atol=1e-6)
    assert torch.count_nonzero(result[:, :, 24:40, 30:34]) > 0


def test_similarity_encoder_returns_unit_embeddings() -> None:
    model = SimilarityClassifier(embedding_size=32).eval()
    embeddings = model.encode(torch.randn(4, 3, 64, 64))

    assert embeddings.shape == (4, 32)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(4), atol=1e-5)


def test_siamese_forward_is_symmetric() -> None:
    model = SimilarityClassifier().eval()
    first = torch.randn(2, 3, 64, 64)
    second = torch.randn(2, 3, 64, 64)

    assert torch.allclose(model(first, second), model(second, first), atol=1e-6)


def test_identical_inputs_have_higher_similarity_than_random_inputs() -> None:
    model = SimilarityClassifier().eval()
    first = torch.randn(3, 3, 64, 64)
    second = torch.randn(3, 3, 64, 64)

    same = model(first, first)
    different = model(first, second)

    assert torch.all(same > different)


def test_transfer_encoder_supports_256px_inputs() -> None:
    model = SimilarityClassifier(
        embedding_size=128,
        architecture="mobilenet_v3_small",
        pretrained_backbone=False,
        input_size=256,
    ).eval()

    with torch.no_grad():
        embeddings = model.encode(torch.randn(2, 3, 256, 256))

    assert embeddings.shape == (2, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_checkpoint_factory_preserves_transfer_architecture() -> None:
    model = similarity_model_from_checkpoint(
        {
            "architecture": "mobilenet_v3_small",
            "embedding_size": 128,
            "input_size": 256,
        }
    )

    assert model.architecture == "mobilenet_v3_small"
    assert model.embedding_size == 128
    assert model.input_size == 256
