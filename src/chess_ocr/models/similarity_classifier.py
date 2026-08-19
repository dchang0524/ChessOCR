"""Siamese encoder for comparing chess-square appearances within one theme."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from chess_ocr.models.background_normalizer import SquareBackgroundNormalizer

EMBEDDING_SIZE = 64


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Return the compact convolution block used by the square models."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SimilarityClassifier(nn.Module):
    """Encode squares and predict whether two squares show the same piece.

    The two Siamese branches are calls to the same :meth:`encode` method and
    therefore share every parameter. During board inference each square is
    encoded once and cosine similarities are calculated from the embeddings.
    """

    def __init__(self, embedding_size: int = EMBEDDING_SIZE) -> None:
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        self.embedding_size = embedding_size
        self.background_normalizer = SquareBackgroundNormalizer()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, embedding_size)

        # sigmoid(scale * (cosine - threshold)); both values are learned.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.similarity_threshold = nn.Parameter(torch.tensor(0.5))

    def encode(self, squares: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings shaped ``(batch, embedding_size)``."""
        normalised = self.background_normalizer(squares)
        features = self.features(normalised)
        pooled = self.pool(features).flatten(1)
        return F.normalize(self.projection(pooled), p=2, dim=1)

    def similarity_logits(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """Return raw same-piece logits for two aligned embedding batches."""
        if embedding_a.shape != embedding_b.shape:
            raise ValueError(
                f"Embedding shapes must match, got {embedding_a.shape} and {embedding_b.shape}"
            )
        cosine = (embedding_a * embedding_b).sum(dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        threshold = self.similarity_threshold.clamp(-1.0, 1.0)
        return scale * (cosine - threshold)

    def forward(self, square_a: torch.Tensor, square_b: torch.Tensor) -> torch.Tensor:
        """Return one raw same-piece logit for each input pair."""
        return self.similarity_logits(self.encode(square_a), self.encode(square_b))


class SimilarityEncoderExport(nn.Module):
    """ONNX export wrapper exposing only the per-square encoder."""

    def __init__(self, model: SimilarityClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        """Return embeddings for an arbitrary square batch."""
        return self.model.encode(squares)
