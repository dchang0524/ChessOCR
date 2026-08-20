"""Siamese encoder for comparing chess-square appearances within one theme."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

EMBEDDING_SIZE = 64
TRANSFER_EMBEDDING_SIZE = 128
SIMILARITY_ARCHITECTURES = ("compact", "mobilenet_v3_small")


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

    def __init__(
        self,
        embedding_size: int = EMBEDDING_SIZE,
        architecture: str = "compact",
        pretrained_backbone: bool = False,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if architecture not in SIMILARITY_ARCHITECTURES:
            raise ValueError(
                f"Unknown similarity architecture {architecture!r}; "
                f"expected one of {SIMILARITY_ARCHITECTURES}"
            )
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.embedding_size = embedding_size
        self.architecture = architecture
        self.input_size = input_size
        if architecture == "compact":
            self.features = nn.Sequential(
                _conv_block(3, 32),
                _conv_block(32, 64),
                _conv_block(64, 128),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(128, embedding_size)
            self.register_buffer("imagenet_mean", torch.empty(0), persistent=False)
            self.register_buffer("imagenet_std", torch.empty(0), persistent=False)
        else:
            weights = (
                MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            )
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.pool = backbone.avgpool
            self.projection = nn.Sequential(
                nn.Linear(576, 256),
                nn.LayerNorm(256),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Linear(256, embedding_size),
            )
            self.register_buffer(
                "imagenet_mean",
                torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "imagenet_std",
                torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                persistent=False,
            )

        # sigmoid(scale * (cosine - threshold)); both values are learned.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.similarity_threshold = nn.Parameter(torch.tensor(0.5))

    def encode(self, squares: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings shaped ``(batch, embedding_size)``."""
        if self.architecture == "mobilenet_v3_small":
            # Dataset/browser tensors use [-1, 1]. Convert them to the
            # ImageNet normalization expected by the pretrained backbone.
            squares = ((squares * 0.5 + 0.5) - self.imagenet_mean) / self.imagenet_std
        features = self.features(squares)
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


def similarity_model_from_checkpoint(
    checkpoint: dict[str, object],
) -> nn.Module:
    """Construct a similarity model matching checkpoint architecture metadata."""
    if checkpoint.get("architecture") == "dinov2_vits14_joint":
        from chess_ocr.models.dino_joint_classifier import dino_joint_model_from_checkpoint

        return dino_joint_model_from_checkpoint(checkpoint)
    return SimilarityClassifier(
        embedding_size=int(checkpoint.get("embedding_size", EMBEDDING_SIZE)),
        architecture=str(checkpoint.get("architecture", "compact")),
        pretrained_backbone=False,
        input_size=int(checkpoint.get("input_size", 64)),
    )
