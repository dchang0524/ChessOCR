"""Compact CNN that classifies a single 64x64 chessboard square."""

from __future__ import annotations

import torch
from torch import nn

from chess_ocr.data.labels import NUM_CLASSES

INPUT_SIZE = 64


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Return two 3x3 conv+Batch Normalization + ReLU layers followed by 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SquareClassifier(nn.Module):
    """Classify a 3x64x64 RGB square into one of the 13 piece/empty classes.

    ``forward`` returns raw logits. Softmax is applied by the caller (see
    :class:`~chess_ocr.inference.board_predictor.BoardPredictor`) so the model
    stays compatible with :class:`torch.nn.CrossEntropyLoss`.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> None:
        """Initialise the network.

        Args:
            num_classes: Number of output classes; defaults to the 13 classes
                defined in :mod:`chess_ocr.data.labels`.
            dropout: Dropout probability applied before the classifier head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:asdfasd
            x: Batch of images shaped ``(batch, 3, 64, 64)``.

        Returns:
            Raw logits shaped ``(batch, num_classes)``.

        Raises:
            ValueError: If ``x`` is not a 4-D tensor with three channels.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input of shape (batch, 3, H, W), got {tuple(x.shape)}")
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)
