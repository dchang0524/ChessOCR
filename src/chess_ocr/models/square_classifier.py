"""Compact CNN that classifies a single 64x64 chessboard square."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from chess_ocr.data.labels import NUM_CLASSES

INPUT_SIZE = 64
CLASSIFIER_ARCHITECTURES = ("compact", "mobilenet_v3_small")


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

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        architecture: str = "compact",
        pretrained_backbone: bool = False,
        input_size: int = INPUT_SIZE,
    ) -> None:
        """Initialise the network.

        Args:
            num_classes: Number of output classes; defaults to the 13 classes
                defined in :mod:`chess_ocr.data.labels`.
            dropout: Dropout probability applied before the classifier head.
        """
        super().__init__()
        if architecture not in CLASSIFIER_ARCHITECTURES:
            raise ValueError(
                f"Unknown classifier architecture {architecture!r}; "
                f"expected one of {CLASSIFIER_ARCHITECTURES}"
            )
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.num_classes = num_classes
        self.architecture = architecture
        self.input_size = input_size
        if architecture == "compact":
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
            self.register_buffer("imagenet_mean", torch.empty(0), persistent=False)
            self.register_buffer("imagenet_std", torch.empty(0), persistent=False)
        else:
            weights = (
                MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            )
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.pool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(576, 256),
                nn.LayerNorm(256),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Batch of images shaped ``(batch, 3, 64, 64)``.

        Returns:
            Raw logits shaped ``(batch, num_classes)``.

        Raises:
            ValueError: If ``x`` is not a 4-D tensor with three channels.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input of shape (batch, 3, H, W), got {tuple(x.shape)}")
        if self.architecture == "mobilenet_v3_small":
            x = ((x * 0.5 + 0.5) - self.imagenet_mean) / self.imagenet_std
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)


def square_classifier_from_checkpoint(
    checkpoint: dict[str, object],
) -> nn.Module:
    """Construct the classifier architecture described by checkpoint metadata."""
    if checkpoint.get("architecture") == "dinov2_vits14_joint":
        from chess_ocr.models.dino_joint_classifier import dino_joint_model_from_checkpoint

        return dino_joint_model_from_checkpoint(checkpoint)
    class_names = list(checkpoint.get("class_names", []))
    return SquareClassifier(
        num_classes=len(class_names) or NUM_CLASSES,
        architecture=str(checkpoint.get("architecture", "compact")),
        pretrained_backbone=False,
        input_size=int(checkpoint.get("input_size", INPUT_SIZE)),
    )
