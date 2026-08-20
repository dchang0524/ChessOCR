"""Joint DINOv2 classifier and similarity encoder for chess squares."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_ocr.data.labels import NUM_CLASSES

DINO_ARCHITECTURE = "dinov2_vits14_joint"
DINO_MODEL_NAME = "dinov2_vits14"
DINO_REPOSITORY = "facebookresearch/dinov2"
DINO_FEATURE_SIZE = 384
DINO_INPUT_SIZE = 224
DINO_EMBEDDING_SIZE = 128


def load_dinov2_backbone(pretrained: bool = True) -> nn.Module:
    """Load Meta's official DINOv2 ViT-S/14 backbone through PyTorch Hub."""
    return torch.hub.load(
        DINO_REPOSITORY,
        DINO_MODEL_NAME,
        pretrained=pretrained,
        trust_repo=True,
    )


class PatchShapeCombiner(nn.Module):
    """Combine DINO's global token with its most relevant local shape patches.

    A learned query attends over all patch tokens. This gives the chess-specific
    head a direct path to emphasize small discriminating parts such as a slit,
    cross, crown point, or knight profile instead of relying only on DINO's
    global class token.
    """

    def __init__(self, feature_size: int = DINO_FEATURE_SIZE, dropout: float = 0.1) -> None:
        super().__init__()
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        if feature_size % 6 != 0:
            raise ValueError("feature_size must be divisible by six attention heads")
        self.query = nn.Parameter(torch.empty(1, 1, feature_size))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            feature_size,
            num_heads=6,
            dropout=dropout,
            batch_first=True,
        )
        self.combiner = nn.Sequential(
            nn.LayerNorm(feature_size * 2),
            nn.Linear(feature_size * 2, feature_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feature_size),
        )

    def forward(
        self, class_token: torch.Tensor, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return one chess-specific shape representation per image."""
        if class_token.dim() != 2 or patch_tokens.dim() != 3:
            raise ValueError("Expected class tokens (B, D) and patch tokens (B, P, D)")
        if class_token.shape[0] != patch_tokens.shape[0]:
            raise ValueError("Class and patch token batch sizes must match")
        query = self.query.expand(class_token.shape[0], -1, -1)
        attended, _ = self.attention(query, patch_tokens, patch_tokens, need_weights=False)
        return self.combiner(torch.cat((class_token, attended[:, 0]), dim=1))


class DinoJointClassifier(nn.Module):
    """Share one DINOv2 shape backbone between classification and grouping."""

    architecture = DINO_ARCHITECTURE

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_size: int = DINO_EMBEDDING_SIZE,
        input_size: int = DINO_INPUT_SIZE,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        backbone: nn.Module | None = None,
        feature_size: int = DINO_FEATURE_SIZE,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if input_size <= 0 or input_size % 14 != 0:
            raise ValueError("DINO input_size must be positive and divisible by 14")
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.feature_size = feature_size
        self.features = backbone if backbone is not None else load_dinov2_backbone(
            pretrained_backbone
        )
        self.shape_combiner = PatchShapeCombiner(feature_size, dropout)
        self.classifier = nn.Linear(feature_size, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_size),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.similarity_threshold = nn.Parameter(torch.tensor(0.5))
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

    def _backbone_tokens(self, squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if squares.dim() != 4 or squares.shape[1] != 3:
            raise ValueError(
                f"Expected input of shape (batch, 3, H, W), got {tuple(squares.shape)}"
            )
        if squares.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} inputs, "
                f"got {tuple(squares.shape[-2:])}"
            )
        normalized = ((squares * 0.5 + 0.5) - self.imagenet_mean) / self.imagenet_std
        outputs: Any = self.features.forward_features(normalized)
        if not isinstance(outputs, dict):
            raise TypeError("DINO backbone forward_features must return a token dictionary")
        return outputs["x_norm_clstoken"], outputs["x_norm_patchtokens"]

    def shape_features(self, squares: torch.Tensor) -> torch.Tensor:
        """Return the combined global/local shape representation."""
        class_token, patch_tokens = self._backbone_tokens(squares)
        return self.shape_combiner(class_token, patch_tokens)

    def classify_and_encode(self, squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and L2-normalized similarity embeddings."""
        shape_features = self.shape_features(squares)
        logits = self.classifier(shape_features)
        embeddings = F.normalize(self.projection(shape_features), p=2, dim=1)
        return logits, embeddings

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        """Return 13-class logits for compatibility with square inference."""
        logits, _ = self.classify_and_encode(squares)
        return logits

    def encode(self, squares: torch.Tensor) -> torch.Tensor:
        """Return embeddings for compatibility with grouped inference."""
        _, embeddings = self.classify_and_encode(squares)
        return embeddings

    def similarity_logits(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """Return learned same-piece logits for aligned embedding batches."""
        if embedding_a.shape != embedding_b.shape:
            raise ValueError("Embedding shapes must match")
        cosine = (embedding_a * embedding_b).sum(dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        threshold = self.similarity_threshold.clamp(-1.0, 1.0)
        return scale * (cosine - threshold)


def dino_joint_model_from_checkpoint(checkpoint: dict[str, object]) -> DinoJointClassifier:
    """Construct the joint model described by checkpoint metadata."""
    class_names = list(checkpoint.get("class_names", []))
    return DinoJointClassifier(
        num_classes=len(class_names) or NUM_CLASSES,
        embedding_size=int(checkpoint.get("embedding_size", DINO_EMBEDDING_SIZE)),
        input_size=int(checkpoint.get("input_size", DINO_INPUT_SIZE)),
        pretrained_backbone=False,
    )
