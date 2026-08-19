"""Shared, differentiable square-background neutralization."""

from __future__ import annotations

import torch
from torch import nn


class SquareBackgroundNormalizer(nn.Module):
    """Replace a square's board colour with neutral gray.

    Inputs use the project's normalized ``[-1, 1]`` RGB range. The background
    colour is estimated from four corner patches. A soft colour-distance mask
    preserves piece pixels and anti-aliased edges. Foreground is expressed as a
    residual from the local background, while background-like pixels map to
    zero (neutral gray).

    Keeping this as a parameter-free ``nn.Module`` makes it part of both model
    graphs: the identical operation runs during training, Python inference, and
    exported ONNX inference.
    """

    def __init__(
        self,
        corner_size: int = 4,
        background_distance: float = 0.06,
        foreground_distance: float = 0.25,
    ) -> None:
        super().__init__()
        if corner_size <= 0:
            raise ValueError("corner_size must be positive")
        if not 0 <= background_distance < foreground_distance:
            raise ValueError("Require 0 <= background_distance < foreground_distance")
        self.corner_size = corner_size
        self.background_distance = background_distance
        self.foreground_distance = foreground_distance

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        """Return squares with their estimated board colour neutralized."""
        if squares.dim() != 4 or squares.shape[1] != 3:
            raise ValueError(
                f"Expected input of shape (batch, 3, H, W), got {tuple(squares.shape)}"
            )
        patch = self.corner_size
        if squares.shape[2] < patch * 2 or squares.shape[3] < patch * 2:
            raise ValueError("Square is too small for the configured corner patches")

        corners = torch.cat(
            (
                squares[:, :, :patch, :patch].flatten(2),
                squares[:, :, :patch, -patch:].flatten(2),
                squares[:, :, -patch:, :patch].flatten(2),
                squares[:, :, -patch:, -patch:].flatten(2),
            ),
            dim=2,
        )
        background = corners.mean(dim=2, keepdim=True).unsqueeze(-1)
        distance = torch.linalg.vector_norm(squares - background, dim=1, keepdim=True)
        foreground = (
            (distance - self.background_distance)
            / (self.foreground_distance - self.background_distance)
        ).clamp(0.0, 1.0)
        # Encode foreground relative to the local board colour. Multiplying the
        # original RGB value by the mask leaves visible checkerboard artefacts
        # on textured or gradient themes; the residual pushes that variation
        # toward neutral gray while retaining the piece's light/dark contrast.
        return ((squares - background) * foreground).clamp(-1.0, 1.0)
