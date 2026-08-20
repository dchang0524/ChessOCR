from __future__ import annotations

import pandas as pd
import torch
from scripts.train_joint_dino import _split_theme_families
from torch import nn

from chess_ocr.models.dino_joint_classifier import DinoJointClassifier, PatchShapeCombiner
from chess_ocr.training.joint_trainer import JointTrainer


class FakeDinoBackbone(nn.Module):
    def __init__(self, feature_size: int = 12) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_size)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = images.mean(dim=(2, 3))
        class_token = self.projection(pooled)
        patch_tokens = class_token.unsqueeze(1).expand(-1, 4, -1)
        return {
            "x_norm_clstoken": class_token,
            "x_norm_patchtokens": patch_tokens,
        }


def test_shape_combiner_uses_global_and_patch_tokens() -> None:
    combiner = PatchShapeCombiner(feature_size=12, dropout=0.0)
    combined = combiner(torch.randn(2, 12), torch.randn(2, 9, 12))
    assert combined.shape == (2, 12)


def test_joint_model_returns_logits_and_normalized_embeddings() -> None:
    model = DinoJointClassifier(
        num_classes=13,
        embedding_size=8,
        input_size=28,
        dropout=0.0,
        pretrained_backbone=False,
        backbone=FakeDinoBackbone(),
        feature_size=12,
    )
    logits, embeddings = model.classify_and_encode(torch.randn(3, 3, 28, 28))
    assert logits.shape == (3, 13)
    assert embeddings.shape == (3, 8)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(3))
    assert model.encode(torch.randn(2, 3, 28, 28)).shape == (2, 8)


def test_consistency_loss_ignores_non_cross_background_pairs() -> None:
    logits_a = torch.randn(2, 13)
    logits_b = torch.randn(2, 13)
    embeddings_a = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    embeddings_b = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    loss = JointTrainer._consistency_loss(
        logits_a,
        logits_b,
        embeddings_a,
        embeddings_b,
        torch.zeros(2),
    )
    assert loss.item() == 0.0


def test_consistency_loss_rewards_matching_cross_background_outputs() -> None:
    logits = torch.randn(2, 13)
    embeddings = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    matching = JointTrainer._consistency_loss(
        logits,
        logits,
        embeddings,
        embeddings,
        torch.ones(2),
    )
    changed = JointTrainer._consistency_loss(
        logits,
        -logits,
        embeddings,
        -embeddings,
        torch.ones(2),
    )
    assert matching.abs().item() < 1e-6
    assert changed > matching


def test_theme_family_split_holds_out_every_palette(tmp_path) -> None:
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "theme": [
                "alpha_classic",
                "alpha_blue",
                "spatial_classic",
                "spatial_green",
            ]
        }
    ).to_csv(metadata, index=False)
    train, validation = _split_theme_families(metadata, ["spatial"])
    assert train == {"alpha_classic", "alpha_blue"}
    assert validation == {"spatial_classic", "spatial_green"}
