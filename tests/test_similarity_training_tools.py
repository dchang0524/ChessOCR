from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ocr.models.similarity_classifier import SimilarityClassifier
from chess_ocr.training.similarity_trainer import SimilarityTrainer
from scripts.sample_training_augmentations import VARIANT_NAMES, build_montage
from scripts.train_similarity import load_initial_model


def test_load_initial_model_keeps_encoder_and_resets_calibrated_boundary(
    tmp_path: Path,
) -> None:
    source = SimilarityClassifier()
    with torch.no_grad():
        source.projection.weight.fill_(0.125)
        source.similarity_threshold.fill_(0.97)
    checkpoint = tmp_path / "similarity.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "embedding_size": source.embedding_size,
            "epoch": 2,
            "similarity_threshold": 0.97,
        },
        checkpoint,
    )

    restored, start_epoch = load_initial_model(checkpoint)

    assert start_epoch == 2
    assert torch.equal(restored.projection.weight, source.projection.weight)
    assert float(restored.similarity_threshold.detach()) == 0.5


def test_similarity_trainer_continues_epoch_numbering(tmp_path: Path) -> None:
    pairs = TensorDataset(
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 3, 64, 64),
        torch.tensor([0.0, 1.0]),
    )
    loader = DataLoader(pairs, batch_size=2)
    checkpoint = tmp_path / "continued.pt"
    trainer = SimilarityTrainer(SimilarityClassifier(), checkpoint, device="cpu")

    history = trainer.fit(loader, loader, epochs=1, start_epoch=2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert history.epochs[0].epoch == 3
    assert history.best_epoch == 3
    assert saved["epoch"] == 3


def test_augmentation_montage_contains_four_columns() -> None:
    variants = {
        name: Image.new("RGB", (64, 64), (index * 40, 20, 30))
        for index, name in enumerate(VARIANT_NAMES)
    }

    montage = build_montage([variants, variants])

    assert montage.size == (34 + 256 * 4, 38 + 256 * 2)
