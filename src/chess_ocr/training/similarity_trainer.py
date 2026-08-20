"""Training loop for the Siamese square-similarity model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from chess_ocr.inference.board_predictor import resolve_device


@dataclass
class SimilarityEpochResult:
    """Metrics for one similarity-training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


@dataclass
class SimilarityTrainingHistory:
    """Similarity training history and best checkpoint information."""

    epochs: list[SimilarityEpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    checkpoint_path: Path | None = None


class SimilarityTrainer:
    """Train a Siamese model with balanced binary cross entropy."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_metadata: dict[str, object] | None = None,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.history = SimilarityTrainingHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0,
    ) -> SimilarityTrainingHistory:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Training similarity model on device: {self.device}")
        final_epoch = start_epoch + epochs
        for epoch in range(start_epoch + 1, final_epoch + 1):
            set_epoch = getattr(train_loader.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch - 1)
            started = time.perf_counter()
            train_loss, train_accuracy = self._run_epoch(train_loader, training=True)
            val_loss, val_accuracy = self._run_epoch(val_loader, training=False)
            duration = time.perf_counter() - started
            result = SimilarityEpochResult(
                epoch, train_loss, train_accuracy, val_loss, val_accuracy, duration
            )
            self.history.epochs.append(result)
            improved = val_accuracy > self.history.best_val_accuracy
            if improved:
                self.history.best_epoch = epoch
                self.history.best_val_accuracy = val_accuracy
                self.save_checkpoint(epoch, val_accuracy)
            marker = " <- best" if improved else ""
            print(
                f"epoch {epoch:3d}/{final_epoch} | train {train_loss:.4f} "
                f"{train_accuracy:.4f} | "
                f"val {val_loss:.4f} {val_accuracy:.4f} | {duration:.1f}s{marker}"
            )
        return self.history

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        seen = 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                if len(batch) == 3:
                    square_a, square_b, targets = batch
                    pair_weights = torch.ones_like(targets)
                elif len(batch) == 4:
                    square_a, square_b, targets, pair_weights = batch
                else:
                    raise ValueError(
                        "Similarity batches must contain squares A/B, targets, "
                        "and optional pair weights"
                    )
                square_a = square_a.to(self.device)
                square_b = square_b.to(self.device)
                targets = targets.float().to(self.device)
                pair_weights = pair_weights.float().to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(square_a, square_b)
                per_pair_loss = self.criterion(logits, targets)
                loss = (per_pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1)
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_size = targets.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_size
                correct += int(((logits >= 0) == (targets >= 0.5)).sum().cpu())
                seen += batch_size
        if seen == 0:
            raise ValueError("Data loader produced no pairs")
        return total_loss / seen, correct / seen

    def save_checkpoint(self, epoch: int, validation_accuracy: float) -> None:
        model = self.model
        threshold = getattr(model, "similarity_threshold", None)
        payload = {
            "model_state_dict": model.state_dict(),
            "embedding_size": int(getattr(model, "embedding_size", 64)),
            "input_size": int(getattr(model, "input_size", 64)),
            "architecture": str(getattr(model, "architecture", "compact")),
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "similarity_threshold": (
                float(threshold.detach().clamp(-1, 1).cpu()) if threshold is not None else 0.5
            ),
            **self.checkpoint_metadata,
        }
        torch.save(payload, self.checkpoint_path)
        self.history.checkpoint_path = self.checkpoint_path
