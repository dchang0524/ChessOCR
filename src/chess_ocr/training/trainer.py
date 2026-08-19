"""Training loop for the square classifier."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from chess_ocr.data.labels import CLASS_NAMES
from chess_ocr.data.square_dataset import INPUT_SIZE
from chess_ocr.inference.board_predictor import resolve_device

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LABEL_SMOOTHING = 0.05


@dataclass
class EpochResult:
    """Metrics for a single epoch.

    Attributes:
        epoch: 1-based epoch number.
        train_loss: Mean training loss.
        train_accuracy: Training accuracy in ``[0, 1]``.
        val_loss: Mean validation loss.
        val_accuracy: Validation accuracy in ``[0, 1]``.
        seconds: Wall-clock duration of the epoch.
    """

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


@dataclass
class TrainingHistory:
    """History of every epoch plus the best checkpoint seen.

    Attributes:
        epochs: One :class:`EpochResult` per completed epoch.
        best_epoch: Epoch number with the highest validation accuracy.
        best_val_accuracy: The highest validation accuracy observed.
        checkpoint_path: Where the best checkpoint was written.
    """

    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    checkpoint_path: Path | None = None


class Trainer:
    """Own the model, optimiser, loss, device, history, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str | Path = "models/square_classifier.pt",
        device: str | torch.device | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
        class_names: list[str] | None = None,
        input_size: int = INPUT_SIZE,
        checkpoint_metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: The classifier to train.
            checkpoint_path: Where the best checkpoint is written.
            device: ``"cpu"``, ``"cuda"``, ``"mps"``, a :class:`torch.device`, or
                ``None`` to auto-detect (CUDA, then Apple MPS, then CPU).
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            label_smoothing: Label smoothing for the cross-entropy loss.
            class_names: Class ordering stored in the checkpoint.
            input_size: Square input size stored in the checkpoint.
            checkpoint_metadata: Additional provenance stored in each checkpoint.
        """
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.class_names = class_names or list(CLASS_NAMES)
        self.input_size = input_size
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        reserved = {
            "model_state_dict",
            "class_names",
            "input_size",
            "epoch",
            "validation_accuracy",
        }
        conflicts = reserved.intersection(self.checkpoint_metadata)
        if conflicts:
            raise ValueError(f"checkpoint_metadata uses reserved keys: {sorted(conflicts)}")
        self.history = TrainingHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train for ``epochs`` epochs, checkpointing the best validation accuracy.

        Args:
            train_loader: Loader over the training split.
            val_loader: Loader over the validation split.
            epochs: Number of epochs to run.
            verbose: Print per-epoch progress.

        Returns:
            The :class:`TrainingHistory` accumulated during this call.

        Raises:
            ValueError: If ``epochs`` is not positive.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = self._run_epoch(train_loader, training=True)
            val_loss, val_accuracy = self._run_epoch(val_loader, training=False)
            duration = time.perf_counter() - started

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                seconds=duration,
            )
            self.history.epochs.append(result)

            improved = val_accuracy > self.history.best_val_accuracy
            if improved:
                self.history.best_val_accuracy = val_accuracy
                self.history.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, validation_accuracy=val_accuracy)

            if verbose:
                marker = "  <- best, checkpoint saved" if improved else ""
                print(
                    f"epoch {epoch:3d}/{epochs} | "
                    f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
                    f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
                    f"{duration:.1f}s{marker}"
                )

        if verbose:
            print(
                f"Best validation accuracy {self.history.best_val_accuracy:.4f} "
                f"at epoch {self.history.best_epoch} -> {self.checkpoint_path}"
            )
        return self.history

    def save_checkpoint(self, epoch: int, validation_accuracy: float) -> Path:
        """Write a checkpoint dictionary (not bare weights) to disk.

        Args:
            epoch: Epoch number being saved.
            validation_accuracy: Validation accuracy at that epoch.

        Returns:
            The path the checkpoint was written to.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "input_size": self.input_size,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            **self.checkpoint_metadata,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.checkpoint_path)
        self.history.checkpoint_path = self.checkpoint_path
        return self.checkpoint_path

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        """Run one pass over ``loader`` and return ``(mean_loss, accuracy)``."""
        self.model.train(training)
        total_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        seen = 0

        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.detach() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum()
                seen += batch_size

        if seen == 0:
            raise ValueError("Data loader produced no samples")
        return float(total_loss.cpu()) / seen, int(correct.cpu()) / seen
