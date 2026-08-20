"""Training loop for the joint DINO classification and grouping model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chess_ocr.data.labels import CLASS_NAMES
from chess_ocr.inference.board_predictor import resolve_device


@dataclass
class JointEpochResult:
    """Classification, similarity, and consistency metrics for one epoch."""

    epoch: int
    train_loss: float
    train_class_accuracy: float
    train_pair_accuracy: float
    train_consistency_loss: float
    val_loss: float
    val_class_accuracy: float
    val_pair_accuracy: float
    val_consistency_loss: float
    seconds: float


@dataclass
class JointTrainingHistory:
    """History and best-checkpoint information for joint training."""

    epochs: list[JointEpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_val_class_accuracy: float = 0.0
    checkpoint_path: Path | None = None


class JointTrainer:
    """Optimize semantic labels, same-piece similarity, and background invariance."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-3,
        backbone_learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        similarity_loss_weight: float = 0.5,
        consistency_loss_weight: float = 0.5,
        checkpoint_metadata: dict[str, object] | None = None,
    ) -> None:
        if similarity_loss_weight < 0 or consistency_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self.classification_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.pair_criterion = nn.BCEWithLogitsLoss(reduction="none")
        backbone_parameters = [
            parameter for parameter in model.features.parameters() if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        parameter_groups: list[dict[str, object]] = []
        if backbone_parameters:
            parameter_groups.append(
                {"params": backbone_parameters, "lr": backbone_learning_rate}
            )
        if head_parameters:
            parameter_groups.append({"params": head_parameters, "lr": learning_rate})
        if not parameter_groups:
            raise ValueError("Model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
        self.similarity_loss_weight = similarity_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.history = JointTrainingHistory()

    @staticmethod
    def _consistency_loss(
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        cross_background: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize prediction changes for matching pieces on opposite colors."""
        selected = cross_background >= 0.5
        if not bool(selected.any()):
            return logits_a.sum() * 0.0
        probabilities_a = logits_a[selected].softmax(dim=1)
        probabilities_b = logits_b[selected].softmax(dim=1)
        midpoint = ((probabilities_a + probabilities_b) * 0.5).clamp_min(1e-7)
        class_consistency = 0.5 * (
            F.kl_div(midpoint.log(), probabilities_a, reduction="batchmean")
            + F.kl_div(midpoint.log(), probabilities_b, reduction="batchmean")
        )
        embedding_consistency = (
            1.0 - (embeddings_a[selected] * embeddings_b[selected]).sum(dim=1)
        ).mean()
        return class_consistency + embedding_consistency

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0,
    ) -> JointTrainingHistory:
        """Train and retain the checkpoint with best validation class accuracy."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Training joint DINO model on device: {self.device}")
        final_epoch = start_epoch + epochs
        for epoch in range(start_epoch + 1, final_epoch + 1):
            set_epoch = getattr(train_loader.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch - 1)
            started = time.perf_counter()
            train = self._run_epoch(train_loader, training=True)
            val = self._run_epoch(val_loader, training=False)
            seconds = time.perf_counter() - started
            result = JointEpochResult(epoch, *train, *val, seconds)
            self.history.epochs.append(result)
            improved = val[1] > self.history.best_val_class_accuracy
            if improved:
                self.history.best_epoch = epoch
                self.history.best_val_class_accuracy = val[1]
                self.save_checkpoint(epoch, val[1], val[2])
            marker = " <- best" if improved else ""
            print(
                f"epoch {epoch:3d}/{final_epoch} | "
                f"train loss {train[0]:.4f} class {train[1]:.4f} "
                f"pair {train[2]:.4f} consistency {train[3]:.4f} | "
                f"val loss {val[0]:.4f} class {val[1]:.4f} "
                f"pair {val[2]:.4f} consistency {val[3]:.4f} | "
                f"{seconds:.1f}s{marker}"
            )
        return self.history

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> tuple[float, float, float, float]:
        self.model.train(training)
        total_loss = 0.0
        total_consistency = 0.0
        class_correct = 0
        pair_correct = 0
        image_count = 0
        pair_count = 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                if len(batch) != 7:
                    raise ValueError(
                        "Joint batches must contain two squares, pair target/weight, "
                        "two class labels, and a cross-background flag"
                    )
                (
                    square_a,
                    square_b,
                    pair_targets,
                    pair_weights,
                    labels_a,
                    labels_b,
                    cross_background,
                ) = (value.to(self.device) for value in batch)
                pair_targets = pair_targets.float()
                pair_weights = pair_weights.float()
                cross_background = cross_background.float()
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                all_squares = torch.cat((square_a, square_b), dim=0)
                all_logits, all_embeddings = self.model.classify_and_encode(all_squares)
                logits_a, logits_b = all_logits.chunk(2)
                embeddings_a, embeddings_b = all_embeddings.chunk(2)
                pair_logits = self.model.similarity_logits(embeddings_a, embeddings_b)
                class_loss = 0.5 * (
                    self.classification_criterion(logits_a, labels_a)
                    + self.classification_criterion(logits_b, labels_b)
                )
                per_pair_loss = self.pair_criterion(pair_logits, pair_targets)
                pair_loss = (
                    (per_pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1)
                )
                consistency_loss = self._consistency_loss(
                    logits_a,
                    logits_b,
                    embeddings_a,
                    embeddings_b,
                    cross_background,
                )
                loss = (
                    class_loss
                    + self.similarity_loss_weight * pair_loss
                    + self.consistency_loss_weight * consistency_loss
                )
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_pairs = pair_targets.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_pairs
                total_consistency += float(consistency_loss.detach().cpu()) * batch_pairs
                class_correct += int((logits_a.argmax(1) == labels_a).sum().cpu())
                class_correct += int((logits_b.argmax(1) == labels_b).sum().cpu())
                pair_correct += int(
                    ((pair_logits >= 0) == (pair_targets >= 0.5)).sum().cpu()
                )
                image_count += batch_pairs * 2
                pair_count += batch_pairs
        if pair_count == 0:
            raise ValueError("Data loader produced no pairs")
        return (
            total_loss / pair_count,
            class_correct / image_count,
            pair_correct / pair_count,
            total_consistency / pair_count,
        )

    def save_checkpoint(
        self, epoch: int, validation_class_accuracy: float, validation_pair_accuracy: float
    ) -> None:
        """Save all backbone and head weights in one self-contained checkpoint."""
        threshold = self.model.similarity_threshold.detach().clamp(-1, 1)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "embedding_size": int(self.model.embedding_size),
            "input_size": int(self.model.input_size),
            "architecture": str(self.model.architecture),
            "epoch": epoch,
            "validation_accuracy": validation_class_accuracy,
            "validation_pair_accuracy": validation_pair_accuracy,
            "similarity_threshold": float(threshold.cpu()),
            **self.checkpoint_metadata,
        }
        torch.save(payload, self.checkpoint_path)
        self.history.checkpoint_path = self.checkpoint_path
