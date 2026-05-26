"""
Training loop for the LSTM fall detection model.

Implements the complete training pipeline with early stopping,
learning rate scheduling, class-weighted loss, and metric tracking.

Reference: research/8.2 - Adam optimizer, lr=0.001, early stopping
patience=15, weighted CrossEntropyLoss.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training orchestrator for the fall detection LSTM.

    Parameters
    ----------
    model : torch.nn.Module
        The LSTM model to train.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization coefficient.
    epochs : int
        Maximum number of training epochs.
    early_stopping_patience : int
        Stop training after this many epochs without val loss improvement.
    class_weights : torch.Tensor, optional
        Class weights for imbalanced loss. Shape (num_classes,).
    device : str
        Training device: 'cpu' or 'cuda'.
    scheduler_patience : int
        ReduceLROnPlateau patience.
    scheduler_factor : float
        ReduceLROnPlateau reduction factor.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cpu",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._epochs = epochs
        self._patience = early_stopping_patience

        # Loss function with optional class weights
        weight = class_weights.to(self._device) if class_weights is not None else None
        self._criterion = nn.CrossEntropyLoss(weight=weight)

        # DECISION: Adam optimizer per research/8.2
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # DECISION: ReduceLROnPlateau per research/8.2
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )

        self._best_val_loss = float("inf")
        self._best_state = None
        self._epochs_without_improvement = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Execute the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        dict
            Training history with keys: 'train_loss', 'val_loss',
            'train_acc', 'val_acc', 'lr'.
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        for epoch in range(1, self._epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)

            current_lr = self._optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            self._scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{self._epochs} - "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                    f"lr={current_lr:.6f}"
                )

            if self._epochs_without_improvement >= self._patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self._patience} epochs)"
                )
                break

        # Restore best model weights
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
            logger.info(
                f"Restored best model (val_loss={self._best_val_loss:.4f})"
            )

        return history

    def _train_epoch(self, loader: DataLoader) -> tuple:
        """Run one training epoch. Returns (loss, accuracy)."""
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels in loader:
            sequences = sequences.to(self._device)
            labels = labels.to(self._device)

            self._optimizer.zero_grad()
            logits = self._model(sequences)
            loss = self._criterion(logits, labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _validate_epoch(self, loader: DataLoader) -> tuple:
        """Run one validation epoch. Returns (loss, accuracy)."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in loader:
                sequences = sequences.to(self._device)
                labels = labels.to(self._device)

                logits = self._model(sequences)
                loss = self._criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model weights and training state.

        Parameters
        ----------
        path : Path
            Destination path for the checkpoint file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "best_val_loss": self._best_val_loss,
        }
        torch.save(checkpoint, str(path))
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model weights and training state from a checkpoint.

        Parameters
        ----------
        path : Path
            Path to the checkpoint file.
        """
        path = Path(path)
        checkpoint = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Checkpoint loaded from {path}")

    @property
    def model(self) -> nn.Module:
        """Return the model instance."""
        return self._model
