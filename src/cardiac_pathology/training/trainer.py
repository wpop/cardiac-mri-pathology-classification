"""Reusable PyTorch training loop for cardiac pathology classifiers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cardiac_pathology.training.early_stopping import EarlyStopping
from cardiac_pathology.training.initialization import (
    CompatibilityReport,
    InitializationStrategy,
)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Aggregate metrics computed for one training or validation epoch."""

    epoch: int
    loss: float
    accuracy: float
    macro_f1: float

    def as_metadata(self) -> dict[str, int | float]:
        """Convert epoch metrics to JSON-compatible metadata."""
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


@dataclass(slots=True)
class TrainingHistory:
    """Training and validation metrics collected across epochs."""

    training: list[EpochMetrics] = field(default_factory=list)
    validation: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_macro_f1: float | None = None
    stopped_early: bool = False

    def as_metadata(self) -> dict[str, object]:
        """Convert training history to JSON-compatible metadata."""
        return {
            "training": [metrics.as_metadata() for metrics in self.training],
            "validation": [metrics.as_metadata() for metrics in self.validation],
            "best_epoch": self.best_epoch,
            "best_validation_macro_f1": self.best_validation_macro_f1,
            "stopped_early": self.stopped_early,
        }


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration for reusable Phase 6 model fitting."""

    num_epochs: int
    num_classes: int
    checkpoint_dir: Path
    initialization_strategy: InitializationStrategy
    seed: int


class Trainer:
    """Train and validate a PyTorch classifier using validation Macro F1 selection."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        config: TrainerConfig,
        early_stopping: EarlyStopping,
        initialization_report: CompatibilityReport,
        training_config_metadata: Mapping[str, object],
    ) -> None:
        """Initialize the reusable trainer.

        Args:
            model: Classifier accepting tensors with shape ``[N, C, D, H, W]``
                and returning raw logits with shape ``[N, num_classes]``.
            optimizer: Optimizer used to update model parameters.
            device: Device on which batches and the model are evaluated.
            config: Training-loop configuration including epoch count,
                checkpoint directory, initialization strategy, and seed.
            early_stopping: Validation Macro F1 early-stopping tracker.
            initialization_report: Report describing random or pretrained
                initialization decisions for checkpoint metadata.
            training_config_metadata: JSON-compatible training configuration
                metadata to store with checkpoints.

        Returns:
            None.
        """
        if config.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.early_stopping = early_stopping
        self.initialization_report = initialization_report
        self.training_config_metadata = dict(training_config_metadata)
        self.loss_function = nn.CrossEntropyLoss()

    def fit(
        self,
        train_loader: DataLoader[object],
        validation_loader: DataLoader[object],
    ) -> TrainingHistory:
        """Train the model and save the best checkpoint by validation Macro F1.

        Args:
            train_loader: DataLoader yielding batches of
                ``(inputs, targets)`` where inputs are ``[N, C, D, H, W]`` tensors
                and targets are class-index tensors with shape ``[N]``.
            validation_loader: DataLoader yielding validation batches with the
                same tensor semantics as ``train_loader``.

        Returns:
            Training history containing per-epoch training and validation
            metrics, the best epoch, best validation Macro F1, and early-stop
            status.
        """
        history = TrainingHistory()
        best_validation_macro_f1: float | None = None

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader=train_loader, epoch=epoch)
            validation_metrics = self.validate_epoch(
                validation_loader=validation_loader,
                epoch=epoch,
            )
            history.training.append(train_metrics)
            history.validation.append(validation_metrics)

            improved = self.early_stopping.step(validation_metrics.macro_f1, epoch)
            if improved:
                best_validation_macro_f1 = validation_metrics.macro_f1
                history.best_epoch = epoch
                history.best_validation_macro_f1 = validation_metrics.macro_f1
                self.save_checkpoint(
                    epoch=epoch,
                    best_validation_macro_f1=validation_metrics.macro_f1,
                )

            if self.early_stopping.should_stop:
                history.stopped_early = True
                break

        if best_validation_macro_f1 is None:
            raise RuntimeError("Training completed without a validation checkpoint")
        return history

    def train_epoch(
        self,
        train_loader: DataLoader[object],
        epoch: int,
    ) -> EpochMetrics:
        """Run one supervised training epoch.

        Args:
            train_loader: DataLoader yielding input tensors with shape
                ``[N, C, D, H, W]`` and target class-index tensors with shape
                ``[N]``.
            epoch: One-based epoch index used in returned metrics.

        Returns:
            Aggregated training loss, accuracy, and Macro F1 for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions: list[int] = []
        targets: list[int] = []

        for batch in train_loader:
            inputs, target = unpack_batch(batch)
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            validate_logits(logits)
            loss = self.loss_function(logits, target)
            validate_loss(loss)
            loss.backward()
            self.optimizer.step()

            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())

        return build_epoch_metrics(
            epoch=epoch,
            total_loss=total_loss,
            total_samples=total_samples,
            predictions=predictions,
            targets=targets,
            num_classes=self.config.num_classes,
        )

    def validate_epoch(
        self,
        validation_loader: DataLoader[object],
        epoch: int,
    ) -> EpochMetrics:
        """Run one validation epoch without updating model parameters.

        Args:
            validation_loader: DataLoader yielding input tensors with shape
                ``[N, C, D, H, W]`` and target class-index tensors with shape
                ``[N]``.
            epoch: One-based epoch index used in returned metrics.

        Returns:
            Aggregated validation loss, accuracy, and Macro F1 for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions: list[int] = []
        targets: list[int] = []

        with torch.inference_mode():
            for batch in validation_loader:
                inputs, target = unpack_batch(batch)
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                logits = self.model(inputs)
                validate_logits(logits)
                loss = self.loss_function(logits, target)
                validate_loss(loss)

                batch_size = int(target.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_samples += batch_size
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
                targets.extend(target.detach().cpu().tolist())

        return build_epoch_metrics(
            epoch=epoch,
            total_loss=total_loss,
            total_samples=total_samples,
            predictions=predictions,
            targets=targets,
            num_classes=self.config.num_classes,
        )

    def save_checkpoint(self, epoch: int, best_validation_macro_f1: float) -> Path:
        """Save the best validation checkpoint as state dictionaries and metadata.

        Args:
            epoch: Epoch that produced the best validation Macro F1.
            best_validation_macro_f1: Best validation Macro F1 observed so far.

        Returns:
            Path to the saved checkpoint file.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / "best_checkpoint.pt"
        payload = self.build_checkpoint_payload(
            epoch=epoch,
            best_validation_macro_f1=best_validation_macro_f1,
        )
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def build_checkpoint_payload(
        self,
        epoch: int,
        best_validation_macro_f1: float,
    ) -> dict[str, object]:
        """Build a checkpoint payload without serializing the model object.

        Args:
            epoch: Epoch that produced the best validation Macro F1.
            best_validation_macro_f1: Best validation Macro F1 observed so far.

        Returns:
            Dictionary containing model state, optimizer state, epoch, best
            validation Macro F1, training config, initialization strategy, seed,
            and initialization compatibility metadata.
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "training_config": dict(self.training_config_metadata),
            "initialization_strategy": self.config.initialization_strategy,
            "seed": self.config.seed,
            "initialization_report": self.initialization_report.as_metadata(),
        }


def unpack_batch(batch: object) -> tuple[Tensor, Tensor]:
    """Extract input and target tensors from a DataLoader batch.

    Args:
        batch: Batch object expected to contain exactly ``(inputs, targets)``.

    Returns:
        Input tensor with shape ``[N, C, D, H, W]`` and target tensor with shape
        ``[N]``.
    """
    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise TypeError("batch must be a two-item sequence of tensors")
    inputs = batch[0]
    targets = batch[1]
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("batch inputs and targets must be tensors")
    return inputs, targets.long()


def validate_logits(logits: Tensor) -> None:
    """Validate model logits emitted by a training or validation step.

    Args:
        logits: Raw class logits with shape ``[N, num_classes]``.

    Returns:
        None.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, num_classes]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")


def validate_loss(loss: Tensor) -> None:
    """Validate a scalar supervised loss value.

    Args:
        loss: Scalar loss tensor.

    Returns:
        None.
    """
    if loss.ndim != 0:
        raise ValueError("loss must be a scalar tensor")
    if not torch.isfinite(loss):
        raise ValueError("loss must be finite")


def build_epoch_metrics(
    epoch: int,
    total_loss: float,
    total_samples: int,
    predictions: Sequence[int],
    targets: Sequence[int],
    num_classes: int,
) -> EpochMetrics:
    """Build aggregate epoch metrics from collected predictions and targets.

    Args:
        epoch: One-based epoch index.
        total_loss: Sum of batch losses weighted by batch size.
        total_samples: Number of samples observed in the epoch.
        predictions: Predicted class indices.
        targets: Ground-truth class indices.
        num_classes: Total number of classes used for Macro F1.

    Returns:
        Aggregated epoch metrics.
    """
    if total_samples < 1:
        raise ValueError("epoch must contain at least one sample")
    accuracy, macro_f1 = accuracy_and_macro_f1(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
    )
    return EpochMetrics(
        epoch=epoch,
        loss=total_loss / total_samples,
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def accuracy_and_macro_f1(
    predictions: Sequence[int],
    targets: Sequence[int],
    num_classes: int,
) -> tuple[float, float]:
    """Compute accuracy and unweighted Macro F1 without external metric packages.

    Args:
        predictions: Predicted class indices.
        targets: Ground-truth class indices.
        num_classes: Number of classes over which Macro F1 is averaged.

    Returns:
        Tuple ``(accuracy, macro_f1)``.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not predictions:
        raise ValueError("predictions must be non-empty")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    correct = sum(
        1 for prediction, target in zip(predictions, targets, strict=True) if prediction == target
    )
    f1_values: list[float] = []
    for class_index in range(num_classes):
        true_positive = sum(
            1
            for prediction, target in zip(predictions, targets, strict=True)
            if prediction == class_index and target == class_index
        )
        false_positive = sum(
            1
            for prediction, target in zip(predictions, targets, strict=True)
            if prediction == class_index and target != class_index
        )
        false_negative = sum(
            1
            for prediction, target in zip(predictions, targets, strict=True)
            if prediction != class_index and target == class_index
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)

    return correct / len(targets), sum(f1_values) / num_classes
