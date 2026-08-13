"""Early stopping utilities for validation-metric based training."""

from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class EarlyStopping:
    """Track validation-metric improvements and request training termination.

    The tracker is maximization-oriented because Phase 6 checkpoint selection is
    based on validation Macro F1. A metric is considered improved only when it is
    greater than the previous best value by more than ``min_delta``.
    """

    patience: int
    min_delta: float = 0.0
    best_metric: float | None = None
    best_epoch: int | None = None
    non_improving_epochs: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        """Validate early-stopping configuration values.

        Returns:
            None.
        """
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def step(self, metric: float, epoch: int) -> bool:
        """Update the tracker with one validation metric value.

        Args:
            metric: Validation metric to maximize. Phase 6 uses validation Macro
                F1, represented as a finite scalar.
            epoch: Zero-based or one-based epoch index supplied by the caller.

        Returns:
            True when ``metric`` is a new best value, otherwise False. The
            ``should_stop`` attribute is set to True once the number of
            consecutive non-improving epochs reaches ``patience``.
        """
        if not isfinite(metric):
            raise ValueError("metric must be finite")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        if self.best_metric is None or metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.non_improving_epochs = 0
            self.should_stop = False
            return True

        self.non_improving_epochs += 1
        if self.non_improving_epochs >= self.patience:
            self.should_stop = True
        return False
