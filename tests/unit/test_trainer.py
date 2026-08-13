"""Unit tests for Phase 6 trainer utilities."""

from pathlib import Path

import torch
from torch import nn

from cardiac_pathology.training.early_stopping import EarlyStopping
from cardiac_pathology.training.initialization import CompatibilityReport
from cardiac_pathology.training.trainer import (
    EpochMetrics,
    Trainer,
    TrainerConfig,
    TrainingHistory,
    accuracy_and_macro_f1,
)


def test_accuracy_and_macro_f1_compute_expected_values() -> None:
    """Trainer metric helper computes deterministic accuracy and Macro F1."""
    accuracy, macro_f1 = accuracy_and_macro_f1(
        predictions=(0, 1, 1, 2),
        targets=(0, 1, 2, 2),
        num_classes=3,
    )

    assert accuracy == 0.75
    assert macro_f1 == (1.0 + (2.0 / 3.0) + (2.0 / 3.0)) / 3.0


def test_epoch_metrics_metadata_is_json_compatible() -> None:
    """Epoch metrics expose a stable JSON-compatible metadata shape."""
    metrics = EpochMetrics(epoch=2, loss=0.25, accuracy=0.75, macro_f1=0.5)

    assert metrics.as_metadata() == {
        "epoch": 2,
        "loss": 0.25,
        "accuracy": 0.75,
        "macro_f1": 0.5,
    }


def test_training_history_metadata_contains_complete_epoch_history() -> None:
    """Training history metadata includes epoch lists and selection state."""
    history = TrainingHistory(
        training=[EpochMetrics(epoch=1, loss=0.4, accuracy=0.5, macro_f1=0.25)],
        validation=[EpochMetrics(epoch=1, loss=0.3, accuracy=0.75, macro_f1=0.6)],
        best_epoch=1,
        best_validation_macro_f1=0.6,
        stopped_early=True,
    )

    assert history.as_metadata() == {
        "training": [
            {
                "epoch": 1,
                "loss": 0.4,
                "accuracy": 0.5,
                "macro_f1": 0.25,
            }
        ],
        "validation": [
            {
                "epoch": 1,
                "loss": 0.3,
                "accuracy": 0.75,
                "macro_f1": 0.6,
            }
        ],
        "best_epoch": 1,
        "best_validation_macro_f1": 0.6,
        "stopped_early": True,
    }


def test_checkpoint_payload_contains_required_metadata(tmp_path: Path) -> None:
    """Trainer checkpoint payload stores state dictionaries and Phase 6 metadata."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        config=TrainerConfig(
            num_epochs=1,
            num_classes=2,
            checkpoint_dir=tmp_path,
            initialization_strategy="random",
            seed=42,
        ),
        early_stopping=EarlyStopping(patience=1),
        initialization_report=CompatibilityReport(
            initialization_strategy="random",
            entries=(),
        ),
        training_config_metadata={
            "batch_size": 2,
            "optimizer": "adamw",
        },
    )

    payload = trainer.build_checkpoint_payload(epoch=1, best_validation_macro_f1=0.5)

    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert payload["epoch"] == 1
    assert payload["best_validation_macro_f1"] == 0.5
    assert payload["initialization_strategy"] == "random"
    assert payload["seed"] == 42
    assert payload["training_config"] == {"batch_size": 2, "optimizer": "adamw"}
    assert payload["initialization_report"] == {
        "initialization_strategy": "random",
        "transferred_count": 0,
        "adapted_count": 0,
        "skipped_count": 0,
        "missing_count": 0,
        "parameters": [],
    }
