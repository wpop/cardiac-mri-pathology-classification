"""Unit tests for Phase 6 training-run artifact helpers."""

import json
from pathlib import Path

from cardiac_pathology.training import EpochMetrics, TrainingHistory
from scripts.run_training import (
    build_run_summary_payload,
    experiment_dir_for_run,
    save_run_summary,
)


def test_random_and_pretrained_experiment_dirs_are_distinct(tmp_path: Path) -> None:
    """Fold-specific random and pretrained outputs never share a directory."""
    random_dir = experiment_dir_for_run(
        checkpoint_root=tmp_path,
        fold_index=0,
        initialization_strategy="random",
    )
    pretrained_dir = experiment_dir_for_run(
        checkpoint_root=tmp_path,
        fold_index=0,
        initialization_strategy="pretrained",
    )

    assert random_dir == tmp_path / "fold_0" / "random"
    assert pretrained_dir == tmp_path / "fold_0" / "pretrained"
    assert random_dir != pretrained_dir


def test_run_summary_payload_contains_required_top_level_metadata() -> None:
    """Run summary payload exposes required reproducibility fields directly."""
    history = build_history()

    payload = build_run_summary_payload(
        fold_index=0,
        seed=42,
        inner_train_patient_ids=("patient001", "patient002"),
        inner_validation_patient_ids=("patient003",),
        training_config={"batch_size": 2, "initialization_strategy": "random"},
        initialization_strategy="random",
        initialization_report={"initialization_strategy": "random", "parameters": []},
        history=history,
    )

    assert payload["fold_index"] == 0
    assert payload["seed"] == 42
    assert payload["inner_train_patient_ids"] == ["patient001", "patient002"]
    assert payload["inner_validation_patient_ids"] == ["patient003"]
    assert payload["training_config"] == {
        "batch_size": 2,
        "initialization_strategy": "random",
    }
    assert payload["initialization_strategy"] == "random"
    assert payload["initialization_report"] == {
        "initialization_strategy": "random",
        "parameters": [],
    }
    assert payload["history"] == history.as_metadata()
    assert payload["best_epoch"] == 1
    assert payload["best_validation_macro_f1"] == 0.7
    assert payload["stopped_early"] is False


def test_save_run_summary_writes_valid_json(tmp_path: Path) -> None:
    """Run summary persistence writes a valid JSON artifact."""
    summary_path = save_run_summary(
        output_dir=tmp_path / "fold_0" / "pretrained",
        fold_index=0,
        seed=42,
        inner_train_patient_ids=("patient001",),
        inner_validation_patient_ids=("patient002",),
        training_config={"batch_size": 2, "initialization_strategy": "pretrained"},
        initialization_strategy="pretrained",
        initialization_report={"initialization_strategy": "pretrained", "parameters": []},
        history=build_history(),
    )

    with summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert summary_path == tmp_path / "fold_0" / "pretrained" / "run_summary.json"
    assert payload["initialization_strategy"] == "pretrained"
    assert payload["best_epoch"] == 1
    assert payload["best_validation_macro_f1"] == 0.7


def build_history() -> TrainingHistory:
    """Build deterministic non-medical epoch metadata for artifact tests."""
    return TrainingHistory(
        training=[EpochMetrics(epoch=1, loss=0.5, accuracy=0.5, macro_f1=0.4)],
        validation=[EpochMetrics(epoch=1, loss=0.25, accuracy=0.8, macro_f1=0.7)],
        best_epoch=1,
        best_validation_macro_f1=0.7,
        stopped_early=False,
    )
