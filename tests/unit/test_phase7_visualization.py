"""Unit tests for Phase 7 visualization helpers."""

import json
from pathlib import Path

import pytest

from cardiac_pathology.training.phase7_visualization import (
    FIGURE_FILENAMES,
    Phase7EpochHistory,
    binary_roc_curve_points,
    figure_output_paths,
    find_epoch,
    load_fold_visualization_data,
    render_per_class_metrics,
)


def test_figure_output_paths_are_deterministic(tmp_path: Path) -> None:
    """All required figure filenames are emitted under the requested directory."""
    output_paths = figure_output_paths(tmp_path)

    assert [path.name for path in output_paths] == list(FIGURE_FILENAMES)
    assert all(path.parent == tmp_path for path in output_paths)


def test_load_fold_visualization_data_reads_histories_and_best_epochs(tmp_path: Path) -> None:
    """Fold visualization loading preserves persisted history values."""
    fold_paths = write_fold_summary_fixture(tmp_path)

    folds = load_fold_visualization_data(fold_paths)

    assert [fold.fold_index for fold in folds] == [0, 1, 2, 3, 4]
    assert folds[0].best_epoch == 2
    assert folds[0].epochs[0].training_loss == pytest.approx(1.0)
    assert folds[0].epochs[1].validation_macro_f1 == pytest.approx(0.4)


def test_find_epoch_returns_selected_epoch() -> None:
    """Best epoch markers resolve to the persisted epoch record."""
    epochs = (
        Phase7EpochHistory(1, 1.0, 1.1, 0.1, 0.2),
        Phase7EpochHistory(2, 0.8, 0.9, 0.3, 0.4),
    )

    selected = find_epoch(epochs, 2)

    assert selected.validation_macro_f1 == pytest.approx(0.4)


def test_binary_roc_curve_points_group_tied_thresholds() -> None:
    """ROC rendering groups tied probability thresholds."""
    false_positive_rates, true_positive_rates = binary_roc_curve_points(
        labels=[1, 0, 1, 0],
        scores=[0.9, 0.8, 0.8, 0.2],
    )

    assert false_positive_rates == [0.0, 0.0, 0.5, 1.0]
    assert true_positive_rates == [0.0, 0.5, 1.0, 1.0]


def test_render_per_class_metrics_writes_png(tmp_path: Path) -> None:
    """Per-class metric rendering uses stored metric values and writes a PNG."""
    output_path = tmp_path / "phase7_per_class_metrics.png"

    render_per_class_metrics(metrics=build_metrics_payload(), output_path=output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def write_fold_summary_fixture(tmp_path: Path) -> dict[int, Path]:
    """Write five minimal Phase 7 fold summaries with history arrays."""
    fold_paths: dict[int, Path] = {}
    for fold_index in range(5):
        path = tmp_path / f"fold_{fold_index}" / "phase7_fold_summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "fold_index": fold_index,
                    "best_epoch": 2,
                    "history": {
                        "training": [
                            {"epoch": 1, "loss": 1.0, "macro_f1": 0.1},
                            {"epoch": 2, "loss": 0.8, "macro_f1": 0.3},
                        ],
                        "validation": [
                            {"epoch": 1, "loss": 1.2, "macro_f1": 0.2},
                            {"epoch": 2, "loss": 0.9, "macro_f1": 0.4},
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        fold_paths[fold_index] = path
    return fold_paths


def build_metrics_payload() -> dict[str, object]:
    """Build minimal metric payload for figure-rendering tests."""
    return {
        "class_order": [
            {"class_index": 0, "class_name": "NOR"},
            {"class_index": 1, "class_name": "DCM"},
            {"class_index": 2, "class_name": "HCM"},
            {"class_index": 3, "class_name": "MINF"},
            {"class_index": 4, "class_name": "RV"},
        ],
        "pooled_oof": {
            "per_class": [
                {
                    "class_index": 0,
                    "class_name": "NOR",
                    "precision": 0.5,
                    "recall": 0.6,
                    "f1": 0.55,
                },
                {
                    "class_index": 1,
                    "class_name": "DCM",
                    "precision": 0.7,
                    "recall": 0.8,
                    "f1": 0.75,
                },
                {
                    "class_index": 2,
                    "class_name": "HCM",
                    "precision": 0.6,
                    "recall": 0.5,
                    "f1": 0.55,
                },
                {
                    "class_index": 3,
                    "class_name": "MINF",
                    "precision": 0.4,
                    "recall": 0.45,
                    "f1": 0.42,
                },
                {
                    "class_index": 4,
                    "class_name": "RV",
                    "precision": 0.9,
                    "recall": 0.85,
                    "f1": 0.87,
                },
            ],
            "confusion_matrix": [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            "ovr_auroc": {"per_class": [], "macro_average": None},
        },
    }
