"""Generate deterministic Phase 7 cross-validation visual outputs."""

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cardiac_pathology.training.phase7_oof_pooling import (
    EXPECTED_NUM_CLASSES,
    EXPECTED_NUM_FOLDS,
    EXPECTED_TOTAL_RECORDS,
    Phase7OofRecord,
    oof_record_from_mapping,
    validate_oof_record,
)

FIGURE_FILENAMES = (
    "phase7_loss_history.png",
    "phase7_macro_f1_history.png",
    "phase7_pooled_confusion_matrix.png",
    "phase7_pooled_ovr_roc.png",
    "phase7_per_class_metrics.png",
)


@dataclass(frozen=True, slots=True)
class Phase7EpochHistory:
    """Training and validation history for one epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    training_macro_f1: float
    validation_macro_f1: float


@dataclass(frozen=True, slots=True)
class Phase7FoldVisualizationData:
    """Loaded history metadata for one Phase 7 fold."""

    fold_index: int
    best_epoch: int
    epochs: tuple[Phase7EpochHistory, ...]


class Phase7VisualizationBuilder:
    """Render Phase 7 final figures from frozen JSON artifacts."""

    def __init__(
        self,
        *,
        fold_summary_paths: Mapping[int, Path],
        metrics_path: Path,
        pooled_oof_path: Path,
        figures_dir: Path,
    ) -> None:
        self.fold_summary_paths = dict(fold_summary_paths)
        self.metrics_path = metrics_path
        self.pooled_oof_path = pooled_oof_path
        self.figures_dir = figures_dir

    @classmethod
    def from_phase7_checkpoint_root(
        cls,
        *,
        checkpoint_root: Path,
        figures_dir: Path | None = None,
    ) -> "Phase7VisualizationBuilder":
        """Build a renderer for the frozen Phase 7 checkpoint layout."""
        return cls(
            fold_summary_paths={
                fold_index: checkpoint_root
                / f"fold_{fold_index}"
                / "pretrained"
                / "phase7_fold_summary.json"
                for fold_index in range(EXPECTED_NUM_FOLDS)
            },
            metrics_path=checkpoint_root / "phase7_metrics.json",
            pooled_oof_path=checkpoint_root / "pooled_oof_predictions.json",
            figures_dir=figures_dir if figures_dir is not None else checkpoint_root / "figures",
        )

    def build_all(self) -> tuple[Path, ...]:
        """Render all required Phase 7 PNG figures."""
        fold_histories = load_fold_visualization_data(self.fold_summary_paths)
        metrics = load_metrics_artifact(self.metrics_path)
        pooled_oof_records = load_pooled_oof_records(self.pooled_oof_path)
        validate_visualization_inputs(
            fold_histories=fold_histories,
            metrics=metrics,
            pooled_oof_records=pooled_oof_records,
        )

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_paths = figure_output_paths(self.figures_dir)
        render_loss_history(
            fold_histories=fold_histories,
            output_path=output_paths[0],
        )
        render_macro_f1_history(
            fold_histories=fold_histories,
            output_path=output_paths[1],
        )
        render_confusion_matrix(
            metrics=metrics,
            output_path=output_paths[2],
        )
        render_ovr_roc(
            metrics=metrics,
            pooled_oof_records=pooled_oof_records,
            output_path=output_paths[3],
        )
        render_per_class_metrics(
            metrics=metrics,
            output_path=output_paths[4],
        )
        return output_paths


def figure_output_paths(figures_dir: Path) -> tuple[Path, ...]:
    """Return the deterministic Phase 7 figure output paths."""
    return tuple(figures_dir / filename for filename in FIGURE_FILENAMES)


def load_fold_visualization_data(
    fold_summary_paths: Mapping[int, Path],
) -> tuple[Phase7FoldVisualizationData, ...]:
    """Load training and validation histories from the five fold summary files."""
    expected_fold_indices = set(range(EXPECTED_NUM_FOLDS))
    if set(fold_summary_paths) != expected_fold_indices:
        raise ValueError(
            f"Expected fold summaries for folds 0..4, got {sorted(fold_summary_paths)}"
        )

    folds: list[Phase7FoldVisualizationData] = []
    for expected_fold_index in sorted(fold_summary_paths):
        path = fold_summary_paths[expected_fold_index]
        with path.open("r", encoding="utf-8") as file:
            summary = require_mapping(json.load(file), path)
        fold_index = require_int(summary, "fold_index", path)
        if fold_index != expected_fold_index:
            raise ValueError(f"{path} has fold_index {fold_index}, expected {expected_fold_index}")
        best_epoch = require_int(summary, "best_epoch", path)
        history = require_mapping(summary.get("history"), path)
        training = require_mapping_list(history, "training", path)
        validation = require_mapping_list(history, "validation", path)
        if len(training) != len(validation):
            raise ValueError(f"{path} training and validation histories must have equal length")

        epochs: list[Phase7EpochHistory] = []
        for train_epoch, validation_epoch in zip(training, validation, strict=True):
            epoch = require_int(train_epoch, "epoch", path)
            validation_epoch_index = require_int(validation_epoch, "epoch", path)
            if validation_epoch_index != epoch:
                raise ValueError(f"{path} training/validation epoch indices differ")
            epochs.append(
                Phase7EpochHistory(
                    epoch=epoch,
                    training_loss=require_float(train_epoch, "loss", path),
                    validation_loss=require_float(validation_epoch, "loss", path),
                    training_macro_f1=require_float(train_epoch, "macro_f1", path),
                    validation_macro_f1=require_float(validation_epoch, "macro_f1", path),
                )
            )
        folds.append(
            Phase7FoldVisualizationData(
                fold_index=fold_index,
                best_epoch=best_epoch,
                epochs=tuple(epochs),
            )
        )
    return tuple(folds)


def load_metrics_artifact(path: Path) -> Mapping[str, object]:
    """Load the authoritative final Phase 7 metrics artifact."""
    with path.open("r", encoding="utf-8") as file:
        return require_mapping(json.load(file), path)


def load_pooled_oof_records(path: Path) -> tuple[Phase7OofRecord, ...]:
    """Load pooled OOF records for ROC curve rendering."""
    with path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected pooled OOF records list in {path}")

    records: list[Phase7OofRecord] = []
    for record_index, raw_record in enumerate(raw_records):
        record = oof_record_from_mapping(
            raw_record=raw_record,
            path=path,
            record_index=record_index,
        )
        validate_oof_record(record=record, path=path, record_index=record_index)
        records.append(record)
    return tuple(records)


def validate_visualization_inputs(
    *,
    fold_histories: Sequence[Phase7FoldVisualizationData],
    metrics: Mapping[str, object],
    pooled_oof_records: Sequence[Phase7OofRecord],
) -> None:
    """Validate visualization inputs without redefining metrics."""
    if len(fold_histories) != EXPECTED_NUM_FOLDS:
        raise ValueError(f"Expected {EXPECTED_NUM_FOLDS} fold histories")
    if [fold.fold_index for fold in fold_histories] != list(range(EXPECTED_NUM_FOLDS)):
        raise ValueError("Fold histories must be ordered 0..4")
    if len(pooled_oof_records) != EXPECTED_TOTAL_RECORDS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_RECORDS} pooled OOF records")
    if len({record.patient_id for record in pooled_oof_records}) != EXPECTED_TOTAL_RECORDS:
        raise ValueError("Pooled OOF records must contain 100 unique patients")

    class_order = require_mapping_list(metrics, "class_order", Path("phase7_metrics.json"))
    class_indices = [
        require_int(entry, "class_index", Path("phase7_metrics.json")) for entry in class_order
    ]
    if class_indices != list(range(EXPECTED_NUM_CLASSES)):
        raise ValueError("phase7_metrics.json class_order must be 0..4")
    pooled_oof = require_mapping(metrics.get("pooled_oof"), Path("phase7_metrics.json"))
    confusion_matrix = require_matrix(pooled_oof, "confusion_matrix", Path("phase7_metrics.json"))
    if len(confusion_matrix) != EXPECTED_NUM_CLASSES:
        raise ValueError("confusion matrix must have five rows")


def render_loss_history(
    *,
    fold_histories: Sequence[Phase7FoldVisualizationData],
    output_path: Path,
) -> Path:
    """Render training and validation loss histories for all folds."""
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for fold in fold_histories:
        epochs = [epoch.epoch for epoch in fold.epochs]
        axis.plot(
            epochs,
            [epoch.training_loss for epoch in fold.epochs],
            linewidth=1.6,
            label=f"Fold {fold.fold_index} train",
        )
        axis.plot(
            epochs,
            [epoch.validation_loss for epoch in fold.epochs],
            linestyle="--",
            linewidth=1.6,
            label=f"Fold {fold.fold_index} val",
        )
    axis.set_title("Phase 7 Loss History")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("CrossEntropyLoss")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    save_figure(figure, output_path)
    return output_path


def render_macro_f1_history(
    *,
    fold_histories: Sequence[Phase7FoldVisualizationData],
    output_path: Path,
) -> Path:
    """Render training and validation Macro F1 histories and selected best epochs."""
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for fold in fold_histories:
        epochs = [epoch.epoch for epoch in fold.epochs]
        validation_values = [epoch.validation_macro_f1 for epoch in fold.epochs]
        axis.plot(
            epochs,
            [epoch.training_macro_f1 for epoch in fold.epochs],
            linewidth=1.6,
            label=f"Fold {fold.fold_index} train",
        )
        axis.plot(
            epochs,
            validation_values,
            linestyle="--",
            linewidth=1.6,
            label=f"Fold {fold.fold_index} val",
        )
        best_epoch = find_epoch(fold.epochs, fold.best_epoch)
        axis.scatter(
            [best_epoch.epoch],
            [best_epoch.validation_macro_f1],
            marker="o",
            s=38,
            edgecolors="black",
            zorder=5,
        )
    axis.set_title("Phase 7 Macro F1 History")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    save_figure(figure, output_path)
    return output_path


def render_confusion_matrix(*, metrics: Mapping[str, object], output_path: Path) -> Path:
    """Render the pooled 5x5 confusion matrix from phase7_metrics.json."""
    pooled_oof = require_mapping(metrics.get("pooled_oof"), output_path)
    matrix = require_matrix(pooled_oof, "confusion_matrix", output_path)
    class_names = class_names_from_metrics(metrics)

    figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Phase 7 Pooled Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(EXPECTED_NUM_CLASSES), labels=class_names)
    axis.set_yticks(range(EXPECTED_NUM_CLASSES), labels=class_names)
    max_value = max(max(row) for row in matrix)
    threshold = max_value / 2.0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    save_figure(figure, output_path)
    return output_path


def render_ovr_roc(
    *,
    metrics: Mapping[str, object],
    pooled_oof_records: Sequence[Phase7OofRecord],
    output_path: Path,
) -> Path:
    """Render pooled one-vs-rest ROC curves with stored per-class AUROC labels."""
    auroc_by_class = auroc_values_from_metrics(metrics)
    class_names = class_names_from_metrics(metrics)
    figure, axis = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    for class_index, class_name in enumerate(class_names):
        labels = [
            1 if record.true_class_index == class_index else 0 for record in pooled_oof_records
        ]
        scores = [record.probabilities[class_index] for record in pooled_oof_records]
        false_positive_rates, true_positive_rates = binary_roc_curve_points(
            labels=labels,
            scores=scores,
        )
        auroc = auroc_by_class[class_index]
        auroc_label = "undefined" if auroc is None else f"{auroc:.3f}"
        axis.plot(
            false_positive_rates,
            true_positive_rates,
            linewidth=1.8,
            label=f"{class_name} AUROC {auroc_label}",
        )
    axis.plot([0, 1], [0, 1], color="0.45", linestyle=":", linewidth=1.2)
    axis.set_title("Phase 7 Pooled OvR ROC")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    save_figure(figure, output_path)
    return output_path


def render_per_class_metrics(*, metrics: Mapping[str, object], output_path: Path) -> Path:
    """Render pooled per-class Precision, Recall, and F1 from phase7_metrics.json."""
    pooled_oof = require_mapping(metrics.get("pooled_oof"), output_path)
    per_class = require_mapping_list(pooled_oof, "per_class", output_path)
    class_names = [require_str(entry, "class_name", output_path) for entry in per_class]
    x_positions = list(range(len(class_names)))
    bar_width = 0.25

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for offset_index, metric_name in enumerate(("precision", "recall", "f1")):
        offset = (offset_index - 1) * bar_width
        axis.bar(
            [position + offset for position in x_positions],
            [require_float(entry, metric_name, output_path) for entry in per_class],
            width=bar_width,
            label=metric_name.title(),
        )
    axis.set_title("Phase 7 Pooled Per-Class Metrics")
    axis.set_xlabel("Class")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x_positions, labels=class_names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)
    return output_path


def binary_roc_curve_points(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Compute ROC curve points from binary labels and class probabilities."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    if positive_count == 0 or negative_count == 0:
        return [0.0, 1.0], [0.0, 1.0]

    pairs = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    false_positive_rates = [0.0]
    true_positive_rates = [0.0]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            if pairs[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        false_positive_rates.append(false_positives / negative_count)
        true_positive_rates.append(true_positives / positive_count)
    return false_positive_rates, true_positive_rates


def find_epoch(
    epochs: Sequence[Phase7EpochHistory],
    selected_epoch: int,
) -> Phase7EpochHistory:
    """Find a one-based epoch history record."""
    for epoch in epochs:
        if epoch.epoch == selected_epoch:
            return epoch
    raise ValueError(f"Selected best epoch {selected_epoch} is absent from history")


def class_names_from_metrics(metrics: Mapping[str, object]) -> tuple[str, ...]:
    """Read fixed class names from the authoritative metrics artifact."""
    class_order = require_mapping_list(metrics, "class_order", Path("phase7_metrics.json"))
    return tuple(
        require_str(entry, "class_name", Path("phase7_metrics.json")) for entry in class_order
    )


def auroc_values_from_metrics(metrics: Mapping[str, object]) -> dict[int, float | None]:
    """Read stored per-class AUROC values from the authoritative metrics artifact."""
    pooled_oof = require_mapping(metrics.get("pooled_oof"), Path("phase7_metrics.json"))
    ovr_auroc = require_mapping(pooled_oof.get("ovr_auroc"), Path("phase7_metrics.json"))
    per_class = require_mapping_list(ovr_auroc, "per_class", Path("phase7_metrics.json"))
    values: dict[int, float | None] = {}
    for entry in per_class:
        class_index = require_int(entry, "class_index", Path("phase7_metrics.json"))
        auroc = entry.get("auroc")
        if auroc is not None and not isinstance(auroc, int | float):
            raise ValueError("AUROC values must be numeric or null")
        values[class_index] = None if auroc is None else float(auroc)
    return values


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save and close a Matplotlib figure deterministically."""
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def require_mapping(value: object, path: Path) -> Mapping[str, object]:
    """Return a JSON object as a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def require_mapping_list(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
) -> list[Mapping[str, object]]:
    """Read a required list of JSON mappings."""
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list in {path}")
    return [require_mapping(item, path) for item in value]


def require_matrix(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
) -> list[list[int]]:
    """Read a required integer matrix."""
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a matrix in {path}")
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or not all(isinstance(item, int) for item in row):
            raise ValueError(f"{key} must contain integer rows in {path}")
        matrix.append(row)
    return matrix


def require_int(mapping: Mapping[str, object], key: str, path: Path) -> int:
    """Read a required integer value."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer in {path}")
    return value


def require_float(mapping: Mapping[str, object], key: str, path: Path) -> float:
    """Read a required finite numeric value."""
    value = mapping.get(key)
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number in {path}")
    return float(value)


def require_str(mapping: Mapping[str, object], key: str, path: Path) -> str:
    """Read a required string value."""
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string in {path}")
    return value
