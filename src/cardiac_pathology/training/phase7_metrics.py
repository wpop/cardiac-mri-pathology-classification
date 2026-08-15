"""Compute final Phase 7 cross-validation metrics from frozen artifacts."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cardiac_pathology.training.phase7_oof_pooling import (
    EXPECTED_NUM_CLASSES,
    EXPECTED_NUM_FOLDS,
    EXPECTED_TOTAL_RECORDS,
    Phase7OofRecord,
    oof_record_from_mapping,
    validate_oof_record,
)

CLASS_NAMES = {
    0: "NOR",
    1: "DCM",
    2: "HCM",
    3: "MINF",
    4: "RV",
}


@dataclass(frozen=True, slots=True)
class Phase7FoldMetric:
    """Outer-test metrics loaded from one fold summary."""

    fold_index: int
    accuracy: float
    macro_f1: float

    def as_metadata(self) -> dict[str, int | float]:
        """Convert fold metrics to JSON metadata."""
        return {
            "fold_index": self.fold_index,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


@dataclass(frozen=True, slots=True)
class Phase7ClassMetrics:
    """Pooled OOF metrics for one class."""

    class_index: int
    class_name: str
    precision: float
    recall: float
    f1: float

    def as_metadata(self) -> dict[str, int | str | float]:
        """Convert class metrics to JSON metadata."""
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class Phase7AurocMetrics:
    """Pooled one-vs-rest AUROC metrics."""

    per_class: tuple[tuple[int, str, float | None], ...]
    macro_average: float | None

    def as_metadata(self) -> dict[str, object]:
        """Convert AUROC metrics to JSON metadata."""
        return {
            "per_class": [
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "auroc": auroc,
                }
                for class_index, class_name, auroc in self.per_class
            ],
            "macro_average": self.macro_average,
        }


class Phase7MetricsBuilder:
    """Build deterministic Phase 7 metric JSON from final CV artifacts."""

    def __init__(
        self,
        *,
        fold_summary_paths: Mapping[int, Path],
        pooled_oof_path: Path,
        output_path: Path,
    ) -> None:
        self.fold_summary_paths = dict(fold_summary_paths)
        self.pooled_oof_path = pooled_oof_path
        self.output_path = output_path

    @classmethod
    def from_phase7_checkpoint_root(
        cls,
        *,
        checkpoint_root: Path,
        output_path: Path | None = None,
    ) -> "Phase7MetricsBuilder":
        """Build a metrics builder for the frozen Phase 7 checkpoint layout."""
        return cls(
            fold_summary_paths={
                fold_index: checkpoint_root
                / f"fold_{fold_index}"
                / "pretrained"
                / "phase7_fold_summary.json"
                for fold_index in range(EXPECTED_NUM_FOLDS)
            },
            pooled_oof_path=checkpoint_root / "pooled_oof_predictions.json",
            output_path=(
                output_path if output_path is not None else checkpoint_root / "phase7_metrics.json"
            ),
        )

    def build_and_save(self) -> Path:
        """Build and persist deterministic final Phase 7 metrics."""
        payload = self.build_metrics_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.output_path

    def build_metrics_payload(self) -> dict[str, object]:
        """Build final Phase 7 metric payload without writing it."""
        fold_metrics = load_fold_metrics(self.fold_summary_paths)
        pooled_records = load_pooled_oof_records(self.pooled_oof_path)
        validate_final_metric_inputs(fold_metrics=fold_metrics, pooled_records=pooled_records)

        accuracy_values = [metric.accuracy for metric in fold_metrics]
        macro_f1_values = [metric.macro_f1 for metric in fold_metrics]
        class_metrics = compute_pooled_per_class_metrics(pooled_records)
        confusion_matrix = compute_confusion_matrix(pooled_records)
        auroc_metrics = compute_pooled_ovr_auroc(pooled_records)

        return {
            "class_order": [
                {"class_index": class_index, "class_name": CLASS_NAMES[class_index]}
                for class_index in range(EXPECTED_NUM_CLASSES)
            ],
            "folds": [metric.as_metadata() for metric in fold_metrics],
            "fold_summary_statistics": {
                "accuracy_mean": arithmetic_mean(accuracy_values),
                "accuracy_sample_std_ddof_1": sample_standard_deviation(accuracy_values),
                "macro_f1_mean": arithmetic_mean(macro_f1_values),
                "macro_f1_sample_std_ddof_1": sample_standard_deviation(macro_f1_values),
            },
            "pooled_oof": {
                "num_records": len(pooled_records),
                "num_unique_patients": len({record.patient_id for record in pooled_records}),
                "per_class": [metric.as_metadata() for metric in class_metrics],
                "confusion_matrix": confusion_matrix,
                "ovr_auroc": auroc_metrics.as_metadata(),
            },
        }


def load_fold_metrics(fold_summary_paths: Mapping[int, Path]) -> tuple[Phase7FoldMetric, ...]:
    """Load per-fold Accuracy and Macro F1 from the five fold summary files."""
    expected_fold_indices = set(range(EXPECTED_NUM_FOLDS))
    if set(fold_summary_paths) != expected_fold_indices:
        raise ValueError(
            f"Expected fold summaries for folds 0..4, got {sorted(fold_summary_paths)}"
        )

    metrics: list[Phase7FoldMetric] = []
    for expected_fold_index in sorted(fold_summary_paths):
        path = fold_summary_paths[expected_fold_index]
        with path.open("r", encoding="utf-8") as file:
            raw_summary = json.load(file)
        summary = require_json_mapping(raw_summary, path)
        fold_index = require_json_int(summary, "fold_index", path)
        if fold_index != expected_fold_index:
            raise ValueError(f"{path} has fold_index {fold_index}, expected {expected_fold_index}")
        metrics.append(
            Phase7FoldMetric(
                fold_index=fold_index,
                accuracy=require_json_float(summary, "outer_test_accuracy", path),
                macro_f1=require_json_float(summary, "outer_test_macro_f1", path),
            )
        )
    return tuple(metrics)


def load_pooled_oof_records(path: Path) -> tuple[Phase7OofRecord, ...]:
    """Load and validate pooled OOF records for final metric computation."""
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


def validate_final_metric_inputs(
    *,
    fold_metrics: Sequence[Phase7FoldMetric],
    pooled_records: Sequence[Phase7OofRecord],
) -> None:
    """Validate final metric input cardinality and class coverage."""
    if len(fold_metrics) != EXPECTED_NUM_FOLDS:
        raise ValueError(f"Expected exactly 5 fold summaries, got {len(fold_metrics)}")
    if [metric.fold_index for metric in fold_metrics] != list(range(EXPECTED_NUM_FOLDS)):
        raise ValueError("Fold summaries must be ordered 0..4")
    if len(pooled_records) != EXPECTED_TOTAL_RECORDS:
        raise ValueError(f"Expected exactly 100 pooled OOF records, got {len(pooled_records)}")

    patient_ids = [record.patient_id for record in pooled_records]
    if len(set(patient_ids)) != EXPECTED_TOTAL_RECORDS:
        raise ValueError("Pooled OOF records must contain 100 unique patients")

    true_classes = {record.true_class_index for record in pooled_records}
    expected_classes = set(range(EXPECTED_NUM_CLASSES))
    if true_classes != expected_classes:
        raise ValueError(f"Expected all five true classes 0..4, got {sorted(true_classes)}")


def compute_pooled_per_class_metrics(
    records: Sequence[Phase7OofRecord],
) -> tuple[Phase7ClassMetrics, ...]:
    """Compute pooled per-class Precision, Recall, and F1 in fixed class order."""
    metrics: list[Phase7ClassMetrics] = []
    for class_index in range(EXPECTED_NUM_CLASSES):
        true_positive = sum(
            1
            for record in records
            if (
                record.predicted_class_index == class_index
                and record.true_class_index == class_index
            )
        )
        false_positive = sum(
            1
            for record in records
            if (
                record.predicted_class_index == class_index
                and record.true_class_index != class_index
            )
        )
        false_negative = sum(
            1
            for record in records
            if (
                record.predicted_class_index != class_index
                and record.true_class_index == class_index
            )
        )
        metrics.append(
            Phase7ClassMetrics(
                class_index=class_index,
                class_name=CLASS_NAMES[class_index],
                precision=safe_divide(true_positive, true_positive + false_positive),
                recall=safe_divide(true_positive, true_positive + false_negative),
                f1=safe_divide(
                    2 * true_positive,
                    (2 * true_positive) + false_positive + false_negative,
                ),
            )
        )
    return tuple(metrics)


def compute_confusion_matrix(records: Sequence[Phase7OofRecord]) -> list[list[int]]:
    """Compute one pooled 5x5 confusion matrix with rows=true, columns=predicted."""
    matrix = [[0 for _ in range(EXPECTED_NUM_CLASSES)] for _ in range(EXPECTED_NUM_CLASSES)]
    for record in records:
        validate_class_index(record.true_class_index, field_name="true_class_index")
        validate_class_index(record.predicted_class_index, field_name="predicted_class_index")
        matrix[record.true_class_index][record.predicted_class_index] += 1
    return matrix


def compute_pooled_ovr_auroc(records: Sequence[Phase7OofRecord]) -> Phase7AurocMetrics:
    """Compute pooled one-vs-rest AUROC from OOF class probabilities."""
    per_class: list[tuple[int, str, float | None]] = []
    defined_values: list[float] = []
    for class_index in range(EXPECTED_NUM_CLASSES):
        labels = [1 if record.true_class_index == class_index else 0 for record in records]
        scores = [record.probabilities[class_index] for record in records]
        auroc = binary_auroc(labels=labels, scores=scores)
        per_class.append((class_index, CLASS_NAMES[class_index], auroc))
        if auroc is not None:
            defined_values.append(auroc)
    return Phase7AurocMetrics(
        per_class=tuple(per_class),
        macro_average=arithmetic_mean(defined_values) if defined_values else None,
    )


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute binary AUROC using average ranks, returning None when undefined."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    if positive_count == 0 or negative_count == 0:
        return None

    ranked_scores = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0 for _ in scores]
    index = 0
    while index < len(ranked_scores):
        tie_end = index + 1
        while tie_end < len(ranked_scores) and ranked_scores[tie_end][1] == ranked_scores[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for ranked_index in range(index, tie_end):
            original_index = ranked_scores[ranked_index][0]
            ranks[original_index] = average_rank
        index = tie_end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)) / (
        positive_count * negative_count
    )


def arithmetic_mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean for a non-empty numeric sequence."""
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Compute sample standard deviation with ddof=1."""
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    mean = arithmetic_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def safe_divide(numerator: int, denominator: int) -> float:
    """Divide two counts, returning 0.0 for a zero denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_class_index(value: int, *, field_name: str) -> None:
    """Validate a class index against the fixed Phase 7 class order."""
    if value not in CLASS_NAMES:
        raise ValueError(f"{field_name} must be in class order 0..4, got {value}")


def require_json_mapping(value: object, path: Path) -> Mapping[str, object]:
    """Return a parsed JSON file as a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def require_json_int(mapping: Mapping[str, object], key: str, path: Path) -> int:
    """Read a required integer field from a JSON mapping."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer in {path}")
    return value


def require_json_float(mapping: Mapping[str, object], key: str, path: Path) -> float:
    """Read a required finite numeric field from a JSON mapping."""
    value = mapping.get(key)
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number in {path}")
    return float(value)
