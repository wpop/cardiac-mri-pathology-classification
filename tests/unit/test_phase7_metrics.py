"""Unit tests for Phase 7 final metric math."""

import pytest

from cardiac_pathology.training.phase7_metrics import (
    arithmetic_mean,
    binary_auroc,
    compute_confusion_matrix,
    compute_pooled_ovr_auroc,
    compute_pooled_per_class_metrics,
    sample_standard_deviation,
)
from cardiac_pathology.training.phase7_oof_pooling import Phase7OofRecord


def test_fold_summary_statistics_use_arithmetic_mean_and_sample_std() -> None:
    """Across-fold summary statistics use arithmetic mean and ddof=1 sample std."""
    values = [0.5, 0.7, 0.9, 0.6, 0.8]

    assert arithmetic_mean(values) == pytest.approx(0.7)
    assert sample_standard_deviation(values) == pytest.approx(0.158113883008419)


def test_pooled_per_class_metrics_use_pooled_counts_and_zero_denominator() -> None:
    """Per-class Precision/Recall/F1 are computed once from pooled counts."""
    records = (
        build_record(true_class_index=0, predicted_class_index=0),
        build_record(true_class_index=0, predicted_class_index=1),
        build_record(true_class_index=1, predicted_class_index=1),
        build_record(true_class_index=1, predicted_class_index=2),
    )

    metrics = compute_pooled_per_class_metrics(records)

    assert metrics[0].precision == pytest.approx(1.0)
    assert metrics[0].recall == pytest.approx(0.5)
    assert metrics[0].f1 == pytest.approx(2 / 3)
    assert metrics[1].precision == pytest.approx(0.5)
    assert metrics[1].recall == pytest.approx(0.5)
    assert metrics[1].f1 == pytest.approx(0.5)
    assert metrics[3].precision == 0.0
    assert metrics[3].recall == 0.0
    assert metrics[3].f1 == 0.0


def test_confusion_matrix_uses_fixed_class_order() -> None:
    """Pooled confusion matrix uses rows=true and columns=predicted in class order 0..4."""
    records = (
        build_record(true_class_index=0, predicted_class_index=0),
        build_record(true_class_index=0, predicted_class_index=1),
        build_record(true_class_index=3, predicted_class_index=4),
    )

    matrix = compute_confusion_matrix(records)

    assert matrix == [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]


def test_binary_auroc_uses_rank_average_for_ties() -> None:
    """Binary AUROC is rank based and handles tied scores by average rank."""
    assert binary_auroc(labels=[1, 1, 0, 0], scores=[0.9, 0.8, 0.7, 0.1]) == pytest.approx(1.0)
    assert binary_auroc(labels=[1, 0], scores=[0.5, 0.5]) == pytest.approx(0.5)


def test_binary_auroc_returns_none_when_undefined() -> None:
    """AUROC remains undefined when positives or negatives are absent."""
    assert binary_auroc(labels=[1, 1], scores=[0.8, 0.9]) is None
    assert binary_auroc(labels=[0, 0], scores=[0.1, 0.2]) is None


def test_pooled_ovr_auroc_macro_averages_defined_classes_only() -> None:
    """Macro OvR AUROC is the unweighted mean over mathematically defined classes."""
    records = (
        build_record(
            true_class_index=0,
            predicted_class_index=0,
            probabilities=(0.9, 0.1, 0, 0, 0),
        ),
        build_record(
            true_class_index=1,
            predicted_class_index=1,
            probabilities=(0.2, 0.8, 0, 0, 0),
        ),
        build_record(
            true_class_index=1,
            predicted_class_index=1,
            probabilities=(0.1, 0.7, 0, 0, 0),
        ),
    )

    metrics = compute_pooled_ovr_auroc(records)

    assert metrics.per_class[0][2] == pytest.approx(1.0)
    assert metrics.per_class[1][2] == pytest.approx(1.0)
    assert metrics.per_class[2][2] is None
    assert metrics.macro_average == pytest.approx(1.0)


def build_record(
    *,
    true_class_index: int,
    predicted_class_index: int,
    probabilities: tuple[float, ...] | None = None,
) -> Phase7OofRecord:
    """Build one in-memory OOF record for metric math tests."""
    default_probabilities = [0.0 for _ in range(5)]
    default_probabilities[predicted_class_index] = 1.0
    return Phase7OofRecord(
        fold_index=0,
        patient_id="patient001",
        true_class_index=true_class_index,
        predicted_class_index=predicted_class_index,
        logits=(0.0, 0.0, 0.0, 0.0, 0.0),
        probabilities=probabilities if probabilities is not None else tuple(default_probabilities),
    )
