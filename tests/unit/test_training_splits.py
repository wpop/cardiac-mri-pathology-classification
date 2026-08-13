"""Unit tests for Phase 6 inner validation splitting."""

from dataclasses import dataclass

import pytest

from cardiac_pathology.training.splits import (
    InnerValidationSplit,
    create_inner_validation_split,
    validate_inner_split,
)


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Label-only record used to test patient-level split mechanics."""

    patient_id: str
    class_index: int


def build_split_records() -> tuple[SplitRecord, ...]:
    """Build deterministic label-only records for split unit tests.

    Returns:
        Records with ACDC-style identifiers and class labels, but no medical
        images, tensors, preprocessing outputs, or clinical measurements.
    """
    records: list[SplitRecord] = []
    for class_index in range(5):
        for class_offset in range(4):
            patient_number = (class_index * 10) + class_offset + 1
            records.append(
                SplitRecord(
                    patient_id=f"patient{patient_number:03d}",
                    class_index=class_index,
                )
            )
    return tuple(records)


def test_inner_validation_split_is_deterministic_and_stratified() -> None:
    """The inner split is deterministic and reserves one validation case per class."""
    records = build_split_records()
    outer_test_ids = ("patient091", "patient092")

    first = create_inner_validation_split(
        development_patients=records,
        outer_test_patient_ids=outer_test_ids,
        validation_fraction=0.25,
        random_seed=42,
    )
    second = create_inner_validation_split(
        development_patients=records,
        outer_test_patient_ids=outer_test_ids,
        validation_fraction=0.25,
        random_seed=42,
    )

    assert first == second
    assert len(first.validation_patient_ids) == 5
    assert len(first.train_patient_ids) == 15


def test_inner_validation_split_prevents_train_validation_and_outer_test_leakage() -> None:
    """Inner train and validation IDs remain disjoint from each other and outer test IDs."""
    records = build_split_records()
    split = create_inner_validation_split(
        development_patients=records,
        outer_test_patient_ids=("patient091", "patient092"),
        validation_fraction=0.25,
        random_seed=42,
    )

    train_ids = set(split.train_patient_ids)
    validation_ids = set(split.validation_patient_ids)
    outer_test_ids = set(split.outer_test_patient_ids)

    assert not train_ids & validation_ids
    assert not train_ids & outer_test_ids
    assert not validation_ids & outer_test_ids


def test_inner_validation_split_rejects_outer_test_overlap() -> None:
    """The inner split rejects development records that contain outer test IDs."""
    records = build_split_records()

    with pytest.raises(ValueError, match="outer test"):
        create_inner_validation_split(
            development_patients=records,
            outer_test_patient_ids=("patient001",),
            validation_fraction=0.25,
            random_seed=42,
        )


def test_validate_inner_split_rejects_existing_leakage() -> None:
    """Existing split validation rejects train-validation leakage."""
    split = InnerValidationSplit(
        train_patient_ids=("patient001", "patient002"),
        validation_patient_ids=("patient002",),
        outer_test_patient_ids=("patient003",),
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_inner_split(split)
