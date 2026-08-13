"""Deterministic inner validation splitting for Phase 6 training."""

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cardiac_pathology.data.patient_fold import PatientFold


class LabeledPatientRecord(Protocol):
    """Patient-level record carrying the fields needed for stratified splitting."""

    @property
    def patient_id(self) -> str:
        """Return the patient identifier."""
        ...

    @property
    def class_index(self) -> int:
        """Return the diagnostic class index."""
        ...


@dataclass(frozen=True, slots=True)
class InnerValidationSplit:
    """Patient-level train/validation split inside one outer development fold."""

    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    outer_test_patient_ids: tuple[str, ...]


def create_inner_validation_split(
    development_patients: Sequence[LabeledPatientRecord],
    outer_test_patient_ids: Sequence[str],
    validation_fraction: float,
    random_seed: int,
) -> InnerValidationSplit:
    """Create a deterministic stratified validation split within development patients.

    Args:
        development_patients: Patients available for model fitting in one outer
            fold. These are the outer training patients and must not include any
            outer test patients.
        outer_test_patient_ids: Patient IDs reserved for final outer-fold testing.
        validation_fraction: Fraction of each class to reserve for validation.
            The value must be in the open interval ``(0, 1)``.
        random_seed: Seed used to shuffle patients within each class.

    Returns:
        Inner train/validation split with sorted patient ID tuples.
    """
    if not development_patients:
        raise ValueError("development_patients must be non-empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")

    outer_test_set = set(outer_test_patient_ids)
    development_ids = [patient.patient_id for patient in development_patients]
    if len(set(development_ids)) != len(development_ids):
        raise ValueError("development_patients contain duplicate patient IDs")
    if set(development_ids) & outer_test_set:
        raise ValueError("development patients overlap the outer test set")

    patient_ids_by_class: defaultdict[int, list[str]] = defaultdict(list)
    for patient in development_patients:
        patient_ids_by_class[patient.class_index].append(patient.patient_id)

    rng = random.Random(random_seed)
    validation_ids: list[str] = []
    for class_index in sorted(patient_ids_by_class):
        class_patient_ids = sorted(patient_ids_by_class[class_index], key=patient_id_sort_key)
        if len(class_patient_ids) < 2:
            raise ValueError("Each class must have at least two development patients")
        rng.shuffle(class_patient_ids)
        validation_count = round(len(class_patient_ids) * validation_fraction)
        validation_count = max(1, min(len(class_patient_ids) - 1, validation_count))
        validation_ids.extend(class_patient_ids[:validation_count])

    validation_set = set(validation_ids)
    train_ids = sorted(
        (patient_id for patient_id in development_ids if patient_id not in validation_set),
        key=patient_id_sort_key,
    )
    validation_ids = sorted(validation_ids, key=patient_id_sort_key)

    split = InnerValidationSplit(
        train_patient_ids=tuple(train_ids),
        validation_patient_ids=tuple(validation_ids),
        outer_test_patient_ids=tuple(sorted(outer_test_set, key=patient_id_sort_key)),
    )
    validate_inner_split(split)
    return split


def validate_inner_split(split: InnerValidationSplit) -> None:
    """Validate train/validation disjointness and outer-test isolation.

    Args:
        split: Inner validation split to validate.

    Returns:
        None.
    """
    train_ids = set(split.train_patient_ids)
    validation_ids = set(split.validation_patient_ids)
    outer_test_ids = set(split.outer_test_patient_ids)
    if train_ids & validation_ids:
        raise ValueError("Inner train and validation patient IDs overlap")
    if train_ids & outer_test_ids:
        raise ValueError("Inner train patient IDs overlap the outer test set")
    if validation_ids & outer_test_ids:
        raise ValueError("Inner validation patient IDs overlap the outer test set")


def load_outer_folds(path: Path) -> tuple[PatientFold, ...]:
    """Load a deterministic outer-fold artifact produced by Phase 2.

    Args:
        path: JSON artifact path containing a ``folds`` list with patient-level
            train and test ID fields.

    Returns:
        Tuple of ``PatientFold`` objects sorted by fold index.
    """
    with path.open("r", encoding="utf-8") as file:
        raw_artifact = json.load(file)
    artifact = require_mapping(raw_artifact, path)
    raw_folds = artifact.get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError(f"Expected folds list in {path}")

    folds = tuple(
        patient_fold_from_mapping(require_mapping(raw_fold, path)) for raw_fold in raw_folds
    )
    return tuple(sorted(folds, key=lambda fold: fold.fold_index))


def patient_fold_from_mapping(raw_fold: Mapping[str, object]) -> PatientFold:
    """Build a ``PatientFold`` from one parsed JSON mapping.

    Args:
        raw_fold: Parsed fold mapping from the Phase 2 split artifact.

    Returns:
        Validated patient fold.
    """
    fold_index = raw_fold.get("fold_index")
    train_ids = raw_fold.get("train_patient_ids")
    test_ids = raw_fold.get("test_patient_ids")
    if not isinstance(fold_index, int):
        raise ValueError("fold_index must be an integer")
    if not isinstance(train_ids, list) or not all(isinstance(value, str) for value in train_ids):
        raise ValueError("train_patient_ids must be a list of strings")
    if not isinstance(test_ids, list) or not all(isinstance(value, str) for value in test_ids):
        raise ValueError("test_patient_ids must be a list of strings")
    return PatientFold(
        fold_index=fold_index,
        train_patient_ids=tuple(sorted(train_ids, key=patient_id_sort_key)),
        test_patient_ids=tuple(sorted(test_ids, key=patient_id_sort_key)),
    )


def require_mapping(value: object, path: Path) -> Mapping[str, object]:
    """Return a parsed JSON value as a string-keyed mapping.

    Args:
        value: Parsed JSON object.
        path: Source path used for error reporting.

    Returns:
        Mapping with string keys and object values.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def patient_id_sort_key(patient_id: str) -> int:
    """Return the numeric sort key for ACDC-style patient identifiers.

    Args:
        patient_id: Patient identifier such as ``patient001``.

    Returns:
        Numeric patient suffix.
    """
    return int(patient_id.removeprefix("patient"))
