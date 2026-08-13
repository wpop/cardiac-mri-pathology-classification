"""Build reproducible Phase 7 inner-validation split manifests."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.data.patient_fold import PatientFold
from cardiac_pathology.training.splits import (
    InnerValidationSplit,
    create_inner_validation_split,
    load_outer_folds,
)

EXPECTED_NUM_FOLDS = 5
EXPECTED_NUM_PATIENTS = 100
EXPECTED_OUTER_DEVELOPMENT_SIZE = 80
EXPECTED_OUTER_TEST_SIZE = 20
EXPECTED_INNER_TRAIN_SIZE = 60
EXPECTED_INNER_VALIDATION_SIZE = 20
EXPECTED_OUTER_DEVELOPMENT_PER_CLASS = 16
EXPECTED_OUTER_TEST_PER_CLASS = 4
EXPECTED_INNER_TRAIN_PER_CLASS = 12
EXPECTED_INNER_VALIDATION_PER_CLASS = 4


class Phase7SplitManifestBuilder:
    """Build and persist deterministic Phase 7 patient split metadata."""

    def __init__(
        self,
        dataset_dir: Path,
        class_mapping_path: Path,
        outer_split_path: Path,
        outer_split_reference: str,
        output_path: Path,
        seed: int,
        validation_fraction: float,
    ) -> None:
        """Initialize the Phase 7 split manifest builder.

        Args:
            dataset_dir: Real labeled ACDC training dataset directory.
            class_mapping_path: Authoritative class mapping JSON path.
            outer_split_path: Existing deterministic Phase 2 outer-fold artifact.
            outer_split_reference: Repository-relative outer split artifact name
                stored in the generated manifest.
            output_path: Destination JSON path for persisted Phase 7 inner splits.
            seed: Deterministic inner-split random seed.
            validation_fraction: Fraction of each outer-development class reserved
                for inner validation.
        """
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        self._dataset_dir = dataset_dir
        self._class_mapping_path = class_mapping_path
        self._outer_split_path = outer_split_path
        self._outer_split_reference = outer_split_reference
        self._output_path = output_path
        self._seed = seed
        self._validation_fraction = validation_fraction

    def build_and_save(self) -> Path:
        """Build the deterministic manifest and save it as formatted JSON.

        Returns:
            Path of the generated Phase 7 split manifest.
        """
        artifact = self.build_artifact()

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self._output_path

    def build_artifact(self) -> dict[str, Any]:
        """Build deterministic inner splits for all existing outer folds.

        Returns:
            JSON-serializable Phase 7 split manifest.
        """
        indexer = AcdcDatasetIndexer(
            dataset_dir=self._dataset_dir,
            class_mapping_path=self._class_mapping_path,
        )
        patients = indexer.index_patients()
        patient_by_id = {patient.patient_id: patient for patient in patients}
        class_mapping = indexer.load_class_mapping()
        outer_folds = load_outer_folds(self._outer_split_path)
        self._validate_cohort_and_outer_folds(
            patients=patients,
            patient_by_id=patient_by_id,
            outer_folds=outer_folds,
        )

        fold_records: list[dict[str, Any]] = []
        for fold in outer_folds:
            fold_records.append(
                self._build_fold_record(
                    fold=fold,
                    patient_by_id=patient_by_id,
                    class_mapping=class_mapping,
                )
            )
        self._validate_pooled_outer_test_coverage(
            patients=patients,
            outer_folds=outer_folds,
        )

        return {
            "class_mapping": {
                str(index): class_name for index, class_name in class_mapping.items()
            },
            "dataset": "ACDC",
            "folds": fold_records,
            "num_folds": len(outer_folds),
            "num_patients": len(patients),
            "outer_split_artifact": self._outer_split_reference,
            "seed": self._seed,
            "validation_fraction": self._validation_fraction,
        }

    def _build_fold_record(
        self,
        fold: PatientFold,
        patient_by_id: Mapping[str, AcdcPatient],
        class_mapping: Mapping[int, str],
    ) -> dict[str, Any]:
        """Build one fold manifest record from the existing outer split."""
        development_patients = self._patients_for_ids(
            patient_ids=fold.train_patient_ids,
            patient_by_id=patient_by_id,
        )

        inner_split = create_inner_validation_split(
            development_patients=development_patients,
            outer_test_patient_ids=fold.test_patient_ids,
            validation_fraction=self._validation_fraction,
            random_seed=self._seed,
        )

        self._validate_fold_invariants(
            fold=fold,
            inner_split=inner_split,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
        )

        return {
            "fold_index": fold.fold_index,
            "outer_development_patient_ids": list(fold.train_patient_ids),
            "outer_test_patient_ids": list(fold.test_patient_ids),
            "inner_train_patient_ids": list(inner_split.train_patient_ids),
            "inner_validation_patient_ids": list(inner_split.validation_patient_ids),
            "outer_development_class_counts": self._class_counts(
                patient_ids=fold.train_patient_ids,
                patient_by_id=patient_by_id,
                class_mapping=class_mapping,
            ),
            "outer_test_class_counts": self._class_counts(
                patient_ids=fold.test_patient_ids,
                patient_by_id=patient_by_id,
                class_mapping=class_mapping,
            ),
            "inner_train_class_counts": self._class_counts(
                patient_ids=inner_split.train_patient_ids,
                patient_by_id=patient_by_id,
                class_mapping=class_mapping,
            ),
            "inner_validation_class_counts": self._class_counts(
                patient_ids=inner_split.validation_patient_ids,
                patient_by_id=patient_by_id,
                class_mapping=class_mapping,
            ),
        }

    def _patients_for_ids(
        self,
        patient_ids: Sequence[str],
        patient_by_id: Mapping[str, AcdcPatient],
    ) -> tuple[AcdcPatient, ...]:
        """Resolve ordered patient records from persisted identifiers."""
        missing_ids = [patient_id for patient_id in patient_ids if patient_id not in patient_by_id]
        if missing_ids:
            raise ValueError(f"Unknown patient IDs: {missing_ids}")

        return tuple(patient_by_id[patient_id] for patient_id in patient_ids)

    def _class_counts(
        self,
        patient_ids: Sequence[str],
        patient_by_id: Mapping[str, AcdcPatient],
        class_mapping: Mapping[int, str],
    ) -> dict[str, int]:
        """Count diagnostic classes for selected patient identifiers."""
        counts = Counter(patient_by_id[patient_id].class_index for patient_id in patient_ids)
        return {
            class_name: counts[class_index] for class_index, class_name in class_mapping.items()
        }

    def _validate_cohort_and_outer_folds(
        self,
        patients: Sequence[AcdcPatient],
        patient_by_id: Mapping[str, AcdcPatient],
        outer_folds: Sequence[PatientFold],
    ) -> None:
        """Validate whole-cohort and outer-fold Phase 7 contract invariants."""
        patient_ids = tuple(patient.patient_id for patient in patients)
        self._validate_unique_ids(
            partition_name="indexed cohort",
            patient_ids=patient_ids,
        )
        self._validate_size(
            partition_name="indexed cohort",
            actual=len(patient_ids),
            expected=EXPECTED_NUM_PATIENTS,
        )
        self._validate_size(
            partition_name="outer folds",
            actual=len(outer_folds),
            expected=EXPECTED_NUM_FOLDS,
        )

        if set(patient_by_id) != set(patient_ids):
            raise ValueError("Indexed cohort patient lookup does not match indexed patient IDs")

    def _validate_fold_invariants(
        self,
        fold: PatientFold,
        inner_split: InnerValidationSplit,
        patient_by_id: Mapping[str, AcdcPatient],
        class_mapping: Mapping[int, str],
    ) -> None:
        """Validate one Phase 7 outer fold and its generated inner split."""
        fold_label = f"fold {fold.fold_index}"
        outer_development_ids = tuple(fold.train_patient_ids)
        outer_test_ids = tuple(fold.test_patient_ids)
        inner_train_ids = tuple(inner_split.train_patient_ids)
        inner_validation_ids = tuple(inner_split.validation_patient_ids)

        for partition_name, patient_ids in [
            (f"{fold_label} outer-development", outer_development_ids),
            (f"{fold_label} outer-test", outer_test_ids),
            (f"{fold_label} inner-train", inner_train_ids),
            (f"{fold_label} inner-validation", inner_validation_ids),
        ]:
            self._validate_unique_ids(
                partition_name=partition_name,
                patient_ids=patient_ids,
            )
            self._validate_known_ids(
                partition_name=partition_name,
                patient_ids=patient_ids,
                patient_by_id=patient_by_id,
            )

        outer_development_set = set(outer_development_ids)
        outer_test_set = set(outer_test_ids)
        inner_train_set = set(inner_train_ids)
        inner_validation_set = set(inner_validation_ids)

        self._validate_disjoint(
            left_name=f"{fold_label} outer-development",
            left_ids=outer_development_set,
            right_name="outer-test",
            right_ids=outer_test_set,
        )
        self._validate_disjoint(
            left_name=f"{fold_label} inner-train",
            left_ids=inner_train_set,
            right_name="inner-validation",
            right_ids=inner_validation_set,
        )
        self._validate_disjoint(
            left_name=f"{fold_label} inner-train",
            left_ids=inner_train_set,
            right_name="outer-test",
            right_ids=outer_test_set,
        )
        self._validate_disjoint(
            left_name=f"{fold_label} inner-validation",
            left_ids=inner_validation_set,
            right_name="outer-test",
            right_ids=outer_test_set,
        )

        if inner_train_set | inner_validation_set != outer_development_set:
            raise ValueError(
                f"{fold_label}: inner train and validation do not exactly cover outer-development"
            )

        self._validate_size(
            partition_name=f"{fold_label} outer-development",
            actual=len(outer_development_ids),
            expected=EXPECTED_OUTER_DEVELOPMENT_SIZE,
        )
        self._validate_size(
            partition_name=f"{fold_label} outer-test",
            actual=len(outer_test_ids),
            expected=EXPECTED_OUTER_TEST_SIZE,
        )
        self._validate_size(
            partition_name=f"{fold_label} inner-train",
            actual=len(inner_train_ids),
            expected=EXPECTED_INNER_TRAIN_SIZE,
        )
        self._validate_size(
            partition_name=f"{fold_label} inner-validation",
            actual=len(inner_validation_ids),
            expected=EXPECTED_INNER_VALIDATION_SIZE,
        )

        self._validate_class_counts(
            partition_name=f"{fold_label} outer-development",
            patient_ids=outer_development_ids,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
            expected_per_class=EXPECTED_OUTER_DEVELOPMENT_PER_CLASS,
        )
        self._validate_class_counts(
            partition_name=f"{fold_label} outer-test",
            patient_ids=outer_test_ids,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
            expected_per_class=EXPECTED_OUTER_TEST_PER_CLASS,
        )
        self._validate_class_counts(
            partition_name=f"{fold_label} inner-train",
            patient_ids=inner_train_ids,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
            expected_per_class=EXPECTED_INNER_TRAIN_PER_CLASS,
        )
        self._validate_class_counts(
            partition_name=f"{fold_label} inner-validation",
            patient_ids=inner_validation_ids,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
            expected_per_class=EXPECTED_INNER_VALIDATION_PER_CLASS,
        )

    def _validate_pooled_outer_test_coverage(
        self,
        patients: Sequence[AcdcPatient],
        outer_folds: Sequence[PatientFold],
    ) -> None:
        """Validate that pooled outer-test IDs cover the indexed cohort exactly once."""
        indexed_patient_ids = tuple(patient.patient_id for patient in patients)
        pooled_outer_test_ids = [
            patient_id for fold in outer_folds for patient_id in fold.test_patient_ids
        ]
        self._validate_size(
            partition_name="pooled outer-test patient IDs",
            actual=len(pooled_outer_test_ids),
            expected=EXPECTED_NUM_PATIENTS,
        )
        self._validate_unique_ids(
            partition_name="pooled outer-test patient IDs",
            patient_ids=pooled_outer_test_ids,
        )
        if set(pooled_outer_test_ids) != set(indexed_patient_ids):
            raise ValueError("Pooled outer-test patient IDs do not cover indexed cohort exactly")

    def _validate_class_counts(
        self,
        partition_name: str,
        patient_ids: Sequence[str],
        patient_by_id: Mapping[str, AcdcPatient],
        class_mapping: Mapping[int, str],
        expected_per_class: int,
    ) -> None:
        """Validate that every authoritative class has the expected partition count."""
        actual_counts = self._class_counts(
            patient_ids=patient_ids,
            patient_by_id=patient_by_id,
            class_mapping=class_mapping,
        )
        expected_counts = {class_name: expected_per_class for class_name in class_mapping.values()}
        if actual_counts != expected_counts:
            raise ValueError(
                f"{partition_name} class counts mismatch: expected {expected_counts}, "
                f"got {actual_counts}"
            )

    def _validate_known_ids(
        self,
        partition_name: str,
        patient_ids: Sequence[str],
        patient_by_id: Mapping[str, AcdcPatient],
    ) -> None:
        """Validate that a partition only references indexed real cohort IDs."""
        missing_ids = [patient_id for patient_id in patient_ids if patient_id not in patient_by_id]
        if missing_ids:
            raise ValueError(f"{partition_name} contains unknown patient IDs: {missing_ids}")

    def _validate_unique_ids(self, partition_name: str, patient_ids: Sequence[str]) -> None:
        """Validate that a partition contains no duplicate patient IDs."""
        duplicate_ids = [
            patient_id for patient_id, count in Counter(patient_ids).items() if count > 1
        ]
        if duplicate_ids:
            raise ValueError(f"{partition_name} contains duplicate patient IDs: {duplicate_ids}")

    def _validate_disjoint(
        self,
        left_name: str,
        left_ids: set[str],
        right_name: str,
        right_ids: set[str],
    ) -> None:
        """Validate that two named patient ID sets are disjoint."""
        overlap = sorted(left_ids & right_ids)
        if overlap:
            raise ValueError(f"{left_name} overlaps {right_name}: {overlap}")

    def _validate_size(self, partition_name: str, actual: int, expected: int) -> None:
        """Validate a frozen Phase 7 partition size."""
        if actual != expected:
            raise ValueError(f"{partition_name} size mismatch: expected {expected}, got {actual}")
