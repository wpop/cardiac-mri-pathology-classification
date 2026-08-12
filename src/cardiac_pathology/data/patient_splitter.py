"""Patient-level stratified splitting utilities."""

from collections import Counter
from collections.abc import Sequence

from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

from cardiac_pathology.data.acdc_patient import AcdcPatient
from cardiac_pathology.data.patient_fold import PatientFold


class PatientSplitter:
    """Create deterministic stratified outer folds at patient level."""

    def __init__(self, num_folds: int, random_seed: int) -> None:
        if num_folds < 2:
            raise ValueError("num_folds must be >= 2")
        self.num_folds = num_folds
        self.random_seed = random_seed

    def split(self, patients: Sequence[AcdcPatient]) -> tuple[PatientFold, ...]:
        """Split patients into deterministic stratified outer folds."""
        if not patients:
            raise ValueError("Cannot split an empty patient collection")

        ordered_patients = tuple(sorted(patients, key=self._patient_sort_key))
        patient_ids = [patient.patient_id for patient in ordered_patients]
        class_indices = [patient.class_index for patient in ordered_patients]

        class_counts = Counter(class_indices)
        if min(class_counts.values()) < self.num_folds:
            raise ValueError(
                "Each class must have at least num_folds patients for stratified splitting"
            )

        splitter = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        folds: list[PatientFold] = []
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(patient_ids, class_indices)
        ):
            train_ids = tuple(
                sorted(
                    (patient_ids[index] for index in train_indices),
                    key=self._patient_id_sort_key,
                )
            )
            test_ids = tuple(
                sorted(
                    (patient_ids[index] for index in test_indices),
                    key=self._patient_id_sort_key,
                )
            )
            folds.append(
                PatientFold(
                    fold_index=fold_index,
                    train_patient_ids=train_ids,
                    test_patient_ids=test_ids,
                )
            )

        self.validate_folds(folds=tuple(folds), all_patient_ids=tuple(patient_ids))
        return tuple(folds)

    def validate_folds(
        self,
        folds: tuple[PatientFold, ...],
        all_patient_ids: tuple[str, ...],
    ) -> None:
        """Validate hard leakage and reproducibility invariants for outer folds."""
        if len(folds) != self.num_folds:
            raise ValueError(f"Expected {self.num_folds} folds, got {len(folds)}")

        all_patient_set = set(all_patient_ids)
        outer_test_ids: list[str] = []
        for fold in folds:
            train_ids = set(fold.train_patient_ids)
            test_ids = set(fold.test_patient_ids)
            if train_ids & test_ids:
                raise ValueError(f"Fold {fold.fold_index}: train/test overlap")
            if train_ids | test_ids != all_patient_set:
                raise ValueError(f"Fold {fold.fold_index}: train/test union mismatch")
            outer_test_ids.extend(fold.test_patient_ids)

        if len(outer_test_ids) != len(all_patient_ids):
            raise ValueError("Outer test folds do not cover each patient exactly once")
        if set(outer_test_ids) != all_patient_set:
            raise ValueError("Outer test patient set differs from indexed patient set")
        duplicate_test_ids = [
            patient_id for patient_id, count in Counter(outer_test_ids).items() if count != 1
        ]
        if duplicate_test_ids:
            raise ValueError(f"Patient appears in multiple outer test folds: {duplicate_test_ids}")

    def _patient_sort_key(self, patient: AcdcPatient) -> int:
        """Sort patients by numeric patient identifier."""
        return self._patient_id_sort_key(patient.patient_id)

    def _patient_id_sort_key(self, patient_id: str) -> int:
        """Sort patient IDs such as patient001 by numeric suffix."""
        return int(patient_id.removeprefix("patient"))
