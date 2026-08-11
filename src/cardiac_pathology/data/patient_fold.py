"""Immutable patient-level outer fold representation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatientFold:
    """One leakage-safe outer fold of patient IDs."""

    fold_index: int
    train_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fold identity and duplicate-free patient lists."""
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if len(set(self.train_patient_ids)) != len(self.train_patient_ids):
            raise ValueError(f"Fold {self.fold_index}: duplicate train patient ID")
        if len(set(self.test_patient_ids)) != len(self.test_patient_ids):
            raise ValueError(f"Fold {self.fold_index}: duplicate test patient ID")
        overlap = set(self.train_patient_ids) & set(self.test_patient_ids)
        if overlap:
            raise ValueError(f"Fold {self.fold_index}: train/test leakage: {sorted(overlap)}")
