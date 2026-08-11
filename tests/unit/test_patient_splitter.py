"""Tests for patient-level stratified splitting using real ACDC metadata."""

from collections import Counter
from pathlib import Path

import pytest

from cardiac_pathology.data import AcdcDatasetIndexer, PatientSplitter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"


def test_patient_splitter_real_acdc_metadata() -> None:
    """Splitter creates deterministic leakage-free folds on real indexed metadata."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    patients = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH).index_patients()
    folds = PatientSplitter(num_folds=5, random_seed=42).split(patients)
    regenerated = PatientSplitter(num_folds=5, random_seed=42).split(patients)

    assert folds == regenerated
    assert len(folds) == 5

    all_patient_ids = {patient.patient_id for patient in patients}
    outer_test_ids: list[str] = []
    patient_by_id = {patient.patient_id: patient for patient in patients}
    for fold in folds:
        assert set(fold.train_patient_ids).isdisjoint(fold.test_patient_ids)
        assert set(fold.train_patient_ids) | set(fold.test_patient_ids) == all_patient_ids
        outer_test_ids.extend(fold.test_patient_ids)
        assert len(fold.train_patient_ids) == 80
        assert len(fold.test_patient_ids) == 20

        test_counts = Counter(
            patient_by_id[patient_id].class_name for patient_id in fold.test_patient_ids
        )
        train_counts = Counter(
            patient_by_id[patient_id].class_name for patient_id in fold.train_patient_ids
        )
        assert set(test_counts.values()) == {4}
        assert set(train_counts.values()) == {16}

    assert Counter(outer_test_ids) == Counter(all_patient_ids)
