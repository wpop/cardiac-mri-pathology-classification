"""Integration tests for real ACDC patient indexing and folds."""

from collections import Counter
from pathlib import Path

import pytest

from cardiac_pathology.data import AcdcDatasetIndexer, PatientSplitter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"


def test_real_acdc_dataset_indexing_and_splits() -> None:
    """Validate real ACDC indexing and leakage-safe 5-fold splits."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()

    assert len(patients) == 100
    assert patients[0].patient_id == "patient001"
    assert patients[-1].patient_id == "patient100"
    assert len({patient.patient_id for patient in patients}) == 100
    assert dict(Counter(patient.class_name for patient in patients)) == {
        class_name: 20 for class_name in class_mapping.values()
    }

    for patient in patients:
        assert patient.ed_path.is_file()
        assert patient.es_path.is_file()
        assert patient.cine_4d_path.is_file()
        assert patient.info_path.is_file()
        patient_paths = (
            patient.ed_path,
            patient.es_path,
            patient.cine_4d_path,
            patient.info_path,
        )
        assert not any(str(path).endswith("_gt.nii.gz") for path in patient_paths)

    folds = PatientSplitter(num_folds=5, random_seed=42).split(patients)
    assert len(folds) == 5

    all_patient_ids = {patient.patient_id for patient in patients}
    patient_by_id = {patient.patient_id: patient for patient in patients}
    outer_test_ids: list[str] = []
    for fold in folds:
        assert set(fold.train_patient_ids).isdisjoint(fold.test_patient_ids)
        assert set(fold.train_patient_ids) | set(fold.test_patient_ids) == all_patient_ids
        assert len(fold.train_patient_ids) == 80
        assert len(fold.test_patient_ids) == 20
        outer_test_ids.extend(fold.test_patient_ids)

        test_counts = Counter(
            patient_by_id[patient_id].class_name for patient_id in fold.test_patient_ids
        )
        train_counts = Counter(
            patient_by_id[patient_id].class_name for patient_id in fold.train_patient_ids
        )
        assert dict(test_counts) == {class_name: 4 for class_name in class_mapping.values()}
        assert dict(train_counts) == {class_name: 16 for class_name in class_mapping.values()}

    assert Counter(outer_test_ids) == Counter(all_patient_ids)
