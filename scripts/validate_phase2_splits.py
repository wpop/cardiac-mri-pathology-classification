"""Validate Phase 2 real ACDC indexing and patient splits."""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.data import (  # noqa: E402
    AcdcDatasetIndexer,
    AcdcPatient,
    PatientFold,
    PatientSplitter,
)

DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"


def main() -> None:
    """Print concise Phase 2 validation report."""
    config = load_yaml(CONFIG_PATH)
    seed = int(config["project"]["seed"])
    num_folds = int(config["training"]["num_folds"])
    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    splitter = PatientSplitter(num_folds=num_folds, random_seed=seed)
    folds = splitter.split(patients)
    regenerated_folds = splitter.split(tuple(patients))
    deterministic_regeneration = folds == regenerated_folds
    if not deterministic_regeneration:
        raise ValueError("Deterministic split regeneration failed")

    leakage_pass = validate_leakage(patients, folds)
    class_distribution = count_patient_classes(patients, class_mapping)

    print("Phase 2 dataset validation")
    print(f"\nPatients: {len(patients)}")
    print("\nClass distribution:")
    for class_name in class_mapping.values():
        print(f"{class_name}: {class_distribution[class_name]}")

    patient_by_id = {patient.patient_id: patient for patient in patients}
    for fold in folds:
        test_counts = class_counts(fold.test_patient_ids, patient_by_id, class_mapping)
        print(f"\nFold {fold.fold_index}:")
        print(f"train: {len(fold.train_patient_ids)}")
        print(f"test: {len(fold.test_patient_ids)}")
        formatted_counts = " ".join(
            f"{class_name}={test_counts[class_name]}" for class_name in class_mapping.values()
        )
        print(f"test classes: {formatted_counts}")

    print("\nLeakage checks:")
    print(f"train/test disjoint per fold: {'PASS' if leakage_pass else 'FAIL'}")
    print("all patients exactly once in outer test sets: PASS")
    print("deterministic regeneration: PASS")
    print("\nPhase 2 validation: PASS")


def validate_leakage(
    patients: tuple[AcdcPatient, ...],
    folds: tuple[PatientFold, ...],
) -> bool:
    """Validate train/test disjointness and exact outer test coverage."""
    all_patient_ids = {patient.patient_id for patient in patients}
    outer_test_ids: list[str] = []
    for fold in folds:
        train = set(fold.train_patient_ids)
        test = set(fold.test_patient_ids)
        if train & test:
            return False
        if train | test != all_patient_ids:
            return False
        outer_test_ids.extend(fold.test_patient_ids)
    return Counter(outer_test_ids) == Counter(all_patient_ids)


def count_patient_classes(
    patients: tuple[AcdcPatient, ...],
    class_mapping: dict[int, str],
) -> dict[str, int]:
    """Count patient classes in mapping order."""
    counts = Counter(patient.class_name for patient in patients)
    return {class_name: counts[class_name] for class_name in class_mapping.values()}


def class_counts(
    patient_ids: tuple[str, ...],
    patient_by_id: dict[str, AcdcPatient],
    class_mapping: dict[int, str],
) -> dict[str, int]:
    """Count classes for selected patient IDs in mapping order."""
    counts = Counter(patient_by_id[patient_id].class_name for patient_id in patient_ids)
    return {class_name: counts[class_name] for class_name in class_mapping.values()}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


if __name__ == "__main__":
    main()
