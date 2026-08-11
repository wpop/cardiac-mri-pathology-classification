"""Build deterministic patient-level ACDC outer fold split artifact."""

import json
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
OUTPUT_PATH = REPOSITORY_ROOT / "artifacts/dataset_splits/acdc_5fold_seed42.json"


def main() -> None:
    """Build and save deterministic ACDC 5-fold patient splits."""
    config = load_yaml(CONFIG_PATH)
    seed = int(config["project"]["seed"])
    num_folds = int(config["training"]["num_folds"])

    indexer = AcdcDatasetIndexer(
        dataset_dir=DATASET_DIR,
        class_mapping_path=CLASS_MAPPING_PATH,
    )
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    splitter = PatientSplitter(num_folds=num_folds, random_seed=seed)
    folds = splitter.split(patients)

    artifact = build_split_artifact(
        patients=patients,
        folds=folds,
        class_mapping=class_mapping,
        seed=seed,
        num_folds=num_folds,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT_PATH}")


def build_split_artifact(
    patients: tuple[AcdcPatient, ...],
    folds: tuple[PatientFold, ...],
    class_mapping: dict[int, str],
    seed: int,
    num_folds: int,
) -> dict[str, Any]:
    """Build JSON-serializable deterministic split artifact."""
    patient_by_id = {patient.patient_id: patient for patient in patients}
    class_mapping_json = {str(index): name for index, name in class_mapping.items()}
    fold_records = []
    for fold in folds:
        fold_records.append(
            {
                "fold_index": fold.fold_index,
                "train_patient_ids": list(fold.train_patient_ids),
                "test_patient_ids": list(fold.test_patient_ids),
                "train_class_counts": class_counts(
                    fold.train_patient_ids,
                    patient_by_id,
                    class_mapping,
                ),
                "test_class_counts": class_counts(
                    fold.test_patient_ids,
                    patient_by_id,
                    class_mapping,
                ),
            }
        )

    return {
        "class_mapping": class_mapping_json,
        "dataset": "ACDC",
        "folds": fold_records,
        "num_folds": num_folds,
        "num_patients": len(patients),
        "seed": seed,
    }


def class_counts(
    patient_ids: tuple[str, ...],
    patient_by_id: dict[str, AcdcPatient],
    class_mapping: dict[int, str],
) -> dict[str, int]:
    """Count classes in mapping order for selected patient IDs."""
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
