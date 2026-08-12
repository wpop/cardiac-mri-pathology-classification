"""Generate deterministic golden preprocessing tensor fixtures from real ACDC data."""

import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.data import AcdcDatasetIndexer  # noqa: E402
from cardiac_pathology.preprocessing import (  # noqa: E402
    GoldenReferenceGenerator,
    PatientPreprocessor,
    PreprocessingConfig,
)

DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
PHASE3_VALIDATION_CSV = REPOSITORY_ROOT / "artifacts/preprocessing_validation/phase3_validation.csv"
PREPROCESSING_CONTRACT_PATH = REPOSITORY_ROOT / "docs/preprocessing_contract.md"
GOLDEN_DIR = REPOSITORY_ROOT / "tests/fixtures/golden"


def main() -> None:
    """Generate golden fixtures and verify deterministic regeneration."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    config = preprocessing_config_from_yaml(CONFIG_PATH)
    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    preprocessor = PatientPreprocessor(config)
    generator = GoldenReferenceGenerator(
        output_dir=GOLDEN_DIR,
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )

    first = generator.generate(patients, class_mapping, preprocessor)
    first_hashes = dict(first.file_hashes)
    second = generator.generate(patients, class_mapping, preprocessor)
    deterministic = first_hashes == second.file_hashes

    print("Golden preprocessing reference generation")
    print(f"\nManifest: {second.manifest_path}")
    print(f"Patients: {len(second.selected_patient_ids)}")
    print("Selected patients:")
    for patient_id in second.selected_patient_ids:
        print(f"  {patient_id}")
    print(f"\nGolden regeneration: {'PASS' if deterministic else 'FAIL'}")

    if not deterministic:
        raise RuntimeError("Golden regeneration was not byte-identical")


def preprocessing_config_from_yaml(path: Path) -> PreprocessingConfig:
    """Construct PreprocessingConfig from configs/default.yaml."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    preprocessing = config["preprocessing"]
    target_spacing = preprocessing["target_spacing_mm"]
    target_shape = preprocessing["target_shape"]
    intensity = preprocessing["intensity"]
    return PreprocessingConfig(
        target_orientation=preprocessing["orientation"],
        target_spacing_xyz=(
            float(target_spacing["x"]),
            float(target_spacing["y"]),
            float(target_spacing["z"]),
        ),
        target_shape_dhw=(
            int(target_shape["d"]),
            int(target_shape["h"]),
            int(target_shape["w"]),
        ),
        lower_percentile=float(intensity["lower_percentile"]),
        upper_percentile=float(intensity["upper_percentile"]),
        epsilon=float(intensity["epsilon"]),
        z_padding_value=float(preprocessing["padding"]["z_value"]),
    )


if __name__ == "__main__":
    main()
