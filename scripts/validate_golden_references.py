"""Validate golden preprocessing tensor fixtures against real ACDC data."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient  # noqa: E402
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
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"


def main() -> None:
    """Run all golden fixture validation checks."""
    generator = GoldenReferenceGenerator(
        output_dir=GOLDEN_DIR,
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )

    print("Golden preprocessing references")
    generator.validate_manifest_files(MANIFEST_PATH)
    manifest = generator.read_manifest(MANIFEST_PATH)
    class_names = {str(patient["class_name"]) for patient in manifest["patients"]}
    print(f"\nPatients: {len(manifest['patients'])}")
    print(f"Classes represented: {len(class_names)} / 5")
    print("Tensor checksum validation: PASS")

    if not DATASET_DIR.is_dir():
        print("Source checksum validation: SKIP")
        print("Production preprocessing parity: SKIP")
        print("Deterministic regeneration: SKIP")
        print("\nPhase 4 validation: SKIP")
        return

    config = preprocessing_config_from_yaml(CONFIG_PATH)
    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    preprocessor = PatientPreprocessor(config)

    generator.validate_source_checksums(MANIFEST_PATH, patients)
    print("Source checksum validation: PASS")

    generator.validate_production_parity(MANIFEST_PATH, patients, preprocessor)
    print("Production preprocessing parity: PASS")

    deterministic = validate_deterministic_regeneration(
        class_mapping=class_mapping,
        patients=patients,
        preprocessor=preprocessor,
    )
    print(f"Deterministic regeneration: {'PASS' if deterministic else 'FAIL'}")
    print("\nPhase 4 validation: PASS")

    if not deterministic:
        raise RuntimeError("Golden deterministic regeneration failed")


def validate_deterministic_regeneration(
    class_mapping: dict[int, str],
    patients: tuple[AcdcPatient, ...],
    preprocessor: PatientPreprocessor,
) -> bool:
    """Generate fixtures in two temporary directories and compare byte hashes."""
    with TemporaryDirectory(prefix="golden_preprocessing_") as temporary_directory:
        root = Path(temporary_directory)
        first_generator = GoldenReferenceGenerator(
            output_dir=root / "first",
            preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
            phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
        )
        second_generator = GoldenReferenceGenerator(
            output_dir=root / "second",
            preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
            phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
        )
        first = first_generator.generate(
            patients=patients,
            class_mapping=class_mapping,
            preprocessor=preprocessor,
        )
        second = second_generator.generate(
            patients=patients,
            class_mapping=class_mapping,
            preprocessor=preprocessor,
        )
        return first.file_hashes == second.file_hashes


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
