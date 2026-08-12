"""Integration tests for golden deterministic preprocessing fixtures."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from cardiac_pathology.data import AcdcDatasetIndexer
from cardiac_pathology.preprocessing import (
    GoldenReferenceGenerator,
    PatientPreprocessor,
    PreprocessingConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
PHASE3_VALIDATION_CSV = REPOSITORY_ROOT / "artifacts/preprocessing_validation/phase3_validation.csv"
PREPROCESSING_CONTRACT_PATH = REPOSITORY_ROOT / "docs/preprocessing_contract.md"
GOLDEN_DIR = REPOSITORY_ROOT / "tests/fixtures/golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"


def test_golden_manifest_integrity() -> None:
    """Golden manifest and tensor files are internally consistent."""
    generator = GoldenReferenceGenerator(
        output_dir=GOLDEN_DIR,
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )
    generator.validate_manifest_files(MANIFEST_PATH)

    manifest = generator.read_manifest(MANIFEST_PATH)
    assert manifest["dataset"] == "ACDC"
    assert manifest["fixture_version"] == 1
    assert manifest["tensor_shape"] == [2, 14, 144, 144]
    assert manifest["dtype"] == "float32"
    assert manifest["source_array_order"] == "XYZ"
    assert manifest["model_array_order"] == "ZYX"
    assert manifest["channel_semantics"] == {"0": "ED", "1": "ES"}

    patients = manifest["patients"]
    assert len(patients) >= 5
    assert [patient["patient_id"] for patient in patients] == sorted(
        patient["patient_id"] for patient in patients
    )
    assert {patient["class_name"] for patient in patients} == {"NOR", "DCM", "HCM", "MINF", "RV"}

    for patient in patients:
        tensor_path = GOLDEN_DIR / patient["tensor_filename"]
        tensor = np.load(tensor_path, allow_pickle=False)
        assert tensor.shape == (2, 14, 144, 144)
        assert tensor.dtype == np.float32
        assert tensor.flags.c_contiguous
        assert np.isfinite(tensor).all()
        assert patient["normalization"]["scope"] == "joint_ed_es"
        assert not Path(patient["tensor_filename"]).is_absolute()
        assert not Path(patient["source_ed_filename"]).is_absolute()
        assert not Path(patient["source_es_filename"]).is_absolute()
        assert not Path(patient["info_filename"]).is_absolute()


def test_golden_fixtures_match_real_acdc_preprocessing() -> None:
    """Golden fixtures exactly match current production preprocessing on real ACDC data."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    preprocessor = PatientPreprocessor(load_preprocessing_config())
    generator = GoldenReferenceGenerator(
        output_dir=GOLDEN_DIR,
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )

    generator.validate_source_checksums(MANIFEST_PATH, patients)
    generator.validate_production_parity(MANIFEST_PATH, patients, preprocessor)


def test_golden_generation_is_byte_deterministic(tmp_path: Path) -> None:
    """Golden generation is byte-identical across independent output directories."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    preprocessor = PatientPreprocessor(load_preprocessing_config())

    first_generator = GoldenReferenceGenerator(
        output_dir=tmp_path / "first",
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )
    second_generator = GoldenReferenceGenerator(
        output_dir=tmp_path / "second",
        preprocessing_contract_path=PREPROCESSING_CONTRACT_PATH,
        phase3_validation_csv_path=PHASE3_VALIDATION_CSV,
    )

    first = first_generator.generate(patients, class_mapping, preprocessor)
    second = second_generator.generate(patients, class_mapping, preprocessor)

    assert first.selected_patient_ids == second.selected_patient_ids
    assert first.file_hashes == second.file_hashes


def load_preprocessing_config() -> PreprocessingConfig:
    """Load frozen preprocessing config from default.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
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
