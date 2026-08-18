"""Integration test for real ACDC PyTorch and ONNX classifier parity."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.deployment.parity_validator import ParityValidator
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
CHECKPOINT_PATH = REPOSITORY_ROOT / "artifacts/checkpoints/phase9/classifier.pt"
ONNX_MODEL_PATH = REPOSITORY_ROOT / "artifacts/deployment/classifier.onnx"


def test_real_acdc_phase9_onnx_parity_on_one_patient_per_class() -> None:
    """Validate PyTorch and ONNX Runtime logits on five real preprocessed patients."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Phase 9 checkpoint unavailable: {CHECKPOINT_PATH}")
    if not ONNX_MODEL_PATH.is_file():
        pytest.skip(f"ONNX deployment model unavailable: {ONNX_MODEL_PATH}")

    indexer = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH)
    patients = indexer.index_patients()
    class_mapping = indexer.load_class_mapping()
    selected_patients = one_patient_per_class(patients, class_mapping)
    preprocessor = PatientPreprocessor(load_preprocessing_config())
    validator = ParityValidator()

    for patient in selected_patients:
        preprocessing_result = preprocessor.preprocess(patient)
        parity_result = validator.validate(
            patient_id=patient.patient_id,
            preprocessed_tensor=preprocessing_result.tensor,
            checkpoint_path=CHECKPOINT_PATH,
            onnx_model_path=ONNX_MODEL_PATH,
        )

        assert parity_result.patient_id == patient.patient_id
        assert parity_result.pytorch_output_shape == (1, 5)
        assert parity_result.onnx_output_shape == (1, 5)
        assert parity_result.pytorch_output_dtype == "float32"
        assert parity_result.onnx_output_dtype == "float32"
        assert parity_result.pytorch_output_is_finite
        assert parity_result.onnx_output_is_finite
        assert parity_result.predicted_class_index_matches
        assert parity_result.maximum_absolute_error <= 1e-5
        assert np.isfinite(parity_result.mean_absolute_error)
        assert parity_result.mean_absolute_error >= 0.0
        assert np.isfinite(parity_result.maximum_relative_error)
        assert parity_result.maximum_relative_error >= 0.0


def one_patient_per_class(
    patients: tuple[AcdcPatient, ...],
    class_mapping: dict[int, str],
) -> tuple[AcdcPatient, ...]:
    """Select the first deterministic real patient for each configured class."""
    patients_by_class_index: dict[int, AcdcPatient] = {}
    for patient in patients:
        patients_by_class_index.setdefault(patient.class_index, patient)

    selected_patients = tuple(
        patients_by_class_index[class_index] for class_index in sorted(class_mapping)
    )
    if len(selected_patients) != 5:
        raise ValueError(
            f"Expected one patient from each of five classes, got {len(selected_patients)}."
        )
    return selected_patients


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
