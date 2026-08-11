"""Integration tests for deterministic preprocessing on real ACDC patients."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"


def test_real_acdc_preprocessing_contract() -> None:
    """All real ACDC patients satisfy the frozen preprocessing contract."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    patients = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH).index_patients()
    preprocessor = PatientPreprocessor(load_preprocessing_config())

    assert len(patients) == 100
    for patient in patients:
        result = preprocessor.preprocess(patient)
        assert result.tensor.shape == (2, 14, 144, 144)
        assert result.tensor.dtype == np.float32
        assert result.tensor.flags.c_contiguous
        assert np.isfinite(result.tensor).all()
        assert result.valid_depth <= 14
        assert result.crop_end_xy[0] - result.crop_start_xy[0] == 144
        assert result.crop_end_xy[1] - result.crop_start_xy[1] == 144
        assert result.normalization_std > preprocessor.config.epsilon


def test_real_acdc_preprocessing_repeatability() -> None:
    """One real patient per class preprocesses identically on repeated runs."""
    if not DATASET_DIR.is_dir():
        pytest.skip(f"Real ACDC dataset unavailable: {DATASET_DIR}")

    patients = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH).index_patients()
    selected_patients = one_patient_per_class(patients)
    preprocessor = PatientPreprocessor(load_preprocessing_config())

    for patient in selected_patients:
        first = preprocessor.preprocess(patient)
        second = preprocessor.preprocess(patient)
        assert np.array_equal(first.tensor, second.tensor)
        assert first.patient_id == second.patient_id
        assert first.source_shape_xyz == second.source_shape_xyz
        assert first.source_spacing_xyz == second.source_spacing_xyz
        assert first.resampled_shape_xyz == second.resampled_shape_xyz
        assert first.target_spacing_xyz == second.target_spacing_xyz
        assert first.crop_start_xy == second.crop_start_xy
        assert first.crop_end_xy == second.crop_end_xy
        assert first.valid_depth == second.valid_depth
        assert first.z_padding_lower == second.z_padding_lower
        assert first.z_padding_upper == second.z_padding_upper
        assert first.clip_lower == second.clip_lower
        assert first.clip_upper == second.clip_upper
        assert first.normalization_mean == second.normalization_mean
        assert first.normalization_std == second.normalization_std


def one_patient_per_class(patients: tuple[AcdcPatient, ...]) -> tuple[AcdcPatient, ...]:
    """Select the first deterministic real patient from each diagnostic class."""
    selected: dict[str, AcdcPatient] = {}
    for patient in patients:
        selected.setdefault(patient.class_name, patient)
    return tuple(selected[class_name] for class_name in sorted(selected))


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
