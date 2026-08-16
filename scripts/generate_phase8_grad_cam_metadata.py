"""Generate reproducibility metadata for the frozen Phase 8 Grad-CAM cases."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import Tensor, nn

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.explainability.grad_cam_3d import GradCam3D
from cardiac_pathology.explainability.grad_cam_visualizer import GradCamVisualizer
from cardiac_pathology.models.resnet3d18 import ResNet3D18
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig
from cardiac_pathology.preprocessing.preprocessing_result import PreprocessingResult
from cardiac_pathology.training.phase7_cross_validation import load_best_checkpoint
from cardiac_pathology.training.phase7_oof_pooling import Phase7OofRecord, oof_record_from_mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/checkpoints/phase7"
POOLED_OOF_PATH = CHECKPOINT_ROOT / "pooled_oof_predictions.json"
OUTPUT_PATH = REPOSITORY_ROOT / "artifacts/phase8_grad_cam/phase8_grad_cam_metadata.json"
FIGURE_DIR = REPOSITORY_ROOT / "artifacts/phase8_grad_cam/figures"

FROZEN_PATIENT_IDS = (
    "patient070",
    "patient063",
    "patient010",
    "patient005",
    "patient023",
    "patient022",
    "patient043",
    "patient044",
    "patient091",
    "patient083",
)
TARGET_LAYER_NAME = "layer4.1"
EXPECTED_ACTIVATION_SHAPE = (512, 7, 5, 5)
EXPECTED_CAM_OUTPUT_SHAPE = (14, 144, 144)
OOF_TOLERANCE = 1.0e-6


def main() -> None:
    """Build and persist deterministic Phase 8 Grad-CAM metadata."""
    config = load_yaml_mapping(CONFIG_PATH)
    class_names = load_class_names(CLASS_MAPPING_PATH)
    patients = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH).index_patients()
    patient_by_id = {patient.patient_id: patient for patient in patients}
    oof_by_patient_id = load_pooled_oof_records(POOLED_OOF_PATH)
    preprocessor = PatientPreprocessor(preprocessing_config_from_mapping(config))
    model_config = require_mapping(config, "model")
    visualizer = GradCamVisualizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strict_model_oof_verification = device.type == "cuda"

    records: list[dict[str, object]] = []
    for patient_id in FROZEN_PATIENT_IDS:
        patient = patient_by_id[patient_id]
        oof_record = oof_by_patient_id[patient_id]
        checkpoint_path = checkpoint_path_for_fold(oof_record.fold_index)
        preprocessing_result = preprocessor.preprocess(patient)
        input_tensor = torch.from_numpy(preprocessing_result.tensor).unsqueeze(0).to(device)

        model = build_model(model_config)
        load_best_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
        model.to(device)
        logits = compute_logits(model, input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        verify_oof_reproduction(
            patient_id=patient_id,
            oof_record=oof_record,
            checkpoint_path=checkpoint_path,
            logits=logits.squeeze(0),
            probabilities=probabilities,
            strict_model_oof_verification=strict_model_oof_verification,
        )

        target_class_index = oof_record.predicted_class_index
        grad_cam = GradCam3D(model, model.layer4[1]).generate(
            input_tensor,
            target_class=target_class_index,
        )
        cam = grad_cam.squeeze(0)
        validate_expected_shapes(model, input_tensor, cam)

        valid_start = preprocessing_result.z_padding_lower
        valid_end = valid_start + preprocessing_result.valid_depth
        selected_slice = visualizer.select_slice(cam, valid_start, valid_end)

        records.append(
            build_metadata_record(
                patient=patient,
                oof_record=oof_record,
                class_names=class_names,
                checkpoint_path=checkpoint_path,
                confidence=oof_record.probabilities[target_class_index],
                target_class_index=target_class_index,
                preprocessing_result=preprocessing_result,
                selected_slice=selected_slice,
            )
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} Phase 8 Grad-CAM metadata records: {OUTPUT_PATH}")


def build_model(model_config: Mapping[str, object]) -> ResNet3D18:
    """Create the frozen custom 3D ResNet classifier."""
    return ResNet3D18(
        input_channels=require_int(model_config, "input_channels"),
        num_classes=require_int(model_config, "num_classes"),
    )


def compute_logits(model: nn.Module, input_tensor: Tensor) -> Tensor:
    """Run a deterministic eval forward pass and return raw logits."""
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
    if logits.shape != (1, 5):
        raise ValueError(f"Expected logits shape [1, 5], got {tuple(logits.shape)}.")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    return logits.detach().cpu()


def verify_oof_reproduction(
    *,
    patient_id: str,
    oof_record: Phase7OofRecord,
    checkpoint_path: Path,
    logits: Tensor,
    probabilities: Tensor,
    strict_model_oof_verification: bool,
) -> None:
    """Verify recomputed predictions reproduce the frozen Phase 7 OOF record."""
    if int(torch.argmax(probabilities).detach().cpu()) != oof_record.predicted_class_index:
        raise ValueError(f"{patient_id}: predicted class does not reproduce Phase 7 OOF.")

    if not strict_model_oof_verification:
        fold_oof_record = load_fold_oof_record(
            checkpoint_path.parent / "oof_predictions.json",
            patient_id=patient_id,
        )
        verify_oof_records_match(patient_id=patient_id, pooled=oof_record, fold=fold_oof_record)
        return

    expected_logits = torch.tensor(oof_record.logits, dtype=logits.dtype)
    expected_probabilities = torch.tensor(oof_record.probabilities, dtype=probabilities.dtype)
    if not torch.allclose(logits, expected_logits, atol=OOF_TOLERANCE, rtol=0.0):
        raise ValueError(f"{patient_id}: logits do not reproduce Phase 7 OOF within 1e-6.")
    if not torch.allclose(probabilities, expected_probabilities, atol=OOF_TOLERANCE, rtol=0.0):
        raise ValueError(f"{patient_id}: probabilities do not reproduce Phase 7 OOF within 1e-6.")


def validate_expected_shapes(model: ResNet3D18, input_tensor: Tensor, cam: Tensor) -> None:
    """Validate frozen Phase 8 target activation and CAM output shapes."""
    activation_shape: tuple[int, ...] | None = None

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        nonlocal activation_shape
        if not isinstance(output, Tensor):
            raise TypeError("Target layer output must be a tensor.")
        activation_shape = tuple(output.shape[1:])

    handle = model.layer4[1].register_forward_hook(capture_activation)
    try:
        model.eval()
        with torch.inference_mode():
            model(input_tensor)
    finally:
        handle.remove()

    if activation_shape != EXPECTED_ACTIVATION_SHAPE:
        raise ValueError(
            f"Expected activation shape {EXPECTED_ACTIVATION_SHAPE}, got {activation_shape}."
        )
    if tuple(cam.shape) != EXPECTED_CAM_OUTPUT_SHAPE:
        raise ValueError(f"Expected CAM shape {EXPECTED_CAM_OUTPUT_SHAPE}, got {tuple(cam.shape)}.")


def build_metadata_record(
    *,
    patient: AcdcPatient,
    oof_record: Phase7OofRecord,
    class_names: Mapping[int, str],
    checkpoint_path: Path,
    confidence: float,
    target_class_index: int,
    preprocessing_result: PreprocessingResult,
    selected_slice: int,
) -> dict[str, object]:
    """Build one deterministic Phase 8 metadata record."""
    valid_slice_start = preprocessing_result.z_padding_lower
    valid_slice_end_exclusive = valid_slice_start + preprocessing_result.valid_depth
    true_class_name = class_names[oof_record.true_class_index]
    predicted_class_name = class_names[oof_record.predicted_class_index]
    target_class_name = class_names[target_class_index]
    return {
        "patient_id": patient.patient_id,
        "fold_index": oof_record.fold_index,
        "checkpoint_path": checkpoint_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "true_class_index": oof_record.true_class_index,
        "true_class_name": true_class_name,
        "predicted_class_index": oof_record.predicted_class_index,
        "predicted_class_name": predicted_class_name,
        "confidence": confidence,
        "grad_cam_target_class_index": target_class_index,
        "grad_cam_target_class_name": target_class_name,
        "target_layer": TARGET_LAYER_NAME,
        "activation_shape": list(EXPECTED_ACTIVATION_SHAPE),
        "cam_output_shape": list(EXPECTED_CAM_OUTPUT_SHAPE),
        "normalization_rule": "ReLU -> per-patient min-max normalization -> trilinear upsampling",
        "upsampling": "trilinear, align_corners=False",
        "valid_slice_start": valid_slice_start,
        "valid_slice_end_exclusive": valid_slice_end_exclusive,
        "selected_slice": selected_slice,
        "slice_selection_rule": "maximum mean CAM within non-padded Z range",
        "attribution_semantics": "joint ED+ES model attribution",
        "figure_path": (FIGURE_DIR / f"{patient.patient_id}_grad_cam.png")
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
    }


def checkpoint_path_for_fold(fold_index: int) -> Path:
    """Return the frozen Phase 7 best checkpoint path for one fold."""
    return CHECKPOINT_ROOT / f"fold_{fold_index}" / "pretrained" / "best_checkpoint.pt"


def load_pooled_oof_records(path: Path) -> dict[str, Phase7OofRecord]:
    """Load frozen pooled Phase 7 OOF records keyed by patient ID."""
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected OOF record list in {path}.")
    records: dict[str, Phase7OofRecord] = {}
    for record_index, raw_record in enumerate(raw_records):
        record = oof_record_from_mapping(
            raw_record=raw_record,
            path=path,
            record_index=record_index,
        )
        records[record.patient_id] = record
    return records


def load_fold_oof_record(path: Path, *, patient_id: str) -> Phase7OofRecord:
    """Load one per-fold OOF record for a frozen Phase 8 patient."""
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected OOF record list in {path}.")
    for record_index, raw_record in enumerate(raw_records):
        record = oof_record_from_mapping(
            raw_record=raw_record,
            path=path,
            record_index=record_index,
        )
        if record.patient_id == patient_id:
            return record
    raise ValueError(f"{patient_id} is absent from {path}.")


def verify_oof_records_match(
    *,
    patient_id: str,
    pooled: Phase7OofRecord,
    fold: Phase7OofRecord,
) -> None:
    """Verify pooled OOF metadata exactly preserves the per-fold OOF record."""
    if pooled.fold_index != fold.fold_index:
        raise ValueError(f"{patient_id}: pooled and fold OOF fold_index differ.")
    if pooled.true_class_index != fold.true_class_index:
        raise ValueError(f"{patient_id}: pooled and fold OOF true_class_index differ.")
    if pooled.predicted_class_index != fold.predicted_class_index:
        raise ValueError(f"{patient_id}: pooled and fold OOF predicted_class_index differ.")
    logit_pairs = zip(pooled.logits, fold.logits, strict=True)
    if any(abs(actual - expected) > OOF_TOLERANCE for actual, expected in logit_pairs):
        raise ValueError(f"{patient_id}: pooled and fold OOF logits differ.")
    if any(
        abs(actual - expected) > OOF_TOLERANCE
        for actual, expected in zip(pooled.probabilities, fold.probabilities, strict=True)
    ):
        raise ValueError(f"{patient_id}: pooled and fold OOF probabilities differ.")


def load_class_names(path: Path) -> dict[int, str]:
    """Load class names keyed by integer class index."""
    raw_mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected class mapping object in {path}.")
    return {
        int(index): require_str_value(name, f"{path}:{index}")
        for index, name in raw_mapping.items()
    }


def preprocessing_config_from_mapping(config: Mapping[str, object]) -> PreprocessingConfig:
    """Construct the frozen preprocessing config from the project YAML."""
    preprocessing = require_mapping(config, "preprocessing")
    target_spacing = require_mapping(preprocessing, "target_spacing_mm")
    target_shape = require_mapping(preprocessing, "target_shape")
    intensity = require_mapping(preprocessing, "intensity")
    padding = require_mapping(preprocessing, "padding")
    return PreprocessingConfig(
        target_orientation=require_str(preprocessing, "orientation"),
        target_spacing_xyz=(
            require_float(target_spacing, "x"),
            require_float(target_spacing, "y"),
            require_float(target_spacing, "z"),
        ),
        target_shape_dhw=(
            require_int(target_shape, "d"),
            require_int(target_shape, "h"),
            require_int(target_shape, "w"),
        ),
        lower_percentile=require_float(intensity, "lower_percentile"),
        upper_percentile=require_float(intensity, "upper_percentile"),
        epsilon=require_float(intensity, "epsilon"),
        z_padding_value=require_float(padding, "z_value"),
    )


def load_yaml_mapping(path: Path) -> Mapping[str, object]:
    """Load a YAML file as a string-keyed mapping."""
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string-keyed mapping in {path}.")
    return cast(Mapping[str, object], value)


def require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read a required nested string-keyed mapping."""
    value = mapping.get(key)
    if not isinstance(value, dict) or not all(isinstance(child_key, str) for child_key in value):
        raise ValueError(f"{key} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def require_str(mapping: Mapping[str, object], key: str) -> str:
    """Read a required string field."""
    return require_str_value(mapping.get(key), key)


def require_str_value(value: Any, name: str) -> str:
    """Validate a required non-empty string value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def require_int(mapping: Mapping[str, object], key: str) -> int:
    """Read a required integer field."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    return value


def require_float(mapping: Mapping[str, object], key: str) -> float:
    """Read a required finite numeric field."""
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite.")
    return numeric


if __name__ == "__main__":
    main()
