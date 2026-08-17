"""Phase 9 final production-model training orchestration."""

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.models.resnet3d18 import ResNet3D18
from cardiac_pathology.preprocessing import PatientPreprocessor
from cardiac_pathology.training.early_stopping import EarlyStopping
from cardiac_pathology.training.initialization import (
    CompatibilityReport,
    InitializationStrategy,
    initialize_model,
    set_deterministic_seed,
)
from cardiac_pathology.training.phase7_cross_validation import (
    build_loader,
    build_optimizer,
    load_yaml_mapping,
    patients_for_ids,
    preprocessing_config_from_mapping,
    require_float,
    require_int,
    require_mapping,
    require_str,
    reset_training_rng_after_initialization,
    training_augmentation_from_mapping,
)
from cardiac_pathology.training.splits import load_outer_folds, patient_id_sort_key
from cardiac_pathology.training.trainer import EpochMetrics, Trainer, TrainerConfig

PHASE9_CHECKPOINT_ROOT = Path("artifacts/checkpoints/phase9")
PHASE9_OUTER_SPLIT_PATH = Path("artifacts/dataset_splits/acdc_5fold_seed42.json")
PREPROCESSING_CONTRACT_PATH = "docs/preprocessing_contract.md"
CLASSIFIER_FILENAME = "classifier.pt"
SUMMARY_FILENAME = "phase9_training_summary.json"
PHASE9_ARCHITECTURE = "ResNet3D18"
PHASE9_INPUT_CHANNELS = 2
PHASE9_NUM_CLASSES = 5
PHASE9_SEED = 42
PHASE9_FINAL_EPOCHS = 25
PHASE9_BATCH_SIZE = 2
PHASE9_OPTIMIZER = "adamw"
PHASE9_LEARNING_RATE = 1.0e-4
PHASE9_WEIGHT_DECAY = 1.0e-4
PHASE9_INITIALIZATION_STRATEGY: InitializationStrategy = "pretrained"
PHASE9_PATIENT_COUNT = 100
PHASE9_CLASS_PATIENT_COUNT = 20
PHASE9_FOLD_COUNT = 5
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True, slots=True)
class Phase9ProductionArtifacts:
    """Deterministic artifact paths for final production training."""

    output_dir: Path
    classifier_path: Path
    summary_path: Path


def phase9_artifacts(repository_root: Path) -> Phase9ProductionArtifacts:
    """Build deterministic artifact paths for Phase 9 production training."""
    output_dir = repository_root / PHASE9_CHECKPOINT_ROOT

    return Phase9ProductionArtifacts(
        output_dir=output_dir,
        classifier_path=output_dir / CLASSIFIER_FILENAME,
        summary_path=output_dir / SUMMARY_FILENAME,
    )


@dataclass(frozen=True, slots=True)
class Phase9ProductionResult:
    """Artifacts produced by one completed Phase 9 production run."""

    classifier_path: Path
    summary_path: Path


class Phase9ProductionTrainer:
    """Train the final classifier on the complete labeled ACDC cohort."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_dir: Path,
        config_path: Path,
        class_mapping_path: Path,
    ) -> None:
        """Initialize Phase 9 production-training orchestration."""
        self.repository_root = repository_root
        self.dataset_dir = dataset_dir
        self.config_path = config_path
        self.class_mapping_path = class_mapping_path

    def run(self) -> Phase9ProductionResult:
        """Train and persist the frozen Phase 9 production model."""
        config = load_yaml_mapping(self.config_path)
        project_config = require_mapping(config, "project")
        training_config = require_mapping(config, "training")
        validate_phase9_training_contract(
            config=config,
            project_config=project_config,
            training_config=training_config,
        )
        validate_phase9_cuda_runtime_preconditions(training_config)

        artifacts = phase9_artifacts(self.repository_root)
        validate_no_phase9_artifacts(artifacts)

        indexer = AcdcDatasetIndexer(
            dataset_dir=self.dataset_dir,
            class_mapping_path=self.class_mapping_path,
        )
        indexed_patients = indexer.index_patients()
        class_mapping = indexer.load_class_mapping()
        patient_ids = load_phase9_patient_ids(self.repository_root / PHASE9_OUTER_SPLIT_PATH)
        class_distribution = validate_phase9_patient_contract(
            indexed_patients=indexed_patients,
            production_patient_ids=patient_ids,
            class_mapping=class_mapping,
        )

        patient_by_id = {patient.patient_id: patient for patient in indexed_patients}
        train_patients = patients_for_ids(patient_ids, patient_by_id)

        preprocessing_config = preprocessing_config_from_mapping(config)
        preprocessor = PatientPreprocessor(preprocessing_config)
        training_augmentation = training_augmentation_from_mapping(config)
        train_loader = build_loader(
            patients=train_patients,
            preprocessor=preprocessor,
            batch_size=PHASE9_BATCH_SIZE,
            shuffle=True,
            num_workers=require_int(training_config, "num_workers"),
            seed=PHASE9_SEED,
            augmentation=training_augmentation,
        )

        set_deterministic_seed(PHASE9_SEED)
        model = ResNet3D18(
            input_channels=PHASE9_INPUT_CHANNELS,
            num_classes=PHASE9_NUM_CLASSES,
        )
        initialization_report = initialize_model(model, PHASE9_INITIALIZATION_STRATEGY)
        reset_training_rng_after_initialization(PHASE9_SEED)
        optimizer = build_optimizer(
            model=model,
            optimizer_name=PHASE9_OPTIMIZER,
            learning_rate=PHASE9_LEARNING_RATE,
            weight_decay=PHASE9_WEIGHT_DECAY,
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device(require_str(training_config, "device")),
            config=TrainerConfig(
                num_epochs=PHASE9_FINAL_EPOCHS,
                num_classes=PHASE9_NUM_CLASSES,
                checkpoint_dir=artifacts.output_dir,
                initialization_strategy=PHASE9_INITIALIZATION_STRATEGY,
                seed=PHASE9_SEED,
            ),
            early_stopping=EarlyStopping(patience=1),
            initialization_report=initialization_report,
            training_config_metadata=phase9_training_config_metadata(
                num_workers=require_int(training_config, "num_workers"),
                device=require_str(training_config, "device"),
            ),
        )

        training_history = [
            trainer.train_epoch(train_loader=train_loader, epoch=epoch)
            for epoch in range(1, PHASE9_FINAL_EPOCHS + 1)
        ]

        save_phase9_classifier(
            path=artifacts.classifier_path,
            model=trainer.model,
            patient_ids=patient_ids,
            class_distribution=class_distribution,
            class_mapping=class_mapping,
        )
        save_phase9_summary(
            path=artifacts.summary_path,
            classifier_path=artifacts.classifier_path,
            patient_ids=patient_ids,
            class_distribution=class_distribution,
            training_history=training_history,
            initialization_report=initialization_report,
            num_workers=require_int(training_config, "num_workers"),
            device=require_str(training_config, "device"),
        )
        return Phase9ProductionResult(
            classifier_path=artifacts.classifier_path,
            summary_path=artifacts.summary_path,
        )


def validate_phase9_training_contract(
    *,
    config: Mapping[str, object],
    project_config: Mapping[str, object],
    training_config: Mapping[str, object],
) -> None:
    """Fail before production training if runtime settings drift from Phase 9."""
    if require_int(project_config, "seed") != PHASE9_SEED:
        raise ValueError(f"Phase 9 requires project.seed == {PHASE9_SEED}")

    training_expected_values: dict[str, object] = {
        "initialization_strategy": PHASE9_INITIALIZATION_STRATEGY,
        "batch_size": PHASE9_BATCH_SIZE,
        "learning_rate": PHASE9_LEARNING_RATE,
        "weight_decay": PHASE9_WEIGHT_DECAY,
        "optimizer": PHASE9_OPTIMIZER,
        "num_workers": 0,
        "device": "cuda",
    }
    for key, expected in training_expected_values.items():
        validate_phase9_expected_value(
            training_config,
            key=key,
            expected=expected,
            config_key=f"training.{key}",
        )

    preprocessing_config = require_mapping(config, "preprocessing")
    validate_phase9_expected_value(
        preprocessing_config,
        key="orientation",
        expected="LPS",
        config_key="preprocessing.orientation",
    )

    target_spacing = require_mapping(preprocessing_config, "target_spacing_mm")
    for key, expected in {"x": 1.5, "y": 1.5, "z": 7.5}.items():
        validate_phase9_expected_value(
            target_spacing,
            key=key,
            expected=expected,
            config_key=f"preprocessing.target_spacing_mm.{key}",
        )

    target_shape = require_mapping(preprocessing_config, "target_shape")
    for key, expected in {"d": 14, "h": 144, "w": 144}.items():
        validate_phase9_expected_value(
            target_shape,
            key=key,
            expected=expected,
            config_key=f"preprocessing.target_shape.{key}",
        )

    crop = require_mapping(preprocessing_config, "crop")
    validate_phase9_expected_value(
        crop,
        key="strategy",
        expected="fov_center",
        config_key="preprocessing.crop.strategy",
    )

    interpolation = require_mapping(preprocessing_config, "interpolation")
    validate_phase9_expected_value(
        interpolation,
        key="image",
        expected="linear",
        config_key="preprocessing.interpolation.image",
    )

    intensity = require_mapping(preprocessing_config, "intensity")
    intensity_expected_values: dict[str, object] = {
        "normalization_scope": "joint_ed_es",
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
        "epsilon": 1.0e-6,
    }
    for key, expected in intensity_expected_values.items():
        validate_phase9_expected_value(
            intensity,
            key=key,
            expected=expected,
            config_key=f"preprocessing.intensity.{key}",
        )

    padding = require_mapping(preprocessing_config, "padding")
    padding_expected_values: dict[str, object] = {
        "z_value": 0.0,
        "apply_after_normalization": True,
    }
    for key, expected in padding_expected_values.items():
        validate_phase9_expected_value(
            padding,
            key=key,
            expected=expected,
            config_key=f"preprocessing.padding.{key}",
        )

    array_config = require_mapping(preprocessing_config, "array")
    for key, expected in {"source_order": "XYZ", "model_order": "ZYX", "dtype": "float32"}.items():
        validate_phase9_expected_value(
            array_config,
            key=key,
            expected=expected,
            config_key=f"preprocessing.array.{key}",
        )

    augmentation = require_mapping(training_config, "augmentation")
    augmentation_expected_values: dict[str, object] = {
        "enabled": True,
        "rotation_degrees": 5.0,
        "translation_mm": 5.0,
        "intensity_scale_min": 0.95,
        "intensity_scale_max": 1.05,
        "gaussian_noise_std": 0.01,
        "transform_probability": 0.5,
    }
    for key, expected in augmentation_expected_values.items():
        validate_phase9_expected_value(
            augmentation,
            key=key,
            expected=expected,
            config_key=f"training.augmentation.{key}",
        )


def validate_phase9_expected_value(
    mapping: Mapping[str, object],
    *,
    key: str,
    expected: object,
    config_key: str,
) -> None:
    """Validate one frozen Phase 9 config value."""
    if isinstance(expected, bool):
        actual = mapping.get(key)
        matches = isinstance(actual, bool) and actual == expected
    elif isinstance(expected, str):
        actual = require_str(mapping, key)
        matches = actual == expected
    elif isinstance(expected, int):
        actual = require_int(mapping, key)
        matches = actual == expected
    elif isinstance(expected, float):
        actual = require_float(mapping, key)
        matches = actual == expected
    else:
        raise TypeError(f"Unsupported expected Phase 9 value for {config_key}")

    if not matches:
        raise ValueError(f"Phase 9 requires {config_key} == {expected!r}; got {actual!r}")


def validate_no_phase9_artifacts(artifacts: Phase9ProductionArtifacts) -> None:
    """Prevent accidental overwrite of completed Phase 9 production artifacts."""
    existing_paths = [
        path for path in (artifacts.classifier_path, artifacts.summary_path) if path.exists()
    ]
    if existing_paths:
        formatted_paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Completed Phase 9 production artifacts already exist: {formatted_paths}"
        )


def validate_phase9_cuda_runtime_preconditions(training_config: Mapping[str, object]) -> None:
    """Validate CUDA and cuBLAS preconditions before Phase 9 training starts."""
    device = require_str(training_config, "device")
    if device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 9 requires CUDA, but torch.cuda.is_available() is false")
    actual_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas_config != EXPECTED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Phase 9 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before training starts"
        )


def load_phase9_patient_ids(split_path: Path) -> tuple[str, ...]:
    """Load and validate the authoritative Phase 9 production patient IDs."""
    outer_folds = load_outer_folds(split_path)
    if len(outer_folds) != PHASE9_FOLD_COUNT:
        raise ValueError(f"Phase 9 requires exactly {PHASE9_FOLD_COUNT} outer folds")

    persisted_ids = [patient_id for fold in outer_folds for patient_id in fold.test_patient_ids]
    unique_ids = set(persisted_ids)
    if len(persisted_ids) != PHASE9_PATIENT_COUNT:
        raise ValueError(
            f"Phase 9 requires {PHASE9_PATIENT_COUNT} total outer-test entries; "
            f"got {len(persisted_ids)}"
        )
    if len(unique_ids) != PHASE9_PATIENT_COUNT:
        duplicate_ids = sorted(
            (patient_id for patient_id, count in Counter(persisted_ids).items() if count > 1),
            key=patient_id_sort_key,
        )
        raise ValueError(
            f"Phase 9 production split contains duplicate patient IDs: {duplicate_ids}"
        )

    return tuple(sorted(unique_ids, key=patient_id_sort_key))


def validate_phase9_patient_contract(
    *,
    indexed_patients: Sequence[AcdcPatient],
    production_patient_ids: Sequence[str],
    class_mapping: Mapping[int, str],
) -> dict[str, int]:
    """Validate that indexed ACDC patients exactly match the production cohort."""
    if len(production_patient_ids) != PHASE9_PATIENT_COUNT:
        raise ValueError(
            f"Phase 9 requires {PHASE9_PATIENT_COUNT} production patient IDs; "
            f"got {len(production_patient_ids)}"
        )
    if len(set(production_patient_ids)) != PHASE9_PATIENT_COUNT:
        raise ValueError("Phase 9 production patient IDs must be unique")

    indexed_ids = {patient.patient_id for patient in indexed_patients}
    production_id_set = set(production_patient_ids)
    if indexed_ids != production_id_set:
        missing_ids = sorted(production_id_set - indexed_ids, key=patient_id_sort_key)
        extra_ids = sorted(indexed_ids - production_id_set, key=patient_id_sort_key)
        raise ValueError(
            "Indexed ACDC patient IDs do not match the persisted Phase 9 set: "
            f"missing={missing_ids}, extra={extra_ids}"
        )

    expected_class_indices = set(range(PHASE9_NUM_CLASSES))
    if set(class_mapping) != expected_class_indices:
        raise ValueError(f"Phase 9 requires class mapping indices {sorted(expected_class_indices)}")

    class_counts = Counter(patient.class_index for patient in indexed_patients)
    if set(class_counts) != expected_class_indices:
        raise ValueError("Phase 9 production cohort must represent all five classes")
    if any(count != PHASE9_CLASS_PATIENT_COUNT for count in class_counts.values()):
        raise ValueError(
            "Phase 9 production cohort requires exactly "
            f"{PHASE9_CLASS_PATIENT_COUNT} patients per class"
        )

    return {
        class_mapping[class_index]: class_counts[class_index]
        for class_index in sorted(class_mapping)
    }


def phase9_training_config_metadata(*, num_workers: int, device: str) -> dict[str, object]:
    """Build JSON-compatible metadata for the frozen Phase 9 training contract."""
    return {
        "final_epochs": PHASE9_FINAL_EPOCHS,
        "seed": PHASE9_SEED,
        "initialization_strategy": PHASE9_INITIALIZATION_STRATEGY,
        "optimizer": PHASE9_OPTIMIZER,
        "learning_rate": PHASE9_LEARNING_RATE,
        "weight_decay": PHASE9_WEIGHT_DECAY,
        "batch_size": PHASE9_BATCH_SIZE,
        "num_workers": num_workers,
        "device": device,
        "validation_set": None,
        "test_set": None,
        "early_stopping": None,
        "checkpoint_selection": None,
    }


def save_phase9_classifier(
    *,
    path: Path,
    model: torch.nn.Module,
    patient_ids: Sequence[str],
    class_distribution: Mapping[str, int],
    class_mapping: Mapping[int, str],
) -> Path:
    """Persist the final Phase 9 production classifier payload."""
    if path.exists():
        raise FileExistsError(f"Phase 9 classifier already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": PHASE9_ARCHITECTURE,
            "input_channels": PHASE9_INPUT_CHANNELS,
            "num_classes": PHASE9_NUM_CLASSES,
            "initialization_strategy": PHASE9_INITIALIZATION_STRATEGY,
            "seed": PHASE9_SEED,
            "final_epochs": PHASE9_FINAL_EPOCHS,
            "patient_count": PHASE9_PATIENT_COUNT,
            "patient_ids": list(patient_ids),
            "class_distribution": dict(class_distribution),
            "class_mapping": dict(class_mapping),
            "preprocessing_contract": PREPROCESSING_CONTRACT_PATH,
        },
        path,
    )
    return path


def save_phase9_summary(
    *,
    path: Path,
    classifier_path: Path,
    patient_ids: Sequence[str],
    class_distribution: Mapping[str, int],
    training_history: Sequence[EpochMetrics],
    initialization_report: CompatibilityReport,
    num_workers: int,
    device: str,
) -> Path:
    """Persist Phase 9 reproducibility and training diagnostics."""
    if path.exists():
        raise FileExistsError(f"Phase 9 training summary already exists: {path}")

    payload = {
        "final_epochs": PHASE9_FINAL_EPOCHS,
        "seed": PHASE9_SEED,
        "initialization_strategy": PHASE9_INITIALIZATION_STRATEGY,
        "optimizer_settings": phase9_training_config_metadata(
            num_workers=num_workers,
            device=device,
        ),
        "patient_ids": list(patient_ids),
        "class_distribution": dict(class_distribution),
        "training_history": [metrics.as_metadata() for metrics in training_history],
        "classifier_path": str(classifier_path),
        "initialization_report": initialization_report.as_metadata(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
