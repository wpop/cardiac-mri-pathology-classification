"""Phase 7 cross-validation orchestration without final pooled evaluation."""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.models.resnet3d18 import ResNet3D18
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig
from cardiac_pathology.training.augmentation_config import TrainingAugmentationConfig
from cardiac_pathology.training.dataset import PreprocessedPatientDataset
from cardiac_pathology.training.early_stopping import EarlyStopping
from cardiac_pathology.training.initialization import (
    CompatibilityReport,
    InitializationStrategy,
    initialize_model,
    set_deterministic_seed,
)
from cardiac_pathology.training.splits import load_outer_folds, patient_id_sort_key
from cardiac_pathology.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingHistory,
    accuracy_and_macro_f1,
    unpack_batch,
    validate_logits,
)
from cardiac_pathology.training.training_augmentation import TrainingAugmentation

EXPECTED_PHASE7_CHECKPOINT_ROOT = Path("artifacts/checkpoints/phase7")
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DEFAULT_OUTER_SPLIT_PATH = Path("artifacts/dataset_splits/acdc_5fold_seed42.json")
DEFAULT_INNER_SPLIT_PATH = Path("artifacts/dataset_splits/acdc_5fold_inner_validation_seed42.json")
PHASE7_SUMMARY_FILENAME = "phase7_fold_summary.json"
PHASE7_OOF_PREDICTIONS_FILENAME = "oof_predictions.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.pt"


@dataclass(frozen=True, slots=True)
class Phase7FoldSplit:
    """Persisted Phase 7 split assignments for one outer fold."""

    fold_index: int
    inner_train_patient_ids: tuple[str, ...]
    inner_validation_patient_ids: tuple[str, ...]
    outer_test_patient_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase7FoldArtifacts:
    """Deterministic artifact paths for one Phase 7 fold."""

    output_dir: Path
    checkpoint_path: Path
    oof_predictions_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class Phase7PredictionRecord:
    """Patient-level out-of-fold prediction record for later pooled evaluation."""

    fold_index: int
    patient_id: str
    true_class_index: int
    predicted_class_index: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]

    def as_metadata(self) -> dict[str, object]:
        """Convert the prediction record to deterministic JSON metadata."""
        return {
            "fold_index": self.fold_index,
            "patient_id": self.patient_id,
            "true_class_index": self.true_class_index,
            "predicted_class_index": self.predicted_class_index,
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True, slots=True)
class Phase7FoldResult:
    """Persisted result metadata for one Phase 7 fold."""

    summary_path: Path
    oof_predictions_path: Path
    best_checkpoint_path: Path
    outer_test_accuracy: float
    outer_test_macro_f1: float


class Phase7FoldRunner:
    """Execute one frozen Phase 7 outer fold using persisted inner assignments."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_dir: Path,
        config: Mapping[str, object],
        class_mapping_path: Path,
        fold_split: Phase7FoldSplit,
    ) -> None:
        self.repository_root = repository_root
        self.dataset_dir = dataset_dir
        self.config = config
        self.class_mapping_path = class_mapping_path
        self.fold_split = fold_split

    def run(self) -> Phase7FoldResult:
        """Train one fold and then evaluate the held-out outer test partition."""
        project_config = require_mapping(self.config, "project")
        training_config = require_mapping(self.config, "training")
        model_config = require_mapping(self.config, "model")

        validate_phase7_training_contract(
            project_config=project_config,
            training_config=training_config,
        )
        validate_cuda_runtime_preconditions(training_config)

        seed = require_int(project_config, "seed")
        initialization_strategy = require_initialization_strategy(training_config)
        artifacts = phase7_artifacts(
            checkpoint_root=self.repository_root / EXPECTED_PHASE7_CHECKPOINT_ROOT,
            fold_index=self.fold_split.fold_index,
            initialization_strategy=initialization_strategy,
        )
        validate_no_completed_phase7_artifacts(artifacts)

        patients = self._index_patients()
        patient_by_id = {patient.patient_id: patient for patient in patients}
        train_patients = patients_for_ids(self.fold_split.inner_train_patient_ids, patient_by_id)
        validation_patients = patients_for_ids(
            self.fold_split.inner_validation_patient_ids,
            patient_by_id,
        )
        outer_test_patients = patients_for_ids(
            self.fold_split.outer_test_patient_ids,
            patient_by_id,
        )

        preprocessing_config = preprocessing_config_from_mapping(self.config)
        preprocessor = PatientPreprocessor(preprocessing_config)
        training_augmentation = training_augmentation_from_mapping(self.config)

        train_loader = build_loader(
            patients=train_patients,
            preprocessor=preprocessor,
            batch_size=require_int(training_config, "batch_size"),
            shuffle=True,
            num_workers=require_int(training_config, "num_workers"),
            seed=seed,
            augmentation=training_augmentation,
        )
        validation_loader = build_loader(
            patients=validation_patients,
            preprocessor=preprocessor,
            batch_size=require_int(training_config, "batch_size"),
            shuffle=False,
            num_workers=require_int(training_config, "num_workers"),
            seed=seed,
            augmentation=None,
        )

        set_deterministic_seed(seed)
        model = self._build_model(model_config)
        initialization_report = initialize_model(model, initialization_strategy)
        reset_training_rng_after_initialization(seed)
        optimizer = build_optimizer(
            model=model,
            optimizer_name=require_str(training_config, "optimizer"),
            learning_rate=require_float(training_config, "learning_rate"),
            weight_decay=require_float(training_config, "weight_decay"),
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device(require_str(training_config, "device")),
            config=TrainerConfig(
                num_epochs=require_int(training_config, "num_epochs"),
                num_classes=require_int(model_config, "num_classes"),
                checkpoint_dir=artifacts.output_dir,
                initialization_strategy=initialization_strategy,
                seed=seed,
            ),
            early_stopping=EarlyStopping(
                patience=require_int(training_config, "early_stopping_patience"),
                min_delta=require_float(training_config, "early_stopping_min_delta"),
            ),
            initialization_report=initialization_report,
            training_config_metadata=dict(training_config),
        )
        history = trainer.fit(train_loader=train_loader, validation_loader=validation_loader)

        outer_test_model = self._build_model(model_config)
        load_best_checkpoint(
            model=outer_test_model,
            checkpoint_path=artifacts.checkpoint_path,
            device=torch.device(require_str(training_config, "device")),
        )
        prediction_records = evaluate_outer_test(
            model=outer_test_model,
            patients=outer_test_patients,
            preprocessor=preprocessor,
            device=torch.device(require_str(training_config, "device")),
            num_classes=require_int(model_config, "num_classes"),
            fold_index=self.fold_split.fold_index,
        )
        outer_test_accuracy, outer_test_macro_f1 = accuracy_and_macro_f1(
            predictions=[record.predicted_class_index for record in prediction_records],
            targets=[record.true_class_index for record in prediction_records],
            num_classes=require_int(model_config, "num_classes"),
        )

        save_oof_predictions(artifacts.oof_predictions_path, prediction_records)
        save_phase7_summary(
            path=artifacts.summary_path,
            fold_split=self.fold_split,
            history=history,
            outer_test_accuracy=outer_test_accuracy,
            outer_test_macro_f1=outer_test_macro_f1,
            checkpoint_path=artifacts.checkpoint_path,
            seed=seed,
            initialization_strategy=initialization_strategy,
            initialization_report=initialization_report,
            training_config=training_config,
        )
        return Phase7FoldResult(
            summary_path=artifacts.summary_path,
            oof_predictions_path=artifacts.oof_predictions_path,
            best_checkpoint_path=artifacts.checkpoint_path,
            outer_test_accuracy=outer_test_accuracy,
            outer_test_macro_f1=outer_test_macro_f1,
        )

    def _index_patients(self) -> tuple[AcdcPatient, ...]:
        """Index real ACDC patients using the configured class mapping."""
        indexer = AcdcDatasetIndexer(
            dataset_dir=self.dataset_dir,
            class_mapping_path=self.class_mapping_path,
        )
        return indexer.index_patients()

    @staticmethod
    def _build_model(model_config: Mapping[str, object]) -> ResNet3D18:
        """Construct the custom project ResNet3D18."""
        return ResNet3D18(
            input_channels=require_int(model_config, "input_channels"),
            num_classes=require_int(model_config, "num_classes"),
        )


class Phase7CrossValidationOrchestrator:
    """Select and run persisted Phase 7 folds in deterministic order."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_dir: Path,
        config_path: Path,
        class_mapping_path: Path,
        outer_split_path: Path,
        inner_split_path: Path,
    ) -> None:
        self.repository_root = repository_root
        self.dataset_dir = dataset_dir
        self.config_path = config_path
        self.class_mapping_path = class_mapping_path
        self.outer_split_path = outer_split_path
        self.inner_split_path = inner_split_path

    def run(self, fold_index: int | None = None) -> tuple[Phase7FoldResult, ...]:
        """Run one requested fold or all five persisted folds."""
        config = load_yaml_mapping(self.config_path)
        folds = load_phase7_fold_splits(
            inner_split_path=self.inner_split_path,
            outer_split_path=self.outer_split_path,
        )
        selected_folds = select_phase7_folds(folds=folds, fold_index=fold_index)
        results: list[Phase7FoldResult] = []
        for split in selected_folds:
            runner = Phase7FoldRunner(
                repository_root=self.repository_root,
                dataset_dir=self.dataset_dir,
                config=config,
                class_mapping_path=self.class_mapping_path,
                fold_split=split,
            )
            results.append(runner.run())
        return tuple(results)


def load_phase7_fold_splits(
    *,
    inner_split_path: Path,
    outer_split_path: Path,
) -> tuple[Phase7FoldSplit, ...]:
    """Load persisted Phase 7 inner assignments and validate them against outer folds."""
    with inner_split_path.open("r", encoding="utf-8") as file:
        raw_inner = json.load(file)
    inner_artifact = require_json_mapping(raw_inner, inner_split_path)
    raw_folds = inner_artifact.get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError(f"Expected folds list in {inner_split_path}")

    outer_folds = {fold.fold_index: fold for fold in load_outer_folds(outer_split_path)}
    fold_splits: list[Phase7FoldSplit] = []
    for raw_fold in raw_folds:
        fold_mapping = require_json_mapping(raw_fold, inner_split_path)
        fold_index = require_json_int(fold_mapping, "fold_index", inner_split_path)
        outer_fold = outer_folds.get(fold_index)
        if outer_fold is None:
            raise ValueError(f"Inner fold {fold_index} is absent from {outer_split_path}")

        outer_development_ids = require_json_str_tuple(
            fold_mapping,
            "outer_development_patient_ids",
            inner_split_path,
        )
        outer_test_ids = require_json_str_tuple(
            fold_mapping,
            "outer_test_patient_ids",
            inner_split_path,
        )
        inner_train_ids = require_json_str_tuple(
            fold_mapping,
            "inner_train_patient_ids",
            inner_split_path,
        )
        inner_validation_ids = require_json_str_tuple(
            fold_mapping,
            "inner_validation_patient_ids",
            inner_split_path,
        )
        validate_phase7_fold_assignments(
            fold_index=fold_index,
            outer_development_ids=outer_development_ids,
            outer_test_ids=outer_test_ids,
            inner_train_ids=inner_train_ids,
            inner_validation_ids=inner_validation_ids,
            persisted_outer_train_ids=outer_fold.train_patient_ids,
            persisted_outer_test_ids=outer_fold.test_patient_ids,
        )
        fold_splits.append(
            Phase7FoldSplit(
                fold_index=fold_index,
                inner_train_patient_ids=inner_train_ids,
                inner_validation_patient_ids=inner_validation_ids,
                outer_test_patient_ids=outer_test_ids,
            )
        )

    return tuple(sorted(fold_splits, key=lambda fold: fold.fold_index))


def validate_phase7_fold_assignments(
    *,
    fold_index: int,
    outer_development_ids: Sequence[str],
    outer_test_ids: Sequence[str],
    inner_train_ids: Sequence[str],
    inner_validation_ids: Sequence[str],
    persisted_outer_train_ids: Sequence[str],
    persisted_outer_test_ids: Sequence[str],
) -> None:
    """Validate persisted inner assignments against the frozen outer split."""
    partitions = {
        "outer-development": tuple(outer_development_ids),
        "outer-test": tuple(outer_test_ids),
        "inner-train": tuple(inner_train_ids),
        "inner-validation": tuple(inner_validation_ids),
    }
    for partition_name, patient_ids in partitions.items():
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError(f"fold {fold_index} {partition_name} contains duplicate patient IDs")

    outer_development_set = set(outer_development_ids)
    outer_test_set = set(outer_test_ids)
    inner_train_set = set(inner_train_ids)
    inner_validation_set = set(inner_validation_ids)

    if outer_development_set != set(persisted_outer_train_ids):
        raise ValueError(f"fold {fold_index} inner manifest disagrees with outer train IDs")
    if outer_test_set != set(persisted_outer_test_ids):
        raise ValueError(f"fold {fold_index} inner manifest disagrees with outer test IDs")
    if inner_train_set & inner_validation_set:
        raise ValueError(f"fold {fold_index} inner train and validation patient IDs overlap")
    if inner_train_set & outer_test_set:
        raise ValueError(f"fold {fold_index} inner train overlaps outer test")
    if inner_validation_set & outer_test_set:
        raise ValueError(f"fold {fold_index} inner validation overlaps outer test")
    if inner_train_set | inner_validation_set != outer_development_set:
        raise ValueError(f"fold {fold_index} inner train/validation do not cover development IDs")


def select_phase7_folds(
    *,
    folds: Sequence[Phase7FoldSplit],
    fold_index: int | None,
) -> tuple[Phase7FoldSplit, ...]:
    """Select one fold or all folds in deterministic persisted order."""
    sorted_folds = tuple(sorted(folds, key=lambda fold: fold.fold_index))
    if fold_index is None:
        return sorted_folds
    selected = tuple(fold for fold in sorted_folds if fold.fold_index == fold_index)
    if not selected:
        available = ", ".join(str(fold.fold_index) for fold in sorted_folds)
        raise ValueError(f"Fold index {fold_index} is not available; expected one of: {available}")
    return selected


def validate_phase7_training_contract(
    *,
    project_config: Mapping[str, object],
    training_config: Mapping[str, object],
) -> None:
    """Fail before training if runtime config drifts from the frozen Phase 7 contract."""
    expected_values: dict[str, object] = {
        "initialization_strategy": "pretrained",
        "batch_size": 2,
        "num_epochs": 50,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "optimizer": "adamw",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "num_workers": 0,
        "checkpoint_dir": EXPECTED_PHASE7_CHECKPOINT_ROOT.as_posix(),
        "device": "cuda",
    }
    if require_int(project_config, "seed") != 42:
        raise ValueError("Phase 7 requires project.seed == 42")
    for key, expected in expected_values.items():
        actual = training_config.get(key)
        if actual != expected:
            raise ValueError(f"Phase 7 requires training.{key} == {expected!r}; got {actual!r}")


def validate_cuda_runtime_preconditions(training_config: Mapping[str, object]) -> None:
    """Validate CUDA and cuBLAS preconditions before any Phase 7 training starts."""
    device = require_str(training_config, "device")
    if device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 requires CUDA, but torch.cuda.is_available() is false")
    actual_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas_config != EXPECTED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Phase 7 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before training starts"
        )


def validate_no_completed_phase7_artifacts(artifacts: Phase7FoldArtifacts) -> None:
    """Prevent accidental overwrite of completed Phase 7 fold artifacts."""
    existing_paths = [
        path
        for path in (
            artifacts.checkpoint_path,
            artifacts.oof_predictions_path,
            artifacts.summary_path,
        )
        if path.exists()
    ]
    if existing_paths:
        formatted_paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Completed Phase 7 fold artifacts already exist: {formatted_paths}")


def phase7_artifacts(
    *,
    checkpoint_root: Path,
    fold_index: int,
    initialization_strategy: InitializationStrategy,
) -> Phase7FoldArtifacts:
    """Build deterministic Phase 7 artifact paths for one fold."""
    expected_suffix = EXPECTED_PHASE7_CHECKPOINT_ROOT.as_posix()
    if checkpoint_root.as_posix().endswith("artifacts/checkpoints/phase6"):
        raise ValueError("Phase 7 must not use Phase 6 checkpoint paths")
    if not checkpoint_root.as_posix().endswith(expected_suffix):
        raise ValueError(
            f"Phase 7 checkpoint root must end with {expected_suffix}, got {checkpoint_root}"
        )
    output_dir = checkpoint_root / f"fold_{fold_index}" / initialization_strategy
    return Phase7FoldArtifacts(
        output_dir=output_dir,
        checkpoint_path=output_dir / BEST_CHECKPOINT_FILENAME,
        oof_predictions_path=output_dir / PHASE7_OOF_PREDICTIONS_FILENAME,
        summary_path=output_dir / PHASE7_SUMMARY_FILENAME,
    )


def build_loader(
    patients: Sequence[AcdcPatient],
    preprocessor: PatientPreprocessor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    augmentation: TrainingAugmentation | None,
) -> DataLoader[object]:
    """Build a deterministic DataLoader over preprocessed patient samples."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = PreprocessedPatientDataset(
        patients=patients,
        preprocessor=preprocessor,
        augmentation=augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def evaluate_outer_test(
    *,
    model: nn.Module,
    patients: Sequence[AcdcPatient],
    preprocessor: PatientPreprocessor,
    device: torch.device,
    num_classes: int,
    fold_index: int,
) -> tuple[Phase7PredictionRecord, ...]:
    """Evaluate one outer-test partition after loading the selected checkpoint."""
    dataset = PreprocessedPatientDataset(
        patients=patients,
        preprocessor=preprocessor,
        augmentation=None,
    )
    model.to(device)
    model.eval()
    records: list[Phase7PredictionRecord] = []
    with torch.inference_mode():
        for index, patient in enumerate(patients):
            inputs, target = unpack_batch(dataset[index])
            logits = model(inputs.unsqueeze(0).to(device))
            validate_logits(logits)
            if logits.shape[1] != num_classes:
                raise ValueError(f"logits must contain {num_classes} class scores")
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_index = int(probabilities.argmax(dim=1).detach().cpu().item())
            records.append(
                Phase7PredictionRecord(
                    fold_index=fold_index,
                    patient_id=patient.patient_id,
                    true_class_index=int(target.detach().cpu().item()),
                    predicted_class_index=predicted_class_index,
                    logits=tuple(float(value) for value in logits.squeeze(0).detach().cpu()),
                    probabilities=tuple(
                        float(value) for value in probabilities.squeeze(0).detach().cpu()
                    ),
                )
            )
    return tuple(records)


def load_best_checkpoint(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load the best validation-Macro-F1 checkpoint into a fresh model."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Malformed checkpoint payload: {checkpoint_path}")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError(f"Checkpoint lacks model_state_dict: {checkpoint_path}")
    model.load_state_dict(cast(dict[str, Tensor], model_state_dict), strict=True)


def save_oof_predictions(
    path: Path,
    prediction_records: Sequence[Phase7PredictionRecord],
) -> Path:
    """Persist deterministic patient-level OOF prediction records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.as_metadata() for record in prediction_records], indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_phase7_summary(
    *,
    path: Path,
    fold_split: Phase7FoldSplit,
    history: TrainingHistory,
    outer_test_accuracy: float,
    outer_test_macro_f1: float,
    checkpoint_path: Path,
    seed: int,
    initialization_strategy: InitializationStrategy,
    initialization_report: CompatibilityReport,
    training_config: Mapping[str, object],
) -> Path:
    """Persist per-fold Phase 7 metadata after outer-test evaluation."""
    payload = {
        "fold_index": fold_split.fold_index,
        "inner_train_patient_ids": list(fold_split.inner_train_patient_ids),
        "inner_validation_patient_ids": list(fold_split.inner_validation_patient_ids),
        "outer_test_patient_ids": list(fold_split.outer_test_patient_ids),
        "best_epoch": history.best_epoch,
        "best_validation_macro_f1": history.best_validation_macro_f1,
        "outer_test_accuracy": outer_test_accuracy,
        "outer_test_macro_f1": outer_test_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "seed": seed,
        "initialization_strategy": initialization_strategy,
        "initialization_report": initialization_report.as_metadata(),
        "training_config": dict(training_config),
        "history": history.as_metadata(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_optimizer(
    *,
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """Build the frozen Phase 7 optimizer."""
    if optimizer_name.lower() != "adamw":
        raise ValueError("Only adamw is supported by Phase 7")
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def patients_for_ids(
    patient_ids: Sequence[str],
    patient_by_id: Mapping[str, AcdcPatient],
) -> tuple[AcdcPatient, ...]:
    """Select patients by identifier while preserving persisted order."""
    missing_ids = [patient_id for patient_id in patient_ids if patient_id not in patient_by_id]
    if missing_ids:
        raise ValueError(f"Unknown patient IDs: {missing_ids}")
    return tuple(patient_by_id[patient_id] for patient_id in patient_ids)


def preprocessing_config_from_mapping(config: Mapping[str, object]) -> PreprocessingConfig:
    """Construct the frozen preprocessing configuration from parsed YAML."""
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


def training_augmentation_from_mapping(config: Mapping[str, object]) -> TrainingAugmentation | None:
    """Construct the configured training-only augmentation pipeline."""
    training = require_mapping(config, "training")
    augmentation = require_mapping(training, "augmentation")

    enabled = augmentation.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("training.augmentation.enabled must be a boolean")
    if not enabled:
        return None

    preprocessing = require_mapping(config, "preprocessing")
    target_spacing = require_mapping(preprocessing, "target_spacing_mm")
    spacing_x = require_float(target_spacing, "x")
    spacing_y = require_float(target_spacing, "y")

    if spacing_x != spacing_y:
        raise ValueError("TrainingAugmentation currently requires equal X/Y target spacing")

    augmentation_config = TrainingAugmentationConfig(
        rotation_degrees=require_float(augmentation, "rotation_degrees"),
        translation_mm=require_float(augmentation, "translation_mm"),
        intensity_scale_min=require_float(augmentation, "intensity_scale_min"),
        intensity_scale_max=require_float(augmentation, "intensity_scale_max"),
        gaussian_noise_std=require_float(augmentation, "gaussian_noise_std"),
        transform_probability=require_float(augmentation, "transform_probability"),
    )
    return TrainingAugmentation(config=augmentation_config, in_plane_spacing_mm=spacing_x)


def reset_training_rng_after_initialization(seed: int) -> None:
    """Reset stochastic training RNG streams after initialization."""
    set_deterministic_seed(seed)


def load_yaml_mapping(path: Path) -> Mapping[str, object]:
    """Load a YAML file as a string-keyed mapping."""
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read a nested mapping from a parsed configuration mapping."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    if not all(isinstance(child_key, str) for child_key in value):
        raise ValueError(f"{key} must contain string keys")
    return value


def require_str(mapping: Mapping[str, object], key: str) -> str:
    """Read a required string value from a mapping."""
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def require_int(mapping: Mapping[str, object], key: str) -> int:
    """Read a required integer value from a mapping."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def require_float(mapping: Mapping[str, object], key: str) -> float:
    """Read a required numeric value from a mapping."""
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def require_initialization_strategy(mapping: Mapping[str, object]) -> InitializationStrategy:
    """Read the configured initialization strategy."""
    value = require_str(mapping, "initialization_strategy")
    if value == "random":
        return "random"
    if value == "pretrained":
        return "pretrained"
    raise ValueError("initialization_strategy must be random or pretrained")


def require_json_mapping(value: object, path: Path) -> Mapping[str, object]:
    """Return a parsed JSON value as a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def require_json_int(mapping: Mapping[str, object], key: str, path: Path) -> int:
    """Read a required integer from a JSON mapping."""
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer in {path}")
    return value


def require_json_str_tuple(
    mapping: Mapping[str, object],
    key: str,
    path: Path,
) -> tuple[str, ...]:
    """Read a required string tuple from a JSON mapping."""
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings in {path}")
    return tuple(sorted(value, key=patient_id_sort_key))
