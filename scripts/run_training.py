"""Run one Phase 6 training job on real preprocessed ACDC patients."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from cardiac_pathology.data import AcdcDatasetIndexer, AcdcPatient
from cardiac_pathology.models.resnet3d18 import ResNet3D18
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig
from cardiac_pathology.training import (
    EarlyStopping,
    InitializationStrategy,
    PreprocessedPatientDataset,
    Trainer,
    TrainerConfig,
    TrainingAugmentation,
    TrainingAugmentationConfig,
    TrainingHistory,
    create_inner_validation_split,
    initialize_model,
    load_outer_folds,
    set_deterministic_seed,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
DEFAULT_CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
DEFAULT_SPLIT_ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/dataset_splits/acdc_5fold_seed42.json"
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"


def main() -> None:
    """Run the configured single-fold training job."""
    args = parse_args()
    config = load_yaml_mapping(args.config)
    project_config = require_mapping(config, "project")
    data_config = require_mapping(config, "data")
    training_config = require_mapping(config, "training")
    model_config = require_mapping(config, "model")

    seed = require_int(project_config, "seed")
    set_deterministic_seed(seed)

    dataset_dir = dataset_dir_from_args(args.dataset_dir, data_config)
    indexer = AcdcDatasetIndexer(dataset_dir=dataset_dir, class_mapping_path=args.class_mapping)
    patients = indexer.index_patients()
    patient_by_id = {patient.patient_id: patient for patient in patients}

    folds = load_outer_folds(args.split_artifact)
    fold_index = require_int(training_config, "pilot_fold_index")
    selected_fold = next((fold for fold in folds if fold.fold_index == fold_index), None)
    if selected_fold is None:
        raise ValueError(f"Outer fold {fold_index} is not present in {args.split_artifact}")

    development_patients = patients_for_ids(selected_fold.train_patient_ids, patient_by_id)
    inner_split = create_inner_validation_split(
        development_patients=development_patients,
        outer_test_patient_ids=selected_fold.test_patient_ids,
        validation_fraction=require_float(training_config, "validation_fraction"),
        random_seed=seed,
    )
    train_patients = patients_for_ids(inner_split.train_patient_ids, patient_by_id)
    validation_patients = patients_for_ids(inner_split.validation_patient_ids, patient_by_id)

    preprocessing_config = preprocessing_config_from_mapping(config)
    preprocessor = PatientPreprocessor(preprocessing_config)
    training_augmentation = training_augmentation_from_mapping(config)
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

    model = ResNet3D18(
        input_channels=require_int(model_config, "input_channels"),
        num_classes=require_int(model_config, "num_classes"),
    )
    initialization_strategy = require_initialization_strategy(training_config)
    initialization_report = initialize_model(model, initialization_strategy)
    reset_training_rng_after_initialization(seed)

    experiment_dir = experiment_dir_for_run(
        checkpoint_root=Path(require_str(training_config, "checkpoint_dir")),
        fold_index=fold_index,
        initialization_strategy=initialization_strategy,
    )

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
            checkpoint_dir=experiment_dir,
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

    save_run_summary(
        output_dir=experiment_dir,
        fold_index=fold_index,
        seed=seed,
        inner_train_patient_ids=inner_split.train_patient_ids,
        inner_validation_patient_ids=inner_split.validation_patient_ids,
        training_config=training_config,
        initialization_strategy=initialization_strategy,
        initialization_report=initialization_report.as_metadata(),
        history=history,
    )

    print(f"Fold {fold_index} best validation Macro F1: {history.best_validation_macro_f1:.6f}")
    print(f"Best epoch: {history.best_epoch}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single training run.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument("--split-artifact", type=Path, default=DEFAULT_SPLIT_ARTIFACT_PATH)
    return parser.parse_args()


def dataset_dir_from_args(cli_dataset_dir: Path | None, data_config: Mapping[str, object]) -> Path:
    """Resolve the real ACDC dataset directory.

    Args:
        cli_dataset_dir: Optional command-line dataset directory.
        data_config: Parsed ``data`` configuration mapping.

    Returns:
        Dataset directory path.
    """
    if cli_dataset_dir is not None:
        return cli_dataset_dir
    configured_path = data_config.get("raw_data_dir")
    if configured_path is None:
        return DEFAULT_DATASET_DIR
    if not isinstance(configured_path, str):
        raise ValueError("data.raw_data_dir must be a string or null")
    return Path(configured_path)


def build_loader(
    patients: Sequence[AcdcPatient],
    preprocessor: PatientPreprocessor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    augmentation: TrainingAugmentation | None,
) -> DataLoader[object]:
    """Build a deterministic DataLoader over real preprocessed patient samples.

    Args:
        patients: Real indexed ACDC patients.
        preprocessor: Frozen patient preprocessing pipeline.
        batch_size: Number of patient samples per batch.
        shuffle: Whether to shuffle samples.
        num_workers: Number of DataLoader worker processes.
        seed: Torch generator seed.

    Returns:
        DataLoader yielding ``(inputs, targets)`` batches.
    """
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


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """Build the configured optimizer.

    Args:
        model: Model whose parameters will be optimized.
        optimizer_name: Optimizer name. Phase 6A supports ``adamw``.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.

    Returns:
        Configured optimizer.
    """
    if optimizer_name.lower() != "adamw":
        raise ValueError("Only adamw is supported by the Phase 6A training CLI")
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def patients_for_ids(
    patient_ids: Sequence[str],
    patient_by_id: Mapping[str, AcdcPatient],
) -> tuple[AcdcPatient, ...]:
    """Select patients by identifier while preserving the requested order.

    Args:
        patient_ids: Patient IDs to select.
        patient_by_id: Mapping from patient ID to indexed patient record.

    Returns:
        Tuple of selected patients.
    """
    return tuple(patient_by_id[patient_id] for patient_id in patient_ids)


def preprocessing_config_from_mapping(config: Mapping[str, object]) -> PreprocessingConfig:
    """Construct the frozen preprocessing configuration from parsed YAML.

    Args:
        config: Parsed repository configuration mapping.

    Returns:
        Preprocessing configuration consumed by the existing production
        preprocessor.
    """
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


def training_augmentation_from_mapping(
    config: Mapping[str, object],
) -> TrainingAugmentation | None:
    """Construct the configured training-only augmentation pipeline.

    Args:
        config: Parsed repository configuration mapping.

    Returns:
        Training augmentation when enabled, otherwise None.
    """
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

    return TrainingAugmentation(
        config=augmentation_config,
        in_plane_spacing_mm=spacing_x,
    )


def experiment_dir_for_run(
    checkpoint_root: Path,
    fold_index: int,
    initialization_strategy: InitializationStrategy,
) -> Path:
    """Return the experiment-specific checkpoint directory for one Phase 6 run.

    Args:
        checkpoint_root: Root checkpoint directory from the training config.
        fold_index: Outer pilot fold index.
        initialization_strategy: Model initialization strategy for this run.

    Returns:
        Directory dedicated to one fold and initialization strategy.
    """
    return checkpoint_root / f"fold_{fold_index}" / initialization_strategy


def reset_training_rng_after_initialization(seed: int) -> None:
    """Reset stochastic training RNG streams after model initialization.

    Strategy-specific initialization may consume different PyTorch RNG internally
    before training begins. Re-seeding here preserves already-initialized model
    parameters while making subsequent stochastic training operations comparable
    across initialization strategies.

    Args:
        seed: Experiment seed used for stochastic training operations.

    Returns:
        None.
    """
    set_deterministic_seed(seed)


def build_run_summary_payload(
    fold_index: int,
    seed: int,
    inner_train_patient_ids: Sequence[str],
    inner_validation_patient_ids: Sequence[str],
    training_config: Mapping[str, object],
    initialization_strategy: InitializationStrategy,
    initialization_report: Mapping[str, object],
    history: TrainingHistory,
) -> dict[str, object]:
    """Build JSON-compatible reproducibility metadata for one Phase 6 run.

    Args:
        fold_index: Outer fold used for the initialization pilot.
        seed: Training seed.
        inner_train_patient_ids: Patient IDs used for optimization.
        inner_validation_patient_ids: Patient IDs used for model selection.
        training_config: Frozen training configuration for the run.
        initialization_strategy: Model initialization strategy for this run.
        initialization_report: Initialization compatibility metadata.
        history: Complete epoch history returned by the trainer.

    Returns:
        JSON-compatible run summary payload.
    """
    history_metadata = history.as_metadata()
    return {
        "fold_index": fold_index,
        "seed": seed,
        "inner_train_patient_ids": list(inner_train_patient_ids),
        "inner_validation_patient_ids": list(inner_validation_patient_ids),
        "training_config": dict(training_config),
        "initialization_strategy": initialization_strategy,
        "initialization_report": dict(initialization_report),
        "history": history_metadata,
        "best_epoch": history.best_epoch,
        "best_validation_macro_f1": history.best_validation_macro_f1,
        "stopped_early": history.stopped_early,
    }


def save_run_summary(
    output_dir: Path,
    fold_index: int,
    seed: int,
    inner_train_patient_ids: Sequence[str],
    inner_validation_patient_ids: Sequence[str],
    training_config: Mapping[str, object],
    initialization_strategy: InitializationStrategy,
    initialization_report: Mapping[str, object],
    history: TrainingHistory,
) -> Path:
    """Persist reproducible metadata for one Phase 6 training run.

    Args:
        output_dir: Experiment-specific output directory.
        fold_index: Outer fold used for the initialization pilot.
        seed: Training seed.
        inner_train_patient_ids: Patient IDs used for optimization.
        inner_validation_patient_ids: Patient IDs used for model selection.
        training_config: Frozen training configuration for the run.
        initialization_strategy: Model initialization strategy for this run.
        initialization_report: Initialization compatibility metadata.
        history: Per-epoch training and validation history.

    Returns:
        Path to the written JSON summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"

    payload = build_run_summary_payload(
        fold_index=fold_index,
        seed=seed,
        inner_train_patient_ids=inner_train_patient_ids,
        inner_validation_patient_ids=inner_validation_patient_ids,
        training_config=training_config,
        initialization_strategy=initialization_strategy,
        initialization_report=initialization_report,
        history=history,
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    return summary_path


def load_yaml_mapping(path: Path) -> Mapping[str, object]:
    """Load a YAML file as a string-keyed mapping.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping.
    """
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


def require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read a nested mapping from a parsed configuration mapping.

    Args:
        mapping: Parent configuration mapping.
        key: Required child key.

    Returns:
        Child mapping.
    """
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    if not all(isinstance(child_key, str) for child_key in value):
        raise ValueError(f"{key} must contain string keys")
    return value


def require_str(mapping: Mapping[str, object], key: str) -> str:
    """Read a required string value from a mapping.

    Args:
        mapping: Parsed configuration mapping.
        key: Required key.

    Returns:
        String value.
    """
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def require_int(mapping: Mapping[str, object], key: str) -> int:
    """Read a required integer value from a mapping.

    Args:
        mapping: Parsed configuration mapping.
        key: Required key.

    Returns:
        Integer value.
    """
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def require_float(mapping: Mapping[str, object], key: str) -> float:
    """Read a required numeric value from a mapping.

    Args:
        mapping: Parsed configuration mapping.
        key: Required key.

    Returns:
        Floating-point value.
    """
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def require_initialization_strategy(mapping: Mapping[str, object]) -> InitializationStrategy:
    """Read the configured initialization strategy.

    Args:
        mapping: Parsed training configuration mapping.

    Returns:
        Initialization strategy, either ``random`` or ``pretrained``.
    """
    value = require_str(mapping, "initialization_strategy")
    if value == "random":
        return "random"
    if value == "pretrained":
        return "pretrained"
    raise ValueError("initialization_strategy must be random or pretrained")


if __name__ == "__main__":
    main()
