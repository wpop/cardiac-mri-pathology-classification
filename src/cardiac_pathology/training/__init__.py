"""Training infrastructure for cardiac MRI pathology classification."""

from cardiac_pathology.training.augmentation_config import TrainingAugmentationConfig
from cardiac_pathology.training.dataset import PreprocessedPatientDataset
from cardiac_pathology.training.early_stopping import EarlyStopping
from cardiac_pathology.training.initialization import (
    CompatibilityReport,
    InitializationStrategy,
    ParameterCompatibility,
    adapt_rgb_stem_to_two_channels,
    apply_pretrained_initialization,
    initialize_model,
    set_deterministic_seed,
)
from cardiac_pathology.training.phase7_cross_validation import (
    Phase7CrossValidationOrchestrator,
    Phase7FoldArtifacts,
    Phase7FoldResult,
    Phase7FoldRunner,
    Phase7FoldSplit,
    Phase7PredictionRecord,
)
from cardiac_pathology.training.phase7_split_manifest_builder import Phase7SplitManifestBuilder
from cardiac_pathology.training.splits import (
    InnerValidationSplit,
    create_inner_validation_split,
    load_outer_folds,
    validate_inner_split,
)
from cardiac_pathology.training.trainer import (
    EpochMetrics,
    Trainer,
    TrainerConfig,
    TrainingHistory,
    accuracy_and_macro_f1,
)
from cardiac_pathology.training.training_augmentation import TrainingAugmentation

__all__ = [
    "CompatibilityReport",
    "EarlyStopping",
    "EpochMetrics",
    "InitializationStrategy",
    "InnerValidationSplit",
    "ParameterCompatibility",
    "Phase7SplitManifestBuilder",
    "Phase7CrossValidationOrchestrator",
    "Phase7FoldArtifacts",
    "Phase7FoldResult",
    "Phase7FoldRunner",
    "Phase7FoldSplit",
    "Phase7PredictionRecord",
    "PreprocessedPatientDataset",
    "Trainer",
    "TrainerConfig",
    "TrainingAugmentation",
    "TrainingAugmentationConfig",
    "TrainingHistory",
    "accuracy_and_macro_f1",
    "adapt_rgb_stem_to_two_channels",
    "apply_pretrained_initialization",
    "create_inner_validation_split",
    "initialize_model",
    "load_outer_folds",
    "set_deterministic_seed",
    "validate_inner_split",
]
