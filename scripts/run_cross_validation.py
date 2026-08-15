"""Run frozen Phase 7 cross-validation folds."""

import argparse
from pathlib import Path

from cardiac_pathology.training import Phase7CrossValidationOrchestrator
from cardiac_pathology.training.phase7_cross_validation import (
    DEFAULT_INNER_SPLIT_PATH,
    DEFAULT_OUTER_SPLIT_PATH,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
DEFAULT_CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"


def main() -> None:
    """Run the requested frozen Phase 7 fold or all persisted folds."""
    args = parse_args()
    orchestrator = Phase7CrossValidationOrchestrator(
        repository_root=REPOSITORY_ROOT,
        dataset_dir=args.dataset_dir,
        config_path=args.config,
        class_mapping_path=args.class_mapping,
        outer_split_path=args.outer_split_artifact,
        inner_split_path=args.inner_split_artifact,
    )
    results = orchestrator.run(fold_index=args.fold_index)
    for result in results:
        print(f"Saved Phase 7 summary: {result.summary_path}")
        print(f"Saved Phase 7 OOF predictions: {result.oof_predictions_path}")


def parse_args() -> argparse.Namespace:
    """Parse Phase 7 cross-validation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument(
        "--outer-split-artifact",
        type=Path,
        default=REPOSITORY_ROOT / DEFAULT_OUTER_SPLIT_PATH,
    )
    parser.add_argument(
        "--inner-split-artifact",
        type=Path,
        default=REPOSITORY_ROOT / DEFAULT_INNER_SPLIT_PATH,
    )
    parser.add_argument("--fold-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
