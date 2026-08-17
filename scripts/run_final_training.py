"""Run frozen Phase 9 final production training."""

import argparse
from pathlib import Path

from cardiac_pathology.training.phase9_production_training import Phase9ProductionTrainer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
DEFAULT_CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"


def main() -> None:
    """Run Phase 9 final production training."""
    args = parse_args()
    trainer = Phase9ProductionTrainer(
        repository_root=REPOSITORY_ROOT,
        dataset_dir=args.dataset_dir,
        config_path=args.config,
        class_mapping_path=args.class_mapping,
    )
    result = trainer.run()
    print(f"Saved Phase 9 classifier: {result.classifier_path}")
    print(f"Saved Phase 9 training summary: {result.summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse Phase 9 final-training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASS_MAPPING_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
