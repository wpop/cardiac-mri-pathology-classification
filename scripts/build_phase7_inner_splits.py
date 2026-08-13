"""Build deterministic Phase 7 inner validation split artifact."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from cardiac_pathology.training import Phase7SplitManifestBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
OUTER_SPLIT_PATH = REPOSITORY_ROOT / "artifacts/dataset_splits/acdc_5fold_seed42.json"
OUTPUT_PATH = REPOSITORY_ROOT / "artifacts/dataset_splits/acdc_5fold_inner_validation_seed42.json"


def main() -> None:
    """Build and save deterministic Phase 7 inner validation splits."""
    config = load_yaml(CONFIG_PATH)
    project_config = require_mapping(config, "project")
    training_config = require_mapping(config, "training")

    seed = require_int(project_config, "seed")
    validation_fraction = require_float(training_config, "validation_fraction")

    builder = Phase7SplitManifestBuilder(
        dataset_dir=DATASET_DIR,
        class_mapping_path=CLASS_MAPPING_PATH,
        outer_split_path=OUTER_SPLIT_PATH,
        outer_split_reference=str(OUTER_SPLIT_PATH.relative_to(REPOSITORY_ROOT)),
        output_path=OUTPUT_PATH,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    output_path = builder.build_and_save()
    print(f"Saved: {output_path}")


def load_yaml(path: Path) -> Mapping[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in data):
        raise ValueError(f"Expected string keys in {path}")

    return data


def require_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return one required nested mapping."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    if not all(isinstance(child_key, str) for child_key in value):
        raise ValueError(f"{key} must contain string keys")
    return value


def require_int(mapping: Mapping[str, Any], key: str) -> int:
    """Return one required integer configuration value."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def require_float(mapping: Mapping[str, Any], key: str) -> float:
    """Return one required numeric configuration value as float."""
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


if __name__ == "__main__":
    main()
