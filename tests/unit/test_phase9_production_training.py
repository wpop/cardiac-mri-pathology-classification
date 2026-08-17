"""Unit tests for Phase 9 production-training contract safeguards."""

import copy
import json
from collections.abc import MutableMapping
from pathlib import Path

import pytest
import torch
from torch import nn

from cardiac_pathology.training.phase7_cross_validation import load_yaml_mapping, require_mapping
from cardiac_pathology.training.phase9_production_training import (
    PHASE9_FINAL_EPOCHS,
    PHASE9_INPUT_CHANNELS,
    PHASE9_NUM_CLASSES,
    PHASE9_SEED,
    PREPROCESSING_CONTRACT_PATH,
    Phase9ProductionArtifacts,
    load_phase9_patient_ids,
    save_phase9_classifier,
    validate_no_phase9_artifacts,
    validate_phase9_training_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"


def test_load_phase9_patient_ids_returns_complete_ordered_production_set(
    tmp_path: Path,
) -> None:
    """A valid five-fold outer manifest yields the deterministic 100-patient set."""
    split_path = write_phase9_split_manifest(tmp_path)

    patient_ids = load_phase9_patient_ids(split_path)

    assert len(patient_ids) == 100
    assert len(set(patient_ids)) == 100
    assert patient_ids == tuple(f"patient{index:03d}" for index in range(1, 101))


def test_load_phase9_patient_ids_rejects_outer_test_duplicates(tmp_path: Path) -> None:
    """A duplicated outer-test patient ID fails before production training."""
    split_path = write_phase9_split_manifest(tmp_path)
    artifact = json.loads(split_path.read_text(encoding="utf-8"))
    artifact["folds"][1]["test_patient_ids"][0] = "patient001"
    duplicated_test_ids = set(artifact["folds"][1]["test_patient_ids"])
    artifact["folds"][1]["train_patient_ids"] = [
        f"patient{index:03d}"
        for index in range(1, 101)
        if f"patient{index:03d}" not in duplicated_test_ids
    ]
    split_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate patient IDs"):
        load_phase9_patient_ids(split_path)


def test_frozen_phase9_training_contract_accepts_default_config() -> None:
    """The current frozen repository config satisfies the Phase 9 contract."""
    config = load_default_config()

    validate_phase9_training_contract(
        config=config,
        project_config=require_mapping(config, "project"),
        training_config=require_mapping(config, "training"),
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("project", "seed"), 7, "project.seed"),
        (("training", "initialization_strategy"), "random", "training.initialization_strategy"),
        (("preprocessing", "target_shape", "d"), 16, "preprocessing.target_shape.d"),
        (("training", "augmentation", "rotation_degrees"), 10.0, "rotation_degrees"),
        (("training", "device"), "cpu", "training.device"),
    ],
)
def test_phase9_training_contract_rejects_config_drift(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    """Representative contract drift fails fast with a clear ValueError."""
    config = copy.deepcopy(load_default_config())
    set_nested_value(config, path, value)

    with pytest.raises(ValueError, match=message):
        validate_phase9_training_contract(
            config=config,
            project_config=require_mapping(config, "project"),
            training_config=require_mapping(config, "training"),
        )


@pytest.mark.parametrize("existing_filename", ["classifier.pt", "phase9_training_summary.json"])
def test_phase9_artifact_overwrite_protection(tmp_path: Path, existing_filename: str) -> None:
    """Existing Phase 9 classifier or summary artifacts block a new run."""
    output_dir = tmp_path / "artifacts/checkpoints/phase9"
    output_dir.mkdir(parents=True)
    existing_path = output_dir / existing_filename
    existing_path.write_bytes(b"existing artifact")

    with pytest.raises(FileExistsError, match=existing_filename):
        validate_no_phase9_artifacts(
            Phase9ProductionArtifacts(
                output_dir=output_dir,
                classifier_path=output_dir / "classifier.pt",
                summary_path=output_dir / "phase9_training_summary.json",
            )
        )


def test_save_phase9_classifier_writes_contract_payload(tmp_path: Path) -> None:
    """The Phase 9 classifier payload stores model state and contract metadata only."""
    path = tmp_path / "classifier.pt"
    model = nn.Linear(2, 5)
    patient_ids = tuple(f"patient{index:03d}" for index in range(1, 101))
    class_distribution = {"NOR": 20, "DCM": 20, "HCM": 20, "MINF": 20, "RV": 20}
    class_mapping = {0: "NOR", 1: "DCM", 2: "HCM", 3: "MINF", 4: "RV"}

    save_phase9_classifier(
        path=path,
        model=model,
        patient_ids=patient_ids,
        class_distribution=class_distribution,
        class_mapping=class_mapping,
    )

    payload = torch.load(path, map_location=torch.device("cpu"))
    assert path.is_file()
    assert "model_state_dict" in payload
    assert payload["architecture"] == "ResNet3D18"
    assert payload["input_channels"] == PHASE9_INPUT_CHANNELS
    assert payload["num_classes"] == PHASE9_NUM_CLASSES
    assert payload["initialization_strategy"] == "pretrained"
    assert payload["seed"] == PHASE9_SEED
    assert payload["final_epochs"] == PHASE9_FINAL_EPOCHS
    assert payload["patient_count"] == 100
    assert payload["class_distribution"] == class_distribution
    assert payload["class_mapping"] == class_mapping
    assert payload["preprocessing_contract"] == PREPROCESSING_CONTRACT_PATH
    assert "optimizer_state_dict" not in payload


def write_phase9_split_manifest(tmp_path: Path) -> Path:
    """Write a minimal valid five-fold Phase 9 split manifest."""
    all_patient_ids = [f"patient{index:03d}" for index in range(1, 101)]
    folds = []
    for fold_index in range(5):
        start = fold_index * 20
        test_patient_ids = all_patient_ids[start : start + 20]
        test_patient_set = set(test_patient_ids)
        folds.append(
            {
                "fold_index": fold_index,
                "train_patient_ids": [
                    patient_id
                    for patient_id in all_patient_ids
                    if patient_id not in test_patient_set
                ],
                "test_patient_ids": test_patient_ids,
            }
        )

    split_path = tmp_path / "acdc_5fold_seed42.json"
    split_path.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    return split_path


def load_default_config() -> MutableMapping[str, object]:
    """Load the repository default config as a mutable mapping for drift tests."""
    return dict(load_yaml_mapping(DEFAULT_CONFIG_PATH))


def set_nested_value(
    mapping: MutableMapping[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    """Set one nested config value for a representative drift case."""
    current = mapping
    for key in path[:-1]:
        next_value = current[key]
        if not isinstance(next_value, dict):
            raise TypeError(f"{key} must contain a nested mapping")
        current = next_value
    current[path[-1]] = value
