"""Unit tests for Phase 1 contract validator architecture checks."""

from pathlib import Path

import yaml

from cardiac_pathology.models.resnet3d18 import ResNet3D18
from scripts.validate_phase1_contract import (
    EXPECTED_FEATURE_SHAPES,
    EXPECTED_INPUT_CHANNELS,
    EXPECTED_STAGE_STRIDES,
    actual_stage_strides,
    collect_model_feature_shapes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_phase1_contract_stage_strides_match_actual_resnet3d18() -> None:
    """Config stride metadata matches the committed ResNet3D18 implementation."""
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    model = ResNet3D18()

    assert config["model"]["stage_strides"] == EXPECTED_STAGE_STRIDES
    assert config["model"]["stage_strides"] == actual_stage_strides(model)


def test_phase7_training_runtime_config_uses_cuda_checkpoint_contract() -> None:
    """Default training runtime follows the Phase 7 CUDA execution contract."""
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))

    assert config["training"]["checkpoint_dir"] == "artifacts/checkpoints/phase7"
    assert config["training"]["device"] == "cuda"


def test_phase1_contract_feature_shapes_include_maxpool_geometry() -> None:
    """Validator feature-map shapes are derived from the real module stack."""
    model = ResNet3D18()

    feature_shapes = collect_model_feature_shapes(
        model=model,
        input_shape=[
            EXPECTED_INPUT_CHANNELS,
            *EXPECTED_FEATURE_SHAPES["input"],
        ],
    )

    assert feature_shapes == EXPECTED_FEATURE_SHAPES
    assert feature_shapes["maxpool"] == [14, 36, 36]
    assert feature_shapes["layer4"] == [7, 5, 5]
