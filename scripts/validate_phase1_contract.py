"""Validate deterministic Phase 1 contract consistency."""

import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = {
    "0": "NOR",
    "1": "DCM",
    "2": "HCM",
    "3": "MINF",
    "4": "RV",
}
EXPECTED_STAGE_STRIDES = [
    [1, 1, 1],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]
EXPECTED_FEATURE_SHAPES = {
    "input": [14, 144, 144],
    "stem": [14, 72, 72],
    "layer1": [14, 72, 72],
    "layer2": [14, 36, 36],
    "layer3": [7, 18, 18],
    "layer4": [4, 9, 9],
}


def main() -> None:
    """Run deterministic Phase 1 contract checks."""
    class_mapping = json.loads(
        (REPOSITORY_ROOT / "configs/class_mapping.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    preprocessing_contract = (
        REPOSITORY_ROOT / "docs/preprocessing_contract.md"
    ).read_text(encoding="utf-8")
    onnx_contract = (
        REPOSITORY_ROOT / "docs/onnx_deployment_contract.md"
    ).read_text(encoding="utf-8")

    assert class_mapping == EXPECTED_CLASSES, class_mapping
    assert config["preprocessing"]["target_shape"] == {"d": 14, "h": 144, "w": 144}
    assert config["preprocessing"]["target_spacing_mm"] == {
        "z": 7.5,
        "y": 1.5,
        "x": 1.5,
    }
    assert config["model"]["input_channels"] == 2
    assert config["model"]["stage_strides"] == EXPECTED_STAGE_STRIDES
    assert "[2, 14, 144, 144]" in preprocessing_contract
    assert "[N, 2, 14, 144, 144]" in preprocessing_contract
    assert "[N, 2, 14, 144, 144]" in onnx_contract

    feature_shapes = calculate_feature_shapes(
        input_shape=EXPECTED_FEATURE_SHAPES["input"],
        stem_stride=config["model"]["stem"]["stride"],
        stage_strides=config["model"]["stage_strides"],
    )
    assert feature_shapes == EXPECTED_FEATURE_SHAPES, feature_shapes

    print("Phase 1 contract validation: PASS")
    print("Feature-map shapes:")
    for name, shape in feature_shapes.items():
        print(f"  {name}: {shape}")


def calculate_feature_shapes(
    input_shape: list[int],
    stem_stride: list[int],
    stage_strides: list[list[int]],
) -> dict[str, list[int]]:
    """Calculate frozen feature-map shapes from integer stride geometry."""
    shapes = {"input": input_shape}
    current = downsample_shape(input_shape, stem_stride)
    shapes["stem"] = current
    for index, stride in enumerate(stage_strides, start=1):
        current = downsample_shape(current, stride)
        shapes[f"layer{index}"] = current
    return shapes


def downsample_shape(shape: list[int], stride: list[int]) -> list[int]:
    """Apply same-padded convolution stride geometry as ceil(size / stride)."""
    return [(size + step - 1) // step for size, step in zip(shape, stride, strict=True)]


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phase 1 contract validation: FAIL ({error})", file=sys.stderr)
        raise
