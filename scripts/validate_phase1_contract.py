"""Validate deterministic Phase 1 contract consistency."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from cardiac_pathology.models.resnet3d18 import ResNet3D18

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
    [1, 2, 2],
    [2, 2, 2],
]
EXPECTED_FEATURE_SHAPES = {
    "input": [14, 144, 144],
    "stem": [14, 72, 72],
    "maxpool": [14, 36, 36],
    "layer1": [14, 36, 36],
    "layer2": [14, 18, 18],
    "layer3": [14, 9, 9],
    "layer4": [7, 5, 5],
}
EXPECTED_INPUT_CHANNELS = 2
EXPECTED_NUM_CLASSES = 5
EXPECTED_STEM_STRIDE = [1, 2, 2]
EXPECTED_MAXPOOL_STRIDE = [1, 2, 2]


def main() -> None:
    """Run deterministic Phase 1 contract checks."""
    class_mapping = json.loads(
        (REPOSITORY_ROOT / "configs/class_mapping.json").read_text(encoding="utf-8")
    )
    config = load_yaml_mapping(REPOSITORY_ROOT / "configs/default.yaml")
    preprocessing_contract = (REPOSITORY_ROOT / "docs/preprocessing_contract.md").read_text(
        encoding="utf-8"
    )
    onnx_contract = (REPOSITORY_ROOT / "docs/onnx_deployment_contract.md").read_text(
        encoding="utf-8"
    )

    assert class_mapping == EXPECTED_CLASSES, class_mapping
    assert config["preprocessing"]["target_shape"] == {"d": 14, "h": 144, "w": 144}
    assert config["preprocessing"]["target_spacing_mm"] == {
        "z": 7.5,
        "y": 1.5,
        "x": 1.5,
    }
    assert config["model"]["input_channels"] == 2
    assert config["model"]["num_classes"] == 5
    assert config["model"]["stage_strides"] == EXPECTED_STAGE_STRIDES
    assert "[2, 14, 144, 144]" in preprocessing_contract
    assert "[N, 2, 14, 144, 144]" in preprocessing_contract
    assert "[N, 2, 14, 144, 144]" in onnx_contract

    model = ResNet3D18(
        input_channels=EXPECTED_INPUT_CHANNELS,
        num_classes=EXPECTED_NUM_CLASSES,
    )
    assert stride_as_list(model.conv1.stride) == EXPECTED_STEM_STRIDE
    assert stride_as_list(model.maxpool.stride) == EXPECTED_MAXPOOL_STRIDE
    assert actual_stage_strides(model) == EXPECTED_STAGE_STRIDES
    assert config["model"]["stem"]["stride"] == stride_as_list(model.conv1.stride)
    assert config["model"]["stage_strides"] == actual_stage_strides(model)

    feature_shapes = collect_model_feature_shapes(
        model=model,
        input_shape=[
            EXPECTED_INPUT_CHANNELS,
            *EXPECTED_FEATURE_SHAPES["input"],
        ],
    )
    assert feature_shapes == EXPECTED_FEATURE_SHAPES, feature_shapes

    print("Phase 1 contract validation: PASS")
    print("Feature-map shapes:")
    for name, shape in feature_shapes.items():
        print(f"  {name}: {shape}")


def actual_stage_strides(model: ResNet3D18) -> list[list[int]]:
    """Return first-block residual stage strides from the actual model modules."""
    return [
        stride_as_list(model.layer1[0].conv1.stride),
        stride_as_list(model.layer2[0].conv1.stride),
        stride_as_list(model.layer3[0].conv1.stride),
        stride_as_list(model.layer4[0].conv1.stride),
    ]


def stride_as_list(stride: int | Sequence[int]) -> list[int]:
    """Return a 3D stride as ``[D, H, W]``."""
    if isinstance(stride, int):
        return [stride, stride, stride]
    stride_list = list(stride)
    if len(stride_list) != 3:
        raise ValueError(f"Expected 3D stride, got {stride_list}")
    return stride_list


def collect_model_feature_shapes(
    model: ResNet3D18,
    input_shape: Sequence[int],
) -> dict[str, list[int]]:
    """Collect frozen feature-map shapes by executing the actual model modules."""
    model.eval()
    input_tensor = torch.zeros((1, *input_shape), dtype=torch.float32)
    shapes = {"input": spatial_shape(input_tensor)}

    with torch.no_grad():
        current: Tensor = model.conv1(input_tensor)
        current = model.bn1(current)
        current = model.relu(current)
        shapes["stem"] = spatial_shape(current)

        current = model.maxpool(current)
        shapes["maxpool"] = spatial_shape(current)

        current = model.layer1(current)
        shapes["layer1"] = spatial_shape(current)

        current = model.layer2(current)
        shapes["layer2"] = spatial_shape(current)

        current = model.layer3(current)
        shapes["layer3"] = spatial_shape(current)

        current = model.layer4(current)
        shapes["layer4"] = spatial_shape(current)

    return shapes


def spatial_shape(tensor: Tensor) -> list[int]:
    """Return ``[D, H, W]`` spatial dimensions from a 5D model tensor."""
    return [int(size) for size in tensor.shape[2:]]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys in {path}")
    return value


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phase 1 contract validation: FAIL ({error})", file=sys.stderr)
        raise
