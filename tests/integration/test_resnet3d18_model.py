"""Integration tests for Phase 5 3D ResNet model components."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from cardiac_pathology.models.basic_block_3d import BasicBlock3D
from cardiac_pathology.models.resnet3d18 import ResNet3D18

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPOSITORY_ROOT / "tests/fixtures/golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
GOLDEN_PATIENT_ID = "patient001"


def load_golden_patient_tensor(patient_id: str = GOLDEN_PATIENT_ID) -> Tensor:
    """Load one real preprocessed ACDC tensor from the golden fixture manifest.

    Args:
        patient_id: Real ACDC patient identifier present in the golden manifest.

    Returns:
        Float tensor with shape ``[2, 14, 144, 144]`` in model order
        ``[C, D, H, W]``.
    """
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    matching_patients = [
        patient for patient in manifest["patients"] if patient["patient_id"] == patient_id
    ]
    if not matching_patients:
        raise ValueError(f"Golden fixture missing patient: {patient_id}")

    tensor_path = GOLDEN_DIR / matching_patients[0]["tensor_filename"]
    array = np.load(tensor_path, allow_pickle=False)
    return torch.from_numpy(array)


def collect_feature_depths(model: ResNet3D18, batch: Tensor) -> dict[str, int]:
    """Collect feature-map depths from one forward pass through ResNet3D18.

    Args:
        model: ResNet3D18 instance to inspect.
        batch: Real ACDC-derived input batch with shape ``[N, 2, 14, 144, 144]``.

    Returns:
        Mapping from model stage name to depth dimension after that stage.
    """
    depths = {"input": int(batch.shape[2])}
    handles: list[RemovableHandle] = []

    def record_depth(stage_name: str) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        """Create a forward hook that records a tensor output depth.

        Args:
            stage_name: Name to use in the collected depth mapping.

        Returns:
            Forward hook compatible with PyTorch modules.
        """

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            """Record the output depth for one module invocation."""
            if not isinstance(output, Tensor):
                raise TypeError(f"{stage_name} output must be a Tensor")
            depths[stage_name] = int(output.shape[2])

        return hook

    for stage_name, module in [
        ("stem", model.relu),
        ("maxpool", model.maxpool),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
    ]:
        handles.append(module.register_forward_hook(record_depth(stage_name)))

    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()

    return depths


def test_golden_patient_tensor_loads_with_frozen_model_shape() -> None:
    """A real preprocessed ACDC golden tensor loads with the frozen model shape."""
    tensor = load_golden_patient_tensor()

    assert tensor.shape == (2, 14, 144, 144)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_resnet3d18_accepts_real_golden_tensor_and_returns_finite_logits() -> None:
    """ResNet3D18 maps one real golden ACDC tensor batch to finite raw logits."""
    patient_tensor = load_golden_patient_tensor()
    batch = patient_tensor.unsqueeze(0)
    model = ResNet3D18()
    model.eval()

    with torch.no_grad():
        logits = model(batch)

    assert batch.shape == (1, 2, 14, 144, 144)
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()
    assert not any(isinstance(module, nn.Softmax) for module in model.modules())

    assert isinstance(model.conv1, nn.Conv3d)
    assert model.conv1.in_channels == 2
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 5


def test_resnet3d18_anisotropic_depth_policy_for_frozen_geometry() -> None:
    """ResNet3D18 preserves depth until layer4 for the frozen ACDC input geometry."""
    patient_tensor = load_golden_patient_tensor()
    batch = patient_tensor.unsqueeze(0)
    model = ResNet3D18()
    model.eval()

    assert model.conv1.stride == (1, 2, 2)
    assert model.maxpool.stride == (1, 2, 2)
    assert isinstance(model.layer1[0], BasicBlock3D)
    assert isinstance(model.layer2[0], BasicBlock3D)
    assert isinstance(model.layer3[0], BasicBlock3D)
    assert isinstance(model.layer4[0], BasicBlock3D)
    assert model.layer1[0].conv1.stride == (1, 1, 1)
    assert model.layer2[0].conv1.stride == (1, 2, 2)
    assert model.layer3[0].conv1.stride == (1, 2, 2)
    assert model.layer4[0].conv1.stride == (2, 2, 2)

    depths = collect_feature_depths(model, batch)

    assert depths == {
        "input": 14,
        "stem": 14,
        "maxpool": 14,
        "layer1": 14,
        "layer2": 14,
        "layer3": 14,
        "layer4": 7,
    }


def test_basic_block_3d_preserves_real_tensor_residual_identity_when_main_path_is_zero() -> None:
    """BasicBlock3D adds the residual identity for a real ACDC-derived tensor."""
    patient_tensor = load_golden_patient_tensor()
    feature_batch = patient_tensor.unsqueeze(0)
    block = BasicBlock3D(in_channels=2, out_channels=2)
    block.eval()

    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
        output = block(feature_batch)

    expected = torch.relu(feature_batch)
    assert output.shape == feature_batch.shape
    assert torch.equal(output, expected)
    assert torch.isfinite(output).all()
