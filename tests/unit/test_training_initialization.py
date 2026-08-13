"""Unit tests for Phase 6 initialization compatibility support."""

import torch
from torch import Tensor, nn

from cardiac_pathology.models.resnet3d18 import ResNet3D18
from cardiac_pathology.training.initialization import (
    adapt_rgb_stem_to_two_channels,
    apply_pretrained_initialization,
    initialize_model,
    source_name_for_target,
)


def test_rgb_stem_adaptation_uses_documented_three_to_two_channel_formula() -> None:
    """The pretrained RGB stem adaptation follows the exact Phase 6 formula."""
    source_weight = torch.arange(2 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 3, 2, 2)

    adapted = adapt_rgb_stem_to_two_channels(source_weight)
    expected = source_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1, 1) * (3.0 / 2.0)

    assert adapted.shape == (2, 2, 3, 2, 2)
    assert torch.equal(adapted, expected)


def test_pretrained_initialization_adapts_stem_and_never_imports_classifier_head() -> None:
    """Compatible pretrained tensors transfer while the classifier head is preserved."""
    model = ResNet3D18()
    original_fc_weight = model.fc.weight.detach().clone()
    original_fc_bias = model.fc.bias.detach().clone()
    source_state = build_compatible_source_state(model)

    report = apply_pretrained_initialization(model=model, source_state_dict=source_state)

    source_stem = source_state["stem.0.weight"]
    expected_stem = adapt_rgb_stem_to_two_channels(source_stem)
    assert torch.equal(model.conv1.weight.detach(), expected_stem)
    assert torch.equal(model.fc.weight.detach(), original_fc_weight)
    assert torch.equal(model.fc.bias.detach(), original_fc_bias)
    assert report.initialization_strategy == "pretrained"
    assert report.adapted_count == 1
    assert report.transferred_count > 0
    assert any(
        entry.target_name == "fc.weight" and entry.action == "skipped" for entry in report.entries
    )


def test_pretrained_compatibility_reports_missing_source_tensors() -> None:
    """Compatibility reporting records target tensors absent from the source state."""
    model = ResNet3D18()

    report = apply_pretrained_initialization(model=model, source_state_dict={})

    assert report.missing_count > 0
    assert report.skipped_count >= 2
    assert any(entry.action == "missing" for entry in report.entries)


def test_random_initialization_preserves_constructor_weights() -> None:
    """Random initialization strategy does not modify model constructor weights."""
    model = ResNet3D18()
    original_stem = model.conv1.weight.detach().clone()

    report = initialize_model(model=model, strategy="random")

    assert torch.equal(model.conv1.weight.detach(), original_stem)
    assert report.initialization_strategy == "random"
    assert report.transferred_count == 0
    assert report.adapted_count == 0


def build_compatible_source_state(model: nn.Module) -> dict[str, Tensor]:
    """Build deterministic source tensors for compatibility unit tests.

    Args:
        model: Target model whose state shapes define compatible test tensors.

    Returns:
        Source-style state dictionary with deterministic non-medical parameter
        tensors. No image tensor or medical sample is created.
    """
    source_state: dict[str, Tensor] = {}
    for target_name, target_tensor in model.state_dict().items():
        if target_name.startswith("fc."):
            source_state[source_name_for_target(target_name)] = torch.full_like(target_tensor, 99.0)
            continue
        if target_name == "conv1.weight":
            element_count = int(target_tensor.shape[0] * 3 * target_tensor.shape[2])
            element_count *= int(target_tensor.shape[3] * target_tensor.shape[4])
            source_state["stem.0.weight"] = torch.arange(
                element_count,
                dtype=target_tensor.dtype,
            ).reshape(target_tensor.shape[0], 3, *target_tensor.shape[2:])
            continue
        source_state[source_name_for_target(target_name)] = torch.zeros_like(target_tensor)
    return source_state
