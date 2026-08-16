"""Unit tests for reusable Phase 8 3D Grad-CAM core behavior."""

import pytest
import torch
from torch import Tensor, nn

from cardiac_pathology.explainability.grad_cam_3d import GradCam3D


class FlipFeatureLayer(nn.Module):
    """Test-only 3D feature layer with deterministic spatial structure."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x: Tensor) -> Tensor:
        first_channel = x * self.scale[0]
        second_channel = torch.flip(x, dims=(-1,)) * self.scale[1]
        return torch.cat((first_channel, second_channel), dim=1)


class TinyRawLogitModel(nn.Module):
    """Small deterministic 3D classifier returning raw logits."""

    def __init__(self) -> None:
        super().__init__()
        self.target_layer = FlipFeatureLayer()
        self.classifier = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        features = self.target_layer(x)
        pooled = features.mean(dim=(2, 3, 4))
        return self.classifier(pooled)


class StridedTargetModel(nn.Module):
    """Model whose target activation is smaller than the input spatial size."""

    def __init__(self) -> None:
        super().__init__()
        self.target_layer = nn.Conv3d(1, 2, kernel_size=1, stride=(1, 2, 2), bias=False)
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.target_layer.weight.copy_(torch.tensor([[[[[1.0]]]], [[[[0.5]]]]]))
            self.classifier.weight.copy_(torch.eye(2))

    def forward(self, x: Tensor) -> Tensor:
        features = self.target_layer(x)
        pooled = features.mean(dim=(2, 3, 4))
        return self.classifier(pooled)


class InvalidActivationModel(nn.Module):
    """Model whose selected target layer emits a non-5D activation."""

    def __init__(self) -> None:
        super().__init__()
        self.target_layer = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(8, 2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.target_layer(x)
        return self.classifier(features)


class InvalidLogitModel(nn.Module):
    """Model that captures a valid activation but returns invalid logits."""

    def __init__(self) -> None:
        super().__init__()
        self.target_layer = nn.Conv3d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.target_layer.weight.fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.target_layer(x)
        return features.mean(dim=(2, 3))


def make_input(batch_size: int = 1, spatial_shape: tuple[int, int, int] = (2, 3, 4)) -> Tensor:
    """Create a deterministic non-medical 5D software test tensor."""
    depth, height, width = spatial_shape
    values = torch.arange(batch_size * depth * height * width, dtype=torch.float32)
    return values.reshape(batch_size, 1, depth, height, width)


def test_basic_valid_generation_returns_normalized_detached_cam() -> None:
    """A valid 3D model/input pair returns finite detached CAMs in [0, 1]."""
    model = TinyRawLogitModel()
    input_tensor = make_input()

    cam = GradCam3D(model, model.target_layer).generate(input_tensor, target_class=0)

    assert cam.shape == (1, 2, 3, 4)
    assert torch.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
    assert cam.requires_grad is False


def test_batch_target_semantics_accept_per_item_and_scalar_targets() -> None:
    """Grad-CAM accepts one class per batch item and one scalar class for all items."""
    model = TinyRawLogitModel()
    grad_cam = GradCam3D(model, model.target_layer)
    input_tensor = make_input(batch_size=2)

    per_item_cam = grad_cam.generate(input_tensor, target_class=torch.tensor([0, 1]))
    scalar_cam = grad_cam.generate(input_tensor, target_class=1)

    assert per_item_cam.shape == (2, 2, 3, 4)
    assert scalar_cam.shape == (2, 2, 3, 4)
    assert torch.isfinite(per_item_cam).all()
    assert torch.isfinite(scalar_cam).all()


def test_batch_target_count_mismatch_is_rejected() -> None:
    """A target list must provide exactly one class index per batch item."""
    model = TinyRawLogitModel()
    input_tensor = make_input(batch_size=2)

    with pytest.raises(ValueError, match="one class index per batch item"):
        GradCam3D(model, model.target_layer).generate(input_tensor, target_class=[0])


@pytest.mark.parametrize(
    ("input_tensor", "expected_exception", "match"),
    [
        (torch.ones(1, 1, 2, 3), ValueError, "shape \\[N, C, D, H, W\\]"),
        (torch.ones(1, 1, 2, 3, 4, dtype=torch.long), TypeError, "floating-point"),
        (torch.full((1, 1, 2, 3, 4), float("inf")), ValueError, "finite"),
    ],
)
def test_input_validation_rejects_invalid_inputs(
    input_tensor: Tensor,
    expected_exception: type[Exception],
    match: str,
) -> None:
    """Input tensors must be finite floating-point 5D tensors."""
    model = TinyRawLogitModel()

    with pytest.raises(expected_exception, match=match):
        GradCam3D(model, model.target_layer).generate(input_tensor, target_class=0)


@pytest.mark.parametrize(
    ("target_class", "expected_exception", "match"),
    [
        (-1, ValueError, "within the model output range"),
        (3, ValueError, "within the model output range"),
        (torch.tensor([0.0]), TypeError, "integer class indices"),
    ],
)
def test_target_class_validation_rejects_invalid_targets(
    target_class: int | Tensor,
    expected_exception: type[Exception],
    match: str,
) -> None:
    """Target class indices must be integer indices in the raw-logit range."""
    model = TinyRawLogitModel()

    with pytest.raises(expected_exception, match=match):
        GradCam3D(model, model.target_layer).generate(make_input(), target_class=target_class)


def test_activation_shape_validation_rejects_non_5d_target_activation() -> None:
    """The selected target layer must produce a 5D activation tensor."""
    model = InvalidActivationModel()

    with pytest.raises(ValueError, match="activation must have shape"):
        GradCam3D(model, model.target_layer).generate(make_input(spatial_shape=(2, 2, 2)), 0)


def test_hook_lifecycle_removes_forward_hook_after_success() -> None:
    """Grad-CAM generation does not leave persistent forward hooks behind."""
    model = TinyRawLogitModel()
    target_layer = model.target_layer

    assert len(target_layer._forward_hooks) == 0
    GradCam3D(model, target_layer).generate(make_input(), target_class=0)
    assert len(target_layer._forward_hooks) == 0


def test_hook_lifecycle_removes_forward_hook_after_failure() -> None:
    """Forward hooks are removed when generation fails after hook registration."""
    model = InvalidLogitModel()
    target_layer = model.target_layer

    assert len(target_layer._forward_hooks) == 0
    with pytest.raises(ValueError, match="model output must have shape"):
        GradCam3D(model, target_layer).generate(make_input(), target_class=0)
    assert len(target_layer._forward_hooks) == 0


def test_generate_does_not_update_model_parameters() -> None:
    """Grad-CAM backward passes must not mutate model parameter values."""
    model = TinyRawLogitModel()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    GradCam3D(model, model.target_layer).generate(make_input(), target_class=0)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])


def test_training_mode_is_restored_after_generation() -> None:
    """A model initially in training mode returns to training mode."""
    model = TinyRawLogitModel()
    model.train()

    GradCam3D(model, model.target_layer).generate(make_input(), target_class=0)

    assert model.training is True


def test_eval_mode_is_preserved_after_generation() -> None:
    """A model initially in eval mode remains in eval mode."""
    model = TinyRawLogitModel()
    model.eval()

    GradCam3D(model, model.target_layer).generate(make_input(), target_class=0)

    assert model.training is False


def test_degenerate_normalization_returns_all_zero_finite_cam() -> None:
    """A zero post-ReLU CAM normalizes safely to an all-zero CAM."""
    model = TinyRawLogitModel()

    cam = GradCam3D(model, model.target_layer).generate(make_input(), target_class=2)

    assert torch.isfinite(cam).all()
    assert torch.equal(cam, torch.zeros_like(cam))
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0


def test_output_is_upsampled_to_input_spatial_dimensions() -> None:
    """The returned CAM uses input spatial dimensions, not target-layer dimensions."""
    model = StridedTargetModel()
    input_tensor = make_input(spatial_shape=(2, 4, 4))

    cam = GradCam3D(model, model.target_layer).generate(input_tensor, target_class=0)

    assert cam.shape == (1, 2, 4, 4)


def test_raw_selected_class_score_controls_cam() -> None:
    """Changing the raw selected logit can change the resulting CAM."""
    model = TinyRawLogitModel()
    input_tensor = make_input()
    normalized_input = (input_tensor[:, 0] - input_tensor.min()) / (
        input_tensor.max() - input_tensor.min()
    )
    expected_class_1 = torch.flip(normalized_input, dims=(-1,))
    grad_cam = GradCam3D(model, model.target_layer)

    class_0_cam = grad_cam.generate(input_tensor, target_class=0)
    class_1_cam = grad_cam.generate(input_tensor, target_class=1)

    assert torch.allclose(class_0_cam, normalized_input)
    assert torch.allclose(class_1_cam, expected_class_1)
    assert not torch.allclose(class_0_cam, class_1_cam)
