"""Unit tests for Phase 8 Grad-CAM slice selection."""

import pytest
import torch

from cardiac_pathology.explainability.grad_cam_visualizer import GradCamVisualizer


def test_select_slice_uses_maximum_mean_cam_inside_valid_range() -> None:
    """The selected slice is the valid slice with the largest mean CAM."""
    cam = torch.zeros(5, 2, 2)
    cam[1] = 0.2
    cam[2] = 0.7
    cam[3] = 0.4

    selected_slice = GradCamVisualizer().select_slice(cam, valid_start=1, valid_end=4)

    assert selected_slice == 2


def test_select_slice_ignores_larger_cam_in_z_padding() -> None:
    """Slices outside the valid non-padded range are not eligible."""
    cam = torch.zeros(5, 2, 2)
    cam[0] = 1.0
    cam[4] = 0.9
    cam[2] = 0.5

    selected_slice = GradCamVisualizer().select_slice(cam, valid_start=1, valid_end=4)

    assert selected_slice == 2


def test_select_slice_tie_selects_smallest_valid_absolute_index() -> None:
    """PyTorch argmax tie behavior selects the first valid slice."""
    cam = torch.zeros(5, 2, 2)
    cam[2] = 0.6
    cam[3] = 0.6

    selected_slice = GradCamVisualizer().select_slice(cam, valid_start=2, valid_end=5)

    assert selected_slice == 2


@pytest.mark.parametrize(
    ("valid_start", "valid_end"),
    [
        (-1, 2),
        (2, 2),
        (3, 2),
        (0, 6),
    ],
)
def test_select_slice_rejects_invalid_valid_depth_ranges(
    valid_start: int,
    valid_end: int,
) -> None:
    """The valid range must satisfy 0 <= valid_start < valid_end <= D."""
    cam = torch.zeros(5, 2, 2)

    with pytest.raises(ValueError, match="0 <= valid_start < valid_end <= D"):
        GradCamVisualizer().select_slice(cam, valid_start=valid_start, valid_end=valid_end)
