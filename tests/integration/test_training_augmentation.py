"""Integration tests for training augmentation using real ACDC golden data."""

from pathlib import Path

import numpy as np
import torch

from cardiac_pathology.training.augmentation_config import TrainingAugmentationConfig
from cardiac_pathology.training.training_augmentation import TrainingAugmentation

GOLDEN_TENSOR_PATH = Path("tests/fixtures/golden/patient001.npy")


def test_training_augmentation_preserves_real_acdc_tensor_contract() -> None:
    """Training augmentation preserves the real ACDC tensor contract."""
    image = torch.from_numpy(np.load(GOLDEN_TENSOR_PATH)).to(dtype=torch.float32)

    config = TrainingAugmentationConfig(
        rotation_degrees=5.0,
        translation_mm=5.0,
        intensity_scale_min=0.95,
        intensity_scale_max=1.05,
        gaussian_noise_std=0.01,
        transform_probability=1.0,
    )
    augmentation = TrainingAugmentation(
        config=config,
        in_plane_spacing_mm=1.5,
    )

    torch.manual_seed(42)
    augmented = augmentation(image)

    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert augmented.device == image.device
    assert augmented.is_contiguous()
    assert torch.isfinite(augmented).all()
    assert not torch.equal(augmented, image)


def test_disabled_training_augmentation_leaves_real_acdc_tensor_unchanged() -> None:
    """Disabled augmentation leaves a real ACDC golden tensor unchanged."""
    image = torch.from_numpy(np.load(GOLDEN_TENSOR_PATH)).to(dtype=torch.float32)

    config = TrainingAugmentationConfig(
        transform_probability=0.0,
    )
    augmentation = TrainingAugmentation(
        config=config,
        in_plane_spacing_mm=1.5,
    )

    augmented = augmentation(image)

    assert torch.equal(augmented, image)
