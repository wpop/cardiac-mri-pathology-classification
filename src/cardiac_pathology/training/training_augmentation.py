"""Conservative training-only augmentation for preprocessed cardiac MRI."""

import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from cardiac_pathology.training.augmentation_config import TrainingAugmentationConfig


class TrainingAugmentation:
    """Apply conservative augmentation to one preprocessed ED/ES patient tensor.

    Spatial transforms are shared by both ED and ES channels and preserve the
    depth axis. Intensity transforms preserve zero-valued background and
    artificial padding.
    """

    def __init__(
        self,
        config: TrainingAugmentationConfig,
        in_plane_spacing_mm: float,
    ) -> None:
        """Initialize training augmentation.

        Args:
            config: Frozen Phase 6 training augmentation configuration.
            in_plane_spacing_mm: Physical spacing of the preprocessed in-plane
                voxels in millimeters.

        Returns:
            None.
        """
        if in_plane_spacing_mm <= 0.0:
            raise ValueError("in_plane_spacing_mm must be positive")

        self.config = config
        self.in_plane_spacing_mm = in_plane_spacing_mm

    def __call__(self, image: Tensor) -> Tensor:
        """Augment one preprocessed patient tensor.

        Args:
            image: Floating-point tensor with shape ``[2, D, H, W]``.

        Returns:
            Augmented tensor with the same shape, dtype, and device.
        """
        self._validate_image(image)

        augmented = image

        apply_rotation = self._sample_probability()
        apply_translation = self._sample_probability()

        if apply_rotation or apply_translation:
            rotation_degrees = (
                self._sample_uniform(
                    -self.config.rotation_degrees,
                    self.config.rotation_degrees,
                    image,
                )
                if apply_rotation
                else 0.0
            )
            translation_x_mm = (
                self._sample_uniform(
                    -self.config.translation_mm,
                    self.config.translation_mm,
                    image,
                )
                if apply_translation
                else 0.0
            )
            translation_y_mm = (
                self._sample_uniform(
                    -self.config.translation_mm,
                    self.config.translation_mm,
                    image,
                )
                if apply_translation
                else 0.0
            )

            augmented = self._apply_spatial_transform(
                image=augmented,
                rotation_degrees=rotation_degrees,
                translation_x_mm=translation_x_mm,
                translation_y_mm=translation_y_mm,
            )

        if self._sample_probability():
            intensity_scale = self._sample_uniform(
                self.config.intensity_scale_min,
                self.config.intensity_scale_max,
                augmented,
            )
            augmented = augmented * intensity_scale

        if self._sample_probability() and self.config.gaussian_noise_std > 0.0:
            foreground_mask = augmented != 0
            noise = torch.randn_like(augmented) * self.config.gaussian_noise_std
            augmented = torch.where(foreground_mask, augmented + noise, augmented)

        if not torch.isfinite(augmented).all():
            raise FloatingPointError("Training augmentation produced non-finite values")

        return augmented.contiguous()

    def _apply_spatial_transform(
        self,
        image: Tensor,
        rotation_degrees: float,
        translation_x_mm: float,
        translation_y_mm: float,
    ) -> Tensor:
        """Apply one shared in-plane affine transform to ED and ES.

        Args:
            image: Tensor with shape ``[2, D, H, W]``.
            rotation_degrees: In-plane rotation angle in degrees.
            translation_x_mm: Horizontal physical translation in millimeters.
            translation_y_mm: Vertical physical translation in millimeters.

        Returns:
            Spatially transformed tensor with unchanged shape.
        """
        _, depth, height, width = image.shape

        angle_radians = math.radians(rotation_degrees)
        cosine = math.cos(angle_radians)
        sine = math.sin(angle_radians)

        translation_x_pixels = translation_x_mm / self.in_plane_spacing_mm
        translation_y_pixels = translation_y_mm / self.in_plane_spacing_mm

        translation_x_normalized = 2.0 * translation_x_pixels / width
        translation_y_normalized = 2.0 * translation_y_pixels / height

        theta = image.new_tensor(
            [
                [
                    cosine,
                    -sine,
                    0.0,
                    translation_x_normalized,
                ],
                [
                    sine,
                    cosine,
                    0.0,
                    translation_y_normalized,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        ).unsqueeze(0)

        batched_image = image.unsqueeze(0)
        grid = functional.affine_grid(
            theta,
            size=[1, image.shape[0], depth, height, width],
            align_corners=False,
        )

        transformed = functional.grid_sample(
            batched_image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return transformed.squeeze(0)

    def _sample_probability(self) -> bool:
        """Sample whether one configured augmentation should be applied."""
        return bool(torch.rand(()) < self.config.transform_probability)

    @staticmethod
    def _sample_uniform(lower: float, upper: float, reference: Tensor) -> float:
        """Sample a scalar uniformly within an inclusive numeric range."""
        sample = torch.rand((), device=reference.device)
        return float((lower + sample * (upper - lower)).item())

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        """Validate the expected preprocessed patient tensor contract."""
        if image.ndim != 4:
            raise ValueError("image must have shape [2, D, H, W]")
        if image.shape[0] != 2:
            raise ValueError("image must contain exactly two ED/ES channels")
        if not image.is_floating_point():
            raise TypeError("image must be a floating-point tensor")
        if not torch.isfinite(image).all():
            raise FloatingPointError("image contains non-finite values")
