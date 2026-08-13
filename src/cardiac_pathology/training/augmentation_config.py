"""Configuration for conservative cardiac MRI training augmentation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingAugmentationConfig:
    """Configuration for training-only ED/ES augmentation.

    Spatial transforms must be shared between ED and ES channels to preserve
    their spatial correspondence. Augmentation is never applied during
    validation or inference.
    """

    rotation_degrees: float = 5.0
    translation_mm: float = 5.0
    intensity_scale_min: float = 0.95
    intensity_scale_max: float = 1.05
    gaussian_noise_std: float = 0.01
    transform_probability: float = 0.5

    def __post_init__(self) -> None:
        """Validate augmentation parameters."""
        if self.rotation_degrees < 0.0:
            raise ValueError("rotation_degrees must be non-negative")
        if self.translation_mm < 0.0:
            raise ValueError("translation_mm must be non-negative")
        if self.intensity_scale_min <= 0.0:
            raise ValueError("intensity_scale_min must be positive")
        if self.intensity_scale_max < self.intensity_scale_min:
            raise ValueError(
                "intensity_scale_max must be greater than or equal to intensity_scale_min"
            )
        if self.gaussian_noise_std < 0.0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if not 0.0 <= self.transform_probability <= 1.0:
            raise ValueError("transform_probability must be between 0 and 1")
