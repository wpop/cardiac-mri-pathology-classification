"""Frozen preprocessing configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Immutable preprocessing contract values."""

    target_orientation: str
    target_spacing_xyz: tuple[float, float, float]
    target_shape_dhw: tuple[int, int, int]
    lower_percentile: float
    upper_percentile: float
    epsilon: float
    z_padding_value: float

    def __post_init__(self) -> None:
        """Validate frozen preprocessing settings."""
        if self.target_orientation != "LPS":
            raise ValueError("target_orientation must be LPS")
        if len(self.target_spacing_xyz) != 3 or any(
            value <= 0 for value in self.target_spacing_xyz
        ):
            raise ValueError("target_spacing_xyz must contain three positive values")
        if self.target_shape_dhw != (14, 144, 144):
            raise ValueError("target_shape_dhw must be frozen as (14, 144, 144)")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
