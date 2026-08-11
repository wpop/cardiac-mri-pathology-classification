"""Deterministic geometric FOV-center XY crop."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CropWindow:
    """Shared XY crop window in XYZ array coordinates."""

    start_xy: tuple[int, int]
    end_xy: tuple[int, int]


class CenterCropper:
    """Apply the frozen 144 x 144 FOV-center crop with no XY padding."""

    def __init__(self, target_h: int, target_w: int) -> None:
        if target_h <= 0 or target_w <= 0:
            raise ValueError("target_h and target_w must be positive")
        self.target_h = target_h
        self.target_w = target_w

    def derive_window(self, array_xyz: NDArray[np.floating]) -> CropWindow:
        """Derive deterministic centered crop window for an XYZ volume."""
        size_x, size_y, _size_z = array_xyz.shape
        if size_x < self.target_w or size_y < self.target_h:
            raise ValueError(
                f"XY padding would be required: source XY={size_x}x{size_y}, "
                f"target={self.target_w}x{self.target_h}"
            )
        crop_x = size_x - self.target_w
        crop_y = size_y - self.target_h
        start_x = crop_x // 2
        start_y = crop_y // 2
        return CropWindow(
            start_xy=(start_x, start_y),
            end_xy=(start_x + self.target_w, start_y + self.target_h),
        )

    def crop(self, array_xyz: NDArray[np.floating], window: CropWindow) -> NDArray[np.floating]:
        """Apply a shared crop window to one XYZ volume."""
        start_x, start_y = window.start_xy
        end_x, end_y = window.end_xy
        cropped = array_xyz[start_x:end_x, start_y:end_y, :]
        if cropped.shape[:2] != (self.target_w, self.target_h):
            raise ValueError(f"Unexpected cropped shape: {cropped.shape}")
        return np.asarray(cropped)
