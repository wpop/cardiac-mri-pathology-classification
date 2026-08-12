"""Typed adapters for third-party analysis dependencies."""

from collections.abc import Callable
from typing import Protocol, cast

import nibabel as nib
import numpy as np


class ImageHeader(Protocol):
    """Structural type for image headers exposing voxel zooms."""

    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel zoom values."""
        ...


class SpatialImage(Protocol):
    """Structural type for NiBabel spatial images used by analysis scripts."""

    @property
    def dataobj(self) -> object:
        """Return the image data object."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image shape."""
        ...

    @property
    def affine(self) -> np.ndarray:
        """Return the image affine."""
        ...

    @property
    def header(self) -> ImageHeader:
        """Return the image header."""
        ...


_aff2axcodes: Callable[[np.ndarray], tuple[str, ...]] = nib.aff2axcodes
_apply_orientation: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
    nib.orientations.apply_orientation
)
_io_orientation: Callable[[np.ndarray], np.ndarray] = nib.orientations.io_orientation
_ornt_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] = nib.orientations.ornt_transform


def image_array(image: SpatialImage) -> np.ndarray:
    """Return an array view of a NiBabel spatial image data object."""
    array: np.ndarray = np.asanyarray(image.dataobj)
    return array


def orientation_codes(image: SpatialImage) -> tuple[str, ...]:
    """Return NiBabel axis codes for a spatial image."""
    return _aff2axcodes(image.affine)


def orientation_transform(source_image: SpatialImage, target_image: SpatialImage) -> np.ndarray:
    """Return the NiBabel orientation transform from source to target image."""
    source_orientation = _io_orientation(source_image.affine)
    target_orientation = _io_orientation(target_image.affine)
    return _ornt_transform(source_orientation, target_orientation)


def apply_orientation(array: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a NiBabel orientation transform to an array."""
    return _apply_orientation(array, transform)


def shape3(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Narrow a shape tuple statically without runtime validation."""
    return tuple3(shape)


def tuple3[TupleValue](
    values: tuple[TupleValue, ...],
) -> tuple[TupleValue, TupleValue, TupleValue]:
    """Narrow a tuple statically without runtime validation."""
    return cast(tuple[TupleValue, TupleValue, TupleValue], values)


def spacing3(image: SpatialImage) -> tuple[float, float, float]:
    """Read three-dimensional voxel spacing from a spatial image."""
    return tuple3(tuple(float(value) for value in image.header.get_zooms()[:3]))
