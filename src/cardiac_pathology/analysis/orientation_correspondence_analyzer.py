"""Audit correspondence between ACDC 4D cine frames and ED/ES volumes."""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrespondenceResult:
    """Exact correspondence result for one standalone frame."""

    direct_shape_equal: bool
    direct_match: bool
    transformed_shape: tuple[int, int, int]
    nib_orientation_match: bool
    nib_max_abs_diff: float
    fallback_match_count: int
    status: str


class OrientationCorrespondenceAnalyzer:
    """Analyze ED/ES frame correspondence using orientation-only transforms."""

    def __init__(
        self,
        dataset_dir: Path,
        output_csv: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_csv = output_csv

    def run(self) -> pd.DataFrame:
        """Run the full correspondence audit and save one CSV row per patient."""
        patient_dirs = sorted(self.dataset_dir.glob("patient*"))

        if not patient_dirs:
            raise RuntimeError(f"No patient directories found in {self.dataset_dir}")

        records = [self._analyze_patient(patient_dir) for patient_dir in patient_dirs]
        dataframe = pd.DataFrame(records)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_csv, index=False)

        return dataframe

    def print_summary(self, dataframe: pd.DataFrame) -> None:
        """Print the concise orientation/cine correspondence gate summary."""
        patient_count = len(dataframe)

        print("Orientation / cine correspondence")

        print("\nOrientation pairs")
        print("Cine orientation -> standalone ED orientation:")
        print(
            self._orientation_pair_counts(
                dataframe,
                "cine_orientation",
                "ed_orientation",
            )
        )
        print("\nCine orientation -> standalone ES orientation:")
        print(
            self._orientation_pair_counts(
                dataframe,
                "cine_orientation",
                "es_orientation",
            )
        )

        print("\nED correspondence")
        print(self._status_counts(dataframe, "ed", patient_count))

        print("\nES correspondence")
        print(self._status_counts(dataframe, "es", patient_count))

        failing_rows = dataframe[
            ~dataframe["ed_correspondence_status"].isin(self._passing_statuses())
            | ~dataframe["es_correspondence_status"].isin(self._passing_statuses())
        ]

        print("\nDataset gate")
        if failing_rows.empty and patient_count == 100:
            print(
                "PASS: all 100 patients have exact ED/ES correspondence using "
                "orientation-only transforms."
            )
        else:
            print(
                "FAIL: one or more patients do not have proven exact ED/ES "
                "correspondence."
            )
            if not failing_rows.empty:
                print("Failing patients:")
                for row in failing_rows.itertuples(index=False):
                    print(
                        f"  {row.patient_id}: "
                        f"ED={row.ed_correspondence_status}, "
                        f"ES={row.es_correspondence_status}"
                    )

    def _analyze_patient(self, patient_dir: Path) -> dict[str, Any]:
        """Analyze one ACDC patient directory."""
        patient_id = patient_dir.name
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        es_frame = int(metadata["ES"])
        group = metadata["Group"]

        cine_path = patient_dir / f"{patient_id}_4d.nii.gz"
        ed_path = patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz"
        es_path = patient_dir / f"{patient_id}_frame{es_frame:02d}.nii.gz"

        for path in [cine_path, ed_path, es_path]:
            if not path.is_file():
                raise FileNotFoundError(f"{patient_id}: missing file {path}")

        cine_image = nib.load(cine_path)
        ed_image = nib.load(ed_path)
        es_image = nib.load(es_path)

        self._validate_images(
            patient_id=patient_id,
            cine_image=cine_image,
            ed_image=ed_image,
            es_image=es_image,
            ed_frame=ed_frame,
            es_frame=es_frame,
        )

        cine_array = np.asanyarray(cine_image.dataobj)
        ed_array = np.asanyarray(ed_image.dataobj)
        es_array = np.asanyarray(es_image.dataobj)
        cine_ed = cine_array[..., ed_frame - 1]
        cine_es = cine_array[..., es_frame - 1]

        ed_result = self._compare_frame(
            cine_frame=cine_ed,
            standalone_frame=ed_array,
            cine_image=cine_image,
            standalone_image=ed_image,
        )
        es_result = self._compare_frame(
            cine_frame=cine_es,
            standalone_frame=es_array,
            cine_image=cine_image,
            standalone_image=es_image,
        )

        return {
            "patient_id": patient_id,
            "class": group,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "cine_orientation": "".join(nib.aff2axcodes(cine_image.affine)),
            "ed_orientation": "".join(nib.aff2axcodes(ed_image.affine)),
            "es_orientation": "".join(nib.aff2axcodes(es_image.affine)),
            "ed_direct_shape_equal": ed_result.direct_shape_equal,
            "es_direct_shape_equal": es_result.direct_shape_equal,
            "ed_direct_match": ed_result.direct_match,
            "es_direct_match": es_result.direct_match,
            "ed_transformed_shape": str(ed_result.transformed_shape),
            "es_transformed_shape": str(es_result.transformed_shape),
            "ed_nib_orientation_match": ed_result.nib_orientation_match,
            "es_nib_orientation_match": es_result.nib_orientation_match,
            "ed_nib_max_abs_diff": ed_result.nib_max_abs_diff,
            "es_nib_max_abs_diff": es_result.nib_max_abs_diff,
            "ed_fallback_match_count": ed_result.fallback_match_count,
            "es_fallback_match_count": es_result.fallback_match_count,
            "ed_correspondence_status": ed_result.status,
            "es_correspondence_status": es_result.status,
        }

    def _validate_images(
        self,
        patient_id: str,
        cine_image: nib.spatialimages.SpatialImage,
        ed_image: nib.spatialimages.SpatialImage,
        es_image: nib.spatialimages.SpatialImage,
        ed_frame: int,
        es_frame: int,
    ) -> None:
        """Validate image dimensions, frame indices, and spatial metadata."""
        if len(cine_image.shape) != 4:
            raise ValueError(f"{patient_id}: expected 4D cine, got {cine_image.shape}")

        if len(ed_image.shape) != 3:
            raise ValueError(f"{patient_id}: expected 3D ED, got {ed_image.shape}")

        if len(es_image.shape) != 3:
            raise ValueError(f"{patient_id}: expected 3D ES, got {es_image.shape}")

        time_points = cine_image.shape[3]
        for frame_name, frame in [("ED", ed_frame), ("ES", es_frame)]:
            frame_index = frame - 1
            if frame_index < 0 or frame_index >= time_points:
                raise ValueError(
                    f"{patient_id}: {frame_name} frame index {frame_index} is "
                    f"outside cine time dimension {time_points}"
                )

        ed_spacing = self._spatial_spacing(ed_image)
        es_spacing = self._spatial_spacing(es_image)
        if ed_image.shape != es_image.shape:
            raise ValueError(
                f"{patient_id}: ED/ES shape mismatch: "
                f"{ed_image.shape} != {es_image.shape}"
            )

        if ed_spacing != es_spacing:
            raise ValueError(
                f"{patient_id}: ED/ES spacing mismatch: "
                f"{ed_spacing} != {es_spacing}"
            )

        ed_orientation = nib.aff2axcodes(ed_image.affine)
        es_orientation = nib.aff2axcodes(es_image.affine)
        if ed_orientation != es_orientation:
            raise ValueError(
                f"{patient_id}: ED/ES orientation mismatch: "
                f"{ed_orientation} != {es_orientation}"
            )

        self._validate_compatible_spatial_metadata(
            patient_id=patient_id,
            cine_image=cine_image,
            standalone_image=ed_image,
            frame_name="ED",
        )
        self._validate_compatible_spatial_metadata(
            patient_id=patient_id,
            cine_image=cine_image,
            standalone_image=es_image,
            frame_name="ES",
        )

    def _validate_compatible_spatial_metadata(
        self,
        patient_id: str,
        cine_image: nib.spatialimages.SpatialImage,
        standalone_image: nib.spatialimages.SpatialImage,
        frame_name: str,
    ) -> None:
        """Check cine and standalone spatial shape/spacing after orientation."""
        transform = self._orientation_transform(cine_image, standalone_image)
        transformed_shape = self._shape_after_orientation(
            cine_image.shape[:3],
            transform,
        )
        transformed_spacing = self._spacing_after_orientation(
            self._spatial_spacing(cine_image),
            transform,
        )
        standalone_spacing = self._spatial_spacing(standalone_image)

        if transformed_shape != standalone_image.shape:
            raise ValueError(
                f"{patient_id}: cine shape is incompatible with standalone "
                f"{frame_name} after orientation transform: "
                f"{transformed_shape} != {standalone_image.shape}"
            )

        if transformed_spacing != standalone_spacing:
            raise ValueError(
                f"{patient_id}: cine spacing is incompatible with standalone "
                f"{frame_name} after orientation transform: "
                f"{transformed_spacing} != {standalone_spacing}"
            )

    def _compare_frame(
        self,
        cine_frame: np.ndarray,
        standalone_frame: np.ndarray,
        cine_image: nib.spatialimages.SpatialImage,
        standalone_image: nib.spatialimages.SpatialImage,
    ) -> CorrespondenceResult:
        """Compare one extracted cine frame to one standalone volume."""
        direct_shape_equal = cine_frame.shape == standalone_frame.shape
        direct_match = direct_shape_equal and np.array_equal(cine_frame, standalone_frame)

        transform = self._orientation_transform(cine_image, standalone_image)
        transformed_frame = nib.orientations.apply_orientation(cine_frame, transform)
        transformed_shape = tuple(int(value) for value in transformed_frame.shape)
        transformed_shape_equal = transformed_frame.shape == standalone_frame.shape
        nib_orientation_match = transformed_shape_equal and np.array_equal(
            transformed_frame,
            standalone_frame,
        )
        nib_max_abs_diff = self._max_abs_diff(transformed_frame, standalone_frame)

        fallback_match_count = 0
        if not nib_orientation_match:
            fallback_match_count = self._count_fallback_matches(
                cine_frame,
                standalone_frame,
            )

        if direct_match:
            status = "direct"
        elif nib_orientation_match:
            status = "nib_orientation"
        elif fallback_match_count == 1:
            status = "fallback_orientation"
        elif fallback_match_count > 1:
            status = "ambiguous"
        else:
            status = "failed"

        return CorrespondenceResult(
            direct_shape_equal=direct_shape_equal,
            direct_match=direct_match,
            transformed_shape=transformed_shape,
            nib_orientation_match=nib_orientation_match,
            nib_max_abs_diff=nib_max_abs_diff,
            fallback_match_count=fallback_match_count,
            status=status,
        )

    def _count_fallback_matches(
        self,
        cine_frame: np.ndarray,
        standalone_frame: np.ndarray,
    ) -> int:
        """Count exhaustive orientation-only permutation/flip matches."""
        match_count = 0
        for axis_order in permutations(range(3)):
            permuted = np.transpose(cine_frame, axes=axis_order)
            if permuted.shape != standalone_frame.shape:
                continue

            for flip_flags in product([False, True], repeat=3):
                candidate = permuted
                for axis, should_flip in enumerate(flip_flags):
                    if should_flip:
                        candidate = np.flip(candidate, axis=axis)

                if np.array_equal(candidate, standalone_frame):
                    match_count += 1

        return match_count

    def _orientation_transform(
        self,
        source_image: nib.spatialimages.SpatialImage,
        target_image: nib.spatialimages.SpatialImage,
    ) -> np.ndarray:
        """Derive the orientation transform from source image to target image."""
        source_orientation = nib.orientations.io_orientation(source_image.affine)
        target_orientation = nib.orientations.io_orientation(target_image.affine)
        return nib.orientations.ornt_transform(source_orientation, target_orientation)

    def _shape_after_orientation(
        self,
        shape: tuple[int, int, int],
        transform: np.ndarray,
    ) -> tuple[int, int, int]:
        """Calculate the spatial shape after an orientation transform."""
        transformed_axes = np.argsort(transform[:, 0].astype(int))
        return tuple(int(shape[axis]) for axis in transformed_axes)

    def _spacing_after_orientation(
        self,
        spacing: tuple[float, float, float],
        transform: np.ndarray,
    ) -> tuple[float, float, float]:
        """Calculate voxel spacing after an orientation transform."""
        transformed_axes = np.argsort(transform[:, 0].astype(int))
        return tuple(float(spacing[axis]) for axis in transformed_axes)

    def _max_abs_diff(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """Return maximum absolute difference when shapes are comparable."""
        if candidate.shape != reference.shape:
            return float("nan")

        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        return float(np.max(np.abs(difference)))

    def _orientation_pair_counts(
        self,
        dataframe: pd.DataFrame,
        source_column: str,
        target_column: str,
    ) -> str:
        """Format orientation pair counts."""
        pair_counts = Counter(
            f"{source} -> {target}"
            for source, target in zip(
                dataframe[source_column],
                dataframe[target_column],
                strict=True,
            )
        )

        return "\n".join(
            f"  {pair}: {count}"
            for pair, count in sorted(pair_counts.items())
        )

    def _status_counts(
        self,
        dataframe: pd.DataFrame,
        prefix: str,
        patient_count: int,
    ) -> str:
        """Format correspondence status counts for ED or ES."""
        direct_count = int(dataframe[f"{prefix}_direct_match"].sum())
        nib_count = int(dataframe[f"{prefix}_nib_orientation_match"].sum())
        fallback_count = int(
            (dataframe[f"{prefix}_correspondence_status"] == "fallback_orientation")
            .sum()
        )
        ambiguous_count = int(
            (dataframe[f"{prefix}_correspondence_status"] == "ambiguous").sum()
        )
        failed_count = int(
            (dataframe[f"{prefix}_correspondence_status"] == "failed").sum()
        )

        return "\n".join(
            [
                f"  direct exact match: {direct_count} / {patient_count}",
                "  exact match after NiBabel orientation transform: "
                f"{nib_count} / {patient_count}",
                "  exact match only through fallback permutation/flip search: "
                f"{fallback_count} / {patient_count}",
                f"  ambiguous: {ambiguous_count} / {patient_count}",
                f"  failed: {failed_count} / {patient_count}",
            ]
        )

    def _spatial_spacing(
        self,
        image: nib.spatialimages.SpatialImage,
    ) -> tuple[float, float, float]:
        """Read image spatial voxel spacing."""
        return tuple(float(value) for value in image.header.get_zooms()[:3])

    def _parse_info_file(self, info_path: Path) -> dict[str, str]:
        """Read key-value metadata from one ACDC Info.cfg file."""
        metadata: dict[str, str] = {}

        with info_path.open("r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split(":", maxsplit=1)
                metadata[key.strip()] = value.strip()

        return metadata

    def _passing_statuses(self) -> set[str]:
        """Return correspondence statuses that prove exact orientation-only match."""
        return {"direct", "nib_orientation", "fallback_orientation"}
