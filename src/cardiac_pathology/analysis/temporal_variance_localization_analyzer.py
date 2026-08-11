"""Temporal-variance localization analysis for real ACDC cine MRI."""

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


class TemporalVarianceLocalizationAnalyzer:
    """Measure high temporal-variance localization relative to FOV center."""

    def __init__(
        self,
        dataset_dir: Path,
        output_csv: Path,
        qa_dir: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_csv = output_csv
        self.qa_dir = qa_dir
        self.generated_qa_patient_ids: list[str] = []

    def run(self) -> pd.DataFrame:
        """Run localization analysis, save CSV, and generate QA figures."""
        patient_dirs = sorted(self.dataset_dir.glob("patient*"))

        if not patient_dirs:
            raise RuntimeError(f"No patient directories found in {self.dataset_dir}")

        records = [self._analyze_patient(patient_dir) for patient_dir in patient_dirs]
        dataframe = pd.DataFrame(records)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_csv, index=False)

        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.generated_qa_patient_ids = self._generate_qa_figures(dataframe)

        return dataframe

    def print_summary(self, dataframe: pd.DataFrame) -> None:
        """Print concise cohort and by-class localization summaries."""
        print("Temporal-variance localization")
        for prefix, label in [("tv05", "Top 5% support"), ("tv01", "Top 1% support")]:
            print(f"\n{label}")
            print(self._support_summary(dataframe, prefix))

        print("\nBy-class summary")
        for prefix, label in [("tv05", "Top 5% support"), ("tv01", "Top 1% support")]:
            print(f"\n{label}:")
            print(self._class_summary(dataframe, prefix))

        print("\nLarge-offset counts")
        for prefix, label in [("tv05", "Top 5% support"), ("tv01", "Top 1% support")]:
            offsets = dataframe[f"{prefix}_euclidean_offset_mm"]
            print(f"{label}:")
            for threshold_mm in [20, 30, 40, 50]:
                print(
                    f"  > {threshold_mm} mm: "
                    f"{int((offsets > threshold_mm).sum())} / {len(dataframe)}"
                )

        print("\nGenerated QA patient IDs")
        for patient_id in self.generated_qa_patient_ids:
            print(f"  {patient_id}")

    def _analyze_patient(self, patient_dir: Path) -> dict[str, Any]:
        """Analyze one patient using raw cine temporal variance."""
        patient_id = patient_dir.name
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        es_frame = int(metadata["ES"])
        ed_index = ed_frame - 1
        es_index = es_frame - 1
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
            ed_index=ed_index,
            es_index=es_index,
        )

        cine_array = np.asanyarray(cine_image.dataobj)
        ed_array = np.asanyarray(ed_image.dataobj)
        es_array = np.asanyarray(es_image.dataobj)

        if not np.array_equal(cine_array[..., ed_index], ed_array):
            raise ValueError(f"{patient_id}: raw cine ED frame does not equal ED volume")

        if not np.array_equal(cine_array[..., es_index], es_array):
            raise ValueError(f"{patient_id}: raw cine ES frame does not equal ES volume")

        spacing_x, spacing_y, _spacing_z = self._spatial_spacing(ed_image)
        size_x, size_y, _size_z = ed_image.shape
        temporal_variance = np.var(cine_array, axis=3, dtype=np.float64)
        localization = self._measure_supports(
            temporal_variance=temporal_variance,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            patient_id=patient_id,
        )

        return {
            "patient_id": patient_id,
            "class": group,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "size_x": size_x,
            "size_y": size_y,
            "fov_x_mm": size_x * spacing_x,
            "fov_y_mm": size_y * spacing_y,
            "spacing_x_mm": spacing_x,
            "spacing_y_mm": spacing_y,
            **localization,
        }

    def _validate_images(
        self,
        patient_id: str,
        cine_image: nib.spatialimages.SpatialImage,
        ed_image: nib.spatialimages.SpatialImage,
        es_image: nib.spatialimages.SpatialImage,
        ed_index: int,
        es_index: int,
    ) -> None:
        """Validate dimensions and raw frame-index availability."""
        if len(cine_image.shape) != 4:
            raise ValueError(f"{patient_id}: expected 4D cine, got {cine_image.shape}")

        if len(ed_image.shape) != 3:
            raise ValueError(f"{patient_id}: expected 3D ED, got {ed_image.shape}")

        if len(es_image.shape) != 3:
            raise ValueError(f"{patient_id}: expected 3D ES, got {es_image.shape}")

        time_points = cine_image.shape[3]
        for frame_name, frame_index in [("ED", ed_index), ("ES", es_index)]:
            if frame_index < 0 or frame_index >= time_points:
                raise ValueError(
                    f"{patient_id}: {frame_name} index {frame_index} is outside "
                    f"cine time dimension {time_points}"
                )

        if cine_image.shape[:3] != ed_image.shape:
            raise ValueError(
                f"{patient_id}: cine spatial shape does not match ED: "
                f"{cine_image.shape[:3]} != {ed_image.shape}"
            )

        if cine_image.shape[:3] != es_image.shape:
            raise ValueError(
                f"{patient_id}: cine spatial shape does not match ES: "
                f"{cine_image.shape[:3]} != {es_image.shape}"
            )

        if ed_image.shape != es_image.shape:
            raise ValueError(
                f"{patient_id}: ED/ES shape mismatch: "
                f"{ed_image.shape} != {es_image.shape}"
            )

        if self._spatial_spacing(ed_image) != self._spatial_spacing(es_image):
            raise ValueError(
                f"{patient_id}: ED/ES spacing mismatch: "
                f"{self._spatial_spacing(ed_image)} != {self._spatial_spacing(es_image)}"
            )

    def _measure_supports(
        self,
        temporal_variance: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        patient_id: str,
    ) -> dict[str, float]:
        """Measure top-5% and top-1% temporal-variance support geometry."""
        positive_variance = temporal_variance[temporal_variance > 0]
        if positive_variance.size == 0:
            raise ValueError(f"{patient_id}: no positive temporal-variance voxels")

        measurements: dict[str, float] = {}
        for prefix, percentile in [("tv05", 95.0), ("tv01", 99.0)]:
            threshold = np.percentile(positive_variance, percentile)
            support = temporal_variance >= threshold
            measurements.update(
                self._measure_single_support(
                    temporal_variance=temporal_variance,
                    support=support,
                    spacing_x=spacing_x,
                    spacing_y=spacing_y,
                    prefix=prefix,
                )
            )

        return measurements

    def _measure_single_support(
        self,
        temporal_variance: np.ndarray,
        support: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        prefix: str,
    ) -> dict[str, float]:
        """Measure one patient-specific high temporal-variance support."""
        size_x, size_y, _size_z = temporal_variance.shape
        center_x_vox = (size_x - 1) / 2
        center_y_vox = (size_y - 1) / 2

        support_x, support_y, _support_z = np.nonzero(support)
        support_weights = temporal_variance[support]
        centroid_x_vox = float(np.average(support_x, weights=support_weights))
        centroid_y_vox = float(np.average(support_y, weights=support_weights))

        min_x_vox = int(np.min(support_x))
        max_x_vox = int(np.max(support_x))
        min_y_vox = int(np.min(support_y))
        max_y_vox = int(np.max(support_y))

        offset_x_mm = abs(centroid_x_vox - center_x_vox) * spacing_x
        offset_y_mm = abs(centroid_y_vox - center_y_vox) * spacing_y
        required_half_width_mm = (
            max(abs(min_x_vox - center_x_vox), abs(max_x_vox - center_x_vox))
            * spacing_x
        )
        required_half_height_mm = (
            max(abs(min_y_vox - center_y_vox), abs(max_y_vox - center_y_vox))
            * spacing_y
        )

        return {
            f"{prefix}_centroid_x_vox": centroid_x_vox,
            f"{prefix}_centroid_y_vox": centroid_y_vox,
            f"{prefix}_offset_x_mm": offset_x_mm,
            f"{prefix}_offset_y_mm": offset_y_mm,
            f"{prefix}_euclidean_offset_mm": float(np.hypot(offset_x_mm, offset_y_mm)),
            f"{prefix}_support_bbox_width_mm": (max_x_vox - min_x_vox) * spacing_x,
            f"{prefix}_support_bbox_height_mm": (max_y_vox - min_y_vox) * spacing_y,
            f"{prefix}_required_center_width_mm": 2 * required_half_width_mm,
            f"{prefix}_required_center_height_mm": 2 * required_half_height_mm,
        }

    def _generate_qa_figures(self, dataframe: pd.DataFrame) -> list[str]:
        """Generate diagnostic QA figures for selected patients."""
        selected_patient_ids = self._select_qa_patient_ids(dataframe)

        for patient_id in selected_patient_ids:
            self._generate_patient_qa(patient_id)

        return selected_patient_ids

    def _select_qa_patient_ids(self, dataframe: pd.DataFrame) -> list[str]:
        """Select requested QA patients while avoiding duplicates where practical."""
        selected: list[str] = []

        largest_offsets = dataframe.nlargest(5, "tv05_euclidean_offset_mm")
        self._append_unique(selected, largest_offsets["patient_id"].tolist())

        dataframe = dataframe.assign(
            tv05_required_center_area_mm2=(
                dataframe["tv05_required_center_width_mm"]
                * dataframe["tv05_required_center_height_mm"]
            )
        )
        largest_crops = dataframe.nlargest(5, "tv05_required_center_area_mm2")
        self._append_unique(selected, largest_crops["patient_id"].tolist())

        cohort_median_offset = dataframe["tv05_euclidean_offset_mm"].median()
        for class_name in sorted(dataframe["class"].unique()):
            class_dataframe = dataframe[dataframe["class"] == class_name].copy()
            class_dataframe["offset_distance_to_median"] = (
                class_dataframe["tv05_euclidean_offset_mm"] - cohort_median_offset
            ).abs()
            patient_id = str(
                class_dataframe.sort_values("offset_distance_to_median")
                .iloc[0]["patient_id"]
            )
            self._append_unique(selected, [patient_id])

        return selected

    def _generate_patient_qa(self, patient_id: str) -> None:
        """Create one temporal-variance QA PNG for one patient."""
        patient_dir = self.dataset_dir / patient_id
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        cine_image = nib.load(patient_dir / f"{patient_id}_4d.nii.gz")
        ed_image = nib.load(patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz")
        cine_array = np.asanyarray(cine_image.dataobj)
        ed_array = np.asanyarray(ed_image.dataobj)
        temporal_variance = np.var(cine_array, axis=3, dtype=np.float64)
        positive_variance = temporal_variance[temporal_variance > 0]
        threshold = np.percentile(positive_variance, 95.0)
        support = temporal_variance >= threshold

        z_index = ed_array.shape[2] // 2
        variance_slice = temporal_variance[:, :, z_index]
        support_slice = support[:, :, z_index]
        spacing_x, spacing_y, _spacing_z = self._spatial_spacing(ed_image)
        support_measurements = self._measure_single_support(
            temporal_variance=temporal_variance,
            support=support,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            prefix="tv05",
        )
        center_x_vox = (ed_array.shape[0] - 1) / 2
        center_y_vox = (ed_array.shape[1] - 1) / 2

        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(np.rot90(ed_array[:, :, z_index]), cmap="gray")
        axes[0].set_title(f"{patient_id} ED mid-Z")
        axes[1].imshow(np.rot90(variance_slice), cmap="magma")
        axes[1].set_title("Temporal variance")
        axes[2].imshow(np.rot90(ed_array[:, :, z_index]), cmap="gray")
        axes[2].imshow(
            np.rot90(np.ma.masked_where(~support_slice, support_slice)),
            cmap="autumn",
            alpha=0.45,
        )
        axes[2].scatter(
            center_x_vox,
            ed_array.shape[1] - 1 - center_y_vox,
            marker="+",
            c="cyan",
            s=90,
            linewidths=2,
            label="FOV center",
        )
        axes[2].scatter(
            support_measurements["tv05_centroid_x_vox"],
            ed_array.shape[1] - 1 - support_measurements["tv05_centroid_y_vox"],
            marker="x",
            c="lime",
            s=90,
            linewidths=2,
            label="TV centroid",
        )
        axes[2].set_title("Top 5% support")
        axes[2].legend(loc="lower right", fontsize=7)

        for axis in axes:
            axis.set_axis_off()

        output_path = self.qa_dir / f"{patient_id}_temporal_variance_qa.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _support_summary(self, dataframe: pd.DataFrame, prefix: str) -> str:
        """Format cohort summary for one support definition."""
        columns = [
            f"{prefix}_euclidean_offset_mm",
            f"{prefix}_required_center_width_mm",
            f"{prefix}_required_center_height_mm",
            f"{prefix}_support_bbox_width_mm",
            f"{prefix}_support_bbox_height_mm",
        ]
        renamed_columns = {
            f"{prefix}_euclidean_offset_mm": "euclidean_offset_mm",
            f"{prefix}_required_center_width_mm": "required_center_width_mm",
            f"{prefix}_required_center_height_mm": "required_center_height_mm",
            f"{prefix}_support_bbox_width_mm": "support_bbox_width_mm",
            f"{prefix}_support_bbox_height_mm": "support_bbox_height_mm",
        }
        return (
            dataframe[columns]
            .rename(columns=renamed_columns)
            .describe(percentiles=[0.05, 0.5, 0.95])
            .loc[["min", "5%", "50%", "mean", "95%", "max"]]
            .rename(index={"5%": "p05", "50%": "median", "95%": "p95"})
            .round(2)
            .to_string()
        )

    def _class_summary(self, dataframe: pd.DataFrame, prefix: str) -> str:
        """Format by-class localization summary for one support definition."""
        summary = dataframe.groupby("class").agg(
            median_centroid_offset_mm=(
                f"{prefix}_euclidean_offset_mm",
                "median",
            ),
            p95_centroid_offset_mm=(
                f"{prefix}_euclidean_offset_mm",
                lambda values: values.quantile(0.95),
            ),
            median_required_center_width_mm=(
                f"{prefix}_required_center_width_mm",
                "median",
            ),
            median_required_center_height_mm=(
                f"{prefix}_required_center_height_mm",
                "median",
            ),
        )
        return summary.round(2).to_string()

    def _parse_info_file(self, info_path: Path) -> dict[str, str]:
        """Read key-value metadata from one ACDC Info.cfg file."""
        metadata: dict[str, str] = {}

        with info_path.open("r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split(":", maxsplit=1)
                metadata[key.strip()] = value.strip()

        return metadata

    def _spatial_spacing(
        self,
        image: nib.spatialimages.SpatialImage,
    ) -> tuple[float, float, float]:
        """Read image spatial voxel spacing."""
        return tuple(float(value) for value in image.header.get_zooms()[:3])

    def _append_unique(self, selected: list[str], patient_ids: list[str]) -> None:
        """Append patient IDs while preserving order and uniqueness."""
        seen = set(selected)
        for patient_id in patient_ids:
            patient_id = str(patient_id)
            if patient_id not in seen:
                selected.append(patient_id)
                seen.add(patient_id)
