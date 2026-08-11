"""Compact deterministic motion-localization analysis for real ACDC cine MRI."""

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("p95", 95.0),
    ("p975", 97.5),
    ("p99", 99.0),
)
METHODS: tuple[tuple[str, str], ...] = (
    ("tv", "Temporal variance"),
    ("edes", "ED-ES difference"),
)


class MotionLocalizationAnalyzer:
    """Compare compact 2D motion-localization candidates on real ACDC data."""

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
        """Run the motion-localization experiment and save CSV/QA artifacts."""
        patient_dirs = sorted(self.dataset_dir.glob("patient*"))
        if not patient_dirs:
            raise RuntimeError(f"No patient directories found in {self.dataset_dir}")

        records = [self._analyze_patient(patient_dir) for patient_dir in patient_dirs]
        dataframe = pd.DataFrame(records)
        dataframe = self._add_cross_method_agreement(dataframe)
        dataframe = self._add_threshold_stability(dataframe)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_csv, index=False)

        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.generated_qa_patient_ids = self._generate_qa_figures(dataframe)

        return dataframe

    def print_summary(self, dataframe: pd.DataFrame) -> None:
        """Print concise cohort summaries for the compact localization analysis."""
        print("Compact motion localization")
        print("\nTemporal variance")
        for threshold_name, _threshold in THRESHOLDS:
            print(f"\n{threshold_name}:")
            print(self._cohort_summary(dataframe, f"tv_{threshold_name}"))

        print("\nED-ES difference")
        for threshold_name, _threshold in THRESHOLDS:
            print(f"\n{threshold_name}:")
            print(self._cohort_summary(dataframe, f"edes_{threshold_name}"))

        print("\nCross-method centroid agreement")
        for threshold_name, _threshold in THRESHOLDS:
            column = f"{threshold_name}_centroid_agreement_mm"
            print(f"\n{threshold_name}:")
            print(self._six_number_summary(dataframe[column]))
            for threshold_mm in [10, 20, 30, 40]:
                print(
                    f"  > {threshold_mm} mm: "
                    f"{int((dataframe[column] > threshold_mm).sum())} / "
                    f"{len(dataframe)}"
                )

        print("\nThreshold stability")
        for method_name, method_label in METHODS:
            print(f"\n{method_label}:")
            columns = [
                f"{method_name}_centroid_displacement_p95_to_p975_mm",
                f"{method_name}_centroid_displacement_p975_to_p99_mm",
                f"{method_name}_centroid_displacement_p95_to_p99_mm",
            ]
            renamed = {
                columns[0]: "95 -> 97.5",
                columns[1]: "97.5 -> 99",
                columns[2]: "95 -> 99",
            }
            print(
                dataframe[columns]
                .rename(columns=renamed)
                .agg(["median", self._p95, "max"])
                .rename(index={"_p95": "p95"})
                .round(2)
                .to_string()
            )

        print("\np97.5 by-class summary")
        for method_name, method_label in METHODS:
            print(f"\n{method_label}:")
            print(self._by_class_summary(dataframe, f"{method_name}_p975"))

        print("\nGenerated QA patient IDs")
        for patient_id in self.generated_qa_patient_ids:
            print(f"  {patient_id}")

    def _analyze_patient(self, patient_dir: Path) -> dict[str, Any]:
        """Analyze one patient using temporal-variance and ED-ES energy maps."""
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
        temporal_variance = np.var(cine_array, axis=3, dtype=np.float64)
        temporal_energy_xy = temporal_variance.sum(axis=2, dtype=np.float64)
        absolute_difference = np.abs(
            ed_array.astype(np.float64) - es_array.astype(np.float64)
        )
        edes_energy_xy = absolute_difference.sum(axis=2, dtype=np.float64)

        record: dict[str, Any] = {
            "patient_id": patient_id,
            "class": group,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "size_x": ed_image.shape[0],
            "size_y": ed_image.shape[1],
            "fov_x_mm": ed_image.shape[0] * spacing_x,
            "fov_y_mm": ed_image.shape[1] * spacing_y,
            "spacing_x_mm": spacing_x,
            "spacing_y_mm": spacing_y,
        }
        record.update(
            self._measure_energy_map(
                energy_xy=temporal_energy_xy,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                column_prefix="tv",
                patient_id=patient_id,
            )
        )
        record.update(
            self._measure_energy_map(
                energy_xy=edes_energy_xy,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                column_prefix="edes",
                patient_id=patient_id,
            )
        )

        return record

    def _measure_energy_map(
        self,
        energy_xy: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        column_prefix: str,
        patient_id: str,
    ) -> dict[str, float | int]:
        """Measure selected largest-energy connected components in one 2D map."""
        positive_energy = energy_xy[energy_xy > 0]
        if positive_energy.size == 0:
            raise ValueError(f"{patient_id}: {column_prefix} map has no positive energy")

        measurements: dict[str, float | int] = {}
        total_map_energy = float(energy_xy.sum(dtype=np.float64))
        for threshold_name, percentile in THRESHOLDS:
            threshold = np.percentile(positive_energy, percentile)
            component_mask = self._largest_energy_component(
                energy_xy=energy_xy,
                threshold=threshold,
            )
            prefix = f"{column_prefix}_{threshold_name}"
            measurements.update(
                self._measure_component(
                    energy_xy=energy_xy,
                    component_mask=component_mask,
                    spacing_x=spacing_x,
                    spacing_y=spacing_y,
                    total_map_energy=total_map_energy,
                    prefix=prefix,
                )
            )

        return measurements

    def _largest_energy_component(
        self,
        energy_xy: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Select the 8-connected threshold component with largest energy sum."""
        threshold_mask = energy_xy >= threshold
        structure = np.ones((3, 3), dtype=int)
        labels, component_count = ndimage.label(threshold_mask, structure=structure)

        if component_count == 0:
            raise ValueError("No connected components found after thresholding")

        component_energies = ndimage.sum(
            energy_xy,
            labels=labels,
            index=np.arange(1, component_count + 1),
        )
        selected_label = int(np.argmax(component_energies) + 1)

        return labels == selected_label

    def _measure_component(
        self,
        energy_xy: np.ndarray,
        component_mask: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        total_map_energy: float,
        prefix: str,
    ) -> dict[str, float | int]:
        """Measure one selected 2D motion-energy component."""
        size_x, size_y = energy_xy.shape
        center_x_vox = (size_x - 1) / 2
        center_y_vox = (size_y - 1) / 2
        component_x, component_y = np.nonzero(component_mask)
        weights = energy_xy[component_mask]
        component_energy = float(weights.sum(dtype=np.float64))
        component_area_pixels = int(component_mask.sum())
        centroid_x_vox = float(np.average(component_x, weights=weights))
        centroid_y_vox = float(np.average(component_y, weights=weights))
        min_x_vox = int(component_x.min())
        max_x_vox = int(component_x.max())
        min_y_vox = int(component_y.min())
        max_y_vox = int(component_y.max())
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
            f"{prefix}_component_area_pixels": component_area_pixels,
            f"{prefix}_component_area_mm2": component_area_pixels
            * spacing_x
            * spacing_y,
            f"{prefix}_component_area_fraction": component_area_pixels
            / float(size_x * size_y),
            f"{prefix}_component_energy": component_energy,
            f"{prefix}_total_map_energy": total_map_energy,
            f"{prefix}_component_energy_fraction": component_energy
            / total_map_energy,
            f"{prefix}_centroid_x_vox": centroid_x_vox,
            f"{prefix}_centroid_y_vox": centroid_y_vox,
            f"{prefix}_offset_x_mm": offset_x_mm,
            f"{prefix}_offset_y_mm": offset_y_mm,
            f"{prefix}_euclidean_offset_mm": float(np.hypot(offset_x_mm, offset_y_mm)),
            f"{prefix}_bbox_min_x_vox": min_x_vox,
            f"{prefix}_bbox_max_x_vox": max_x_vox,
            f"{prefix}_bbox_min_y_vox": min_y_vox,
            f"{prefix}_bbox_max_y_vox": max_y_vox,
            f"{prefix}_bbox_width_pixels": max_x_vox - min_x_vox + 1,
            f"{prefix}_bbox_height_pixels": max_y_vox - min_y_vox + 1,
            f"{prefix}_bbox_width_mm": (max_x_vox - min_x_vox) * spacing_x,
            f"{prefix}_bbox_height_mm": (max_y_vox - min_y_vox) * spacing_y,
            f"{prefix}_required_center_width_mm": 2 * required_half_width_mm,
            f"{prefix}_required_center_height_mm": 2 * required_half_height_mm,
        }

    def _add_cross_method_agreement(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add centroid distance between TV and ED-ES components per threshold."""
        dataframe = dataframe.copy()
        for threshold_name, _threshold in THRESHOLDS:
            dx_mm = (
                dataframe[f"tv_{threshold_name}_centroid_x_vox"]
                - dataframe[f"edes_{threshold_name}_centroid_x_vox"]
            ) * dataframe["spacing_x_mm"]
            dy_mm = (
                dataframe[f"tv_{threshold_name}_centroid_y_vox"]
                - dataframe[f"edes_{threshold_name}_centroid_y_vox"]
            ) * dataframe["spacing_y_mm"]
            dataframe[f"{threshold_name}_centroid_agreement_mm"] = np.hypot(
                dx_mm,
                dy_mm,
            )

        return dataframe

    def _add_threshold_stability(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add centroid displacement across thresholds for each method."""
        dataframe = dataframe.copy()
        for method_name, _method_label in METHODS:
            for start, end, label in [
                ("p95", "p975", "p95_to_p975"),
                ("p975", "p99", "p975_to_p99"),
                ("p95", "p99", "p95_to_p99"),
            ]:
                dx_mm = (
                    dataframe[f"{method_name}_{start}_centroid_x_vox"]
                    - dataframe[f"{method_name}_{end}_centroid_x_vox"]
                ) * dataframe["spacing_x_mm"]
                dy_mm = (
                    dataframe[f"{method_name}_{start}_centroid_y_vox"]
                    - dataframe[f"{method_name}_{end}_centroid_y_vox"]
                ) * dataframe["spacing_y_mm"]
                dataframe[
                    f"{method_name}_centroid_displacement_{label}_mm"
                ] = np.hypot(dx_mm, dy_mm)

        return dataframe

    def _generate_qa_figures(self, dataframe: pd.DataFrame) -> list[str]:
        """Generate comparison QA figures for selected patients."""
        selected_patient_ids = self._select_qa_patient_ids(dataframe)
        for patient_id in selected_patient_ids:
            self._generate_patient_qa(patient_id)

        return selected_patient_ids

    def _select_qa_patient_ids(self, dataframe: pd.DataFrame) -> list[str]:
        """Select QA patients from disagreement, instability, and class median cases."""
        selected: list[str] = []

        largest_disagreement = dataframe.nlargest(5, "p975_centroid_agreement_mm")
        self._append_unique(selected, largest_disagreement["patient_id"].tolist())

        stability_columns = [
            "tv_centroid_displacement_p95_to_p975_mm",
            "tv_centroid_displacement_p975_to_p99_mm",
            "tv_centroid_displacement_p95_to_p99_mm",
            "edes_centroid_displacement_p95_to_p975_mm",
            "edes_centroid_displacement_p975_to_p99_mm",
            "edes_centroid_displacement_p95_to_p99_mm",
        ]
        dataframe = dataframe.assign(
            max_threshold_instability_mm=dataframe[stability_columns].max(axis=1)
        )
        largest_instability = dataframe.nlargest(5, "max_threshold_instability_mm")
        self._append_unique(selected, largest_instability["patient_id"].tolist())

        median_agreement = dataframe["p975_centroid_agreement_mm"].median()
        for class_name in sorted(dataframe["class"].unique()):
            class_dataframe = dataframe[dataframe["class"] == class_name].copy()
            class_dataframe["agreement_distance_to_median"] = (
                class_dataframe["p975_centroid_agreement_mm"] - median_agreement
            ).abs()
            patient_id = str(
                class_dataframe.sort_values("agreement_distance_to_median")
                .iloc[0]["patient_id"]
            )
            self._append_unique(selected, [patient_id])

        return selected

    def _generate_patient_qa(self, patient_id: str) -> None:
        """Create one six-panel diagnostic motion-localization QA PNG."""
        patient_dir = self.dataset_dir / patient_id
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        es_frame = int(metadata["ES"])
        cine_image = nib.load(patient_dir / f"{patient_id}_4d.nii.gz")
        ed_image = nib.load(patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz")
        es_image = nib.load(patient_dir / f"{patient_id}_frame{es_frame:02d}.nii.gz")
        cine_array = np.asanyarray(cine_image.dataobj)
        ed_array = np.asanyarray(ed_image.dataobj)
        es_array = np.asanyarray(es_image.dataobj)
        spacing_x, spacing_y, _spacing_z = self._spatial_spacing(ed_image)
        temporal_variance = np.var(cine_array, axis=3, dtype=np.float64)
        tv_energy_xy = temporal_variance.sum(axis=2, dtype=np.float64)
        edes_energy_xy = np.abs(
            ed_array.astype(np.float64) - es_array.astype(np.float64)
        ).sum(axis=2, dtype=np.float64)
        tv_component = self._largest_energy_component(
            tv_energy_xy,
            np.percentile(tv_energy_xy[tv_energy_xy > 0], 97.5),
        )
        edes_component = self._largest_energy_component(
            edes_energy_xy,
            np.percentile(edes_energy_xy[edes_energy_xy > 0], 97.5),
        )
        tv_measurements = self._measure_component(
            tv_energy_xy,
            tv_component,
            spacing_x,
            spacing_y,
            float(tv_energy_xy.sum(dtype=np.float64)),
            "tv_p975",
        )
        edes_measurements = self._measure_component(
            edes_energy_xy,
            edes_component,
            spacing_x,
            spacing_y,
            float(edes_energy_xy.sum(dtype=np.float64)),
            "edes_p975",
        )

        z_index = ed_array.shape[2] // 2
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = [
            ("ED mid-Z image", ed_array[:, :, z_index], "gray", None),
            ("ES mid-Z image", es_array[:, :, z_index], "gray", None),
            ("Temporal-variance XY energy", tv_energy_xy, "magma", None),
            ("Selected TV p97.5 component", tv_energy_xy, "magma", tv_component),
            ("ED-ES XY energy", edes_energy_xy, "viridis", None),
            ("Selected ED-ES p97.5 component", edes_energy_xy, "viridis", edes_component),
        ]

        for axis, (title, image, cmap, component) in zip(
            axes.ravel(),
            panels,
            strict=True,
        ):
            axis.imshow(np.rot90(image), cmap=cmap)
            if component is not None:
                axis.imshow(
                    np.rot90(np.ma.masked_where(~component, component)),
                    cmap="autumn",
                    alpha=0.45,
                )
            axis.set_title(title)
            self._draw_qa_markers(
                axis=axis,
                size_x=ed_array.shape[0],
                size_y=ed_array.shape[1],
                tv_measurements=tv_measurements,
                edes_measurements=edes_measurements,
                include_tv=title.startswith("Selected TV"),
                include_edes=title.startswith("Selected ED-ES"),
            )
            axis.set_axis_off()

        figure.suptitle(f"{patient_id} diagnostic motion-localization QA")
        output_path = self.qa_dir / f"{patient_id}_motion_localization_qa.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _draw_qa_markers(
        self,
        axis: plt.Axes,
        size_x: int,
        size_y: int,
        tv_measurements: dict[str, float | int],
        edes_measurements: dict[str, float | int],
        include_tv: bool,
        include_edes: bool,
    ) -> None:
        """Draw FOV center, centroids, and optional component bounding boxes."""
        # The display uses np.rot90, so original y maps to inverted display y.
        if include_tv:
            self._draw_component_box(axis, size_y, tv_measurements, "tv_p975", "cyan")
            self._draw_centroid(axis, size_y, tv_measurements, "tv_p975", "cyan", "x")
        if include_edes:
            self._draw_component_box(axis, size_y, edes_measurements, "edes_p975", "lime")
            self._draw_centroid(axis, size_y, edes_measurements, "edes_p975", "lime", "x")

        fov_center_x = (size_x - 1) / 2
        fov_center_y = (size_y - 1) / 2
        axis.scatter(
            fov_center_x,
            size_y - 1 - fov_center_y,
            marker="+",
            c="white",
            s=90,
            linewidths=2,
        )

    def _draw_component_box(
        self,
        axis: plt.Axes,
        size_y: int,
        measurements: dict[str, float | int],
        prefix: str,
        color: str,
    ) -> None:
        """Draw a selected component bounding box after rot90 display."""
        min_x = float(measurements[f"{prefix}_bbox_min_x_vox"])
        max_x = float(measurements[f"{prefix}_bbox_max_x_vox"])
        min_y = float(measurements[f"{prefix}_bbox_min_y_vox"])
        max_y = float(measurements[f"{prefix}_bbox_max_y_vox"])
        axis.add_patch(
            Rectangle(
                (min_x, size_y - 1 - max_y),
                max_x - min_x,
                max_y - min_y,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )

    def _draw_centroid(
        self,
        axis: plt.Axes,
        size_y: int,
        measurements: dict[str, float | int],
        prefix: str,
        color: str,
        marker: str,
    ) -> None:
        """Draw a component centroid after rot90 display."""
        centroid_x = float(measurements[f"{prefix}_centroid_x_vox"])
        centroid_y = float(measurements[f"{prefix}_centroid_y_vox"])
        axis.scatter(
            centroid_x,
            size_y - 1 - centroid_y,
            marker=marker,
            c=color,
            s=75,
            linewidths=2,
        )

    def _cohort_summary(self, dataframe: pd.DataFrame, prefix: str) -> str:
        """Format cohort summary for one method and threshold."""
        columns = [
            f"{prefix}_component_area_fraction",
            f"{prefix}_component_energy_fraction",
            f"{prefix}_bbox_width_mm",
            f"{prefix}_bbox_height_mm",
            f"{prefix}_euclidean_offset_mm",
            f"{prefix}_required_center_width_mm",
            f"{prefix}_required_center_height_mm",
        ]
        renamed = {
            f"{prefix}_component_area_fraction": "component_area_fraction",
            f"{prefix}_component_energy_fraction": "component_energy_fraction",
            f"{prefix}_bbox_width_mm": "bbox_width_mm",
            f"{prefix}_bbox_height_mm": "bbox_height_mm",
            f"{prefix}_euclidean_offset_mm": "centroid_offset_mm",
            f"{prefix}_required_center_width_mm": "required_center_width_mm",
            f"{prefix}_required_center_height_mm": "required_center_height_mm",
        }
        return (
            dataframe[columns]
            .rename(columns=renamed)
            .describe(percentiles=[0.05, 0.5, 0.95])
            .loc[["min", "5%", "50%", "mean", "95%", "max"]]
            .rename(index={"5%": "p05", "50%": "median", "95%": "p95"})
            .round(4)
            .to_string()
        )

    def _six_number_summary(self, series: pd.Series) -> str:
        """Format min/p05/median/mean/p95/max for one series."""
        summary = pd.Series(
            {
                "min": series.min(),
                "p05": series.quantile(0.05),
                "median": series.median(),
                "mean": series.mean(),
                "p95": series.quantile(0.95),
                "max": series.max(),
            }
        )
        return summary.round(2).to_string()

    def _by_class_summary(self, dataframe: pd.DataFrame, prefix: str) -> str:
        """Format p97.5 localization summary by diagnostic class."""
        summary = dataframe.groupby("class").agg(
            median_centroid_offset_mm=(f"{prefix}_euclidean_offset_mm", "median"),
            median_bbox_width_mm=(f"{prefix}_bbox_width_mm", "median"),
            median_bbox_height_mm=(f"{prefix}_bbox_height_mm", "median"),
            median_component_area_fraction=(
                f"{prefix}_component_area_fraction",
                "median",
            ),
            median_component_energy_fraction=(
                f"{prefix}_component_energy_fraction",
                "median",
            ),
        )
        return summary.round(4).to_string()

    def _validate_images(
        self,
        patient_id: str,
        cine_image: nib.spatialimages.SpatialImage,
        ed_image: nib.spatialimages.SpatialImage,
        es_image: nib.spatialimages.SpatialImage,
        ed_index: int,
        es_index: int,
    ) -> None:
        """Validate raw cine and standalone ED/ES dimensions and frame indices."""
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
                    f"{patient_id}: {frame_name} index {frame_index} outside cine "
                    f"time dimension {time_points}"
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

    def _p95(self, values: pd.Series) -> float:
        """Return the 95th percentile for aggregation."""
        return float(values.quantile(0.95))
